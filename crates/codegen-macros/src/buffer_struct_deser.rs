use darling::{FromMeta, ast::NestedMeta};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Error, parse_macro_input, spanned::Spanned};

#[derive(Debug, FromMeta)]
#[darling(rename_all = "snake_case")]
enum ResourceType {
    Buffer,
    Texture,
    Sampler,
    AccelerationStructure,
}

#[derive(Debug, FromMeta)]
struct ResourceArgs {
    bindgroup_index: usize,
    resource_index: usize,
    ty: ResourceType,
}

#[derive(Debug, FromMeta)]
#[darling(rename_all = "lowercase")]
enum FieldAttr {
    Vertex,
    Index,
    Resource(ResourceArgs),
}

impl TryFrom<&syn::Attribute> for FieldAttr {
    type Error = Error;
    fn try_from(value: &syn::Attribute) -> Result<Self, Self::Error> {
        let ident = value
            .path()
            .get_ident()
            .ok_or_else(|| Error::new(value.span(), "expected ident"))?;
        match ident.to_string().as_str() {
            "vertex" => Ok(FieldAttr::Vertex),
            "index" => Ok(FieldAttr::Index),
            "resource" => {
                let meta_list = match &value.meta {
                    syn::Meta::List(list) => list,
                    _ => return Err(Error::new(value.span(), "expected resource(...)")),
                };
                let nested = NestedMeta::parse_meta_list(meta_list.tokens.clone())?;
                ResourceArgs::from_list(&nested)
                    .map(FieldAttr::Resource)
                    .map_err(Error::from)
            }
            _ => Err(Error::new(
                value.span(),
                "expected vertex, index, or resource",
            )),
        }
    }
}

pub fn buffer_struct_deser(item: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(item as DeriveInput);

    let fields = match &parsed.data {
        syn::Data::Struct(syn::DataStruct {
            fields: syn::Fields::Named(named),
            ..
        }) => &named.named,
        _ => unreachable!(),
    };

    let mut field_inits: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut vertex_idx = 0usize;
    let mut index_idx = 0usize;

    for field in fields {
        let ident = field.ident.as_ref().expect("named field");

        for attr in &field.attrs {
            if let Ok(fa) = FieldAttr::try_from(attr) {
                match fa {
                    FieldAttr::Vertex => {
                        field_inits.push(
                            quote! { #ident: value.vertex_buffers[#vertex_idx].clone().into() },
                        );
                        vertex_idx += 1;
                    }
                    FieldAttr::Index => {
                        field_inits.push(
                            quote! { #ident: value.index_buffers[#index_idx].clone().into() },
                        );
                        index_idx += 1;
                    }
                    FieldAttr::Resource(args) => {
                        let bg = args.bindgroup_index;
                        let ri = args.resource_index;
                        let access = match args.ty {
                            ResourceType::Buffer => {
                                quote! { value.resource_groups[#bg].buffers[#ri].clone().into() }
                            }
                            ResourceType::Texture => {
                                quote! { value.resource_groups[#bg].textures[#ri].clone().into() }
                            }
                            ResourceType::Sampler => {
                                quote! { value.resource_groups[#bg].samplers[#ri].clone().into() }
                            }
                            ResourceType::AccelerationStructure => {
                                quote! { value.resource_groups[#bg].acceleration_structures[#ri].clone().into() }
                            }
                        };
                        field_inits.push(quote! { #ident: #access });
                    }
                }
            }
        }
    }

    let name = &parsed.ident;
    let generics = &parsed.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let lifetimes: Vec<_> = parsed.generics.lifetimes().map(|l| &l.lifetime).collect();
    let uni_ty = if lifetimes.is_empty() {
        quote! { memory::buffers::UnifiedShaderBuildData<'_, wgpu::Buffer> }
    } else {
        quote! { memory::buffers::UnifiedShaderBuildData<#(#lifetimes,)* wgpu::Buffer> }
    };

    let output = if parsed.generics.params.is_empty() {
        quote! {
            impl From<#uni_ty> for #name {
                fn from(value: #uni_ty) -> #name {
                    #name {
                        #(#field_inits,)*
                    }
                }
            }
        }
    } else {
        quote! {
            impl #impl_generics From<#uni_ty> for #name #ty_generics #where_clause {
                fn from(value: #uni_ty) -> #name #ty_generics {
                    #name {
                        #(#field_inits,)*
                    }
                }
            }
        }
    };

    output.into()
}
