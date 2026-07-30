use darling::{FromDeriveInput, FromField, ast::Data};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Ident, Type, parse_macro_input};

#[derive(FromField)]
#[darling(attributes(codegen))]
struct Field {
    ident: Option<Ident>,
    ty: Type,
}

#[derive(FromDeriveInput)]
#[darling(supports(struct_named))]
struct StructInput {
    ident: Ident,
    data: Data<(), Field>,
}

pub fn derive_into_wgsl_struct_inner(item: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(item as DeriveInput);
    let struct_input = StructInput::from_derive_input(&parsed).unwrap();

    let struct_name = &struct_input.ident;
    let fields = struct_input.data.take_struct().unwrap().fields;

    let field_insertions: Vec<_> = fields
        .iter()
        .map(|f| {
            let field_name = f.ident.as_ref().unwrap().to_string();
            let field_ty = &f.ty;
            if let Type::Path(p) = field_ty && p.path.segments.last().map(|s| s.ident == "Vec").unwrap_or_default() {
                panic!("runtime-sized Vec<T> not supported in struct; use fixed-size arrays")
            }

            quote! {
                fields.push((#field_name.to_string(), <#field_ty as codegen::asts::IntoWgslStruct>::dt()));
            }
        })
        .collect();

    let struct_ident = struct_name.to_string();
    let make_fn_ident = quote::format_ident!("__make_struct_for_{}", struct_ident);

    let output = quote! {
        #[allow(non_snake_case)]
        fn #make_fn_ident() -> codegen::asts::lowered::Struct {
            let mut fields = Vec::new();
            #(#field_insertions)*
            codegen::asts::lowered::Struct { name: #struct_ident.to_string(), inner: fields }
        }

        impl codegen::asts::IntoWgslStruct for #struct_name {
            fn dt() -> codegen::dt::DType {
                codegen::dt::DType::StructRef { ident: #struct_ident.to_string() }
            }
        }

        codegen::inventory::submit! {
            codegen::asts::WgslStructFactory {
                name: stringify!(#struct_name),
                make: #make_fn_ident,
            }
        }
    };

    output.into()
}
