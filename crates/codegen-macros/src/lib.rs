use darling::{FromDeriveInput, FromField, ast::Data};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    DeriveInput, Expr, GenericArgument, Ident, Lit, PathArguments, Type, TypePath,
    parse_macro_input,
};

#[derive(FromField)]
#[darling(attributes(codegen))]
struct Field {
    ident: Option<Ident>,
    ty: Type,
    atomic: darling::util::Flag,
}

#[derive(FromDeriveInput)]
#[darling(supports(struct_named))]
struct StructInput {
    ident: Ident,
    data: Data<(), Field>,
}

#[proc_macro_derive(IntoWgslStruct, attributes(codegen))]
pub fn derive_into_wgsl_struct(item: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(item as DeriveInput);
    let struct_input = StructInput::from_derive_input(&parsed).unwrap();

    let struct_name = &struct_input.ident;
    let fields = struct_input.data.take_struct().unwrap().fields;

    let field_insertions: Vec<_> = fields
        .iter()
        .map(|f| {
            let field_name = f.ident.as_ref().unwrap().to_string();
            let is_atomic = f.atomic.is_present();
            
            if is_atomic {
                assert!(is_integer_type(&f.ty), 
                    "#[codegen(atomic)] can only be used with integer types (u32, i32), found: {:?}", 
                    f.ty);
            }
            
            let dtype_tokens = parse_type_to_dtype(&f.ty, is_atomic);

            quote! {
                fields.insert(#field_name.to_string(), #dtype_tokens);
            }
        })
        .collect();

    let struct_ident = struct_name.to_string();

    let output = quote! {
        impl From<#struct_name> for (String, codegen::asts::lowered::Struct) {
            fn from(_item: #struct_name) -> Self {
                use codegen::dt::{BasicTy, DType, IntegerTy, MaybeAtomic, VecTy};
                let mut fields = std::collections::HashMap::new();
                #(#field_insertions)*
                let s = codegen::asts::lowered::Struct {
                    inner: fields,
                };
                (#struct_ident.to_string(), s)
            }
        }

        impl codegen::asts::IntoWgslStruct for #struct_name {}
    };

    output.into()
}

fn is_integer_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            let path = &type_path.path;
            if let Some(last_seg) = path.segments.last() {
                let ident = last_seg.ident.to_string();
                matches!(ident.as_str(), "u32" | "i32")
            } else {
                false
            }
        }
        _ => false,
    }
}

fn parse_type_to_dtype(ty: &Type, is_atomic: bool) -> proc_macro2::TokenStream {
    match ty {
        Type::Array(type_array) => {
            let elem_ty = &type_array.elem;

            let elem_dtype = parse_array_inner_type(elem_ty, false);

            let Expr::Lit(syn::ExprLit {
                lit: Lit::Int(len), ..
            }) = &type_array.len
            else {
                panic!("const arr requried");
            };

            let len_val = len.base10_parse::<usize>().unwrap();
            assert!(
                len_val == 2 || len_val == 3,
                "array length err got {}",
                len_val
            );

            let vec_ident = format_ident!("Vec{}", len_val);
            quote! {
                DType::Vector(VecTy::#vec_ident(#elem_dtype))
            }
        }
        Type::Path(type_path) => parse_path_type_to_dtype(type_path, is_atomic),
        _ => panic!("{:?}", ty),
    }
}

fn parse_array_inner_type(ty: &Type, is_atomic: bool) -> proc_macro2::TokenStream {
    match ty {
        Type::Path(type_path) => {
            let path = &type_path.path;
            let last_seg = path.segments.last().unwrap();
            let ident = last_seg.ident.to_string();
            match ident.as_str() {
                "f32" => quote! { BasicTy::F32 },
                "u32" => {
                    if is_atomic {
                        quote! { BasicTy::Integer(IntegerTy::U32) }
                    } else {
                        quote! { BasicTy::Integer(IntegerTy::U32) }
                    }
                }
                "i32" => {
                    if is_atomic {
                        quote! { BasicTy::Integer(IntegerTy::I32) }
                    } else {
                        quote! { BasicTy::Integer(IntegerTy::I32) }
                    }
                }
                _ => panic!(
                    "array got: {}",
                    ident
                ),
            }
        }
        _ => panic!("got: {:?}", ty),
    }
}

fn parse_path_type_to_dtype(type_path: &TypePath, is_atomic: bool) -> proc_macro2::TokenStream {
    let path = &type_path.path;
    let last_seg = path.segments.last().unwrap();
    let ident = &last_seg.ident;

    match ident.to_string().as_str() {
        "f32" => {
            if is_atomic {
                panic!(
                    "atomic<f32> foubd atomics can only be used with integer types (u32, i32)"
                );
            }
            quote! {
                DType::Basic(BasicTy::F32)
            }
        }
        "u32" => {
            if is_atomic {
                quote! {
                    DType::Atomic(IntegerTy::U32)
                }
            } else {
                quote! {
                    DType::Basic(BasicTy::Integer(IntegerTy::U32))
                }
            }
        }
        "i32" => {
            if is_atomic {
                quote! {
                    DType::Atomic(IntegerTy::I32)
                }
            } else {
                quote! {
                    DType::Basic(BasicTy::Integer(IntegerTy::I32))
                }
            }
        }
        "Vec" => {
            let inner = extract_generic_type(&last_seg.arguments);
            let inner_tokens = parse_vec_inner_type(inner);

            quote! {
                DType::Vector(VecTy::Array(MaybeAtomic::Naked(#inner_tokens)))
            }
        }
        _ => {
            let type_ident = format_ident!("{}", ident);
            quote! {{
                struct _AssertIntoWgslStruct where #type_ident: codegen::asts::IntoWgslStruct;
                DType::StructRef { ident: stringify!(#type_ident).to_string() }
            }}
        }
    }
}

fn parse_vec_inner_type(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Path(type_path) => {
            let path = &type_path.path;
            let last_seg = path.segments.last().unwrap();
            let ident = last_seg.ident.to_string();
            match ident.as_str() {
                "f32" => quote! { BasicTy::F32 },
                "u32" => quote! { BasicTy::Integer(IntegerTy::U32) },
                "i32" => quote! { BasicTy::Integer(IntegerTy::I32) },
                _ => panic!("Vec inner type must be f32, u32, or i32, got: {}", ident),
            }
        }
        _ => panic!("Expected path type for Vec inner type, got: {:?}", ty),
    }
}

fn extract_generic_type(args: &PathArguments) -> &Type {
    if let PathArguments::AngleBracketed(bracketed) = args {
        if let Some(GenericArgument::Type(ty)) = bracketed.args.first() {
            return ty;
        }
    }
    panic!("Generic failed ig");
}
