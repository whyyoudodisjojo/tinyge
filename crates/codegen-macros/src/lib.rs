mod struct_parser;

use darling::{FromDeriveInput, FromField, ast::Data};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Ident, Type, parse_macro_input};

use crate::struct_parser::parse_type_to_dtype;

#[derive(FromField)]
#[darling(attributes(codegen))]
struct Field {
    ident: Option<Ident>,
    ty: Type,
    atomic: darling::util::Flag,
    pad: Option<usize>,
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
        .filter_map(|f| {
            if f.pad.is_some() {
                return None;
            }

            let field_name = f.ident.as_ref().unwrap().to_string();
            let is_atomic = f.atomic.is_present();

            let dtype_tokens = parse_type_to_dtype(&f.ty, is_atomic);

            Some(quote! {
                fields.push((#field_name.to_string(), #dtype_tokens));
            })
        })
        .collect();

    let struct_ident = struct_name.to_string();
    let make_fn_ident = quote::format_ident!("__make_struct_for_{}", struct_ident);

    let output = quote! {
        #[allow(non_snake_case)]
        fn #make_fn_ident() -> (String, Vec<(String, codegen::dt::DType)>) {
            use codegen::dt::{BasicTy, BasicTyOrStructRef, DType, IntegerTy, MaybeAtomic, VecTy};
            let mut fields = Vec::new();
            #(#field_insertions)*
            (#struct_ident.to_string(), fields)
        }

        impl codegen::asts::IntoWgslStruct for #struct_name {
            fn dt() -> (String, codegen::asts::lowered::Struct) {
                let (name, fields) = #make_fn_ident();
                (name, codegen::asts::lowered::Struct { inner: fields })
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
