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

    let output = quote! {
        impl From<#struct_name> for (String, codegen::asts::lowered::Struct) {
            fn from(_item: #struct_name) -> Self {
                use codegen::dt::{BasicTy, BasicTyOrStructRef, DType, IntegerTy, MaybeAtomic, VecTy};
                let mut fields = Vec::new();
                #(#field_insertions)*

                let mut result = Vec::new();
                let mut prev_dtype: Option<DType> = None;
                let mut pad_counter = 0usize;

                for (name, dtype) in fields {
                    if let Some(ref prev) = prev_dtype {
                        let padding_needed = codegen::asts::lowered::Struct::required_padding(prev, &dtype);
                        if padding_needed > 0 {
                            let pad_name = format!("__pad_{}", pad_counter);
                            result.push((pad_name, DType::Pad(padding_needed)));
                            pad_counter += 1;
                        }
                    }

                    result.push((name, dtype.clone()));
                    prev_dtype = Some(dtype);
                }

                let s = codegen::asts::lowered::Struct {
                    inner: result,
                };
                (#struct_ident.to_string(), s)
            }
        }

        impl codegen::asts::IntoWgslStruct for #struct_name {}
    };

    output.into()
}
