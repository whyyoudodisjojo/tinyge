use quote::{format_ident, quote};
use syn::{Expr, GenericArgument, Lit, PathArguments, Type, TypePath};

pub fn parse_type_to_dtype(ty: &Type, is_atomic: bool) -> proc_macro2::TokenStream {
    match ty {
        Type::Array(type_array) => {
            let elem_ty = &type_array.elem;
            let elem_dtype = parse_basic_ty(elem_ty);

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

fn parse_basic_ty(ty: &Type) -> proc_macro2::TokenStream {
    match ty {
        Type::Path(type_path) => {
            let path = &type_path.path;
            let last_seg = path.segments.last().unwrap();
            let ident = last_seg.ident.to_string();
            match ident.as_str() {
                "f32" => quote! { BasicTy::F32 },
                "u32" => quote! { BasicTy::Integer(IntegerTy::U32) },
                "i32" => quote! { BasicTy::Integer(IntegerTy::I32) },
                _ => panic!("got: {}", ident),
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
                panic!("atomic<f32> found, atomics can only be used with integer types (u32, i32)");
            }
            quote! { DType::Basic(BasicTy::F32) }
        }
        "u32" => {
            if is_atomic {
                quote! { DType::Atomic(IntegerTy::U32) }
            } else {
                quote! { DType::Basic(BasicTy::Integer(IntegerTy::U32)) }
            }
        }
        "i32" => {
            if is_atomic {
                quote! { DType::Atomic(IntegerTy::I32) }
            } else {
                quote! { DType::Basic(BasicTy::Integer(IntegerTy::I32)) }
            }
        }
        "Vec" => {
            let inner = extract_generic_type(&last_seg.arguments);
            let inner_tokens = parse_basic_ty(inner);

            quote! {
                DType::Vector(VecTy::Array(MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(#inner_tokens))))
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

fn extract_generic_type(args: &PathArguments) -> &Type {
    if let PathArguments::AngleBracketed(bracketed) = args {
        if let Some(GenericArgument::Type(ty)) = bracketed.args.first() {
            return ty;
        }
    }
    panic!("Generic failed ig");
}
