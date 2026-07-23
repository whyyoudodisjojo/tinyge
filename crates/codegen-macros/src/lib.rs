use codegen::asts::lowered::{CustomBufferBindingType, EntrypointData};
use darling::{FromDeriveInput, FromField, FromMeta, ast::Data};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{DeriveInput, FnArg, Ident, Meta, Type, parse_macro_input};

use darling::ast::NestedMeta;

#[derive(darling::FromMeta)]
struct ComputeArgs {
    workgroup_sz: usize,
}

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

#[proc_macro_attribute]
pub fn shader(attr: TokenStream, item: TokenStream) -> TokenStream {
    let meta: Meta = syn::parse(attr).unwrap();
    let ty = match meta {
        Meta::List(list) if list.path.is_ident("compute") => {
            let nested = NestedMeta::parse_meta_list(list.tokens).unwrap();
            let args = ComputeArgs::from_list(&nested).unwrap();
            EntrypointData::Compute {
                workgroup_sz: args.workgroup_sz,
            }
        }
        Meta::Path(path) if path.is_ident("shader") => EntrypointData::Shader,
        _ => panic!("expected compute(...) or shader"),
    };

    let func = parse_macro_input!(item as syn::ItemFn);

    let ident = &func.sig.ident;

    let pascal = ident
        .to_string()
        .split('_')
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().to_string() + c.as_str(),
            }
        })
        .collect::<String>();
    let struct_ident = format_ident!("{}", pascal);
    let args_ident = format_ident!("{}Args", struct_ident);

    let args: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|input| {
            if let FnArg::Typed(pat) = input {
                let name = if let syn::Pat::Ident(ident) = &*pat.pat {
                    ident.ident.to_string()
                } else {
                    panic!("expected named argument");
                };
                let b = pat
                    .attrs
                    .iter()
                    .find(|a| a.path().is_ident("binding"))
                    .and_then(|a| FromMeta::from_meta(&a.meta).ok())?;
                Some((name, b, &pat.ty))
            } else {
                None
            }
        })
        .collect();

    let shared_args: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|input| {
            if let FnArg::Typed(pat) = input {
                let name = if let syn::Pat::Ident(ident) = &*pat.pat {
                    ident.ident.to_string()
                } else {
                    panic!("expected named argument");
                };
                if let Type::Path(p) = &*pat.ty {
                    if let Some(seg) = p.path.segments.last() {
                        if seg.ident == "SharedData" {
                            Some((name, &pat.ty))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    let extra_args: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|input| {
            let FnArg::Typed(pat) = input else {
                return None;
            };
            let name = if let syn::Pat::Ident(ident) = &*pat.pat {
                ident.ident.to_string()
            } else {
                panic!("expected named argument");
            };
            if pat
                .attrs
                .iter()
                .any(|a| a.path().is_ident("binding") || a.path().is_ident("private"))
            {
                return None;
            }
            if let Type::Path(p) = &*pat.ty {
                if let Some(seg) = p.path.segments.last() {
                    if seg.ident == "SharedData" {
                        return None;
                    }
                }
            }
            Some((name, &pat.ty))
        })
        .collect();

    let private_args: Vec<_> = func
        .sig
        .inputs
        .iter()
        .filter_map(|input| {
            let FnArg::Typed(pat) = input else {
                return None;
            };
            let name = if let syn::Pat::Ident(ident) = &*pat.pat {
                ident.ident.to_string()
            } else {
                return None;
            };
            if !pat.attrs.iter().any(|a| a.path().is_ident("private")) {
                return None;
            }
            Some((name, &pat.ty))
        })
        .collect();

    let (private_arg_names, private_arg_tys): (Vec<_>, Vec<_>) = private_args
        .iter()
        .map(|(n, ty)| (n.clone(), ty.as_ref().clone()))
        .unzip();

    let (shared_arg_names, shared_arg_inner_types): (Vec<_>, Vec<_>) = shared_args
        .iter()
        .map(|(n, ty)| {
            let Type::Path(p) = &***ty else {
                panic!("expected SharedData<T>, got {}", quote! { #ty })
            };
            let seg = p.path.segments.last().unwrap();
            assert!(
                seg.ident == "SharedData",
                "expected SharedData, got {}",
                quote! { #ty }
            );
            let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
                panic!("expected SharedData<T>, got {}", quote! { #ty })
            };
            let syn::GenericArgument::Type(inner) = args.args.first().unwrap() else {
                panic!("expected SharedData<T>, got {}", quote! { #ty })
            };
            (n.clone(), inner.clone())
        })
        .unzip();

    let shared_arg_markers: Vec<_> = shared_arg_inner_types
        .iter()
        .enumerate()
        .map(|(i, inner_ty)| {
            let idx = i;
            quote! { codegen::asts::lowered::SharedData::<#inner_ty>::new(#idx) }
        })
        .collect();

    let (arg_names, arg_inner_types): (Vec<_>, Vec<_>) = args
        .iter()
        .map(|(n, _, ty)| {
            let inner_ty = {
                let Type::Path(p) = &***ty else {
                    panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
                };
                let seg = p.path.segments.last().unwrap();
                assert!(
                    seg.ident == "BindedBuffer",
                    "expected BindedBuffer, got {}",
                    quote! { #ty }
                );
                let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
                    panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
                };
                let syn::GenericArgument::Type(inner) = args.args.first().unwrap() else {
                    panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
                };
                inner.clone()
            };
            (n.clone(), inner_ty)
        })
        .unzip();

    let arg_n_idents: Vec<_> = args
        .iter()
        .map(|(n, _, _)| Ident::new(n, ident.span()))
        .collect();

    let arg_struct_f = arg_n_idents
        .iter()
        .zip(arg_inner_types.clone())
        .map(|(n, ty)| {
            quote! {
                pub #n : tinyge_graphics::shaders::buffers::BufferWithType<#ty>
            }
        });

    let extra_arg_tys: Vec<_> = extra_args.iter().map(|(_, ty)| *ty).collect();
    let extra_arg_n_idents: Vec<_> = extra_args
        .iter()
        .map(|(n, _)| Ident::new(n, ident.span()))
        .collect();

    let private_arg_n_idents: Vec<_> = private_args
        .iter()
        .map(|(n, _)| Ident::new(n, ident.span()))
        .collect();

    let mut extra_struct_f: Vec<_> = extra_arg_n_idents
        .iter()
        .zip(extra_arg_tys.iter())
        .map(|(n, ty)| {
            quote! {
                pub #n : #ty
            }
        })
        .collect();

    let mut clean_func = func.clone();
    clean_func.sig.inputs = clean_func
        .sig
        .inputs
        .into_iter()
        .map(|input| match input {
            FnArg::Typed(mut pat) => {
                pat.attrs
                    .retain(|a| !a.path().is_ident("binding") && !a.path().is_ident("private"));
                FnArg::Typed(pat)
            }
            other => other,
        })
        .collect();

    let arg_markers: Vec<_> = args.iter().enumerate().map(|(i, (_, _, ty))| {
        let idx = syn::Index::from(i);
        let Type::Path(p) = &***ty else { panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty }) };
        let seg = p.path.segments.last().unwrap();
        assert!(seg.ident == "BindedBuffer", "expected BindedBuffer, got {}", quote! { #ty });
        let syn::PathArguments::AngleBracketed(args) = &seg.arguments else { panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty }) };
        let syn::GenericArgument::Type(inner) = args.args.first().unwrap() else { panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty }) };
        quote! { codegen::asts::lowered::BindedBuffer::<#inner, #idx>(std::marker::PhantomData) }
    }).collect();

    let extra_arg_self_refs: Vec<_> = extra_arg_n_idents
        .iter()
        .map(|n| {
            quote! { self.#n }
        })
        .collect();

    for (n, ty) in &private_args {
        let field_name = format_ident!("{}", n);
        extra_struct_f.push(quote! { pub #field_name: #ty });
    }

    let arg_binding_tys: Vec<_> = args.iter().map(|(_, b, _)| {
        match b {
            CustomBufferBindingType::Uniform => {
                quote! { codegen::asts::lowered::CustomBufferBindingType::Uniform }
            }
            CustomBufferBindingType::Storage { read_only } => {
                quote! { codegen::asts::lowered::CustomBufferBindingType::Storage { read_only: #read_only } }
            }
        }
    }).collect();

    let arg_group_layout: Vec<_> = args
        .iter()
        .enumerate()
        .map(|(i, (n, b, ty))| {
            let binding_ty = match b {
                CustomBufferBindingType::Uniform => {
                    quote! { wgpu::BufferBindingType::Uniform }
                }
                CustomBufferBindingType::Storage { read_only } => {
                    quote! { wgpu::BufferBindingType::Storage { read_only: #read_only } }
                }
            };
            let buffer_usages = match b {
                CustomBufferBindingType::Uniform => {
                    quote! { wgpu::BufferUsages::UNIFORM }
                }
                CustomBufferBindingType::Storage { .. } => {
                    quote! { wgpu::BufferUsages::STORAGE }
                }
            };
            let i_u32 = i as u32;

            let sz = if let Type::Path(p) = ty.as_ref() {
                let seg = p.path.segments.last().unwrap();
                let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
                    panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
                };
                let syn::GenericArgument::Type(inner) = args.args.first().unwrap() else {
                    panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
                };
                if let Type::Path(inner_p) = inner
                    && inner_p
                        .path
                        .segments
                        .last()
                        .map(|s| s.ident == "Vec")
                        .unwrap_or_default()
                {
                    let size_f_name = format_ident!("{n}_elem_count");
                    extra_struct_f.push(quote! { pub #size_f_name: u64 });
                    quote! { self.#size_f_name * std::mem::size_of::<#inner>() as u64 }
                } else {
                    quote! { std::mem::size_of::<#ty>() as u64 }
                }
            } else {
                quote! { std::mem::size_of::<#ty>() as u64 }
            };

            quote! {
                tinyge_graphics::shaders::descriptors::ResourceBinding {
                    binding: #i_u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: tinyge_graphics::shaders::descriptors::ResourceBindingType::Buffer {
                        ty: #binding_ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: #sz,
                        usages: #buffer_usages,
                        is_input: true,
                    },
                    count: None,
                }
            }
        })
        .collect::<Vec<_>>();
    let struct_def = if extra_struct_f.is_empty() {
        quote! { pub struct #struct_ident; }
    } else {
        quote! { pub struct #struct_ident { #(#extra_struct_f,)* } }
    };
    match ty {
        EntrypointData::Shader => todo!("vertex/fragment shader not yet supported"),
        EntrypointData::Compute { workgroup_sz } => {
            let func_clean = &clean_func;
            quote! {
                #func_clean

                #struct_def

                pub struct #args_ident {
                    #(#arg_struct_f,)*
                }

                impl<'a> tinyge_graphics::shaders::ComputeShader<'a> for #struct_ident {
                    type Args = #args_ident;
                    type Ret = ();

                    fn entry_point(&self) -> &'static str {
                        stringify!(#ident)
                    }

                    fn load_source_code(&self) -> String {
                        let structs = codegen::asts::build_struct_map();

                        let mut ir = codegen::asts::lowered::ShaderIR {
                            structs,
                            binded: vec![],
                            shared_vars: vec![],
                            private_vars: vec![],
                            entrypoint_globals: vec![],
                            functions: vec![],
                        };

                        ir.binded = vec![
                            #(codegen::asts::lowered::BindingMeta {
                                ident: #arg_names.to_string(),
                                ty: #arg_binding_tys,
                                dtype: <#arg_inner_types as codegen::asts::IntoWgslStruct>::dt(),
                            },)*
                        ];

                        ir.shared_vars = vec![
                            #((
                                #shared_arg_names.to_string(),
                                <#shared_arg_inner_types as codegen::asts::IntoWgslStruct>::dt(),
                            ),)*
                        ];

                        ir.private_vars = vec![
                            #((
                                #private_arg_names.to_string(),
                                <#private_arg_tys as codegen::asts::IntoWgslStruct>::dt(),
                            ),)*
                        ];

                        ir.entrypoint_globals = vec![
                            codegen::asts::lowered::EntrypointGlobals::GlobalInvocationId,
                            codegen::asts::lowered::EntrypointGlobals::LocalInvocationId,
                        ];

                        let scope = #ident(#(#arg_markers,)* #(#shared_arg_markers,)* #(#extra_arg_self_refs,)* #(self.#private_arg_n_idents),*);

                        ir.functions.push(
                            codegen::asts::lowered::Functions {
                                args: vec![
                                    #((#arg_names.to_string(), <#arg_inner_types as codegen::asts::IntoWgslStruct>::dt()),)*
                                ],
                                ret: None,
                                ident: stringify!(#ident).to_string(),
                                entrypoint_ty: Some(codegen::asts::lowered::EntrypointData::Compute { workgroup_sz: #workgroup_sz }),
                                body: scope,
                            },
                        );
                        codegen::asts::lowered::renderer::LoweredRenderer { ir: &ir }.translate()
                    }

                    fn resource_buffers_with_bind_group_layouts(
                        &self,
                    ) -> Vec<tinyge_graphics::shaders::descriptors::ResourceGroupLayout<'a>> {
                        vec![
                            tinyge_graphics::shaders::descriptors::ResourceGroupLayout {
                                entries: vec![#(#arg_group_layout,)*],
                            },
                        ]
                    }

                    fn dispatch(
                        &mut self,
                        args: Self::Args,
                        built_data: &mut tinyge_graphics::shaders::ComputeShaderBuiltData<'a>,
                        device: &wgpu::Device,
                        queue: &wgpu::Queue,
                    ) -> Self::Ret {
                        let mut encoder = device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: None }
                        );
                        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(
                            &[#(tinyge_graphics::shaders::buffers::ResourceType::Buffer(args.#arg_n_idents.inner),)*],
                            device,
                        );
                        {
                            let mut pass = encoder.begin_compute_pass(
                                &wgpu::ComputePassDescriptor {
                                    label: None,
                                    timestamp_writes: None,
                                }
                            );
                            pass.set_pipeline(&built_data.pipeline);
                            pass.set_bind_group(0, Some(bind_group), &[]);
                            pass.dispatch_workgroups(#workgroup_sz as u32, 1, 1);
                        }
                        queue.submit(std::iter::once(encoder.finish()));
                    }
                }
            }
        },
    }.into()
}

#[test]
fn test_any_shit() {
    #[repr(C)]
    struct Test<T> {
        x: u32,
        _p: std::marker::PhantomData<T>,
    }

    let t: Test<u32> = Test {
        x: 10,
        _p: std::marker::PhantomData,
    };

    let ptr = &t as *const Test<u32> as *const Test<f32>;
    let new: &Test<f32> = unsafe { &*ptr };

    println!("{}", new.x);
}
