use codegen::asts::lowered::{CustomBufferBindingType, EntrypointData};
use darling::{FromMeta, ast::NestedMeta};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{FnArg, Ident, Meta, Type, parse_macro_input};

#[derive(darling::FromMeta)]
struct ComputeArgs {
    workgroup_sz: usize,
}

pub fn shader_inner(attr: TokenStream, item: TokenStream) -> TokenStream {
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
                let binding_attr = pat.attrs.iter().find(|a| a.path().is_ident("binding"))?;

                let Meta::List(list) = &binding_attr.meta else {
                    return None;
                };
                let items = NestedMeta::parse_meta_list(list.tokens.clone()).ok()?;

                let mut ty_meta = None;
                for item in &items {
                    match item {
                        NestedMeta::Meta(meta) => {
                            ty_meta = Some(meta.clone());
                        }
                        _ => {}
                    }
                }

                let ty_meta = ty_meta?;
                let b: CustomBufferBindingType =
                    FromMeta::from_list(&[NestedMeta::Meta(ty_meta)]).ok()?;
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
        .enumerate()
        .map(|(i, (n, ty))| {
            let ri = syn::Index::from(i);
            quote! {
                #[resource(bindgroup_index = 0, resource_index = #ri, ty = "buffer")]
                pub #n : memory::buffers::BufferWithType<#ty>
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

            let Type::Path(p) = ty.as_ref() else {
                panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
            };
            let seg = p.path.segments.last().unwrap();
            let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
                panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
            };
            let syn::GenericArgument::Type(inner) = args.args.first().unwrap() else {
                panic!("expected BindedBuffer<T, N>, got {}", quote! { #ty })
            };
            let size_f_name = format_ident!("{n}_elem_count");
            let is_storage_vec = match inner {
                Type::Path(p)
                    if p.path
                        .segments
                        .last()
                        .map(|s| s.ident == "Vec")
                        .unwrap_or_default() =>
                {
                    extra_struct_f.push(quote! { pub #size_f_name: u64 });
                    true
                }
                _ => false,
            };
            let sz = if is_storage_vec {
                quote! { self.#size_f_name * <#inner as codegen::asts::IntoWgslStruct>::wgsl_byte_size() }
            } else {
                quote! { <#inner as codegen::asts::IntoWgslStruct>::wgsl_byte_size() }
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

                #[derive(codegen_macros::IntoBufferStruct)]
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
                        built_data: &mut tinyge_graphics::shaders::ComputeShaderBuiltData<Self::Args>,
                        device: &wgpu::Device,
                        queue: &wgpu::Queue,
                    ) -> Self::Ret {
                        let mut encoder = device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: None }
                        );
                        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(
                            &[#(memory::buffers::ResourceType::Buffer(args.#arg_n_idents.inner),)*],
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
