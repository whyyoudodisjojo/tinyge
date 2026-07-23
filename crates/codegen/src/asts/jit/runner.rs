use wgpu::{
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, Device, Queue, ShaderStages,
};

use tinyge_graphics::shaders::{
    ComputeShader, ComputeShaderBuiltData,
    buffers::ResourceType,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};

use crate::asts::jit::JitAST;
use crate::asts::lowered::{
    Accessor, BindingMeta, CustomBufferBindingType, EntrypointData, EntrypointGlobals, Functions,
    LoweredAST, ShaderIR, VarRef, VarRefType,
    renderer::LoweredRenderer,
    scope::{Scope, entrypoint, local},
};
use crate::dt::DType;

pub struct JitRunner<'ast> {
    ast: &'ast JitAST,
    element_count: u32,
    num_vars: usize,
    input_dt: DType,
    output_dt: DType,
    output_size: u64,
}

impl<'ast> JitRunner<'ast> {
    pub fn new(ast: &'ast JitAST, element_count: u32) -> Self {
        let (num_vars, input_dt) = ast.collect_var_info();
        let input_dt = input_dt.expect("JitAST must have at least one Var");
        let output_dt = ast.dt();
        let output_size = (output_dt.byte_size() * element_count as usize) as u64;
        Self {
            ast,
            element_count,
            num_vars,
            input_dt,
            output_dt,
            output_size,
        }
    }

    fn build_shader_ir(
        ast: &JitAST,
        num_vars: usize,
        input_dt: &DType,
        output_dt: &DType,
    ) -> ShaderIR {
        let mut ir = ShaderIR {
            structs: crate::asts::build_struct_map(),
            binded: vec![],
            shared_vars: vec![],
            private_vars: vec![],
            entrypoint_globals: vec![],
            functions: vec![],
        };

        for i in 0..num_vars {
            ir.binded.push(BindingMeta {
                ident: format!("input_{}", i),
                ty: CustomBufferBindingType::Storage { read_only: true },
                dtype: input_dt.clone(),
            });
        }
        ir.binded.push(BindingMeta {
            ident: "output".to_string(),
            ty: CustomBufferBindingType::Storage { read_only: false },
            dtype: output_dt.as_array_dtype(),
        });

        ir.entrypoint_globals = vec![EntrypointGlobals::GlobalInvocationId];

        let mut scope = Scope::new();
        let idx = scope.var(entrypoint(0).f("x").load());
        let mut var_counter = 0;
        let body = JitAST::lower_with_rewrite(
            ast.clone(),
            &mut scope,
            &mut || {
                let binding = var_counter;
                var_counter += 1;
                LoweredAST::Load(VarRefType::Global(VarRef {
                    id: binding,
                    by: vec![Accessor::Index(Box::new(local(idx).load()))],
                }))
            },
            &[],
        );

        let output_store = match body {
            LoweredAST::Group(mut stmts) => {
                let result = stmts.pop().unwrap();
                stmts.push(LoweredAST::Store {
                    var: VarRefType::Global(VarRef {
                        id: num_vars,
                        by: vec![Accessor::Index(Box::new(local(idx).load()))],
                    }),
                    val: Box::new(result),
                });
                LoweredAST::Group(stmts)
            }
            _ => LoweredAST::Store {
                var: VarRefType::Global(VarRef {
                    id: num_vars,
                    by: vec![Accessor::Index(Box::new(local(idx).load()))],
                }),
                val: Box::new(body),
            },
        };
        scope.ast = Some(output_store);

        ir.functions.push(Functions {
            args: vec![],
            ret: None,
            ident: "jit_main".to_string(),
            entrypoint_ty: Some(EntrypointData::Compute { workgroup_sz: 256 }),
            body: scope,
        });

        ir
    }
}

impl<'a> ComputeShader<'a> for JitRunner<'_> {
    type Args = Vec<Buffer>;
    type Ret = Buffer;

    fn entry_point(&self) -> &'static str {
        "jit_main"
    }

    fn load_source_code(&self) -> String {
        let ir = Self::build_shader_ir(self.ast, self.num_vars, &self.input_dt, &self.output_dt);
        LoweredRenderer { ir: &ir }.translate()
    }

    fn resource_buffers_with_bind_group_layouts(&self) -> Vec<ResourceGroupLayout<'a>> {
        let mut entries = Vec::with_capacity(self.num_vars + 1);
        for i in 0..self.num_vars {
            entries.push(ResourceBinding {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: ResourceBindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                    size: 0,
                    usages: BufferUsages::STORAGE,
                    is_input: true,
                },
                count: None,
            });
        }
        entries.push(ResourceBinding {
            binding: self.num_vars as u32,
            visibility: ShaderStages::COMPUTE,
            ty: ResourceBindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
                size: self.output_size,
                usages: BufferUsages::STORAGE,
                is_input: false,
            },
            count: None,
        });
        vec![ResourceGroupLayout { entries }]
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        built_data: &mut ComputeShaderBuiltData<'a>,
        device: &Device,
        queue: &Queue,
    ) -> Self::Ret {
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("jit_output"),
            size: self.output_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut resources: Vec<ResourceType> = args.into_iter().map(ResourceType::Buffer).collect();
        resources.push(ResourceType::Buffer(output_buffer.clone()));

        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(&resources, device);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("jit_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("jit_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&built_data.pipeline);
            pass.set_bind_group(0, Some(bind_group), &[]);
            pass.dispatch_workgroups((self.element_count + 255) / 256, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        output_buffer
    }
}

impl JitAST {
    pub fn realize(&self, device: &Device, queue: &Queue, element_count: u32) -> JitAST {
        let mut bufs = vec![];
        self.collect_var_buffers(&mut bufs);
        let input_bufs: Vec<Buffer> = bufs.into_iter().cloned().collect();

        let mut runner = JitRunner::new(self, element_count);
        let mut built_data = runner.build(device);
        let output = runner.dispatch(input_bufs, &mut built_data, device, queue);

        JitAST::Var {
            buffer: output,
            dtype: self.dt(),
        }
    }
}
