use memory::{
    buffers::{DynamicBindGroup, ResourceType},
    socket::TinyBuffer,
};
use wgpu::{
    BindGroupLayoutDescriptor, BufferBindingType, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, Queue,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use tinyge_graphics::shaders::{
    ComputeShader, ComputeShaderBuiltData,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};

use crate::asts::jit::JitAST;
use crate::asts::lowered::ASTOrConst;
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
                dtype: input_dt.peel_array().as_array_dtype(),
            });
        }
        ir.binded.push(BindingMeta {
            ident: "output".to_string(),
            ty: CustomBufferBindingType::Storage { read_only: false },
            dtype: output_dt.as_array_dtype(),
        });

        ir.entrypoint_globals = vec![EntrypointGlobals::GlobalInvocationId];

        let var_strides = compute_var_strides(ast);

        let mut scope = Scope::new();
        let idx = scope.var(entrypoint(0).f("x").load());
        let mut var_counter = 0;
        let mut var_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let body = JitAST::lower_with_rewrite(
            ast.clone(),
            &mut scope,
            &mut |var_id: usize| {
                let binding = if let Some(&b) = var_map.get(&var_id) {
                    b
                } else {
                    let b = var_counter;
                    var_counter += 1;
                    var_map.insert(var_id, b);
                    b
                };
                let stride = var_strides.get(&var_id).copied().unwrap_or(1);
                if stride > 1 {
                    LoweredAST::Load(VarRefType::Global(VarRef {
                        id: binding,
                        by: vec![],
                    }))
                } else {
                    LoweredAST::Load(VarRefType::Global(VarRef {
                        id: binding,
                        by: vec![Accessor::Index(Box::new(local(idx).load()))],
                    }))
                }
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

#[derive(Clone)]
pub struct JitArgs(pub Vec<TinyBuffer>);

impl<'a> From<memory::buffers::UnifiedShaderBuildData<'a>> for JitArgs {
    fn from(data: memory::buffers::UnifiedShaderBuildData<'a>) -> Self {
        let mut bufs = data.vertex_buffers;
        bufs.extend(data.index_buffers);
        for group in data.resource_groups {
            bufs.extend(group.buffers);
        }
        JitArgs(bufs)
    }
}

impl<'a> ComputeShader<'a> for JitRunner<'_> {
    type Args = JitArgs;
    type Ret = TinyBuffer;

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
                    usages: BufferUsages::STORAGE | BufferUsages::COPY_DST,
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
                usages: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            },
            count: None,
        });
        vec![ResourceGroupLayout { entries }]
    }

    fn build(&self, device: &Device) -> ComputeShaderBuiltData<Self::Args> {
        let resource_buffer_descs = self.resource_buffers_with_bind_group_layouts();

        let bind_group_layouts = resource_buffer_descs
            .iter()
            .map(|l| {
                let bind_group_layout_descriptor = BindGroupLayoutDescriptor {
                    label: None,
                    entries: &l.entries.iter().map(Into::into).collect::<Vec<_>>(),
                };
                device.create_bind_group_layout(&bind_group_layout_descriptor)
            })
            .collect::<Vec<_>>();

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &bind_group_layouts
                .iter()
                .map(|b| Some(b))
                .collect::<Vec<_>>(),
            immediate_size: 0,
        });

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(self.load_source_code())),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader_module,
            entry_point: Some(self.entry_point()),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_groups = bind_group_layouts
            .into_iter()
            .map(DynamicBindGroup::new)
            .collect();

        let mut var_bufs = vec![];
        self.ast.collect_var_buffers(&mut var_bufs);
        let mut buffers: Vec<TinyBuffer> = var_bufs.into_iter().cloned().collect();

        let ResourceBindingType::Buffer { usages, .. } =
            &resource_buffer_descs[0].entries[self.num_vars].ty
        else {
            unreachable!()
        };
        let mut output = TinyBuffer::new(self.output_size, *usages);
        output.build(device);
        buffers.push(output);

        ComputeShaderBuiltData {
            bind_groups,
            pipeline,
            buffers: JitArgs(buffers),
        }
    }

    fn dispatch(
        &mut self,
        mut args: Self::Args,
        built_data: &mut ComputeShaderBuiltData<Self::Args>,
        device: &Device,
        queue: &Queue,
    ) -> Self::Ret {
        let output_buf = args.0.pop().unwrap();
        let mut resources: Vec<ResourceType> =
            args.0.into_iter().map(ResourceType::Buffer).collect();
        resources.push(ResourceType::Buffer(output_buf.clone()));

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

        output_buf
    }
}

fn compute_var_strides(ast: &JitAST) -> std::collections::HashMap<usize, u32> {
    let mut strides = std::collections::HashMap::new();
    compute_var_strides_inner(ast, &mut strides);
    strides
}

fn compute_var_strides_inner(ast: &JitAST, strides: &mut std::collections::HashMap<usize, u32>) {
    match ast {
        JitAST::Var { id, dtype, buffer } => {
            if strides.contains_key(id) {
                return;
            }
            let scalar_byte_size = dtype.peel_all().byte_size() as u64;
            let element_byte_size = dtype.byte_size() as u64;
            let num_elements = buffer.size / element_byte_size;
            let stride = if num_elements > 1 {
                (element_byte_size / scalar_byte_size) as u32
            } else {
                1
            };
            strides.insert(*id, stride);
        }
        JitAST::BinOp { lhs, rhs, .. } => {
            compute_var_strides_inner(lhs, strides);
            compute_var_strides_inner(rhs, strides);
        }
        JitAST::UnaryOp { operand, .. }
        | JitAST::Cast { operand, .. }
        | JitAST::Movement { operand, .. }
        | JitAST::Reduce { operand, .. }
        | JitAST::AllReduce { operand, .. } => {
            compute_var_strides_inner(operand, strides);
        }
        JitAST::Ternary { a, b, c, .. } => {
            compute_var_strides_inner(a, strides);
            compute_var_strides_inner(b, strides);
            compute_var_strides_inner(c, strides);
        }
        JitAST::Const(c) => {
            for d in &c.data {
                if let ASTOrConst::AST(a) = d {
                    compute_var_strides_inner(a, strides);
                }
            }
        }
        JitAST::Lowered { .. } => {}
    }
}

impl JitAST {
    pub fn realize(&self, device: &Device, queue: &Queue, element_count: u32) -> JitAST {
        let mut runner = JitRunner::new(self, element_count);
        let mut built_data = runner.build(device);

        println!("{}", runner.load_source_code());

        let args = built_data.buffers.clone();
        let output = runner.dispatch(args, &mut built_data, device, queue);

        JitAST::Var {
            id: usize::MAX,
            buffer: output,
            dtype: self.dt(),
        }
    }
}
