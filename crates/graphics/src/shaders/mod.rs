pub mod descriptors;

use memory::{
    buffers::{DynamicBindGroup, UnifiedShaderBuildData},
    descriptors::{
        MeshBufferSpecs, ResourceGroupLayout, ShaderPipelineDescriptor, VertexBufferSpec,
    },
};
use wgpu::*;

pub struct ShaderBuiltData<T> {
    pub pipeline: RenderPipeline,
    pub bind_groups: Vec<DynamicBindGroup>,
    pub buffers: T,
}

pub struct ComputeShaderBuiltData<T> {
    pub buffers: T,
    pub bind_groups: Vec<DynamicBindGroup>,
    pub pipeline: ComputePipeline,
}

pub trait Shader<'a> {
    type Args: From<UnifiedShaderBuildData<'a>>;
    fn mesh_buffers_layouts(&self) -> MeshBufferSpecs<'a> {
        MeshBufferSpecs::default()
    }
    fn resource_buffers_with_bind_group_layouts(&self) -> Vec<ResourceGroupLayout<'a>> {
        vec![]
    }
    fn load_source_code(&self) -> String;
    fn shader_pipeline_desc(&self) -> ShaderPipelineDescriptor<'_>;

    fn build(
        &mut self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
        prev_build_data: Option<ShaderBuiltData<Self::Args>>,
    ) -> ShaderBuiltData<Self::Args> {
        let mesh_buffer_specs = self.mesh_buffers_layouts();
        let vertex_layouts = mesh_buffer_specs
            .vertex_buffers
            .iter()
            .cloned()
            .map(|VertexBufferSpec { layout, .. }| layout)
            .collect::<Vec<_>>();

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

        let desc = self.shader_pipeline_desc();

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
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: desc.vertex_entry_point,
                compilation_options: desc.vertex_compilation_options,
                buffers: &vertex_layouts,
            },
            primitive: desc.primitive_state,
            depth_stencil: desc.depth_stencil,
            multisample: desc.multisample,
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: desc.fragment_entry_point,
                compilation_options: desc.fragment_compilation_options,
                targets: &desc
                    .fragment_targets
                    .into_iter()
                    .map(|t| {
                        t.as_ref().map(|t| ColorTargetState {
                            format: *texture_format,
                            blend: t.blend,
                            write_mask: t.write_mask,
                        })
                    })
                    .collect::<Vec<_>>(),
            }),
            multiview_mask: desc.multiview_mask,
            cache,
        });

        let bind_groups = bind_group_layouts
            .into_iter()
            .map(DynamicBindGroup::new)
            .collect();

        let buffers = match prev_build_data {
            Some(prev) => prev.buffers,
            None => {
                UnifiedShaderBuildData::new(&resource_buffer_descs, Some(&mesh_buffer_specs)).into()
            }
        };

        ShaderBuiltData {
            buffers,
            pipeline,
            bind_groups,
        }
    }
}

pub struct ShaderWrapper<S, T>
where
    S: for<'a> Shader<'a>,
{
    pub built_data: Option<ShaderBuiltData<T>>,
    pub shader: S,
}

impl<S> ShaderWrapper<S, <S as Shader<'_>>::Args>
where
    S: for<'a> Shader<'a>,
{
    pub fn new(shader: S) -> Self {
        Self {
            built_data: None,
            shader,
        }
    }

    pub fn build(
        &mut self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) -> bool {
        let buffers_rebuild = self.built_data.is_none();
        self.built_data =
            Some(
                self.shader
                    .build(device, texture_format, cache, self.built_data.take()),
            );
        buffers_rebuild
    }
}

pub struct ComputeShaderWrapper<S, T> {
    built_data: Option<ComputeShaderBuiltData<T>>,
    pub inner: S,
}

impl<S, T> ComputeShaderWrapper<S, T> {
    pub fn built_data(&self) -> &ComputeShaderBuiltData<T> {
        self.built_data.as_ref().unwrap()
    }

    pub fn built_data_mut(&mut self) -> &mut ComputeShaderBuiltData<T> {
        self.built_data.as_mut().unwrap()
    }
}

impl<'a, S> ComputeShaderWrapper<S, S::Args>
where
    S: ComputeShader<'a>,
{
    pub fn new(shader: S, device: &Device) -> Self {
        let buffers = shader.build(device, None);
        Self {
            built_data: Some(buffers),
            inner: shader,
        }
    }

    pub fn recompile(&mut self, device: &Device) {
        let buffer_build_spec = self.inner.build(device, self.built_data.take());
        self.built_data = Some(buffer_build_spec);
    }

    pub fn dispatch(&mut self, args: S::Args, device: &Device, queue: &Queue) -> S::Ret {
        self.inner
            .dispatch(args, self.built_data.as_mut().unwrap(), device, queue)
    }
}

pub trait ComputeShader<'a> {
    type Args: From<UnifiedShaderBuildData<'a>>;
    type Ret;

    fn resource_buffers_with_bind_group_layouts(&self) -> Vec<ResourceGroupLayout<'a>> {
        vec![]
    }
    fn load_source_code(&self) -> String;
    fn entry_point(&self) -> &'static str;

    fn build(
        &self,
        device: &Device,
        prev_build_data: Option<ComputeShaderBuiltData<Self::Args>>,
    ) -> ComputeShaderBuiltData<Self::Args> {
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

        let buffers = match prev_build_data {
            Some(prev) => prev.buffers,
            None => {
                let mut data = UnifiedShaderBuildData::new(&resource_buffer_descs, None);
                for buf in &mut data.vertex_buffers {
                    buf.build(device);
                }
                for buf in &mut data.index_buffers {
                    buf.build(device);
                }
                for group in &mut data.resource_groups {
                    for buf in &mut group.buffers {
                        buf.build(device);
                    }
                }
                data.into()
            }
        };

        ComputeShaderBuiltData {
            bind_groups,
            pipeline,
            buffers,
        }
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        build_data: &mut ComputeShaderBuiltData<Self::Args>,
        device: &Device,
        queue: &Queue,
    ) -> Self::Ret;
}
