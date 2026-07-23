pub mod buffers;
pub mod descriptors;
pub mod manager;

use std::sync::Arc;

use memory::{
    buffers::{BufferBuildSpec, DynamicBindGroup, ResourceGroupBuildSpec},
    descriptors::{
        MeshBufferSpecs, ResourceGroupLayout, ShaderPipelineDescriptor, VertexBufferSpec,
    },
};
use wgpu::*;

pub struct ShaderBuiltData<'a> {
    pub pipeline: RenderPipeline,
    pub buffer_build_spec: BufferBuildSpec<'a>,
    pub bind_groups: Vec<DynamicBindGroup>,
}

pub struct ComputeShaderBuiltData<'a> {
    pub buffer_build_spec: BufferBuildSpec<'a>,
    pub bind_groups: Vec<DynamicBindGroup>,
    pub pipeline: ComputePipeline,
}

pub trait Shader<'a> {
    fn mesh_buffers_layouts(&self) -> MeshBufferSpecs<'a> {
        MeshBufferSpecs::default()
    }
    fn resource_buffers_with_bind_group_layouts(&self) -> Vec<ResourceGroupLayout<'a>> {
        vec![]
    }
    fn load_source_code(&self) -> String;
    fn shader_pipeline_desc(&self) -> ShaderPipelineDescriptor<'_>;

    fn build(
        &self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) -> ShaderBuiltData<'a> {
        let MeshBufferSpecs {
            vertex_buffers: vertex_layouts,
            index_buffer_size,
        } = self.mesh_buffers_layouts();
        let (vertex_layouts, vertex_buffer_sizes) = vertex_layouts
            .into_iter()
            .map(|VertexBufferSpec { layout, size }| (layout, size))
            .collect::<(Vec<_>, Vec<_>)>();

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

        let build_spec = BufferBuildSpec {
            vertex_buffer_szs: vertex_buffer_sizes,
            index_buffer_sz: index_buffer_size,
            resource_buffer: resource_buffer_descs
                .into_iter()
                .zip(bind_group_layouts)
                .map(|(d, l)| ResourceGroupBuildSpec {
                    layout: l,
                    layout_entries: d.entries,
                })
                .collect(),
        };

        let bind_groups = DynamicBindGroup::new_from_buffer_spec(&build_spec);

        ShaderBuiltData {
            pipeline,
            buffer_build_spec: build_spec,
            bind_groups,
        }
    }
}

pub struct ShaderWrapper<'a, S> {
    pub buffer_build_spec: Option<ShaderBuiltData<'a>>,
    pub inner: S,
}

impl<'a, S> ShaderWrapper<'a, S>
where
    S: Shader<'a>,
{
    pub fn new(
        shader: S,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) -> Self {
        let buffer_build_spec = shader.build(device, texture_format, cache);
        Self {
            buffer_build_spec: Some(buffer_build_spec),
            inner: shader,
        }
    }

    pub fn recompile(
        &mut self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) {
        let buffer_build_spec = self.inner.build(device, texture_format, cache);
        self.buffer_build_spec = Some(buffer_build_spec);
    }
}

impl<'a, S: ?Sized> Shader<'a> for Arc<S>
where
    S: Shader<'a>,
{
    fn build(
        &self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) -> ShaderBuiltData<'a> {
        self.as_ref().build(device, texture_format, cache)
    }

    fn load_source_code(&self) -> String {
        self.as_ref().load_source_code()
    }

    fn mesh_buffers_layouts(&self) -> MeshBufferSpecs<'a> {
        self.as_ref().mesh_buffers_layouts()
    }

    fn resource_buffers_with_bind_group_layouts(&self) -> Vec<ResourceGroupLayout<'a>> {
        self.as_ref().resource_buffers_with_bind_group_layouts()
    }

    fn shader_pipeline_desc(&self) -> ShaderPipelineDescriptor<'_> {
        self.as_ref().shader_pipeline_desc()
    }
}

pub struct ComputeShaderWrapper<'a, S> {
    pub buffer_build_spec: ComputeShaderBuiltData<'a>,
    pub inner: S,
}

impl<'a, S> ComputeShaderWrapper<'a, S>
where
    S: ComputeShader<'a>,
{
    pub fn new(shader: S, device: &Device) -> Self {
        let spec = shader.build(device);

        Self {
            buffer_build_spec: spec,
            inner: shader,
        }
    }

    pub fn recompile(&mut self, device: &Device) {
        let buffer_build_spec = self.inner.build(device);
        self.buffer_build_spec = buffer_build_spec;
    }

    pub fn dispatch(&mut self, args: S::Args, device: &Device, queue: &Queue) -> S::Ret {
        self.inner
            .dispatch(args, &mut self.buffer_build_spec, device, queue)
    }
}

pub trait ComputeShader<'a> {
    type Args;
    type Ret;

    fn resource_buffers_with_bind_group_layouts(&self) -> Vec<ResourceGroupLayout<'a>> {
        vec![]
    }
    fn load_source_code(&self) -> String;
    fn entry_point(&self) -> &'static str;

    fn build(&self, device: &Device) -> ComputeShaderBuiltData<'a> {
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

        let build_spec = BufferBuildSpec {
            vertex_buffer_szs: vec![],
            index_buffer_sz: 0,
            resource_buffer: resource_buffer_descs
                .into_iter()
                .zip(bind_group_layouts)
                .map(|(d, l)| ResourceGroupBuildSpec {
                    layout: l,
                    layout_entries: d.entries,
                })
                .collect(),
        };

        let bind_groups = DynamicBindGroup::new_from_buffer_spec(&build_spec);

        ComputeShaderBuiltData {
            bind_groups,
            pipeline,
            buffer_build_spec: build_spec,
        }
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        build_data: &mut ComputeShaderBuiltData<'a>,
        device: &Device,
        queue: &Queue,
    ) -> Self::Ret;
}
