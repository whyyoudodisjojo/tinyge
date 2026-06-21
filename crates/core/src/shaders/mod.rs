pub mod buffers;
pub mod descriptors;
pub mod manager;

use wgpu::*;

use crate::shaders::{
    buffers::ShaderBuffers,
    descriptors::{
        BindGroupLayoutDescriptorOwned, ResourceBufferBindGroupLayoutWithUsages,
        ShaderMeshBufferLayouts, ShaderPipelineDescriptor, ShaderVertexBufferLayout,
    },
};

pub fn align_to_4_bytes(size: u64) -> u64 {
    ((size + 3) / 4) * 4
}

pub struct ShaderBuiltData {
    buffers: ShaderBuffers,
    pipeline: RenderPipeline,
}

pub trait Shader {
    fn mesh_buffers_layouts(&self) -> ShaderMeshBufferLayouts<'static>;
    fn resource_buffers_bind_group_layouts(&self) -> Vec<ResourceBufferBindGroupLayoutWithUsages>;
    fn load_source_code(&self) -> &'static str;
    fn shader_pipeline_desc(&self) -> ShaderPipelineDescriptor<'static>;

    fn build(
        &self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) -> ShaderBuiltData {
        let ShaderMeshBufferLayouts {
            vertex_buffer_layouts: vertex_layouts,
            index_buffer_size,
        } = self.mesh_buffers_layouts();
        let (vertex_layouts, vertex_buffer_sizes): (Vec<VertexBufferLayout<'static>>, Vec<u64>) =
            vertex_layouts
                .into_iter()
                .map(
                    |ShaderVertexBufferLayout {
                         vertex_buffer,
                         vertex_buffer_size,
                     }| (vertex_buffer, vertex_buffer_size),
                )
                .collect::<(Vec<_>, Vec<_>)>();
        let (bind_group_layout_desc, usages, resource_buffer_sizes): (
            Vec<BindGroupLayoutDescriptorOwned>,
            Vec<BufferUsages>,
            Vec<u64>,
        ) = self
            .resource_buffers_bind_group_layouts()
            .into_iter()
            .map(
                |ResourceBufferBindGroupLayoutWithUsages {
                     layout,
                     usages,
                     size,
                 }| { (layout, usages, size) },
            )
            .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

        let bind_group_layouts = bind_group_layout_desc
            .iter()
            .map(|l| device.create_bind_group_layout(&l.into_desc()))
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
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(self.load_source_code())),
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

        let buffers = ShaderBuffers::build(
            device,
            vertex_buffer_sizes,
            index_buffer_size,
            resource_buffer_sizes,
            bind_group_layout_desc,
            bind_group_layouts,
            usages,
        );

        ShaderBuiltData { buffers, pipeline }
    }
}
