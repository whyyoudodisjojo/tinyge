pub mod buffers;
pub mod descriptors;
pub mod manager;
pub mod texture;

use wgpu::*;

use crate::shaders::{
    buffers::{BufferBuildSpec, Buffers, ResourceGroupBuildSpec},
    descriptors::{
        MeshBufferSpecs, ResourceGroupLayout, ShaderPipelineDescriptor, VertexBufferSpec,
    },
};

pub struct ShaderBuiltData {
    buffers: Buffers,
    pipeline: RenderPipeline,
}

pub trait Shader {
    fn mesh_buffers_layouts(&self) -> MeshBufferSpecs<'static>;
    fn resource_buffers_with_bind_group_layouts<'a>(&'a self) -> Vec<ResourceGroupLayout<'a>>;
    fn load_source_code(&self) -> &'static str;
    fn shader_pipeline_desc(&self) -> ShaderPipelineDescriptor<'static>;

    fn build(
        &self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) -> ShaderBuiltData {
        let MeshBufferSpecs {
            vertex_buffers: vertex_layouts,
            index_buffer_size,
        } = self.mesh_buffers_layouts();
        let (vertex_layouts, vertex_buffer_sizes): (Vec<VertexBufferLayout<'static>>, Vec<u64>) =
            vertex_layouts
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

        let buffers = Buffers::build(
            device,
            BufferBuildSpec {
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
            },
        );

        ShaderBuiltData { buffers, pipeline }
    }
}
