use std::num::NonZeroU64;

use tinyge_graphics::shaders::{
    Shader,
    descriptors::{
        ColorTarget, MeshBufferSpecs, ResourceBinding, ResourceBindingType, ResourceGroupLayout,
        ShaderPipelineDescriptor,
    },
};
use wgpu::{
    BlendComponent, BlendState, BufferUsages, ColorWrites, Extent3d, MultisampleState,
    PrimitiveState, SamplerBindingType, SamplerDescriptor, ShaderStages, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDimension,
};

pub struct Sprites {
    pub texture_size: Extent3d,
}

impl Shader for Sprites {
    fn mesh_buffers_layouts(&self) -> MeshBufferSpecs<'static> {
        MeshBufferSpecs::default()
    }

    fn resource_buffers_with_bind_group_layouts<'a>(&'a self) -> Vec<ResourceGroupLayout<'a>> {
        vec![
            ResourceGroupLayout {
                entries: vec![
                    ResourceBinding {
                        binding: 0,
                        visibility: ShaderStages::all(),
                        ty: ResourceBindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(4),
                            size: 4,
                            usages: BufferUsages::UNIFORM,
                        },
                        count: None,
                        create_initial_buffers: true,
                    },
                    ResourceBinding {
                        binding: 1,
                        visibility: ShaderStages::all(),
                        ty: ResourceBindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                            size: 1024 * 32,
                            usages: BufferUsages::STORAGE,
                        },
                        count: None,
                        create_initial_buffers: true,
                    },
                ],
            },
            ResourceGroupLayout {
                entries: vec![
                    ResourceBinding {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: ResourceBindingType::Sampler {
                            ty: SamplerBindingType::Filtering,
                            sampler_descriptor: SamplerDescriptor::default(),
                        },
                        count: None,
                        create_initial_buffers: true,
                    },
                    ResourceBinding {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: ResourceBindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                            texture_descriptor: TextureDescriptor {
                                label: None,
                                size: self.texture_size,
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: TextureDimension::D2,
                                format: TextureFormat::Rgba8UnormSrgb,
                                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                                view_formats: &[],
                            },
                        },
                        count: None,
                        create_initial_buffers: true,
                    },
                ],
            },
        ]
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("../../shaders/sprites.wgsl")
    }

    fn shader_pipeline_desc(&self) -> ShaderPipelineDescriptor<'static> {
        ShaderPipelineDescriptor {
            vertex_entry_point: Some("vs_main"),
            vertex_compilation_options: Default::default(),
            primitive_state: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                conservative: false,
                polygon_mode: wgpu::PolygonMode::Fill,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment_targets: &[Some(ColorTarget {
                blend: Some(BlendState {
                    color: BlendComponent::REPLACE,
                    alpha: BlendComponent::REPLACE,
                }),
                write_mask: ColorWrites::ALL,
            })],
            fragment_compilation_options: Default::default(),
            fragment_entry_point: Some("fs_main"),
            multiview_mask: None,
        }
    }
}
