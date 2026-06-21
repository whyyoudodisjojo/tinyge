use std::num::NonZeroU64;

use tinyge_core::shaders::{
    Shader,
    descriptors::{
        ColorTarget, MeshBufferSpecs, ResourceBinding, ResourceGroupLayout,
        ShaderPipelineDescriptor, VertexBufferSpec,
    },
};
use wgpu::{
    BindingType, BlendComponent, BlendState, BufferUsages, ColorWrites, MultisampleState,
    PrimitiveState, ShaderStages, VertexAttribute, VertexBufferLayout, VertexFormat,
};

use crate::shader::Vertex;

pub const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        color: [0.0, 0.9, 1.0],
    },
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        color: [0.2, 0.9, 0.3],
    },
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        color: [1.0, 0.9, 0.1],
    },
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        color: [1.0, 0.5, 0.1],
    },
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        color: [0.9, 0.2, 0.9],
    },
];

pub const INDICES: &[u16] = &[
    0, 1, 4, // Top triangle
    1, 2, 4, // Left triangle
    2, 3, 4, // Bottom triangle
    0, // Padding for 4-byte alignment
];

pub struct Pentagon;

impl Shader for Pentagon {
    fn mesh_buffers_layouts(&self) -> MeshBufferSpecs<'static> {
        let vertex_sz = (3 * 4) + (3 * 4); // position (3 floats) + color (3 floats) = 24 bytes per vertex
        let vertex_buffer_sz = vertex_sz * VERTICES.len() as u64; // 5 vertices

        let layout = VertexBufferLayout {
            array_stride: vertex_sz,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                },
            ],
        };

        MeshBufferSpecs {
            vertex_buffers: vec![VertexBufferSpec {
                layout,
                size: vertex_buffer_sz,
            }],
            index_buffer_size: (INDICES.len() * 2) as u64, // 9 indices * 2 bytes each = 18 bytes
        }
    }

    fn resource_buffers_with_bind_group_layouts(&self) -> Vec<ResourceGroupLayout> {
        vec![ResourceGroupLayout {
            entries: vec![ResourceBinding {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                count: None,
                usage_overrides: BufferUsages::UNIFORM,
                size: 4,
            }],
        }]
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("../../shaders/triangle.wgsl")
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
