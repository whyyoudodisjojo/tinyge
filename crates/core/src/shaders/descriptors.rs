use std::num::NonZero;

use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BlendState, BufferUsages, ColorWrites,
    DepthStencilState, MultisampleState, PipelineCompilationOptions, PrimitiveState,
    VertexBufferLayout,
};

pub struct ShaderPipelineDescriptor<'a> {
    pub vertex_entry_point: Option<&'a str>,
    pub vertex_compilation_options: PipelineCompilationOptions<'a>,
    pub primitive_state: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment_entry_point: Option<&'a str>,
    pub fragment_targets: &'a [Option<ColorTargetStateData>],
    pub fragment_compilation_options: PipelineCompilationOptions<'a>,
    pub multiview_mask: Option<NonZero<u32>>,
}

pub struct ColorTargetStateData {
    pub blend: Option<BlendState>,
    pub write_mask: ColorWrites,
}

pub struct BindGroupLayoutDescriptorOwned {
    pub entries: Vec<BindGroupLayoutEntry>,
}

impl BindGroupLayoutDescriptorOwned {
    pub fn into_desc<'a>(&'a self) -> BindGroupLayoutDescriptor<'a> {
        BindGroupLayoutDescriptor {
            label: None,
            entries: &self.entries,
        }
    }
}

pub struct ResourceBufferBindGroupLayoutWithUsages {
    pub layout: BindGroupLayoutDescriptorOwned,
    pub usages: BufferUsages,
    pub size: u64,
}

pub struct ShaderVertexBufferLayout<'a> {
    pub vertex_buffer: VertexBufferLayout<'a>,
    pub vertex_buffer_size: u64,
}

pub struct ShaderMeshBufferLayouts<'a> {
    pub vertex_buffer_layouts: Vec<ShaderVertexBufferLayout<'a>>,
    pub index_buffer_size: u64,
}
