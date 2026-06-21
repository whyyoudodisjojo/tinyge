use std::num::{NonZero, NonZeroU32};

use wgpu::{
    BindGroupLayoutEntry, BindingType, BlendState, BufferUsages, ColorWrites, DepthStencilState,
    MultisampleState, PipelineCompilationOptions, PrimitiveState, ShaderStages, VertexBufferLayout,
};

pub struct ShaderPipelineDescriptor<'a> {
    pub vertex_entry_point: Option<&'a str>,
    pub vertex_compilation_options: PipelineCompilationOptions<'a>,
    pub primitive_state: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment_entry_point: Option<&'a str>,
    pub fragment_targets: &'a [Option<ColorTarget>],
    pub fragment_compilation_options: PipelineCompilationOptions<'a>,
    pub multiview_mask: Option<NonZero<u32>>,
}

pub struct ColorTarget {
    pub blend: Option<BlendState>,
    pub write_mask: ColorWrites,
}

pub struct ResourceGroupLayout {
    pub entries: Vec<ResourceBinding>,
}

pub struct ResourceBinding {
    pub binding: u32,
    pub visibility: ShaderStages,
    pub ty: BindingType,
    pub count: Option<NonZeroU32>,
    pub usage_overrides: BufferUsages,
    pub size: u64,
}

impl From<&ResourceBinding> for BindGroupLayoutEntry {
    fn from(binding: &ResourceBinding) -> Self {
        BindGroupLayoutEntry {
            binding: binding.binding,
            visibility: binding.visibility,
            ty: binding.ty,
            count: binding.count,
        }
    }
}

pub struct VertexBufferSpec<'a> {
    pub layout: VertexBufferLayout<'a>,
    pub size: u64,
}

pub struct MeshBufferSpecs<'a> {
    pub vertex_buffers: Vec<VertexBufferSpec<'a>>,
    pub index_buffer_size: u64,
}
