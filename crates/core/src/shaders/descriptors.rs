use std::num::{NonZero, NonZeroU32};

use wgpu::{
    BindGroupLayoutEntry, BindingType, BlendState, BufferBindingType, BufferSize, BufferUsages,
    ColorWrites, DepthStencilState, MultisampleState, PipelineCompilationOptions, PrimitiveState,
    SamplerBindingType, SamplerDescriptor, ShaderStages, StorageTextureAccess, TextureDescriptor,
    TextureFormat, TextureSampleType, TextureViewDimension, VertexBufferLayout,
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

pub struct ResourceGroupLayout<'a> {
    pub entries: Vec<ResourceBinding<'a>>,
}

pub struct ResourceBinding<'a> {
    pub binding: u32,
    pub visibility: ShaderStages,
    pub ty: ResourceBindingType<'a>,
    pub count: Option<NonZeroU32>,
}

impl<'a> From<&ResourceBinding<'a>> for BindGroupLayoutEntry {
    fn from(binding: &ResourceBinding) -> Self {
        BindGroupLayoutEntry {
            binding: binding.binding,
            visibility: binding.visibility,
            ty: (&binding.ty).into(),
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

pub enum ResourceBindingType<'a> {
    Buffer {
        ty: BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: Option<BufferSize>,
        size: u64,
        usages: BufferUsages,
    },
    Sampler {
        ty: SamplerBindingType,
        sampler_descriptor: SamplerDescriptor<'a>,
    },
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
        texture_descriptor: TextureDescriptor<'a>,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: TextureFormat,
        view_dimension: TextureViewDimension,
    },
    AccelerationStructure {
        vertex_return: bool,
    },
    ExternalTexture,
}

impl<'a> From<&ResourceBindingType<'a>> for BindingType {
    fn from(value: &ResourceBindingType) -> Self {
        match value {
            ResourceBindingType::AccelerationStructure { vertex_return } => {
                BindingType::AccelerationStructure {
                    vertex_return: *vertex_return,
                }
            }
            ResourceBindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
                ..
            } => BindingType::Buffer {
                ty: *ty,
                has_dynamic_offset: *has_dynamic_offset,
                min_binding_size: *min_binding_size,
            },
            ResourceBindingType::ExternalTexture => BindingType::ExternalTexture,
            ResourceBindingType::Sampler { ty, .. } => BindingType::Sampler(*ty),
            ResourceBindingType::StorageTexture {
                access,
                format,
                view_dimension,
            } => BindingType::StorageTexture {
                access: *access,
                format: *format,
                view_dimension: *view_dimension,
            },
            ResourceBindingType::Texture {
                sample_type,
                view_dimension,
                multisampled,
                ..
            } => BindingType::Texture {
                sample_type: *sample_type,
                view_dimension: *view_dimension,
                multisampled: *multisampled,
            },
        }
    }
}
