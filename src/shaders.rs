use std::{collections::HashMap, hash::Hash, mem, num::NonZero};

use wgpu::*;

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

pub struct ResourceBufferBindGroupLayoutWithUsages {
    pub layout: BindGroupLayout,
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

pub fn align_to_4_bytes(size: u64) -> u64 {
    ((size + 3) / 4) / 4
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
        let (vertex_layouts, vertex_buffer_sizes): (Vec<VertexBufferLayout<'static>>, Vec<u64>)= vertex_layouts
            .into_iter()
            .map(
                |ShaderVertexBufferLayout {
                     vertex_buffer,
                     vertex_buffer_size,
                 }| (vertex_buffer, vertex_buffer_size),
            )
            .collect::<(Vec<_>, Vec<_>)>();
        let (bind_group_layouts, usages, resource_buffer_sizes): (Vec<BindGroupLayout>, Vec<BufferUsages>, Vec<u64>) = self
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

        let desc = self.shader_pipeline_desc();

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &bind_group_layouts
                .iter()
                .map(|l| Some(l))
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

        let vertex_buffers = vertex_buffer_sizes
            .into_iter()
            .map(|size| {
                device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: align_to_4_bytes(size),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: align_to_4_bytes(index_buffer_size),
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let resource_buffers = resource_buffer_sizes
            .into_iter()
            .zip(usages.into_iter())
            .map(|(size, usage)| {
                device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: align_to_4_bytes(size),
                    usage: usage | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();

        ShaderBuiltData {
            buffers: ShaderBuffers {
                vertex_buffers,
                index_buffer,
                resource_buffers,
            },
            pipeline,
        }
    }
}

pub struct ShaderBuiltData {
    buffers: ShaderBuffers,
    pipeline: RenderPipeline,
}

pub struct ShaderBuffers {
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffer: Buffer,
    pub resource_buffers: Vec<Buffer>,
}

impl ShaderBuffers {
    fn copy_via_encoder(src: &wgpu::Buffer, dst: &wgpu::Buffer, encoder: &mut CommandEncoder) {
        let copy_size = src.size().min(dst.size());
        if copy_size > 0 {
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, copy_size);
        }
    }

    pub fn copy_data_into(&self, new_buffers: &Self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.vertex_buffers
            .iter()
            .zip(new_buffers.vertex_buffers.iter())
            .for_each(|(o, n)| Self::copy_via_encoder(o, n, &mut encoder));
        Self::copy_via_encoder(&self.index_buffer, &new_buffers.index_buffer, &mut encoder);
        self.resource_buffers
            .iter()
            .zip(new_buffers.resource_buffers.iter())
            .for_each(|(o, n)| Self::copy_via_encoder(o, n, &mut encoder));

        queue.submit(std::iter::once(encoder.finish()));
    }
}

pub struct ShaderManager<'a, K> {
    pub compilation_cache: Option<PipelineCache>,
    pub pipeline_cache: HashMap<K, RenderPipeline>,
    pub shaders: HashMap<K, &'a dyn Shader>,
    pub compilation_pending_shaders: HashMap<K, &'a dyn Shader>,
    pub texture_format: Option<TextureFormat>,
}

impl<'a, K> ShaderManager<'a, K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            compilation_cache: None,
            pipeline_cache: HashMap::new(),
            shaders: HashMap::new(),
            compilation_pending_shaders: HashMap::new(),
            texture_format: None,
        }
    }

    pub fn new_with_compilation_cache(
        device: &Device,
        cache_descriptor: PipelineCacheDescriptor,
    ) -> Self {
        Self {
            compilation_cache: Some(unsafe { device.create_pipeline_cache(&cache_descriptor) }),
            pipeline_cache: HashMap::new(),
            shaders: HashMap::new(),
            compilation_pending_shaders: HashMap::new(),
            texture_format: None,
        }
    }

    pub fn update_texture_format(&mut self, texture_format: TextureFormat) {
        self.texture_format = Some(texture_format);
    }

    pub fn register_shader_dyn(&mut self, key: K, shader: &'static dyn Shader) {
        let shader: &dyn Shader = shader;

        self.compilation_pending_shaders.insert(key, shader);
    }

    pub fn register_shader<S>(&mut self, key: K, shader: &'a S)
    where
        S: Shader + Sized,
    {
        let shader: &'a dyn Shader = shader as &'a dyn Shader;

        self.compilation_pending_shaders.insert(key, shader);
    }

    pub fn register_shaders(&mut self, shaders: HashMap<K, &'static dyn Shader>) {
        shaders
            .into_iter()
            .for_each(|(k, s)| self.register_shader_dyn(k, s))
    }

    pub fn recompile_shaders(&mut self, device: &Device) -> Option<HashMap<K, ShaderBuffers>> {
        self.pipeline_cache.clear();

        let pending_shaders = mem::take(&mut self.compilation_pending_shaders);

        self.shaders.extend(pending_shaders);

        Some(
            self.shaders
                .iter()
                .map(|(k, s)| {
                    let build_data = s.build(
                        device,
                        &self.texture_format.unwrap(),
                        self.compilation_cache.as_ref(),
                    );

                    self.pipeline_cache.insert(k.clone(), build_data.pipeline);

                    (k.clone(), build_data.buffers)
                })
                .collect(),
        )
    }
}
