use std::{collections::HashMap, hash::Hash, num::NonZero};

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
    layout: BindGroupLayout,
    usages: BufferUsages,
    size: u64,
}

pub struct ShaderVertexBufferLayout<'a> {
    vertex_buffer: VertexBufferLayout<'a>,
    vertex_buffer_size: u64,
}

pub struct ShaderMeshBufferLayouts<'a> {
    vertex_buffer_layouts: Vec<ShaderVertexBufferLayout<'a>>,
    index_buffer_size: u64,
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
        let (vertex_layouts, vertex_buffer_sizes) = vertex_layouts
            .into_iter()
            .map(
                |ShaderVertexBufferLayout {
                     vertex_buffer,
                     vertex_buffer_size,
                 }| (vertex_buffer, vertex_buffer_size),
            )
            .collect::<(Vec<_>, Vec<_>)>();
        let (bind_group_layouts, usages, resource_buffer_sizes) = self
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
                    size,
                    usage: BufferUsages::VERTEX,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: index_buffer_size,
            usage: BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        let resource_buffers = resource_buffer_sizes
            .into_iter()
            .zip(usages.into_iter())
            .map(|(size, usage)| {
                device.create_buffer(&BufferDescriptor {
                    label: None,
                    size,
                    usage,
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

pub struct ShaderManager<K> {
    pub compilation_cache: Option<PipelineCache>,
    pub pipeline_cache: HashMap<K, RenderPipeline>,
    pub shaders: Vec<(K, &'static dyn Shader)>,
    pub texture_format: TextureFormat,
}

impl<K> ShaderManager<K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    pub fn new(texture_format: TextureFormat) -> Self {
        Self {
            compilation_cache: None,
            pipeline_cache: HashMap::new(),
            shaders: vec![],
            texture_format,
        }
    }

    pub fn new_with_compilation_cache(
        device: &Device,
        cache_descriptor: PipelineCacheDescriptor,
        texture_format: TextureFormat,
    ) -> Self {
        Self {
            compilation_cache: Some(unsafe { device.create_pipeline_cache(&cache_descriptor) }),
            pipeline_cache: HashMap::new(),
            shaders: vec![],
            texture_format,
        }
    }

    pub fn update_texture_format(&mut self, texture_format: TextureFormat) {
        self.texture_format = texture_format;
    }

    pub fn register_shader_dyn(
        &mut self,
        key: K,
        shader: &'static dyn Shader,
        device: &Device,
    ) -> ShaderBuffers {
        let shader: &dyn Shader = shader;
        let ShaderBuiltData { buffers, pipeline } = shader.build(
            device,
            &self.texture_format,
            self.compilation_cache.as_ref(),
        );

        self.pipeline_cache.insert(key.clone(), pipeline);
        self.shaders.push((key, shader));

        buffers
    }

    pub fn register_shader<S>(
        &mut self,
        key: K,
        shader: &'static S,
        device: &Device,
    ) -> ShaderBuffers
    where
        S: Shader + Sized,
    {
        let shader: &dyn Shader = shader;
        let ShaderBuiltData { buffers, pipeline } = shader.build(
            device,
            &self.texture_format,
            self.compilation_cache.as_ref(),
        );

        self.pipeline_cache.insert(key.clone(), pipeline);
        self.shaders.push((key, shader));

        buffers
    }

    pub fn register_shaders(
        &mut self,
        shaders: HashMap<K, &'static dyn Shader>,
        device: &Device,
    ) -> HashMap<K, ShaderBuffers> {
        shaders
            .into_iter()
            .map(|(k, s)| (k.clone(), self.register_shader_dyn(k, s, device)))
            .collect()
    }

    pub fn recompile_shaders(&mut self, device: &Device) -> HashMap<K, ShaderBuffers> {
        self.pipeline_cache.clear();

        self.shaders
            .iter()
            .map(|(k, s)| {
                let ShaderBuiltData { buffers, pipeline } = s.build(
                    device,
                    &self.texture_format,
                    self.compilation_cache.as_ref(),
                );

                self.pipeline_cache.insert(k.clone(), pipeline);

                (k.clone(), buffers)
            })
            .collect()
    }
}
