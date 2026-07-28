pub mod descriptors;
pub mod manager;

use std::sync::{
    Arc, Mutex,
    mpsc::{Receiver, Sender, channel},
};

use memory::{
    buffers::{AccelerationStructures, DynamicBindGroup, ResourceGroup},
    descriptors::{
        MeshBufferSpecs, ResourceBindingType, ResourceGroupLayout, ShaderPipelineDescriptor,
        VertexBufferSpec,
    },
    socket::{TinyBlas, TinyBuffer, TinySampler, TinyTexture, TinyTlas},
    texture::ResourceTexture,
};
use wgpu::*;

pub struct ShaderBuiltData {
    pub pipeline: RenderPipeline,
    pub bind_groups: Vec<DynamicBindGroup>,
}

pub struct ComputeShaderBuiltData<T> {
    pub buffers: T,
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
        &mut self,
        device: &Device,
        texture_format: &TextureFormat,
        cache: Option<&PipelineCache>,
    ) {
        let MeshBufferSpecs {
            vertex_buffers: vertex_layouts,
            ..
        } = self.mesh_buffers_layouts();
        let vertex_layouts = vertex_layouts
            .into_iter()
            .map(|VertexBufferSpec { layout, .. }| layout)
            .collect::<Vec<_>>();

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

        let bind_groups = bind_group_layouts
            .into_iter()
            .map(DynamicBindGroup::new)
            .collect();

        let res = ShaderBuiltData {
            pipeline,
            bind_groups,
        };

        self.handle_recompilation(res);
    }

    fn handle_recompilation(&mut self, built_data: ShaderBuiltData);
}

pub struct ShaderWrapper<S>
where
    S: for<'a> Shader<'a>,
{
    pub shader: Arc<Mutex<S>>,
    sender_tx: Sender<RecompilationData>,
}

impl<S> ShaderWrapper<S>
where
    S: for<'a> Shader<'a>,
{
    pub fn new(shader: Arc<Mutex<S>>) -> (Arc<Self>, Receiver<RecompilationData>) {
        let (tx, rx) = channel();
        (
            Arc::new(Self {
                shader,
                sender_tx: tx,
            }),
            rx,
        )
    }

    pub fn get_sender_tx(&self) -> Sender<RecompilationData> {
        self.sender_tx.clone()
    }

    pub fn watch(rx: Receiver<RecompilationData>, shader: Arc<Mutex<S>>) {
        while let Ok(RecompilationData {
            device,
            texture_format,
            cache,
        }) = rx.recv()
        {
            let mut s = shader.lock().unwrap();
            s.build(&device, &texture_format, cache.as_ref());
        }
    }
}

pub struct RecompilationData {
    device: Device,
    texture_format: TextureFormat,
    cache: Option<PipelineCache>,
}

pub struct ComputeShaderWrapper<S, T> {
    pub built_data: ComputeShaderBuiltData<T>,
    pub inner: S,
}

impl<'a, S> ComputeShaderWrapper<S, S::Args>
where
    S: ComputeShader<'a>,
{
    pub fn new(shader: S, device: &Device) -> Self {
        let buffers = shader.build(device);
        Self {
            built_data: buffers,
            inner: shader,
        }
    }

    pub fn recompile(&mut self, device: &Device) {
        let buffer_build_spec = self.inner.build(device);
        self.built_data = buffer_build_spec;
    }

    pub fn dispatch(&mut self, args: S::Args, device: &Device, queue: &Queue) -> S::Ret {
        self.inner
            .dispatch(args, &mut self.built_data, device, queue)
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

    fn build_sockets_dyn(
        &self,
        resource_group_layout: &[ResourceGroupLayout<'a>],
    ) -> Vec<ResourceGroup<'a>> {
        resource_group_layout
            .iter()
            .map(|r| {
                let mut acc_struct = vec![];
                let mut buffers = vec![];
                let mut samplers = vec![];
                let mut textures = vec![];

                r.entries.iter().for_each(|e| match &e.ty {
                    ResourceBindingType::AccelerationStructure {
                        tlas_desc,
                        blas_desc,
                        blas_geo_sz_desc,
                        ..
                    } => {
                        let tlas = TinyTlas::new(tlas_desc.clone());
                        let blas = TinyBlas::new(blas_desc.clone(), blas_geo_sz_desc.clone());

                        acc_struct.push(AccelerationStructures { blas, tlas })
                    }
                    ResourceBindingType::Buffer { size, usages, .. } => {
                        let buffer = TinyBuffer::new(size.clone(), usages.clone());
                        buffers.push(buffer);
                    }
                    ResourceBindingType::ExternalTexture => todo!(),
                    ResourceBindingType::Sampler {
                        sampler_descriptor, ..
                    } => {
                        let sampler = TinySampler::new(sampler_descriptor.clone());
                        samplers.push(sampler)
                    }
                    ResourceBindingType::StorageTexture { .. } => todo!(),
                    ResourceBindingType::Texture {
                        texture_descriptor, ..
                    } => {
                        let texture = TinyTexture::new(texture_descriptor.clone());
                        textures.push(ResourceTexture {
                            texture,
                            sz: texture_descriptor.size,
                        });
                    }
                });

                ResourceGroup {
                    buffers,
                    textures,
                    samplers,
                    acceleration_structures: acc_struct,
                }
            })
            .collect()
    }

    fn build_sockets(
        &self,
        resource_group_layout: &[ResourceGroupLayout<'a>],
        device: &Device,
    ) -> Self::Args;

    fn build(&self, device: &Device) -> ComputeShaderBuiltData<Self::Args> {
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

        let bind_groups = bind_group_layouts
            .into_iter()
            .map(DynamicBindGroup::new)
            .collect();

        let buffers = self.build_sockets(&resource_buffer_descs, device);

        ComputeShaderBuiltData {
            bind_groups,
            pipeline,
            buffers,
        }
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        build_data: &mut ComputeShaderBuiltData<Self::Args>,
        device: &Device,
        queue: &Queue,
    ) -> Self::Ret;
}
