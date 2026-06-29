use std::{
    hash::{DefaultHasher, Hash, Hasher},
    num::NonZeroUsize,
};

use lru::LruCache;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder, Device, Sampler, wgt::TextureViewDescriptor,
};

use crate::shaders::{
    descriptors::{ResourceBinding, ResourceBindingType},
    texture::ResourceTexture,
};

#[derive(Clone)]
pub struct ResourceGroup {
    pub buffers: Vec<Option<Buffer>>,
    pub textures: Vec<ResourceTexture>,
    pub samplers: Vec<Sampler>,
}

pub struct DynamicBindGroup {
    pub layout: BindGroupLayout,
    pub bind_group_cache: LruCache<u64, BindGroup>,
}

impl DynamicBindGroup {
    pub fn new_from_buffer_spec(spec: &BufferBuildSpec) -> Vec<Self> {
        spec.resource_buffer
            .iter()
            .map(|b| DynamicBindGroup::new(b.layout.clone()))
            .collect()
    }

    pub fn new(layout: BindGroupLayout) -> Self {
        Self {
            layout,
            bind_group_cache: LruCache::new(NonZeroUsize::new(16).unwrap()),
        }
    }

    pub fn key(bufs: &[ResourceType]) -> u64 {
        let mut hasher = DefaultHasher::new();
        bufs.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get_bind_group(&mut self, bufs: &[ResourceType]) -> Option<&BindGroup> {
        let k = Self::key(bufs);

        self.bind_group_cache.get(&k)
    }

    pub fn insert(&mut self, b: &[ResourceType], bind_group: BindGroup) {
        self.bind_group_cache.put(Self::key(b), bind_group);
    }

    pub fn get_or_create_bind_group<'a>(
        &'a mut self,
        buffs: &[ResourceType],
        device: &Device,
    ) -> &'a BindGroup {
        let k = Self::key(buffs);

        if self.bind_group_cache.get(&k).is_none() {
            let b = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &buffs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| BindGroupEntry {
                        binding: i as u32,
                        resource: match b {
                            ResourceType::Buffer(b) => b.as_entire_binding(),
                            ResourceType::Sampler(s) => BindingResource::Sampler(s),
                            ResourceType::Texture(t) => BindingResource::TextureView(&t.view),
                        },
                    })
                    .collect::<Vec<_>>(),
            });

            self.bind_group_cache.put(k, b);
        }

        self.bind_group_cache.get(&k).unwrap()
    }
}

#[derive(Clone)]
pub struct BufferBuildSpec<'a> {
    pub vertex_buffer_szs: Vec<u64>,
    pub index_buffer_sz: u64,
    pub resource_buffer: Vec<ResourceGroupBuildSpec<'a>>,
}

#[derive(Clone)]
pub struct ResourceGroupBuildSpec<'a> {
    pub layout_entries: Vec<ResourceBinding<'a>>,
    pub layout: BindGroupLayout,
}

#[derive(Clone)]
pub struct Buffers {
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub resource_buffers: Vec<ResourceGroup>,
}

pub struct ResourceEntry {
    pub binding: u32,
    pub resource: ResourceType,
}

#[derive(Clone, Hash)]
pub enum ResourceType {
    Buffer(Buffer),
    Sampler(Sampler),
    Texture(ResourceTexture),
}

impl Buffers {
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
        self.index_buffer
            .as_ref()
            .zip(new_buffers.index_buffer.as_ref())
            .map(|(o, n)| Self::copy_via_encoder(&o, &n, &mut encoder));
        self.resource_buffers
            .iter()
            .zip(new_buffers.resource_buffers.iter())
            .for_each(|(o, n)| {
                o.buffers
                    .iter()
                    .zip(n.buffers.iter())
                    .for_each(|(o_buf, n_buf)| {
                        if let (Some(o), Some(n)) = (o_buf, n_buf) {
                            Self::copy_via_encoder(o, n, &mut encoder)
                        }
                    })
            });

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn build(device: &wgpu::Device, spec: &BufferBuildSpec, build_input_only: bool) -> Self {
        let vertex_buffers = spec
            .vertex_buffer_szs
            .iter()
            .map(|size| {
                device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: align_to_4_bytes(*size),
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let index_buffer = if spec.index_buffer_sz > 0 {
            Some(device.create_buffer(&BufferDescriptor {
                label: None,
                size: align_to_4_bytes(spec.index_buffer_sz),
                usage: BufferUsages::INDEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let resource_buffers = spec
            .resource_buffer
            .iter()
            .map(|b| {
                let mut buffers: Vec<Option<Buffer>> = vec![None; b.layout_entries.len()];
                let mut textures: Vec<ResourceTexture> = Vec::new();
                let mut samplers: Vec<Sampler> = Vec::new();

                for (binding_idx, layout_entry) in b.layout_entries.iter().enumerate() {
                    let _binding = binding_idx as u32;

                    match &layout_entry.ty {
                        ResourceBindingType::Buffer {
                            size,
                            usages,
                            is_input,
                            ..
                        } => {
                            if build_input_only && !*is_input || !build_input_only && *is_input {
                                continue;
                            }
                            let buffer = device.create_buffer(&BufferDescriptor {
                                label: None,
                                usage: *usages | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                size: align_to_4_bytes(*size),
                                mapped_at_creation: false,
                            });
                            buffers[binding_idx] = Some(buffer);
                        }
                        ResourceBindingType::Texture {
                            texture_descriptor, ..
                        } => {
                            let texture = device.create_texture(texture_descriptor);
                            let view = texture.create_view(&TextureViewDescriptor::default());
                            textures.push(ResourceTexture {
                                texture,
                                view,
                                sz: texture_descriptor.size,
                            });
                        }
                        ResourceBindingType::Sampler {
                            sampler_descriptor, ..
                        } => {
                            let sampler = device.create_sampler(sampler_descriptor);
                            samplers.push(sampler);
                        }
                        ResourceBindingType::StorageTexture { .. } => {
                            todo!("StorageTexture not yet implemented in build")
                        }
                        ResourceBindingType::AccelerationStructure { .. } => {
                            todo!("AccelerationStructure not yet implemented in build")
                        }
                        ResourceBindingType::ExternalTexture => {
                            todo!("ExternalTexture not yet implemented in build")
                        }
                    };
                }

                ResourceGroup {
                    buffers,
                    textures,
                    samplers,
                }
            })
            .collect();

        Self {
            vertex_buffers,
            index_buffer,
            resource_buffers,
        }
    }
}

pub fn align_to_4_bytes(size: u64) -> u64 {
    ((size + 3) / 4) * 4
}
