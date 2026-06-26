use std::{
    hash::{DefaultHasher, Hash, Hasher},
    num::NonZeroUsize,
};

use lru::LruCache;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder, Device, Sampler, TextureView,
    wgt::TextureViewDescriptor,
};

use crate::shaders::{
    descriptors::{ResourceBinding, ResourceBindingType},
    texture::ResourceTexture,
};

#[derive(Clone)]
pub struct ResourceGroup {
    pub buffers: Vec<Buffer>,
    pub bind_group: DynamicBindGroup,
    pub textures: Vec<ResourceTexture>,
}

#[derive(Clone)]
pub struct DynamicBindGroup {
    pub layout: BindGroupLayout,
    pub bind_group_cache: LruCache<u64, BindGroup>,
}

#[derive(Hash, Clone)]
pub enum BindGroupEntryInput {
    Buffer(Buffer),
    Sampler(Sampler),
    Texture(TextureView),
}

impl DynamicBindGroup {
    pub fn new(layout: BindGroupLayout) -> Self {
        Self {
            layout,
            bind_group_cache: LruCache::new(NonZeroUsize::new(16).unwrap()),
        }
    }

    pub fn key(bufs: &[BindGroupEntryInput]) -> u64 {
        let mut hasher = DefaultHasher::new();
        bufs.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get_bind_group(&mut self, bufs: &[BindGroupEntryInput]) -> Option<&BindGroup> {
        let k = Self::key(bufs);

        self.bind_group_cache.get(&k)
    }

    pub fn peek_last_bind_group(&self) -> Option<&BindGroup> {
        self.bind_group_cache.peek_lru().map(|(_, v)| v)
    }

    pub fn insert(&mut self, b: &[BindGroupEntryInput], bind_group: BindGroup) {
        self.bind_group_cache.put(Self::key(b), bind_group);
    }

    pub fn get_or_create_bind_group(
        &mut self,
        buffs: &[BindGroupEntryInput],
        device: &Device,
    ) -> BindGroup {
        let k = Self::key(buffs);

        let bind_group = match &mut self.bind_group_cache.get(&k) {
            Some(b) => b.clone(),
            None => {
                let b = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &self.layout,
                    entries: &buffs
                        .into_iter()
                        .enumerate()
                        .map(|(i, b)| BindGroupEntry {
                            binding: i as u32,
                            resource: match b {
                                BindGroupEntryInput::Buffer(b) => b.as_entire_binding(),
                                BindGroupEntryInput::Sampler(s) => BindingResource::Sampler(s),
                                BindGroupEntryInput::Texture(t) => BindingResource::TextureView(t),
                            },
                        })
                        .collect::<Vec<_>>(),
                });

                self.bind_group_cache.put(k, b.clone());

                b
            }
        };

        bind_group
    }
}
pub struct BufferBuildSpec<'a> {
    pub vertex_buffer_szs: Vec<u64>,
    pub index_buffer_sz: u64,
    pub resource_buffer: Vec<ResourceGroupBuildSpec<'a>>,
}

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
                    .for_each(|(o, n)| Self::copy_via_encoder(o, n, &mut encoder))
            });

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn build(device: &wgpu::Device, spec: BufferBuildSpec) -> Self {
        let vertex_buffers = spec
            .vertex_buffer_szs
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
            .into_iter()
            .map(|b| {
                let textures = b
                    .layout_entries
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| {
                        let ResourceBindingType::Texture {
                            texture_descriptor, ..
                        } = &l.ty
                        else {
                            return None;
                        };

                        let texture = device.create_texture(&texture_descriptor);
                        let view = texture.create_view(&TextureViewDescriptor::default());

                        Some((
                            i,
                            ResourceTexture {
                                texture,
                                view,
                                sz: texture_descriptor.size,
                            },
                        ))
                    })
                    .collect::<Vec<_>>();

                let sampler = b
                    .layout_entries
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| {
                        let ResourceBindingType::Sampler {
                            sampler_descriptor, ..
                        } = &l.ty
                        else {
                            return None;
                        };

                        Some((i, device.create_sampler(sampler_descriptor)))
                    })
                    .collect::<Vec<_>>();

                let buffers = b
                    .layout_entries
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| {
                        let ResourceBindingType::Buffer { size, usages, .. } = l.ty else {
                            return None;
                        };

                        Some((
                            i,
                            device.create_buffer(&BufferDescriptor {
                                label: None,
                                usage: usages | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                                size: align_to_4_bytes(size),
                                mapped_at_creation: false,
                            }),
                        ))
                    })
                    .collect::<Vec<_>>();

                let buffers: Vec<Buffer> = buffers.into_iter().map(|b| b.1).collect();
                let textures: Vec<ResourceTexture> = textures.into_iter().map(|t| t.1).collect();

                let bind_group_entries: Vec<BindGroupEntryInput> = buffers
                    .iter()
                    .map(|b| BindGroupEntryInput::Buffer(b.clone()))
                    .chain(
                        textures
                            .iter()
                            .map(|t| BindGroupEntryInput::Texture(t.view.clone())),
                    )
                    .chain(
                        sampler
                            .iter()
                            .map(|(_, s)| BindGroupEntryInput::Sampler(s.clone())),
                    )
                    .collect();

                let entries: Vec<BindGroupEntry> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .chain(textures.iter().enumerate().map(|(i, t)| BindGroupEntry {
                        binding: (buffers.len() + i) as u32,
                        resource: BindingResource::TextureView(&t.view),
                    }))
                    .chain(sampler.iter().enumerate().map(|(i, s)| BindGroupEntry {
                        binding: (buffers.len() + textures.len() + i) as u32,
                        resource: BindingResource::Sampler(&s.1),
                    }))
                    .collect();

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &b.layout,
                    entries: &entries,
                });

                let mut cache = LruCache::new(NonZeroUsize::new(16usize).unwrap());
                cache.put(DynamicBindGroup::key(&bind_group_entries), bind_group);

                let bind_group = DynamicBindGroup {
                    layout: b.layout,
                    bind_group_cache: cache,
                };

                ResourceGroup {
                    buffers,
                    bind_group,
                    textures,
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
