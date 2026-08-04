use std::{
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    num::NonZeroUsize,
};

use bytemuck::Pod;
use lru::LruCache;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    BufferUsages, Device,
};

use crate::{
    descriptors::{MeshBufferSpecs, ResourceBinding, ResourceBindingType, ResourceGroupLayout},
    socket::{TinyBlas, TinyBuffer, TinySampler, TinyTexture, TinyTlas},
    texture::ResourceTexture,
};

pub struct UnifiedShaderBuildData<'a, I = ()> {
    pub vertex_buffers: Vec<TinyBuffer<I>>,
    pub index_buffers: Vec<TinyBuffer<I>>,
    pub resource_groups: Vec<ResourceGroup<'a, I>>,
}

impl<'a> UnifiedShaderBuildData<'a> {
    pub fn new(
        resource_group_layouts: &[ResourceGroupLayout<'a>],
        mesh_buffer_specs: Option<&MeshBufferSpecs>,
    ) -> Self {
        let res = resource_group_layouts
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
            .collect();

        let vertex_buffers = mesh_buffer_specs
            .map(|m| {
                m.vertex_buffers
                    .iter()
                    .map(|v| TinyBuffer::new(v.size, BufferUsages::COPY_DST | BufferUsages::VERTEX))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let index_buffers = mesh_buffer_specs
            .map(|m| {
                m.index_buffer_size
                    .iter()
                    .map(|i| TinyBuffer::new(*i, BufferUsages::COPY_DST | BufferUsages::INDEX))
                    .collect()
            })
            .unwrap_or_default();

        UnifiedShaderBuildData {
            vertex_buffers,
            index_buffers,
            resource_groups: res,
        }
    }

    pub fn build(self, device: &Device) -> UnifiedShaderBuildData<'a, Buffer> {
        UnifiedShaderBuildData {
            vertex_buffers: self
                .vertex_buffers
                .into_iter()
                .map(|b| b.build(device))
                .collect(),
            index_buffers: self
                .index_buffers
                .into_iter()
                .map(|b| b.build(device))
                .collect(),
            resource_groups: self
                .resource_groups
                .into_iter()
                .map(|g| ResourceGroup {
                    buffers: g.buffers.into_iter().map(|b| b.build(device)).collect(),
                    textures: g.textures,
                    samplers: g.samplers,
                    acceleration_structures: g.acceleration_structures,
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct ResourceGroup<'a, I = ()> {
    pub buffers: Vec<TinyBuffer<I>>,
    pub textures: Vec<ResourceTexture<'a>>,
    pub samplers: Vec<TinySampler<'a>>,
    pub acceleration_structures: Vec<AccelerationStructures<'a>>,
}

#[derive(Clone)]
pub struct AccelerationStructures<'a> {
    pub blas: TinyBlas<'a>,
    pub tlas: TinyTlas<'a>,
}

pub struct DynamicBindGroup {
    pub layout: BindGroupLayout,
    pub bind_group_cache: LruCache<u64, BindGroup>,
}

impl DynamicBindGroup {
    pub fn new(layout: BindGroupLayout) -> Self {
        Self {
            layout,
            bind_group_cache: LruCache::new(NonZeroUsize::new(16).unwrap()),
        }
    }

    pub fn key<I>(bufs: &[ResourceType<I>]) -> u64 
        where I: Hash
    {
        let mut hasher = DefaultHasher::new();
        bufs.hash(&mut hasher);
        hasher.finish()
    }

    pub fn get_bind_group<I>(&mut self, bufs: &[ResourceType<I>]) -> Option<&BindGroup> 
        where I: Hash
    {
        let k = Self::key(bufs);

        self.bind_group_cache.get(&k)
    }

    pub fn insert<I>(&mut self, b: &[ResourceType<I>], bind_group: BindGroup) 
        where I: Hash
    {
        self.bind_group_cache.put(Self::key(b), bind_group);
    }

    pub fn get_or_create_bind_group<'a>(
        &'a mut self,
        buffs: &[ResourceType<Buffer>],
        device: &Device,
    ) -> &'a BindGroup 
    {
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
                            ResourceType::Buffer(b) => b.raw().as_entire_binding(),
                            ResourceType::Sampler(s) => BindingResource::Sampler(s.raw()),
                            ResourceType::Texture(t) => {
                                BindingResource::TextureView(t.texture.raw_view())
                            }
                            ResourceType::AccelerationStructure(t) => t.raw().as_binding(),
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
pub struct ResourceGroupBuildSpec<'a> {
    pub layout_entries: Vec<ResourceBinding<'a>>,
    pub layout: BindGroupLayout,
}

#[derive(Clone)]
pub struct BufferWithType<T, I = ()> {
    pub inner: TinyBuffer<I>,
    _p_d: PhantomData<T>,
}

impl<T, I> From<TinyBuffer<I>> for BufferWithType<T, I> {
    fn from(value: TinyBuffer<I>) -> Self {
        BufferWithType {
            inner: value,
            _p_d: PhantomData,
        }
    }
}

impl<T> From<Buffer> for BufferWithType<T, Buffer> {
    fn from(buffer: Buffer) -> Self {
        BufferWithType::from(TinyBuffer::<wgpu::Buffer>::from(buffer))
    }
}

pub trait AsByteSlice {
    fn as_byte_slice(&self) -> &[u8];
}

impl<T: Pod> AsByteSlice for T {
    fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl<T: Pod> AsByteSlice for [T] {
    fn as_byte_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }
}

impl<T> BufferWithType<T, Buffer>
where
    T: Pod,
{
    pub fn write<Q: AsByteSlice>(&self, queue: &wgpu::Queue, data: &Q) {
        queue.write_buffer(&self.inner.raw(), 0, data.as_byte_slice());
    }
}

#[derive(Clone)]
pub struct Buffers<'a> {
    pub vertex_buffers: Vec<TinyBuffer>,
    pub index_buffer: Option<TinyBuffer>,
    pub resource_buffers: Vec<ResourceGroup<'a>>,
}

pub struct ResourceEntry<'a, I> {
    pub binding: u32,
    pub resource: ResourceType<'a, I>,
}

#[derive(Clone, Hash)]
pub enum ResourceType<'a, I> {
    Buffer(TinyBuffer<I>),
    Sampler(TinySampler<'a>),
    Texture(ResourceTexture<'a>),
    AccelerationStructure(TinyTlas<'a>),
}

pub fn align_to_4_bytes(size: u64) -> u64 {
    ((size + 3) / 4) * 4
}
