use std::{
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    num::NonZeroUsize,
};

use bytemuck::Pod;
use lru::LruCache;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    Device,
};

use crate::{
    descriptors::{ResourceBinding},
    socket::{TinyBlas, TinyBuffer, TinySampler, TinyTlas},
    texture::ResourceTexture,
};

#[derive(Clone)]
pub struct ResourceGroup<'a> {
    pub buffers: Vec<TinyBuffer>,
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
pub struct BufferWithType<T> {
    pub inner: TinyBuffer,
    _p_d: PhantomData<T>,
}

impl<T> From<TinyBuffer> for BufferWithType<T> {
    fn from(value: TinyBuffer) -> Self {
        BufferWithType {
            inner: value,
            _p_d: PhantomData,
        }
    }
}

impl<T> From<Buffer> for BufferWithType<T> {
    fn from(buffer: Buffer) -> Self {
        BufferWithType::from(TinyBuffer::from(buffer))
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

impl<T> BufferWithType<T>
where
    T: Pod,
{
    pub fn write<Q: AsByteSlice>(&self, queue: &wgpu::Queue, data: &Q) {
        queue.write_buffer(self.inner.raw(), 0, data.as_byte_slice());
    }
}

#[derive(Clone)]
pub struct Buffers<'a> {
    pub vertex_buffers: Vec<TinyBuffer>,
    pub index_buffer: Option<TinyBuffer>,
    pub resource_buffers: Vec<ResourceGroup<'a>>,
}

pub struct ResourceEntry<'a> {
    pub binding: u32,
    pub resource: ResourceType<'a>,
}

#[derive(Clone, Hash)]
pub enum ResourceType<'a> {
    Buffer(TinyBuffer),
    Sampler(TinySampler<'a>),
    Texture(ResourceTexture<'a>),
    AccelerationStructure(TinyTlas<'a>),
}

pub fn align_to_4_bytes(size: u64) -> u64 {
    ((size + 3) / 4) * 4
}
