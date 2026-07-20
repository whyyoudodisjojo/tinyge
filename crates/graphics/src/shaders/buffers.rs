use std::{
    hash::{DefaultHasher, Hash, Hasher}, marker::PhantomData, num::NonZeroUsize,
};

use bytemuck::{Pod};
use codegen::asts::IntoWgslStruct;
use lru::LruCache;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Blas, Buffer, BufferDescriptor, BufferUsages, Device, Sampler, Tlas, wgt::TextureViewDescriptor,
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
    pub acceleration_structures: Vec<AccelerationStructures>,
}

#[derive(Clone)]
pub struct AccelerationStructures {
    pub blas: Blas,
    pub tlas: Tlas,
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
                            ResourceType::AccelerationStructure(t) => t.as_binding(),
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

pub struct BufferWithType<T>
    where T: IntoWgslStruct
{
    pub inner: Buffer,
    _p_d: PhantomData<T>,
}

impl<T> From<Buffer> for BufferWithType<T>
    where T: IntoWgslStruct
{
    fn from(value: Buffer) -> Self {
        BufferWithType { inner: value, _p_d: PhantomData }
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
    T: IntoWgslStruct + Pod,
{
    pub fn write<Q: AsByteSlice>(&self, queue: &wgpu::Queue, data: &Q) {
        queue.write_buffer(&self.inner, 0, data.as_byte_slice());
    }
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
    AccelerationStructure(Tlas),
}

impl Buffers {
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
                let mut acceleration_structures = vec![];

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
                        ResourceBindingType::AccelerationStructure {
                            tlas_desc,
                            blas_desc,
                            blas_geo_sz_desc,
                            ..
                        } => {
                            let tlas = device.create_tlas(tlas_desc);
                            let blas = device.create_blas(blas_desc, blas_geo_sz_desc.clone());

                            acceleration_structures.push(AccelerationStructures { blas, tlas });
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
                    acceleration_structures,
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
