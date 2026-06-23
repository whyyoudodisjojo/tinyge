use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder, Texture, TextureView,
    wgt::TextureViewDescriptor,
};

use crate::shaders::descriptors::{ResourceBinding, ResourceBindingType};

pub struct ResourceGroup {
    pub buffers: Vec<Buffer>,
    pub bind_group: BindGroup,
    pub textures: Vec<ResourceTexture>,
}

pub struct ResourceTexture {
    pub texture: Texture,
    pub view: TextureView,
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

pub struct Buffers {
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffer: Buffer,
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
        Self::copy_via_encoder(&self.index_buffer, &new_buffers.index_buffer, &mut encoder);
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
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: align_to_4_bytes(spec.index_buffer_sz),
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let resource_buffers = spec
            .resource_buffer
            .into_iter()
            .map(|b| {
                let textures = b
                    .layout_entries
                    .iter()
                    .filter_map(|l| {
                        let ResourceBindingType::Texture {
                            texture_descriptor, ..
                        } = &l.ty
                        else {
                            return None;
                        };

                        let texture = device.create_texture(&texture_descriptor);
                        let view = texture.create_view(&TextureViewDescriptor::default());

                        Some(ResourceTexture { texture, view })
                    })
                    .collect::<Vec<_>>();

                let sampler = b
                    .layout_entries
                    .iter()
                    .filter_map(|l| {
                        let ResourceBindingType::Sampler {
                            sampler_descriptor, ..
                        } = &l.ty
                        else {
                            return None;
                        };

                        Some(device.create_sampler(sampler_descriptor))
                    })
                    .collect::<Vec<_>>();

                let buffers = b
                    .layout_entries
                    .iter()
                    .filter_map(|l| {
                        let ResourceBindingType::Buffer { size, usages, .. } = l.ty else {
                            return None;
                        };

                        Some(device.create_buffer(&BufferDescriptor {
                            label: None,
                            usage: usages | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                            size: align_to_4_bytes(size),
                            mapped_at_creation: false,
                        }))
                    })
                    .collect::<Vec<_>>();

                let entries = buffers
                    .iter()
                    .map(|b| b.as_entire_binding())
                    .chain(
                        textures
                            .iter()
                            .map(|t| BindingResource::TextureView(&t.view)),
                    )
                    .chain(sampler.iter().map(|s| BindingResource::Sampler(s)))
                    .enumerate()
                    .map(|(i, b)| BindGroupEntry {
                        binding: i as u32,
                        resource: b,
                    })
                    .collect::<Vec<_>>();

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &b.layout,
                    entries: &entries,
                });

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
