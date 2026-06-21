use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, BufferDescriptor,
    BufferUsages, CommandEncoder,
};

use crate::shaders::{align_to_4_bytes, descriptors::BindGroupLayoutDescriptorOwned};

pub struct ShaderResourceBindGroupsAndBuffers {
    pub buffers: Vec<Buffer>,
    pub bind_group: BindGroup,
}
pub struct ShaderBuffers {
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffer: Buffer,
    pub resource_buffers: Vec<ShaderResourceBindGroupsAndBuffers>,
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
            .for_each(|(o, n)| {
                o.buffers
                    .iter()
                    .zip(n.buffers.iter())
                    .for_each(|(o, n)| Self::copy_via_encoder(o, n, &mut encoder))
            });

        queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn build(
        device: &wgpu::Device,
        vertex_buffer_sizes: Vec<u64>,
        index_buffer_size: u64,
        resource_buffer_sizes: Vec<u64>,
        bind_group_layout_desc: Vec<BindGroupLayoutDescriptorOwned>,
        bind_group_layouts: Vec<BindGroupLayout>,
        usages: Vec<BufferUsages>,
    ) -> Self {
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
            .zip(bind_group_layout_desc.iter())
            .zip(bind_group_layouts.iter())
            .zip(usages.into_iter())
            .map(|(((size, desc), layout), usage)| {
                let buffers = desc
                    .entries
                    .iter()
                    .map(|_| {
                        device.create_buffer(&BufferDescriptor {
                            label: None,
                            size: align_to_4_bytes(size),
                            usage: usage | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        })
                    })
                    .collect::<Vec<_>>();

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &buffers
                        .iter()
                        .enumerate()
                        .map(|(i, b)| BindGroupEntry {
                            binding: i as u32,
                            resource: b.as_entire_binding(),
                        })
                        .collect::<Vec<_>>(),
                });

                ShaderResourceBindGroupsAndBuffers {
                    buffers,
                    bind_group,
                }
            })
            .collect::<Vec<_>>();

        Self {
            vertex_buffers,
            index_buffer,
            resource_buffers,
        }
    }
}
