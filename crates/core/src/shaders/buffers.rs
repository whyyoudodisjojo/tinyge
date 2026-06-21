use std::collections::VecDeque;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, Buffer,
    BufferDescriptor, BufferUsages, CommandEncoder,
};

pub struct ShaderResourceBindGroupsAndBuffers {
    pub buffers: Vec<Buffer>,
    pub bind_group: BindGroup,
}

pub struct ShaderBufferBuildSpec {
    pub vertex_buffer_szs: Vec<u64>,
    pub index_buffer_sz: u64,
    pub resource_buffer: ShaderResourceBuffersBuildSpec,
}

pub struct ShaderResourceBuffersBuildSpec {
    pub bind_groups: Vec<ShaderResourceBindGroupBuildSpec>,
    pub buffers: Vec<ShaderResourceRawBufferBuildSpec>,
}

pub struct ShaderResourceBindGroupBuildSpec {
    pub layout_entries: Vec<BindGroupLayoutEntry>,
    pub layout: BindGroupLayout,
}

pub struct ShaderResourceRawBufferBuildSpec {
    pub usage: BufferUsages,
    pub size: u64,
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

    pub fn build(device: &wgpu::Device, spec: ShaderBufferBuildSpec) -> Self {
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

        let mut resource_buffers_raw = spec
            .resource_buffer
            .buffers
            .into_iter()
            .map(|u| {
                device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: align_to_4_bytes(u.size),
                    usage: u.usage | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect::<VecDeque<_>>();

        let resource_buffers = spec
            .resource_buffer
            .bind_groups
            .into_iter()
            .map(|b| {
                let group_buffers = resource_buffers_raw
                    .drain(0..b.layout_entries.len())
                    .collect::<Vec<_>>();

                let entries = b
                    .layout_entries
                    .into_iter()
                    .zip(group_buffers.iter())
                    .enumerate()
                    .map(|(i, (_entry_spec, buffer))| BindGroupEntry {
                        binding: i as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>();

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &b.layout,
                    entries: &entries,
                });

                ShaderResourceBindGroupsAndBuffers {
                    buffers: group_buffers,
                    bind_group,
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
