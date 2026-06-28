use bytemuck::{Pod, Zeroable};
use glam::{Vec3A, Vec4};

pub mod cpu;
pub mod gpu;

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
pub struct FlattenedBVHNode {
    pub min: Vec4,
    pub max: Vec4,
    pub parent: i32,
    pub left_child: i32,
    pub right_child: i32,
    pub node_type: u32,
}

impl FlattenedBVHNode {
    pub const fn size_in_bytes() -> usize {
        48
    }

    pub fn read_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<FlattenedBVHNode> {
        use std::sync::mpsc;
        use wgpu::{BufferDescriptor, BufferUsages, MapMode};

        let size = buffer.size();

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read FlattenedBVHNodes"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = mpsc::channel();

        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<FlattenedBVHNode> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to read FlattenedBVHNodes back from buffer!");
        }
    }
}

#[repr(C)]
#[derive(Hash, Pod, Zeroable, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Key {
    pub code: u32,
    pub idx: u32,
}

impl Key {
    /// Reads a buffer of Keys from the GPU.
    pub fn read_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<Key> {
        use std::sync::mpsc;
        use wgpu::{BufferDescriptor, BufferUsages, MapMode};

        let size = buffer.size();

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Buffer"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = mpsc::channel();

        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<Key> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to read data back from buffer!");
        }
    }

    pub fn mortonize(mut x: u32) -> u32 {
        x &= 0x000003ff;
        x = (x | (x << 16)) & 0xff0000ff;
        x = (x | (x << 8)) & 0x0300f00f;
        x = (x | (x << 4)) & 0x030c30c3;
        x = (x | (x << 2)) & 0x09249249;
        x
    }

    pub fn new(centroid: Vec3A, global_min: Vec3A, global_max: Vec3A, idx: usize) -> Self {
        let sz = global_max - global_min;
        let mask = sz.cmpgt(Vec3A::ZERO);
        let inv_sz = Vec3A::select(mask, Vec3A::ONE / sz, Vec3A::ZERO);
        let norm = (centroid - global_min) * inv_sz;
        let quant = norm.clamp(Vec3A::ZERO, Vec3A::ONE) * 1023.0;
        let u = quant.as_uvec3();
        let code = (Self::mortonize(u.x) << 2) | (Self::mortonize(u.y) << 1) | Self::mortonize(u.z);
        Self {
            code,
            idx: idx as u32,
        }
    }
}
