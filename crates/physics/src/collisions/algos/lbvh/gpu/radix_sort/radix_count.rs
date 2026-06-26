use std::hash::{DefaultHasher, Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use tinyge_graphics::shaders::{
    ComputeShader,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{
    BindGroup, Buffer, BufferUsages, ComputePassDescriptor, ComputePipeline, Device, Queue,
    ShaderStages, wgt::CommandEncoderDescriptor,
};

use crate::collisions::algos::lbvh::Key;

pub struct RadixSortCount {
    init_data: Option<InitData>,
    num_elems: u64,
}

pub struct InitData {
    in_keys: Buffer,
    global_counter: Buffer,
    params: Buffer,
    bind_group: BindGroup,
    pipeline: ComputePipeline,
    current_keys_hash: u64,
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
struct RadixSortCountParams {
    num_elems: u32,
    shift_bits: u32,
}

#[derive(Hash)]
pub struct RadixSortCountArgs {
    pub in_keys: Vec<Key>,
    pub shift_bits: u32,
}

impl ComputeShader for RadixSortCount {
    type Args = RadixSortCountArgs;
    type Ret = Buffer;
    fn entry_point(&self) -> &'static str {
        "count"
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("../../../shaders/lbvh/radix_sort/count.wgsl")
    }

    fn resource_buffers_with_bind_group_layouts<'a>(
        &'a self,
    ) -> Vec<tinyge_graphics::shaders::descriptors::ResourceGroupLayout<'a>> {
        let keys_bytes = self.num_elems * size_of::<Key>() as u64;

        let threads_per_wg = 256u64;
        let num_workgroups = (self.num_elems + threads_per_wg - 1) / threads_per_wg;
        let global_counters_bytes = num_workgroups * 16 * size_of::<u32>() as u64;

        vec![ResourceGroupLayout {
            entries: vec![
                ResourceBinding {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: keys_bytes,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: global_counters_bytes,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: size_of::<RadixSortCountParams>() as u64,
                        usages: BufferUsages::UNIFORM,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn dispatch(&mut self, args: Self::Args, device: &Device, queue: &Queue) -> Self::Ret {
        if self.init_data.is_none() {
            let built_data = self.build(device);
            let buffers = &built_data.buffers.resource_buffers[0];

            self.init_data = Some(InitData {
                in_keys: buffers.buffers[0].clone(),
                global_counter: buffers.buffers[1].clone(),
                params: buffers.buffers[2].clone(),
                bind_group: buffers.bind_group.clone(),
                current_keys_hash: 0,
                pipeline: built_data.pipeline.clone(),
            });
        }

        let init_data = self.init_data.as_mut().unwrap();
        let threads_per_wg = 256u32;
        let num_wg = (self.num_elems as u32 + threads_per_wg - 1) / threads_per_wg;

        let params = RadixSortCountParams {
            num_elems: self.num_elems as u32,
            shift_bits: args.shift_bits,
        };

        queue.write_buffer(&init_data.params, 0, bytemuck::bytes_of(&params));

        let mut curr_hash = DefaultHasher::new();
        args.in_keys.hash(&mut curr_hash);
        let curr_hash = curr_hash.finish();

        if init_data.current_keys_hash != curr_hash {
            queue.write_buffer(&init_data.in_keys, 0, bytemuck::cast_slice(&args.in_keys));
            init_data.current_keys_hash = curr_hash;
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&init_data.pipeline);
            compute_pass.set_bind_group(0, &init_data.bind_group, &[]);
            compute_pass.dispatch_workgroups(num_wg, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        init_data.global_counter.clone()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Backends, DeviceDescriptor, Instance, InstanceDescriptor, RequestAdapterOptions};

    fn read_buffer_to_vec(device: &Device, queue: &Queue, src_buffer: &Buffer) -> Vec<u32> {
        let size = src_buffer.size();

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Counter Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(src_buffer, 0, &staging_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
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
            let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to read data back from the staging buffer copy!");
        }
    }

    #[test]
    fn test_radix_sort_count_pass() {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Default::default(),
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default()))
                .expect("Failed to find an available GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&DeviceDescriptor::default()))
                .expect("Failed to initialize GPU device");

        let mock_keys = vec![
            Key { code: 5, idx: 0 },  // Digit 5
            Key { code: 12, idx: 1 }, // Digit 12 (C)
            Key { code: 5, idx: 2 },  // Digit 5 (Duplicate)
            Key { code: 0, idx: 3 },  // Digit 0
            Key { code: 15, idx: 4 }, // Digit 15 (F)
        ];

        let mut radix_counter = RadixSortCount {
            init_data: None,
            num_elems: mock_keys.len() as u64,
        };

        let args = RadixSortCountArgs {
            in_keys: mock_keys,
            shift_bits: 0,
        };

        let counter_buffer = radix_counter.dispatch(args, &device, &queue);

        let counters = read_buffer_to_vec(&device, &queue, &counter_buffer);

        assert_eq!(
            counters.len(),
            16,
            "Buffer layout must match 16 buckets per workgroup"
        );

        assert_eq!(counters[0], 1, "Bucket 0 should have exactly 1 item");
        assert_eq!(counters[5], 2, "Bucket 5 should have exactly 2 items");
        assert_eq!(counters[12], 1, "Bucket 12 should have exactly 1 item");
        assert_eq!(counters[15], 1, "Bucket 15 should have exactly 1 item");

        assert_eq!(counters[1], 0, "Bucket 1 should be completely empty");
        assert_eq!(counters[7], 0, "Bucket 7 should be completely empty");
    }
}
