pub mod radix_count;
pub mod radix_scan;
pub mod radix_scatter;

use crate::collisions::algos::lbvh::Key;
use radix_count::{RadixSortCount, RadixSortCountArgs};
use radix_scan::{RadixScan, RadixScanArgs};
use radix_scatter::{RadixScatter, RadixScatterArgs};
use tinyge_graphics::shaders::ComputeShader;

pub struct RadixSort {
    radix_counter: RadixSortCount,
    radix_scan: RadixScan,
    radix_scatter: RadixScatter,
    buffer_a: wgpu::Buffer,
    buffer_b: wgpu::Buffer,
}

impl RadixSort {
    pub fn new(num_elems: u64, device: &wgpu::Device) -> Self {
        let keys_bytes = num_elems * std::mem::size_of::<Key>() as u64;

        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: keys_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix Sort Buffer B"),
            size: keys_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let radix_counter = RadixSortCount {
            init_data: None,
            num_elems,
        };

        let radix_scan = RadixScan {
            num_elems,
            init_data: None,
        };

        let radix_scatter = RadixScatter {
            num_elems,
            init_data: None,
        };

        Self {
            radix_counter,
            radix_scan,
            radix_scatter,
            buffer_a,
            buffer_b,
        }
    }

    pub fn sort_pass(
        &mut self,
        keys: &[Key],
        shift_bits: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
    ) -> wgpu::Buffer {
        let counter_buffer = self.radix_counter.dispatch(
            RadixSortCountArgs {
                in_keys: keys.to_vec(),
                shift_bits,
            },
            device,
            queue,
        );

        let global_counters = self.radix_scan.dispatch(
            RadixScanArgs {
                counter_buf: counter_buffer,
            },
            device,
            queue,
        );

        self.radix_scatter.dispatch(
            RadixScatterArgs {
                in_keys: keys.to_vec(),
                global_counters,
                shift_bits,
                in_buffer: src_buffer.clone(),
                out_buffer: dst_buffer.clone(),
            },
            device,
            queue,
        )
    }

    pub fn sort(
        &mut self,
        keys: &[Key],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> &wgpu::Buffer {
        queue.write_buffer(&self.buffer_a, 0, bytemuck::cast_slice(keys));

        for pass in 0..8 {
            let shift_bits = pass * 4;
            let (src_buffer, dst_buffer) = if pass % 2 == 0 {
                (self.buffer_a.clone(), self.buffer_b.clone())
            } else {
                (self.buffer_b.clone(), self.buffer_a.clone())
            };

            let current_keys = Key::read_buffer(device, queue, &src_buffer);

            self.sort_pass(
                &current_keys,
                shift_bits,
                device,
                queue,
                &src_buffer,
                &dst_buffer,
            );
        }

        &self.buffer_a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collisions::algos::lbvh::Key;
    use wgpu::{Backends, DeviceDescriptor, Instance, InstanceDescriptor, RequestAdapterOptions};

    #[test]
    fn test_full_32_bit_radix_sort() {
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

        let keys = vec![
            Key {
                code: 0x12345678,
                idx: 0,
            },
            Key {
                code: 0x00000005,
                idx: 1,
            },
            Key {
                code: 0xF0000000,
                idx: 2,
            },
            Key {
                code: 0x0F000000,
                idx: 3,
            },
            Key {
                code: 0x00F00000,
                idx: 4,
            },
            Key {
                code: 0x000F0000,
                idx: 5,
            },
            Key {
                code: 0x0000F000,
                idx: 6,
            },
            Key {
                code: 0x00000F00,
                idx: 7,
            },
            Key {
                code: 0x000000F0,
                idx: 8,
            },
            Key {
                code: 0x0000000F,
                idx: 9,
            },
            Key {
                code: 0xFFFFFFFF,
                idx: 10,
            },
            Key {
                code: 0x00000000,
                idx: 11,
            },
            Key {
                code: 0xAAAAAAAA,
                idx: 12,
            },
            Key {
                code: 0x12345678,
                idx: 14,
            },
            Key {
                code: 0x87654321,
                idx: 15,
            },
        ];

        let mut radix_sort = RadixSort::new(keys.len() as u64, &device);
        let sorted_keys_buffer = radix_sort.sort(&keys, &device, &queue);

        let sorted_keys = Key::read_buffer(&device, &queue, sorted_keys_buffer);

        assert_eq!(
            sorted_keys.len(),
            keys.len(),
            "Sorted keys length should match input"
        );

        for i in 1..sorted_keys.len() {
            assert!(
                sorted_keys[i].code >= sorted_keys[i - 1].code,
                "Keys should be in ascending order: {} at index {} is less than {} at index {}",
                sorted_keys[i].code,
                i,
                sorted_keys[i - 1].code,
                i - 1
            );
        }

        let mut original_codes: Vec<u32> = keys.iter().map(|k| k.code).collect();
        let mut sorted_codes: Vec<u32> = sorted_keys.iter().map(|k| k.code).collect();
        original_codes.sort();
        sorted_codes.sort();
        assert_eq!(
            original_codes, sorted_codes,
            "Sorted output should contain the same keys as input"
        );
    }
}
