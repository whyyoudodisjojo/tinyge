pub mod radix_count;
pub mod radix_scan;
pub mod radix_scatter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collisions::algos::lbvh::Key;
    use radix_count::{RadixSortCount, RadixSortCountArgs};
    use radix_scan::{RadixScan, RadixScanArgs};
    use radix_scatter::{RadixScatter, RadixScatterArgs};
    use tinyge_graphics::shaders::ComputeShader;
    use wgpu::{
        Backends, Buffer, BufferDescriptor, BufferUsages, DeviceDescriptor, Instance,
        InstanceDescriptor, RequestAdapterOptions,
    };

    fn read_buffer_to_keys(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &Buffer,
    ) -> Vec<Key> {
        let size = buffer.size();

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
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
            let result: Vec<Key> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to read data back from the staging buffer copy!");
        }
    }

    #[test]
    fn test_full_radix_sort_pass() {
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
            Key { code: 5, idx: 0 },
            Key { code: 12, idx: 1 },
            Key { code: 5, idx: 2 },
            Key { code: 0, idx: 3 },
            Key { code: 15, idx: 4 },
            Key { code: 3, idx: 5 },
            Key { code: 8, idx: 6 },
            Key { code: 2, idx: 7 },
        ];
        let num_elems = keys.len() as u64;

        let mut radix_counter = RadixSortCount {
            init_data: None,
            num_elems,
        };

        let count_args = RadixSortCountArgs {
            in_keys: keys.clone(),
            shift_bits: 0,
        };

        let counter_buffer = radix_counter.dispatch(count_args, &device, &queue);

        let mut radix_scan = RadixScan {
            num_elems,
            init_data: None,
        };

        let scan_args = RadixScanArgs {
            counter_buf: counter_buffer,
        };

        let global_counters = radix_scan.dispatch(scan_args, &device, &queue);

        let mut radix_scatter = RadixScatter {
            num_elems,
            init_data: None,
        };

        let scatter_args = RadixScatterArgs {
            in_keys: keys.clone(),
            global_counters: global_counters.clone(),
            shift_bits: 0,
        };

        let sorted_keys_buffer = radix_scatter.dispatch(scatter_args, &device, &queue);

        let sorted_keys = read_buffer_to_keys(&device, &queue, &sorted_keys_buffer);

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
