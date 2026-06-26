use tinyge_graphics::shaders::{ComputeShader, buffers::DynamicBindGroup};
use wgpu::{Buffer, ComputePipeline, Device};

use crate::collisions::algos::lbvh::gpu::radix_sort::{
    count::RadixSortCountPhase, cumsum::RadixSortCumsumPhase, rearrange::RadixSortRearrangePhase,
};

pub mod count;
pub mod cumsum;
pub mod rearrange;

struct InitData {
    bind_group: DynamicBindGroup,
    pipeline: ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub shift: u32,
    pub num_elems: u32,
}

#[derive(Clone)]
pub struct RadixSortPhaseArgs {
    pub param_buffer: Buffer,
    pub input_arr_buffer: Buffer,
    pub count_arr_buffer: Buffer,
    pub output_arr_buffer: Buffer,
    pub global_offsets_buffer: Buffer,
}
#[derive(Clone)]
pub struct RadixSortInternalBuffers {
    pub param_buffer: Buffer,
    pub count_arr_buffer: Buffer,
    pub output_arr_buffer: Buffer,
    pub global_offsets_buffer: Buffer,
}

pub struct RadixSort {
    count: RadixSortCountPhase,
    cumsum: RadixSortCumsumPhase,
    rearrange: RadixSortRearrangePhase,
    num_elems: u32,
    buffers: RadixSortInternalBuffers,
}

impl RadixSort {
    pub fn new(num_elems: u32, device: &Device) -> Self {
        Self {
            count: RadixSortCountPhase::new(num_elems),
            cumsum: RadixSortCumsumPhase::new(num_elems),
            rearrange: RadixSortRearrangePhase::new(num_elems),
            num_elems,
            buffers: Self::create_buffers(num_elems, device),
        }
    }

    pub fn create_buffers(num_elems: u32, device: &Device) -> RadixSortInternalBuffers {
        let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_arr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let output_arr_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: num_elems as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_offsets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        RadixSortInternalBuffers {
            param_buffer,
            count_arr_buffer,
            output_arr_buffer,
            global_offsets_buffer,
        }
    }

    pub fn sort(&mut self, input_buffer: Buffer, device: &Device, queue: &wgpu::Queue) {
        let ping_buffer = input_buffer;
        let pong_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.num_elems as u64 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut current_input = ping_buffer;
        let mut current_output = pong_buffer;

        for shift in 0..8 {
            let params = Params { shift: shift * 4, num_elems: self.num_elems };
            queue.write_buffer(&self.buffers.param_buffer, 0, bytemuck::bytes_of(&params));

            let args = RadixSortPhaseArgs {
                param_buffer: self.buffers.param_buffer.clone(),
                input_arr_buffer: current_input.clone(),
                count_arr_buffer: self.buffers.count_arr_buffer.clone(),
                output_arr_buffer: current_output.clone(),
                global_offsets_buffer: self.buffers.global_offsets_buffer.clone(),
            };

            self.count.dispatch(args.clone(), device, queue);
            self.cumsum.dispatch(args.clone(), device, queue);
            self.rearrange.dispatch(args.clone(), device, queue);

            std::mem::swap(&mut current_input, &mut current_output);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            &current_input,
            0,
            &self.buffers.output_arr_buffer,
            0,
            self.num_elems as u64 * std::mem::size_of::<u32>() as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    use super::*;

    async fn setup_wgpu() -> (Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create device");

        (device, queue)
    }

    fn create_input_buffer(device: &Device, data: &[u32]) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    async fn read_buffer(
        device: &Device,
        queue: &wgpu::Queue,
        buffer: &Buffer,
        size: u64,
    ) -> Vec<u32> {
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        let index = queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: None,
            })
            .ok();
        receiver.recv().unwrap().expect("Failed to map buffer");

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    #[test]
    fn test_radix_sort() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;

            let input_data: Vec<u32> = vec![
                0x12345678, 0x87654321, 0xABCDEF00, 0x00FEDCBA, 0x55555555, 0xAAAAAAAA, 0x00000000,
                0xFFFFFFFF, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x99999999, 0x88888888,
                0x77777777, 0x66666666,
            ];
            let num_elems = input_data.len() as u32;

            let input_buffer = create_input_buffer(&device, &input_data);

            let mut radix_sort = RadixSort::new(num_elems, &device);
            radix_sort.sort(input_buffer, &device, &queue);

            let output_size = num_elems as u64 * std::mem::size_of::<u32>() as u64;
            let output_data = read_buffer(
                &device,
                &queue,
                &radix_sort.buffers.output_arr_buffer,
                output_size,
            )
            .await;

            let mut expected = input_data.clone();
            expected.sort();

            assert_eq!(
                output_data, expected,
                "Radix sort output doesn't match expected sorted data"
            );
        });
    }
}
