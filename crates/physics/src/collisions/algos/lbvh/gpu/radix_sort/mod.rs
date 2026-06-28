use tinyge_graphics::shaders::{ComputeShaderWrapper, buffers::Buffers};
use wgpu::{Buffer, Device};

use crate::collisions::algos::lbvh::Key;
use crate::collisions::algos::lbvh::gpu::radix_sort::phase::{RadixSortPhase, RadixSortStage};

pub mod phase;

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

pub struct RadixSort<'a> {
    count: ComputeShaderWrapper<'a, RadixSortPhase>,
    cumsum: ComputeShaderWrapper<'a, RadixSortPhase>,
    rearrange: ComputeShaderWrapper<'a, RadixSortPhase>,
    num_elems: u32,
    buffers: RadixSortInternalBuffers,
}

impl<'a> RadixSort<'a> {
    pub fn new(num_elems: u32, device: &Device) -> Self {
        let count = ComputeShaderWrapper::new(
            RadixSortPhase::new(num_elems, RadixSortStage::Count),
            device,
        );
        let cumsum = ComputeShaderWrapper::new(
            RadixSortPhase::new(num_elems, RadixSortStage::Cumsum),
            device,
        );
        let rearrange = ComputeShaderWrapper::new(
            RadixSortPhase::new(num_elems, RadixSortStage::Rearrange),
            device,
        );

        let buffers = Buffers::build(device, &count.buffer_build_spec.buffer_build_spec);

        let buffers = RadixSortInternalBuffers {
            param_buffer: buffers.resource_buffers[0].buffers[0].clone(),
            count_arr_buffer: buffers.resource_buffers[0].buffers[1].clone(),
            output_arr_buffer: buffers.resource_buffers[0].buffers[2].clone(),
            global_offsets_buffer: buffers.resource_buffers[0].buffers[3].clone(),
        };

        Self {
            count,
            cumsum,
            rearrange,
            num_elems,
            buffers,
        }
    }

    pub fn sort(&mut self, input_buffer: Buffer, device: &Device, queue: &wgpu::Queue) {
        let ping_buffer = input_buffer;
        let pong_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.num_elems as u64 * std::mem::size_of::<Key>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut current_input = ping_buffer;
        let mut current_output = pong_buffer;

        for shift in 0..8 {
            let params = Params {
                shift: shift * 4,
                num_elems: self.num_elems,
            };
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
            self.num_elems as u64 * std::mem::size_of::<Key>() as u64,
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

    fn create_input_buffer(device: &Device, data: &[Key]) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    #[test]
    fn test_radix_sort() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;

            let codes: Vec<u32> = vec![
                0x12345678, 0x87654321, 0xABCDEF00, 0x00FEDCBA, 0x55555555, 0xAAAAAAAA, 0x00000000,
                0xFFFFFFFF, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x99999999, 0x88888888,
                0x77777777, 0x66666666,
            ];

            let input_data: Vec<Key> = codes
                .iter()
                .enumerate()
                .map(|(idx, &code)| Key {
                    code,
                    idx: idx as u32,
                })
                .collect();
            let num_elems = input_data.len() as u32;

            let input_buffer = create_input_buffer(&device, &input_data);

            let mut radix_sort = RadixSort::new(num_elems, &device);
            radix_sort.sort(input_buffer, &device, &queue);

            let output_data =
                Key::read_buffer(&device, &queue, &radix_sort.buffers.output_arr_buffer);

            let mut expected: Vec<Key> = input_data.clone();
            expected.sort_by_key(|k| k.code);

            assert_eq!(
                output_data, expected,
                "Radix sort output doesn't match expected sorted data"
            );
        });
    }
}
