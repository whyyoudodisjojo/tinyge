use codegen_macros::{IntoBufferStruct, IntoWgslStruct};
use memory::buffers::BufferWithType;
use tinyge_graphics::shaders::ComputeShaderWrapper;
use wgpu::Device;

use crate::collisions::algos::lbvh::{
    Key,
    gpu::custom::radix_sort::phase::{RadixSortPhase, RadixSortStage},
};

pub mod phase;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, IntoWgslStruct)]
pub struct Params {
    pub shift: u32,
    pub num_elems: u32,
}

#[derive(Clone, IntoBufferStruct)]
pub struct RadixSortPhaseArgs {
    #[resource(bindgroup_index = 0, resource_index = 0, ty = "buffer")]
    pub param_buffer: BufferWithType<Params>,
    #[resource(bindgroup_index = 0, resource_index = 1, ty = "buffer")]
    pub input_arr_buffer: BufferWithType<Vec<Key>>,
    #[resource(bindgroup_index = 0, resource_index = 2, ty = "buffer")]
    pub count_arr_buffer: BufferWithType<[u32; 16]>,
    #[resource(bindgroup_index = 0, resource_index = 3, ty = "buffer")]
    pub output_arr_buffer: BufferWithType<Vec<Key>>,
    #[resource(bindgroup_index = 0, resource_index = 4, ty = "buffer")]
    pub global_offsets_buffer: BufferWithType<[u32; 16]>,
}
#[derive(Clone)]
pub struct RadixSortInternalBuffers {
    pub param_buffer: BufferWithType<Params>,
    pub count_arr_buffer: BufferWithType<[u32; 16]>,
    pub output_arr_buffer: BufferWithType<Vec<Key>>,
    pub global_offsets_buffer: BufferWithType<[u32; 16]>,
}

pub struct RadixSort {
    count: ComputeShaderWrapper<RadixSortPhase, RadixSortPhaseArgs>,
    cumsum: ComputeShaderWrapper<RadixSortPhase, RadixSortPhaseArgs>,
    rearrange: ComputeShaderWrapper<RadixSortPhase, RadixSortPhaseArgs>,
    num_elems: u32,
    buffers: RadixSortInternalBuffers,
}

impl RadixSort {
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

        let buffers = RadixSortInternalBuffers {
            param_buffer: count
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .param_buffer
                .clone(),
            count_arr_buffer: count
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .count_arr_buffer
                .clone(),
            output_arr_buffer: count
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .output_arr_buffer
                .clone(),
            global_offsets_buffer: count
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .global_offsets_buffer
                .clone(),
        };

        Self {
            count,
            cumsum,
            rearrange,
            num_elems,
            buffers,
        }
    }

    pub fn sort(
        &mut self,
        input_buffer: BufferWithType<Vec<Key>>,
        device: &Device,
        queue: &wgpu::Queue,
    ) {
        let ping_buffer = input_buffer.inner.inner.unwrap();
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
            queue.write_buffer(
                self.buffers.param_buffer.inner.raw(),
                0,
                bytemuck::bytes_of(&params),
            );

            let args = RadixSortPhaseArgs {
                param_buffer: self.buffers.param_buffer.clone(),
                input_arr_buffer: BufferWithType::<Vec<Key>>::from(current_input.clone()),
                count_arr_buffer: self.buffers.count_arr_buffer.clone(),
                output_arr_buffer: BufferWithType::<Vec<Key>>::from(current_output.clone()),
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
            self.buffers.output_arr_buffer.inner.raw(),
            0,
            self.num_elems as u64 * std::mem::size_of::<Key>() as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));
    }
}
