use tinyge_graphics::shaders::{
    ComputeShader, ComputeShaderBuiltData,
    buffers::ResourceType,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{BufferUsages, ComputePassDescriptor, ShaderStages, wgt::CommandEncoderDescriptor};

use crate::collisions::algos::lbvh::gpu::radix_sort::{Params, RadixSortPhaseArgs};

pub struct RadixSortCumsumPhase {
    num_elems: u32,
}

impl RadixSortCumsumPhase {
    pub fn new(num_elems: u32) -> Self {
        Self { num_elems }
    }
}
impl<'a> ComputeShader<'a> for RadixSortCumsumPhase {
    type Args = RadixSortPhaseArgs;
    type Ret = ();

    fn entry_point(&self) -> &'static str {
        "cumsum"
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("../../../shaders/lbvh/radix_sort.wgsl")
    }

    fn resource_buffers_with_bind_group_layouts(
        &self,
    ) -> Vec<tinyge_graphics::shaders::descriptors::ResourceGroupLayout<'a>> {
        vec![ResourceGroupLayout {
            entries: vec![
                ResourceBinding {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: size_of::<Params>() as u64,
                        usages: BufferUsages::UNIFORM,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: self.num_elems as u64 * size_of::<u32>() as u64,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: 16 * 4,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: self.num_elems as u64 * size_of::<u32>() as u64,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: 16 * 4,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        built_data: &mut ComputeShaderBuiltData,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(
            &[
                ResourceType::Buffer(args.param_buffer.clone()),
                ResourceType::Buffer(args.input_arr_buffer.clone()),
                ResourceType::Buffer(args.count_arr_buffer.clone()),
                ResourceType::Buffer(args.output_arr_buffer.clone()),
                ResourceType::Buffer(args.global_offsets_buffer.clone()),
            ],
            device,
        );
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&built_data.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}
