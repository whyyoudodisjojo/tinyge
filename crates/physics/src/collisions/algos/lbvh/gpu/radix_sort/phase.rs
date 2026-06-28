use tinyge_graphics::shaders::{
    ComputeShader, ComputeShaderBuiltData,
    buffers::ResourceType,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{BufferUsages, ComputePassDescriptor, ShaderStages, wgt::CommandEncoderDescriptor};

use crate::collisions::algos::lbvh::{
    Key,
    gpu::radix_sort::{Params, RadixSortPhaseArgs},
};

pub enum RadixSortStage {
    Count,
    Cumsum,
    Rearrange,
}

pub struct RadixSortPhase {
    num_elems: u32,
    stage: RadixSortStage,
}

impl RadixSortPhase {
    pub fn new(num_elems: u32, stage: RadixSortStage) -> Self {
        Self { num_elems, stage }
    }
}

impl<'a> ComputeShader<'a> for RadixSortPhase {
    type Args = RadixSortPhaseArgs;
    type Ret = ();

    fn entry_point(&self) -> &'static str {
        match &self.stage {
            RadixSortStage::Count => "count",
            RadixSortStage::Cumsum => "cumsum",
            RadixSortStage::Rearrange => "rearrange",
        }
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
                        is_input: false,
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
                        size: self.num_elems as u64 * size_of::<Key>() as u64,
                        usages: BufferUsages::STORAGE,
                        is_input: true,
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
                        is_input: false,
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
                        size: self.num_elems as u64 * size_of::<Key>() as u64,
                        usages: BufferUsages::STORAGE,
                        is_input: false,
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
                        is_input: false,
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
        let num_wg = match &self.stage {
            RadixSortStage::Count | RadixSortStage::Rearrange => {
                ((self.num_elems + 255) / 256).max(1)
            }
            RadixSortStage::Cumsum => 1,
        };

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(
            &[
                ResourceType::Buffer(args.param_buffer),
                ResourceType::Buffer(args.input_arr_buffer),
                ResourceType::Buffer(args.count_arr_buffer),
                ResourceType::Buffer(args.output_arr_buffer),
                ResourceType::Buffer(args.global_offsets_buffer),
            ],
            device,
        );
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&built_data.pipeline);
            pass.set_bind_group(0, Some(bind_group), &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}
