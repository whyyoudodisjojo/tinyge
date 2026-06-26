use bytemuck::{Pod, Zeroable};
use tinyge_graphics::shaders::{
    ComputeShader,
    buffers::{DynamicBindGroup, ResourceType},
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{
    Buffer, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ShaderStages,
};

use crate::collisions::algos::lbvh::Key;

pub struct RadixScatter {
    pub num_elems: u64,
    pub init_data: Option<InitData>,
}

pub struct InitData {
    pub in_keys: Buffer,
    pub out_keys: Buffer,
    pub params: Buffer,
    pub bind_group: DynamicBindGroup,
    pub pipeline: ComputePipeline,
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
struct RadixScatterParams {
    num_elems: u32,
    shift_bits: u32,
}

pub struct RadixScatterArgs {
    pub in_keys: Vec<Key>,
    pub global_counters: Buffer,
    pub shift_bits: u32,
}

impl ComputeShader for RadixScatter {
    type Args = RadixScatterArgs;
    type Ret = Buffer;

    fn entry_point(&self) -> &'static str {
        "radix_scatter"
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("../../../shaders/lbvh/radix_sort/scatter.wgsl")
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
                        size: keys_bytes,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: global_counters_bytes,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: size_of::<RadixScatterParams>() as u64,
                        usages: BufferUsages::UNIFORM,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        if self.init_data.is_none() {
            let built_data = self.build(device);
            let resource_buffers = &built_data.buffers.resource_buffers[0];

            self.init_data = Some(InitData {
                in_keys: resource_buffers.buffers[0].clone(),
                out_keys: resource_buffers.buffers[1].clone(),
                params: resource_buffers.buffers[3].clone(),
                bind_group: resource_buffers.bind_group.clone(),
                pipeline: built_data.pipeline.clone(),
            });
        }

        let init_data = self.init_data.as_mut().unwrap();

        let threads_per_wg = 256u32;
        let num_wg = (self.num_elems as u32 + threads_per_wg - 1) / threads_per_wg;

        let params = RadixScatterParams {
            num_elems: self.num_elems as u32,
            shift_bits: args.shift_bits,
        };

        queue.write_buffer(&init_data.params, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&init_data.in_keys, 0, bytemuck::cast_slice(&args.in_keys));

        let bind_group_entries = vec![
            ResourceType::Buffer(init_data.in_keys.clone()),
            ResourceType::Buffer(init_data.out_keys.clone()),
            ResourceType::Buffer(args.global_counters.clone()),
            ResourceType::Buffer(init_data.params.clone()),
        ];
        let bind_group = init_data
            .bind_group
            .get_or_create_bind_group(&bind_group_entries, device);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&init_data.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_wg, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        init_data.out_keys.clone()
    }
}
