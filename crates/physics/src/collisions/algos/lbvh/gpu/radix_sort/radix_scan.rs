use bytemuck::{Pod, Zeroable};
use tinyge_graphics::shaders::{
    ComputeShader,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{
    BindGroup, Buffer, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ShaderStages,
};

pub struct RadixScan {
    pub num_elems: u64,
    pub init_data: Option<InitData>,
}

pub struct InitData {
    pub params: Buffer,
    pub counter: Buffer,
    pub bind_group: BindGroup,
    pub pipeline: ComputePipeline,
    pub current_counter_hash: u64,
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
struct RadixScanParams {
    num_elems: u32,
}

pub struct RadixScanArgs {
    pub counter_buf: Buffer,
}

impl ComputeShader for RadixScan {
    type Args = RadixScanArgs;
    type Ret = Buffer;

    fn entry_point(&self) -> &'static str {
        "radix_scan"
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("../../../shaders/lbvh/radix_sort/scan.wgsl")
    }

    fn resource_buffers_with_bind_group_layouts<'a>(&'a self) -> Vec<ResourceGroupLayout<'a>> {
        let threads_per_wg = 256u64;
        let num_workgroups = (self.num_elems + threads_per_wg - 1) / threads_per_wg;
        let counters_bytes = num_workgroups * 16 * size_of::<u32>() as u64;

        vec![ResourceGroupLayout {
            entries: vec![
                ResourceBinding {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: size_of::<RadixScanParams>() as u64,
                        usages: BufferUsages::UNIFORM,
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
                        size: counters_bytes,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        if self.init_data.is_none() {
            let built_data = self.build(device);
            let resource_buffers = &built_data.buffers.resource_buffers[0];

            self.init_data = Some(InitData {
                params: resource_buffers.buffers[0].clone(),
                counter: resource_buffers.buffers[1].clone(),
                bind_group: resource_buffers.bind_group.clone(),
                current_counter_hash: 0,
                pipeline: built_data.pipeline.clone(),
            });
        }

        let init_data = self.init_data.as_mut().unwrap();

        let num_wg = 1u32;

        let params = RadixScanParams {
            num_elems: self.num_elems as u32,
        };

        queue.write_buffer(&init_data.params, 0, bytemuck::bytes_of(&params));

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

        args.counter_buf
    }
}
