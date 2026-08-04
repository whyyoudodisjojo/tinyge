use codegen_macros::IntoBufferStruct;
use memory::buffers::{BufferWithType, ResourceType};
use tinyge_graphics::shaders::{
    ComputeShader, ComputeShaderBuiltData,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{Buffer, BufferUsages, ComputePassDescriptor, ShaderStages, wgt::CommandEncoderDescriptor};

#[derive(IntoBufferStruct)]
pub struct MortonizeArgs {
    #[resource(bindgroup_index = 0, resource_index = 0, ty = "buffer")]
    pub rects_buffer: BufferWithType<Vec<glam::Vec4>, Buffer>,
    #[resource(bindgroup_index = 0, resource_index = 1, ty = "buffer")]
    pub keys_buffer: BufferWithType<Vec<u32>, Buffer>,
    #[resource(bindgroup_index = 0, resource_index = 2, ty = "buffer")]
    pub global_bounds_buffer: BufferWithType<glam::Vec4, Buffer>,
    #[resource(bindgroup_index = 0, resource_index = 3, ty = "buffer")]
    pub num_rects_buffer: BufferWithType<u32, Buffer>,
}

pub struct Mortonize {
    num_rects: u32,
}

impl Mortonize {
    pub fn new(num_rects: u32) -> Self {
        Self { num_rects }
    }
}

impl<'a> ComputeShader<'a> for Mortonize {
    type Args = MortonizeArgs;
    type Ret = ();

    fn entry_point(&self) -> &'static str {
        "generate_morton_keys"
    }

    fn load_source_code(&self) -> String {
        include_str!("../../../../shaders/lbvh/mortonize.wgsl").into()
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: self.num_rects as u64 * 32,
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
                        size: self.num_rects as u64 * 8,
                        usages: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
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
                        size: 32,
                        usages: BufferUsages::UNIFORM,
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
                        size: 4,
                        usages: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        built_data: &mut ComputeShaderBuiltData<Self::Args>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        let num_wg = ((self.num_rects + 255) / 256).max(1);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(
            &[
                ResourceType::Buffer(args.rects_buffer.inner),
                ResourceType::Buffer(args.keys_buffer.inner),
                ResourceType::Buffer(args.global_bounds_buffer.inner),
                ResourceType::Buffer(args.num_rects_buffer.inner),
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
