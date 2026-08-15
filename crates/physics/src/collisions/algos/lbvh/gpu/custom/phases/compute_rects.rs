use codegen_macros::IntoBufferStruct;
use memory::buffers::{BufferWithType, ResourceType};
use tinyge_graphics::shaders::{
    ComputeShader, ComputeShaderBuiltData,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{Buffer, BufferUsages, ComputePassDescriptor, ShaderStages, wgt::CommandEncoderDescriptor};

#[derive(IntoBufferStruct)]
pub struct ComputeRectsArgs {
    #[resource(bindgroup_index = 0, resource_index = 0, ty = "buffer")]
    pub model_verts_buffer: BufferWithType<Vec<[f32; 4]>, Buffer>,
    #[resource(bindgroup_index = 0, resource_index = 1, ty = "buffer")]
    pub model_infos_buffer: BufferWithType<Vec<[u32; 2]>, Buffer>,
    #[resource(bindgroup_index = 0, resource_index = 2, ty = "buffer")]
    pub output_rect_buffer: BufferWithType<Vec<glam::Vec4>, Buffer>,
}

pub struct ComputeRects {
    num_models: u32,
    num_verts: u32,
}

impl ComputeRects {
    pub fn new(num_models: u32, num_verts: u32) -> Self {
        Self {
            num_models,
            num_verts,
        }
    }
}

impl<'a> ComputeShader<'a> for ComputeRects {
    type Args = ComputeRectsArgs;
    type Ret = ();

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
                        size: (16 * self.num_verts) as u64,
                        usages: BufferUsages::STORAGE,
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
                        size: (8 * self.num_models) as u64,
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
                        size: (32 * self.num_models) as u64,
                        usages: BufferUsages::STORAGE,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn entry_point(&self) -> &'static str {
        "compute_rects"
    }

    fn load_source_code(&self) -> String {
        include_str!("../../../../shaders/lbvh/compute_rects.wgsl").into()
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        build_data: &mut ComputeShaderBuiltData<Self::Args>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        let buffers = vec![
            ResourceType::Buffer(args.model_verts_buffer.inner.clone()),
            ResourceType::Buffer(args.model_infos_buffer.inner.clone()),
            ResourceType::Buffer(args.output_rect_buffer.inner.clone()),
        ];

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(
                0,
                build_data.bind_groups[0].get_or_create_bind_group(&buffers, device),
                &[],
            );
            pass.set_pipeline(&build_data.pipeline);
            pass.dispatch_workgroups(self.num_models, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}
