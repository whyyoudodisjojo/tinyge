use tinyge_graphics::shaders::{
    ComputeShader, ComputeShaderBuiltData,
    buffers::{BufferWithType, ResourceType},
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{BufferUsages, ComputePassDescriptor, ShaderStages, wgt::CommandEncoderDescriptor};

use crate::collisions::algos::FlattenedBVHNode;

pub struct BuildTreeArgs {
    pub keys_buffer: BufferWithType<Vec<u32>>,
    pub rects_buffer: BufferWithType<Vec<glam::Vec4>>,
    pub nodes_buffer: BufferWithType<Vec<FlattenedBVHNode>>,
    pub counts_buffer: BufferWithType<Vec<u32>>,
    pub params_buffer: BufferWithType<u32>,
}

pub enum BuildTreeStage {
    BuildLeaves,
    BuildStructure,
    ComputeBounds,
}

pub struct BuildTree {
    num_leaves: u32,
    stage: BuildTreeStage,
}

impl BuildTree {
    pub fn new(num_leaves: u32, stage: BuildTreeStage) -> Self {
        Self { num_leaves, stage }
    }
}

impl<'a> ComputeShader<'a> for BuildTree {
    type Args = BuildTreeArgs;
    type Ret = ();

    fn entry_point(&self) -> &'static str {
        match &self.stage {
            BuildTreeStage::BuildLeaves => "build_leaves",
            BuildTreeStage::BuildStructure => "build_structure",
            BuildTreeStage::ComputeBounds => "compute_bounds",
        }
    }

    fn load_source_code(&self) -> String {
        include_str!("../../../../shaders/lbvh/build_tree.wgsl").to_string()
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
                        size: self.num_leaves as u64 * 8,
                        usages: BufferUsages::STORAGE,
                        is_input: true,
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
                        size: self.num_leaves as u64 * 32,
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
                        size: ((2 * self.num_leaves - 1) as u64 * 48).max(48),
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
                        size: ((self.num_leaves - 1) as u64 * 4).max(4),
                        usages: BufferUsages::STORAGE,
                        is_input: false,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: 4,
                        usages: BufferUsages::UNIFORM,
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
        built_data: &mut ComputeShaderBuiltData<'a>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        let num_wg = match &self.stage {
            BuildTreeStage::BuildLeaves => ((self.num_leaves + 255) / 256).max(1),
            BuildTreeStage::BuildStructure => ((self.num_leaves - 1 + 255) / 256).max(1),
            BuildTreeStage::ComputeBounds => ((self.num_leaves + 255) / 256).max(1),
        };

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(
            &[
                ResourceType::Buffer(args.keys_buffer.inner),
                ResourceType::Buffer(args.rects_buffer.inner),
                ResourceType::Buffer(args.nodes_buffer.inner),
                ResourceType::Buffer(args.counts_buffer.inner),
                ResourceType::Buffer(args.params_buffer.inner),
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
