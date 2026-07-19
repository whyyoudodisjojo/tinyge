use tinyge_graphics::shaders::{
    buffers::{Buffers, ResourceType},
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{BufferUsages, ShaderStages};

use crate::collisions::algos::{
    BVHNode, BVHTree, CpuBVHTraversal, CpuStorage, GpuBVHTraversal, GpuStorage, Ray, TraversalFlow,
};

impl CpuBVHTraversal for BVHTree<CpuStorage> {
    fn traverse_ray<F>(&self, ray: &Ray, mut callback: F)
    where
        F: FnMut(usize, f32, f32) -> TraversalFlow,
    {
        if self.storage.tree.is_empty() {
            return;
        }

        let mut stack = vec![self.storage.root_idx];
        let mut t_max = f32::INFINITY;

        while let Some(current_idx) = stack.pop() {
            let node = &self.storage.tree[current_idx];
            let Some(t_near) = ray.intersects_rect(node.rect(), t_max) else {
                continue;
            };

            match node {
                BVHNode::Leaf { idx, .. } => match callback(*idx, t_near, t_max) {
                    TraversalFlow::Break => return,
                    TraversalFlow::ContinueWithNewMax(new_max) => t_max = new_max,
                    TraversalFlow::Continue => {}
                },
                BVHNode::Internal {
                    left_child,
                    right_child,
                    ..
                } => {
                    let left_node = &self.storage.tree[*left_child];
                    let right_node = &self.storage.tree[*right_child];
                    let t_left = ray.intersects_rect(left_node.rect(), t_max);
                    let t_right = ray.intersects_rect(right_node.rect(), t_max);

                    match (t_left, t_right) {
                        (Some(tl), Some(tr)) => {
                            if tl < tr {
                                stack.push(*right_child);
                                stack.push(*left_child);
                            } else {
                                stack.push(*left_child);
                                stack.push(*right_child);
                            }
                        }
                        (Some(_), None) => stack.push(*left_child),
                        (None, Some(_)) => stack.push(*right_child),
                        (None, None) => {}
                    }
                }
            }
        }
    }
}

pub struct BvhTraversalShader {
    num_rays: u32,
    root_idx: u32,
}

impl BvhTraversalShader {
    pub fn new(num_rays: u32, root_idx: u32) -> Self {
        Self { num_rays, root_idx }
    }
}

pub struct BvhTraversalArgs {
    pub rays_buffer: wgpu::Buffer,
    pub nodes_buffer: wgpu::Buffer,
}

impl<'a> tinyge_graphics::shaders::ComputeShader<'a> for BvhTraversalShader {
    type Args = BvhTraversalArgs;
    type Ret = wgpu::Buffer;

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
                        size: (48 * self.num_rays) as u64,
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
                        size: 0,
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
                        size: (16 * self.num_rays) as u64,
                        usages: BufferUsages::STORAGE,
                        is_input: false,
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
                        usages: BufferUsages::UNIFORM | BufferUsages::STORAGE,
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
                        usages: BufferUsages::UNIFORM | BufferUsages::STORAGE,
                        is_input: false,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn load_source_code(&self) -> String {
        include_str!("shaders/traversal.wgsl").to_string()
    }

    fn entry_point(&self) -> &'static str {
        "traverse"
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        build_data: &mut tinyge_graphics::shaders::ComputeShaderBuiltData<'a>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        use wgpu::{ComputePassDescriptor, wgt::CommandEncoderDescriptor};

        let buffers = Buffers::build(device, &build_data.buffer_build_spec, false);

        let results_buffer = buffers.resource_buffers[0].buffers[2].clone().unwrap();
        let num_rays_buffer = buffers.resource_buffers[0].buffers[3].clone().unwrap();
        let root_idx_buffer = buffers.resource_buffers[0].buffers[4].clone().unwrap();

        queue.write_buffer(&num_rays_buffer, 0, bytemuck::bytes_of(&self.num_rays));
        queue.write_buffer(&root_idx_buffer, 0, bytemuck::bytes_of(&self.root_idx));

        let bind_group_resources = vec![
            ResourceType::Buffer(args.rays_buffer),
            ResourceType::Buffer(args.nodes_buffer),
            ResourceType::Buffer(results_buffer.clone()),
            ResourceType::Buffer(num_rays_buffer),
            ResourceType::Buffer(root_idx_buffer),
        ];

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(
                0,
                build_data.bind_groups[0].get_or_create_bind_group(&bind_group_resources, device),
                &[],
            );
            pass.set_pipeline(&build_data.pipeline);
            pass.dispatch_workgroups((self.num_rays + 255) / 256, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        results_buffer
    }
}

impl GpuBVHTraversal for BVHTree<GpuStorage> {
    fn traverse_gpu(
        &self,
        rays_buffer: &wgpu::Buffer,
        num_rays: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::Buffer {
        use tinyge_graphics::shaders::ComputeShaderWrapper;

        let mut traversal = ComputeShaderWrapper::new(
            BvhTraversalShader::new(num_rays, self.storage.root_idx as u32),
            device,
        );

        traversal.dispatch(
            BvhTraversalArgs {
                rays_buffer: rays_buffer.clone(),
                nodes_buffer: self.storage.nodes_buffer.clone(),
            },
            device,
            queue,
        )
    }
}
