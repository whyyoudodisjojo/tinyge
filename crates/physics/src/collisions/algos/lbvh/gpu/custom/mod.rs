use tinyge_graphics::shaders::{ComputeShaderWrapper, buffers::Buffers};
use wgpu::{Buffer, Device};

use crate::collisions::algos::{
    GpuCollisionAlgorithm,
    lbvh::gpu::custom::{
        phases::{
            build_tree::{BuildTree, BuildTreeArgs, BuildTreeStage},
            compute_rects::{ComputeRects, ComputeRectsArgs},
            mortonize::{Mortonize, MortonizeArgs},
        },
        radix_sort::RadixSort,
    },
};

pub mod phases;
pub mod radix_sort;
pub struct LBVHBuffers {
    pub rects_buffer: Buffer,
    pub keys_buffer: Buffer,
    pub global_bounds_buffer: Buffer,
    pub num_rects_buffer: Buffer,
    pub nodes_buffer: Buffer,
    pub counts_buffer: Buffer,
    pub params_buffer: Buffer,
}

pub struct LBVHBuilder<'a> {
    compute_rects: ComputeShaderWrapper<'a, ComputeRects>,
    mortonize: ComputeShaderWrapper<'a, Mortonize>,
    build_leaves: ComputeShaderWrapper<'a, BuildTree>,
    build_structure: ComputeShaderWrapper<'a, BuildTree>,
    compute_bounds: ComputeShaderWrapper<'a, BuildTree>,
    radix_sort: RadixSort<'a>,
    buffers: LBVHBuffers,
    num_models: u32,
}

impl<'a> LBVHBuilder<'a> {
    pub fn new(num_models: u32, num_verts: u32, device: &Device) -> Self {
        let compute_rects =
            ComputeShaderWrapper::new(ComputeRects::new(num_models, num_verts), device);
        let mortonize = ComputeShaderWrapper::new(Mortonize::new(num_models), device);
        let build_leaves = ComputeShaderWrapper::new(
            BuildTree::new(num_models, BuildTreeStage::BuildLeaves),
            device,
        );
        let build_structure = ComputeShaderWrapper::new(
            BuildTree::new(num_models, BuildTreeStage::BuildStructure),
            device,
        );
        let compute_bounds = ComputeShaderWrapper::new(
            BuildTree::new(num_models, BuildTreeStage::ComputeBounds),
            device,
        );
        let radix_sort = RadixSort::new(num_models, device);

        let compute_rects_buffers =
            Buffers::build(device, &compute_rects.buffer_build_spec.buffer_build_spec);
        let rects_buffer = compute_rects_buffers.resource_buffers[0].buffers[2].clone();

        let mortonize_buffers =
            Buffers::build(device, &mortonize.buffer_build_spec.buffer_build_spec);
        let keys_buffer = mortonize_buffers.resource_buffers[0].buffers[1].clone();
        let global_bounds_buffer = mortonize_buffers.resource_buffers[0].buffers[2].clone();
        let num_rects_buffer = mortonize_buffers.resource_buffers[0].buffers[3].clone();

        let build_tree_buffers =
            Buffers::build(device, &build_leaves.buffer_build_spec.buffer_build_spec);

        let nodes_buffer = build_tree_buffers.resource_buffers[0].buffers[2].clone();
        let counts_buffer = build_tree_buffers.resource_buffers[0].buffers[3].clone();
        let params_buffer = build_tree_buffers.resource_buffers[0].buffers[4].clone();

        let buffers = LBVHBuffers {
            rects_buffer,
            keys_buffer,
            global_bounds_buffer,
            num_rects_buffer,
            nodes_buffer,
            counts_buffer,
            params_buffer,
        };

        Self {
            compute_rects,
            mortonize,
            build_leaves,
            build_structure,
            compute_bounds,
            radix_sort,
            buffers,
            num_models,
        }
    }
}

impl<'a> GpuCollisionAlgorithm for LBVHBuilder<'a> {
    fn build(
        &mut self,
        model_verts_buffer: wgpu::Buffer,
        model_infos_buffer: wgpu::Buffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::Buffer {
        queue.write_buffer(
            &self.buffers.num_rects_buffer,
            0,
            bytemuck::bytes_of(&self.num_models),
        );
        queue.write_buffer(
            &self.buffers.params_buffer,
            0,
            bytemuck::bytes_of(&self.num_models),
        );

        self.compute_rects.dispatch(
            ComputeRectsArgs {
                model_verts_buffer: model_verts_buffer.clone(),
                model_infos_buffer: model_infos_buffer.clone(),
                output_rect_buffer: self.buffers.rects_buffer.clone(),
            },
            device,
            queue,
        );

        self.mortonize.dispatch(
            MortonizeArgs {
                rects_buffer: self.buffers.rects_buffer.clone(),
                keys_buffer: self.buffers.keys_buffer.clone(),
                global_bounds_buffer: self.buffers.global_bounds_buffer.clone(),
                num_rects_buffer: self.buffers.num_rects_buffer.clone(),
            },
            device,
            queue,
        );

        self.radix_sort
            .sort(self.buffers.keys_buffer.clone(), device, queue);

        self.build_leaves.dispatch(
            BuildTreeArgs {
                keys_buffer: self.buffers.keys_buffer.clone(),
                rects_buffer: self.buffers.rects_buffer.clone(),
                nodes_buffer: self.buffers.nodes_buffer.clone(),
                counts_buffer: self.buffers.counts_buffer.clone(),
                params_buffer: self.buffers.params_buffer.clone(),
            },
            device,
            queue,
        );

        self.build_structure.dispatch(
            BuildTreeArgs {
                keys_buffer: self.buffers.keys_buffer.clone(),
                rects_buffer: self.buffers.rects_buffer.clone(),
                nodes_buffer: self.buffers.nodes_buffer.clone(),
                counts_buffer: self.buffers.counts_buffer.clone(),
                params_buffer: self.buffers.params_buffer.clone(),
            },
            device,
            queue,
        );

        self.compute_bounds.dispatch(
            BuildTreeArgs {
                keys_buffer: self.buffers.keys_buffer.clone(),
                rects_buffer: self.buffers.rects_buffer.clone(),
                nodes_buffer: self.buffers.nodes_buffer.clone(),
                counts_buffer: self.buffers.counts_buffer.clone(),
                params_buffer: self.buffers.params_buffer.clone(),
            },
            device,
            queue,
        );

        self.buffers.nodes_buffer.clone()
    }
}

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    use crate::collisions::algos::{
        FlattenedBVHNode, GpuCollisionAlgorithm, lbvh::gpu::custom::LBVHBuilder,
    };

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ModelInfo {
        offset: u32,
        stride: u32,
    }

    async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
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

    #[test]
    fn test_gpu_lbvh_build() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;

            let vertices: Vec<[f32; 3]> = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [2.0, 2.0, 2.0],
                [3.0, 2.0, 2.0],
                [2.5, 3.0, 2.0],
                [-1.0, -1.0, -1.0],
                [0.0, -1.0, -1.0],
                [-0.5, 0.0, -1.0],
                [1.5, 1.5, 1.5],
                [2.5, 1.5, 1.5],
                [2.0, 2.5, 1.5],
            ];

            let model_infos: Vec<ModelInfo> = vec![
                ModelInfo {
                    offset: 0,
                    stride: 3,
                },
                ModelInfo {
                    offset: 3,
                    stride: 3,
                },
                ModelInfo {
                    offset: 6,
                    stride: 3,
                },
                ModelInfo {
                    offset: 9,
                    stride: 3,
                },
            ];

            let num_models = model_infos.len() as u32;
            let num_verts = vertices.len() as u32;

            let model_verts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let model_infos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&model_infos),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let mut builder = LBVHBuilder::new(num_models, num_verts, &device);
            let nodes_buffer =
                builder.build(model_verts_buffer, model_infos_buffer, &device, &queue);

            let nodes = FlattenedBVHNode::read_buffer(&device, &queue, &nodes_buffer);

            for i in 0..num_models as usize {
                assert_eq!(nodes[i].node_type, 0, "Node {} should be a leaf", i);
                assert_eq!(
                    nodes[i].left_child, -1,
                    "Leaf node {} should have left_child=-1",
                    i
                );
                assert_eq!(
                    nodes[i].right_child, -1,
                    "Leaf node {} should have right_child=-1",
                    i
                );
                if nodes.len() > 1 {
                    assert!(
                        nodes[i].parent >= num_models as i32,
                        "Leaf node {} should have internal node as parent",
                        i
                    );
                }
            }

            for i in num_models as usize..nodes.len() {
                assert_eq!(
                    nodes[i].node_type, 1,
                    "Node {} should be internal (type=1), got type={}",
                    i, nodes[i].node_type
                );
                assert!(
                    nodes[i].left_child >= 0,
                    "Internal node {} should have left child, got {}",
                    i,
                    nodes[i].left_child
                );
                assert!(
                    nodes[i].right_child >= 0,
                    "Internal node {} should have right child, got {}",
                    i,
                    nodes[i].right_child
                );
            }

            let root_idx = nodes.len() - 1;
            assert!(
                nodes[root_idx].left_child >= 0 || nodes[root_idx].right_child >= 0,
                "Root should have children"
            );

            assert_eq!(nodes.len(), (2 * num_models - 1) as usize);
        });
    }
}
