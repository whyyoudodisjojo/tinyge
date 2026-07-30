use memory::buffers::BufferWithType;
use tinyge_graphics::shaders::ComputeShaderWrapper;
use wgpu::{Buffer, Device};

use crate::collisions::algos::{
    BVHTree, FlattenedBVHNode, GpuCollisionAlgorithm, GpuStorage,
    lbvh::gpu::custom::{
        phases::{
            build_tree::{BuildTree, BuildTreeArgs, BuildTreeStage},
            compute_rects::{ComputeRects, ComputeRectsArgs},
            mortonize::{Mortonize, MortonizeArgs},
        },
        radix_sort::RadixSort,
    },
};
use crate::collisions::{ModelInfo as CollisionModelInfo, ModelVertex as CollisionModelVertex};

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

pub struct LBVHBuilder {
    compute_rects: ComputeShaderWrapper<ComputeRects, ComputeRectsArgs>,
    mortonize: ComputeShaderWrapper<Mortonize, MortonizeArgs>,
    build_leaves: ComputeShaderWrapper<BuildTree, BuildTreeArgs>,
    build_structure: ComputeShaderWrapper<BuildTree, BuildTreeArgs>,
    compute_bounds: ComputeShaderWrapper<BuildTree, BuildTreeArgs>,
    radix_sort: RadixSort,
    buffers: LBVHBuffers,
    num_models: u32,
}

impl LBVHBuilder {
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

        let rects_buffer = compute_rects
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .output_rect_buffer
            .inner
            .inner
            .clone()
            .unwrap();

        let keys_buffer = mortonize
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .keys_buffer
            .inner
            .inner
            .clone()
            .unwrap();
        let global_bounds_buffer = mortonize
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .global_bounds_buffer
            .inner
            .inner
            .clone()
            .unwrap();
        let num_rects_buffer = mortonize
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .num_rects_buffer
            .inner
            .inner
            .clone()
            .unwrap();

        let nodes_buffer = build_leaves
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .nodes_buffer
            .inner
            .inner
            .clone()
            .unwrap();
        let counts_buffer = build_leaves
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .counts_buffer
            .inner
            .inner
            .clone()
            .unwrap();
        let params_buffer = build_leaves
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .params_buffer
            .inner
            .inner
            .clone()
            .unwrap();

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

impl GpuCollisionAlgorithm for LBVHBuilder {
    fn build(
        &mut self,
        model_verts_buffer: BufferWithType<Vec<CollisionModelVertex>>,
        model_infos_buffer: BufferWithType<Vec<CollisionModelInfo>>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> BVHTree<GpuStorage> {
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
                model_verts_buffer: model_verts_buffer.inner.clone().into(),
                model_infos_buffer: model_infos_buffer.inner.clone().into(),
                output_rect_buffer: self.buffers.rects_buffer.clone().into(),
            },
            device,
            queue,
        );

        self.mortonize.dispatch(
            MortonizeArgs {
                rects_buffer: self.buffers.rects_buffer.clone().into(),
                keys_buffer: self.buffers.keys_buffer.clone().into(),
                global_bounds_buffer: self.buffers.global_bounds_buffer.clone().into(),
                num_rects_buffer: self.buffers.num_rects_buffer.clone().into(),
            },
            device,
            queue,
        );

        self.radix_sort
            .sort(self.buffers.keys_buffer.clone().into(), device, queue);

        self.build_leaves.dispatch(
            BuildTreeArgs {
                keys_buffer: self.buffers.keys_buffer.clone().into(),
                rects_buffer: self.buffers.rects_buffer.clone().into(),
                nodes_buffer: self.buffers.nodes_buffer.clone().into(),
                counts_buffer: self.buffers.counts_buffer.clone().into(),
                params_buffer: self.buffers.params_buffer.clone().into(),
            },
            device,
            queue,
        );

        self.build_structure.dispatch(
            BuildTreeArgs {
                keys_buffer: self.buffers.keys_buffer.clone().into(),
                rects_buffer: self.buffers.rects_buffer.clone().into(),
                nodes_buffer: self.buffers.nodes_buffer.clone().into(),
                counts_buffer: self.buffers.counts_buffer.clone().into(),
                params_buffer: self.buffers.params_buffer.clone().into(),
            },
            device,
            queue,
        );

        self.compute_bounds.dispatch(
            BuildTreeArgs {
                keys_buffer: self.buffers.keys_buffer.clone().into(),
                rects_buffer: self.buffers.rects_buffer.clone().into(),
                nodes_buffer: self.buffers.nodes_buffer.clone().into(),
                counts_buffer: self.buffers.counts_buffer.clone().into(),
                params_buffer: self.buffers.params_buffer.clone().into(),
            },
            device,
            queue,
        );

        BVHTree {
            storage: GpuStorage {
                nodes_buffer: BufferWithType::<FlattenedBVHNode>::from(
                    self.buffers.nodes_buffer.clone(),
                ),
                root_idx: (2 * self.num_models - 1) as usize - 1,
                num_nodes: (2 * self.num_models - 1) as usize,
            },
        }
    }
}
