use tinyge_graphics::shaders::{ComputeShaderWrapper, buffers::Buffers};
use wgpu::{Buffer, Device};

use crate::collisions::algos::{
    BVHTree, GpuCollisionAlgorithm, GpuStorage,
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

        let compute_rects_buffers = Buffers::build(
            device,
            &compute_rects.buffer_build_spec.buffer_build_spec,
            false,
        );
        let rects_buffer = compute_rects_buffers.resource_buffers[0].buffers[2]
            .clone()
            .unwrap();

        let mortonize_buffers = Buffers::build(
            device,
            &mortonize.buffer_build_spec.buffer_build_spec,
            false,
        );
        let keys_buffer = mortonize_buffers.resource_buffers[0].buffers[1]
            .clone()
            .unwrap();
        let global_bounds_buffer = mortonize_buffers.resource_buffers[0].buffers[2]
            .clone()
            .unwrap();
        let num_rects_buffer = mortonize_buffers.resource_buffers[0].buffers[3]
            .clone()
            .unwrap();

        let build_tree_buffers = Buffers::build(
            device,
            &build_leaves.buffer_build_spec.buffer_build_spec,
            false,
        );

        let nodes_buffer = build_tree_buffers.resource_buffers[0].buffers[2]
            .clone()
            .unwrap();
        let counts_buffer = build_tree_buffers.resource_buffers[0].buffers[3]
            .clone()
            .unwrap();
        let params_buffer = build_tree_buffers.resource_buffers[0].buffers[4]
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

impl<'a> GpuCollisionAlgorithm for LBVHBuilder<'a> {
    fn build(
        &mut self,
        model_verts_buffer: wgpu::Buffer,
        model_infos_buffer: wgpu::Buffer,
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

        BVHTree {
            storage: GpuStorage {
                nodes_buffer: self.buffers.nodes_buffer.clone(),
                root_idx: (2 * self.num_models - 1) as usize - 1,
                num_nodes: (2 * self.num_models - 1) as usize,
            },
        }
    }
}


