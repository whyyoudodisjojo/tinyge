use tinyge_graphics::shaders::{ComputeShaderWrapper, buffers::Buffers};
use wgpu::{Buffer, Device};

use crate::collisions::algos::lbvh::gpu::phases::{
    build_tree::{BuildTree, BuildTreeArgs, BuildTreeStage},
    compute_rects::{ComputeRects, ComputeRectsArgs, ComputeRectsStage},
    mortonize::{Mortonize, MortonizeArgs},
};
use crate::collisions::algos::lbvh::gpu::radix_sort::RadixSort;

pub mod build_tree;
pub mod compute_rects;
pub mod mortonize;

pub struct LBVHBuffers {
    pub atomic_bounds_buffer: Buffer,
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
    convert_bounds: ComputeShaderWrapper<'a, ComputeRects>,
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
        let compute_rects = ComputeShaderWrapper::new(
            ComputeRects::new(num_models, num_verts, ComputeRectsStage::ComputeRects),
            device,
        );
        let convert_bounds = ComputeShaderWrapper::new(
            ComputeRects::new(
                num_models,
                num_verts,
                ComputeRectsStage::ConvertAtomicBoundsToBounds,
            ),
            device,
        );
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
        let atomic_bounds_buffer = compute_rects_buffers.resource_buffers[0].buffers[2].clone();
        let rects_buffer = compute_rects_buffers.resource_buffers[0].buffers[3].clone();

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
            atomic_bounds_buffer,
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
            convert_bounds,
            mortonize,
            build_leaves,
            build_structure,
            compute_bounds,
            radix_sort,
            buffers,
            num_models,
        }
    }

    pub fn run(
        &mut self,
        model_verts_buffer: Buffer,
        model_infos_buffer: Buffer,
        device: &Device,
        queue: &wgpu::Queue,
    ) -> Buffer {
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
                output_rect_atomic_buffer: self.buffers.atomic_bounds_buffer.clone(),
                output_rect_buffer: self.buffers.rects_buffer.clone(),
            },
            device,
            queue,
        );

        self.convert_bounds.dispatch(
            ComputeRectsArgs {
                model_verts_buffer: model_verts_buffer.clone(),
                model_infos_buffer: model_infos_buffer.clone(),
                output_rect_atomic_buffer: self.buffers.atomic_bounds_buffer.clone(),
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
