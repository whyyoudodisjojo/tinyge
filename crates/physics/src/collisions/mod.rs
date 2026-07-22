use codegen_macros::IntoWgslStruct;
use tinyge_graphics::shaders::buffers::BufferWithType;
use wgpu::{Device, Queue};

use crate::collisions::algos::{GpuBVHTraversal, GpuCollisionAlgorithm, Ray, RayResult};

pub mod algos;

#[derive(IntoWgslStruct, Clone)]
pub struct ModelVertex {
    pub pos: [f32; 3],
    pub _pad: u32,
}

#[derive(IntoWgslStruct, Clone)]
pub struct ModelInfo {
    pub offset: u32,
    pub stride: u32,
}

pub struct Collisions<Algo, Probe> {
    pub algo: Algo,
    pub probe: Probe,
}

impl<Algo, Probe> Collisions<Algo, Probe>
where
    Algo: GpuCollisionAlgorithm,
    Probe: GpuBVHTraversal,
{
    pub fn run_gpu(
        &mut self,
        model_verts: BufferWithType<Vec<ModelVertex>>,
        model_infos: BufferWithType<Vec<ModelInfo>>,
        device: &Device,
        queue: &Queue,
        rays: BufferWithType<Vec<Ray>>,
    ) -> BufferWithType<RayResult> {
        let res = self.algo.build(model_verts, model_infos, device, queue);
        res.traverse_gpu(
            &rays,
            (rays.inner.size() / size_of::<Ray>() as u64) as u32,
            device,
            queue,
        )
    }
}
