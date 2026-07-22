use std::ops::Add;

use bytemuck::{Pod, Zeroable};
use codegen_macros::IntoWgslStruct;
use glam::{Vec3A, Vec4};
use tinyge_graphics::shaders::buffers::BufferWithType;

pub mod gpu_accelerated;
pub mod lbvh;
pub mod sah;
#[cfg(test)]
pub mod test_utils;
pub mod traversal;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, IntoWgslStruct)]
pub struct GpuRay {
    pub origin: [f32; 4],
    pub dir: [f32; 4],
    pub inv_dir: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, IntoWgslStruct)]
pub struct RayResult {
    pub hit_node_idx: i32,
    pub t_near: f32,
    pub _pad: [u32; 2],
}

#[derive(Clone, Copy, Default, Debug)]
pub struct RectangleBounds {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl RectangleBounds {
    pub fn centroid(&self) -> Vec3A {
        (self.min + self.max) * 0.5
    }

    pub fn surface_area(&self) -> f32 {
        let extents = (self.max - self.min).max(Vec3A::ZERO);

        2.0 * (extents.x * extents.y + extents.y * extents.z + extents.z * extents.x)
    }

    pub const MAX: Self = RectangleBounds {
        min: Vec3A::MAX,
        max: Vec3A::MIN,
    };
}

impl From<&[Vec3A]> for RectangleBounds {
    fn from(vertices: &[Vec3A]) -> RectangleBounds {
        vertices
            .iter()
            .fold(RectangleBounds::MAX, |bounds, &v| RectangleBounds {
                min: bounds.min.min(v),
                max: bounds.max.max(v),
            })
    }
}

impl Add for RectangleBounds {
    type Output = RectangleBounds;
    fn add(self, rhs: RectangleBounds) -> Self::Output {
        RectangleBounds {
            min: self.min.min(rhs.min),
            max: self.max.max(rhs.max),
        }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug, IntoWgslStruct)]
pub struct FlattenedBVHNode {
    pub min: Vec4,
    pub max: Vec4,
    pub parent: i32,
    pub left_child: i32,
    pub right_child: i32,
    pub node_type: u32,
}

impl FlattenedBVHNode {
    pub const fn size_in_bytes() -> usize {
        48
    }
}

pub enum BVHNode {
    Internal {
        rect: RectangleBounds,
        left_child: usize,
        right_child: usize,
    },
    Leaf {
        rect: RectangleBounds,
        idx: usize,
    },
}

impl BVHNode {
    pub fn rect(&self) -> &RectangleBounds {
        match self {
            Self::Internal { rect, .. } => rect,
            Self::Leaf { rect, .. } => rect,
        }
    }
}

impl Default for BVHNode {
    fn default() -> Self {
        Self::Internal {
            rect: RectangleBounds::MAX,
            left_child: Default::default(),
            right_child: Default::default(),
        }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, IntoWgslStruct, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3A,
    pub dir: Vec3A,
    pub inv_dir: Vec3A,
}

impl Ray {
    pub fn new(origin: Vec3A, dir: Vec3A) -> Self {
        let inv_dir = Vec3A::select(
            dir.cmpeq(Vec3A::ZERO),
            Vec3A::splat(f32::INFINITY),
            Vec3A::ONE / dir,
        );

        Self {
            origin,
            dir,
            inv_dir,
        }
    }

    pub fn intersects_rect(&self, rect: &RectangleBounds, t_max: f32) -> Option<f32> {
        let t1 = (rect.min - self.origin) * self.inv_dir;
        let t2 = (rect.max - self.origin) * self.inv_dir;

        let t_min_axes = t1.min(t2);
        let t_max_axes = t1.max(t2);

        let t_near = t_min_axes.max_element();
        let t_far = t_max_axes.min_element();

        if t_near <= t_far && t_far >= 0.0 && t_near < t_max {
            Some(t_near.max(0.0))
        } else {
            None
        }
    }
}

pub enum TraversalFlow {
    Continue,
    ContinueWithNewMax(f32),
    Break,
}

#[derive(Default)]
pub struct CpuStorage {
    pub tree: Vec<BVHNode>,
    pub root_idx: usize,
}

pub struct GpuStorage {
    pub nodes_buffer: BufferWithType<FlattenedBVHNode>,
    pub root_idx: usize,
    pub num_nodes: usize,
}

#[derive(Default)]
pub struct BVHTree<S = CpuStorage> {
    pub storage: S,
}

pub trait CpuBVHTraversal {
    fn traverse_ray<F>(&self, ray: &Ray, callback: F)
    where
        F: FnMut(usize, f32, f32) -> TraversalFlow;
}
pub trait GpuBVHTraversal {
    fn traverse_gpu(
        &self,
        rays_buffer: &BufferWithType<Vec<Ray>>,
        num_rays: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> BufferWithType<RayResult>;
}
pub trait CpuCollisionAlgorithm {
    fn build(&mut self, vertices: Vec<Vec<Vec3A>>) -> BVHTree<CpuStorage>;
}

pub trait GpuCollisionAlgorithm {
    fn build(
        &mut self,
        model_verts_buffer: BufferWithType<Vec<super::ModelVertex>>,
        model_infos_buffer: BufferWithType<Vec<super::ModelInfo>>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> BVHTree<GpuStorage>;
}
