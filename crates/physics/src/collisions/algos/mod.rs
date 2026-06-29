use std::ops::Add;

use bytemuck::{Pod, Zeroable};
use glam::{Vec3A, Vec4};

pub mod lbvh;
pub mod sah;

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
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
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

    pub fn read_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<FlattenedBVHNode> {
        use std::sync::mpsc;
        use wgpu::{BufferDescriptor, BufferUsages, MapMode};

        let size = buffer.size();

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read FlattenedBVHNodes"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = mpsc::channel();

        buffer_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<FlattenedBVHNode> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to read FlattenedBVHNodes back from buffer!");
        }
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

#[derive(Default)]
pub struct BVHTree {
    pub tree: Vec<BVHNode>,
    pub root_idx: usize,
}
pub enum TraversalFlow {
    Continue,
    ContinueWithNewMax(f32),
    Break,
}

impl BVHTree {
    pub fn traverse_ray_cpu<F>(&self, ray: &Ray, mut callback: F)
    where
        F: FnMut(usize, f32, f32) -> TraversalFlow,
    {
        if self.tree.is_empty() {
            return;
        }

        let mut stack = vec![self.root_idx];
        let mut t_max = f32::INFINITY;

        while let Some(current_idx) = stack.pop() {
            let node = &self.tree[current_idx];
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
                    let left_node = &self.tree[*left_child];
                    let right_node = &self.tree[*right_child];
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

pub trait CpuCollisionAlgorithm {
    fn build(&mut self, vertices: Vec<Vec<Vec3A>>) -> BVHTree;
}

pub trait GpuCollisionAlgorithm {
    fn build(
        &mut self,
        model_verts_buffer: wgpu::Buffer,
        model_infos_buffer: wgpu::Buffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::Buffer;
}
