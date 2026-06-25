use std::ops::Add;

use glam::Vec3A;

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

impl Add for RectangleBounds {
    type Output = RectangleBounds;
    fn add(self, rhs: RectangleBounds) -> Self::Output {
        RectangleBounds {
            min: self.min.min(rhs.min),
            max: self.max.max(rhs.max),
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

pub trait CollisionAlgorithm {
    fn build(&mut self, rects: &[RectangleBounds]) -> BVHTree;
}
