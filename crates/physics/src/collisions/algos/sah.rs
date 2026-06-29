use glam::Vec3A;

use crate::collisions::algos::{BVHNode, BVHTree, CpuCollisionAlgorithm, RectangleBounds};

#[derive(Clone, Copy)]
pub struct Bin {
    pub count: usize,
    pub rect: RectangleBounds,
}

#[derive(Default)]
pub struct SAH {
    pub max_bin_count: usize,
}

enum BuildStage {
    ProcessRange {
        start: usize,
        end: usize,
    },
    ComputeBounds {
        internal_idx: usize,
        left_idx: usize,
        right_idx: usize,
    },
}

impl SAH {
    pub fn build_tree(
        &self,
        prim_indices: &mut [usize],
        rects: &[RectangleBounds],
        nodes: &mut Vec<BVHNode>,
    ) -> usize {
        let mut stages = vec![BuildStage::ProcessRange {
            start: 0,
            end: prim_indices.len(),
        }];
        let mut returned_indices = Vec::with_capacity(32);

        while let Some(stage) = stages.pop() {
            match stage {
                BuildStage::ProcessRange { start, end } => {
                    let count = end - start;

                    if count == 1 {
                        let idx = prim_indices[start];
                        returned_indices.push(nodes.len());
                        nodes.push(BVHNode::Leaf {
                            rect: rects[idx],
                            idx,
                        });
                        continue;
                    }

                    let (node_rect, centroid_min, centroid_max) = prim_indices[start..end]
                        .iter()
                        .map(|&idx| (rects[idx], rects[idx].centroid()))
                        .fold(
                            (RectangleBounds::MAX, Vec3A::MAX, Vec3A::MIN),
                            |(node_acc, cmin_acc, cmax_acc), (r, c)| {
                                (node_acc + r, cmin_acc.min(c), cmax_acc.max(c))
                            },
                        );

                    let extent = centroid_max - centroid_min;
                    let axis = if extent.z > extent.x && extent.z > extent.y {
                        2
                    } else if extent.y > extent.x {
                        1
                    } else {
                        0
                    };

                    let axis_min = match axis {
                        0 => centroid_min.x,
                        1 => centroid_min.y,
                        _ => centroid_min.z,
                    };
                    let axis_extent = match axis {
                        0 => extent.x,
                        1 => extent.y,
                        _ => extent.z,
                    };

                    let mut bins = vec![
                        Bin {
                            count: 0,
                            rect: RectangleBounds::MAX,
                        };
                        self.max_bin_count
                    ];

                    prim_indices[start..end].iter().for_each(|&idx| {
                        let c = rects[idx].centroid();
                        let val = match axis {
                            0 => c.x,
                            1 => c.y,
                            _ => c.z,
                        };
                        let bin_idx = (((val - axis_min) / axis_extent)
                            * (self.max_bin_count as f32 - 0.001))
                            as usize;
                        bins[bin_idx].count += 1;
                        bins[bin_idx].rect = bins[bin_idx].rect + rects[idx];
                    });

                    let parent_area = node_rect.surface_area();

                    let mut right_sums = Vec::with_capacity(self.max_bin_count - 1);
                    let mut r_rect = RectangleBounds::MAX;
                    let mut r_count = 0;
                    for i in (1..self.max_bin_count).rev() {
                        r_rect = r_rect + bins[i].rect;
                        r_count += bins[i].count;
                        right_sums.push((r_rect, r_count));
                    }
                    right_sums.reverse();

                    let best_split = (1..self.max_bin_count)
                        .scan((RectangleBounds::MAX, 0), |(l_rect, l_count), i| {
                            *l_rect = *l_rect + bins[i - 1].rect;
                            *l_count += bins[i - 1].count;
                            let (r_rect, r_count) = right_sums[i - 1];
                            let cost = 0.125
                                + (l_rect.surface_area() / parent_area * *l_count as f32)
                                + (r_rect.surface_area() / parent_area * r_count as f32);
                            Some((i, cost))
                        })
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(self.max_bin_count / 2);

                    let local_slice = &mut prim_indices[start..end];
                    let count = local_slice.len();
                    let mut left = Vec::with_capacity(count / 2);
                    let mut right = Vec::with_capacity(count / 2);
                    local_slice.iter().for_each(|&idx| {
                        let val = match axis {
                            0 => rects[idx].centroid().x,
                            1 => rects[idx].centroid().y,
                            _ => rects[idx].centroid().z,
                        };
                        let bin_idx = (((val - axis_min) / axis_extent)
                            * (self.max_bin_count as f32 - 0.001))
                            as usize;
                        if bin_idx < best_split {
                            left.push(idx);
                        } else {
                            right.push(idx);
                        }
                    });

                    if left.is_empty() || right.is_empty() {
                        let idx = prim_indices[start];
                        returned_indices.push(nodes.len());
                        nodes.push(BVHNode::Leaf {
                            rect: node_rect,
                            idx,
                        });
                        continue;
                    }

                    let mid_offset = left.len();
                    local_slice[..mid_offset].copy_from_slice(&left);
                    local_slice[mid_offset..].copy_from_slice(&right);
                    let split = start + mid_offset;

                    let internal_idx = nodes.len();
                    nodes.push(BVHNode::default());

                    stages.push(BuildStage::ComputeBounds {
                        internal_idx,
                        left_idx: returned_indices.len(),
                        right_idx: returned_indices.len() + 1,
                    });
                    stages.push(BuildStage::ProcessRange { start: split, end });
                    stages.push(BuildStage::ProcessRange { start, end: split });
                }
                BuildStage::ComputeBounds {
                    internal_idx,
                    left_idx,
                    right_idx,
                } => {
                    let l_child = returned_indices[left_idx];
                    let r_child = returned_indices[right_idx];
                    let parent_rect = *nodes[l_child].rect() + *nodes[r_child].rect();
                    nodes[internal_idx] = BVHNode::Internal {
                        rect: parent_rect,
                        left_child: l_child,
                        right_child: r_child,
                    };
                    returned_indices.truncate(left_idx);
                    returned_indices.push(internal_idx);
                }
            }
        }

        returned_indices[0]
    }
}

impl CpuCollisionAlgorithm for SAH {
    fn build(&mut self, vertices: Vec<Vec<Vec3A>>) -> BVHTree {
        if vertices.is_empty() {
            return BVHTree::default();
        }

        let rects: Vec<RectangleBounds> = vertices
            .iter()
            .map(|prim_verts| RectangleBounds::from(prim_verts.as_slice()))
            .collect();

        let mut prim_indices: Vec<usize> = (0..rects.len()).collect();
        let mut nodes = Vec::with_capacity(rects.len() * 2);
        let root_idx = self.build_tree(&mut prim_indices, &rects, &mut nodes);

        BVHTree {
            tree: nodes,
            root_idx,
        }
    }
}
