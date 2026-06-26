use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::collisions::algos::{BVHNode, BVHTree, CollisionAlgorithm, RectangleBounds, lbvh::Key};

#[derive(Default)]
pub struct LinearBVH;

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

impl LinearBVH {
    pub fn build_tree(keys: &[Key], rects: &[RectangleBounds], nodes: &mut Vec<BVHNode>) -> usize {
        let mut stages = vec![BuildStage::ProcessRange {
            start: 0,
            end: keys.len(),
        }];
        let mut returned_indices = Vec::with_capacity(32);

        while let Some(stage) = stages.pop() {
            match stage {
                BuildStage::ProcessRange { start, end } => {
                    if end - start == 1 {
                        let idx = keys[start].idx;
                        returned_indices.push(nodes.len());
                        nodes.push(BVHNode::Leaf {
                            rect: rects[idx as usize],
                            idx: idx as usize,
                        });
                        continue;
                    }

                    let first_code = keys[start].code;
                    let last_code = keys[end - 1].code;
                    let common_len = (first_code ^ last_code).leading_zeros();

                    let split = (start..end)
                        .position(|i| {
                            (first_code ^ keys[start + i].code).leading_zeros() <= common_len
                        })
                        .map(|pos| start + pos)
                        .unwrap_or((start + end) / 2);

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

impl CollisionAlgorithm for LinearBVH {
    fn build(&mut self, rects: &[RectangleBounds]) -> BVHTree {
        if rects.is_empty() {
            return BVHTree::default();
        }

        let global_bounds = rects
            .par_iter()
            .cloned()
            .reduce(|| rects[0], |agg, r| agg + r);
        let mut keys: Vec<_> = rects
            .par_iter()
            .enumerate()
            .map(|(id, r)| Key::new(r.centroid(), global_bounds.min, global_bounds.max, id))
            .collect();
        keys.par_sort_unstable_by_key(|k| k.code);

        let mut nodes = Vec::with_capacity(rects.len() * 2);
        let root_idx = Self::build_tree(&keys, rects, &mut nodes);
        BVHTree {
            tree: nodes,
            root_idx,
        }
    }
}
