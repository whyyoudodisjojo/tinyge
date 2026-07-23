use std::collections::HashMap;

use crate::{
    asts::{
        ASTOrConst, AstConst,
        jit::{JitAST, MovOp},
        lowered::{
            Accessor, LoweredAST, VarRef, VarRefType,
            scope::{Scope, entrypoint, local},
        },
    },
    dt::{BasicTy, DType},
};

use super::super::pattern::RewriteRule;

fn coord_linearize(coords: &[LoweredAST], shape: &[usize]) -> LoweredAST {
    if coords.is_empty() {
        return 0u32.into();
    }
    coords
        .iter()
        .enumerate()
        .fold(LoweredAST::from(0u32), |acc, (i, coord)| {
            let stride = shape[i + 1..].iter().product::<usize>() as u32;
            if stride > 0 {
                acc + coord.clone() * stride.into()
            } else {
                acc + coord.clone()
            }
        })
}

pub fn movement(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let (base, chain) = matched.inner_movement_chain();
    if chain.is_empty() {
        return JitAST::graph_rewrite(base.clone(), scope, rules, var_producer);
    }
    let out_shape = matched.shape();
    let base_shape = base.shape();

    let base_loaded = JitAST::graph_rewrite(base.clone(), scope, rules, var_producer);

    let shapes: Vec<Vec<usize>> = std::iter::successors(Some(&matched as &JitAST), |node| {
        if let JitAST::Movement { operand, .. } = node {
            Some(operand.as_ref())
        } else {
            None
        }
    })
    .map(|n| n.shape())
    .collect();

    match &base_loaded {
        LoweredAST::Load(VarRefType::Global(vr)) => {
            let binding_id = vr.id;
            let thread_id = entrypoint(0).f("x").load();

            let coord = {
                let linear = thread_id.clone();
                if out_shape.len() <= 1 {
                    vec![linear]
                } else {
                    let (mut coords, rem_var) = (0..out_shape.len() - 1).fold(
                        (vec![], scope.var(linear)),
                        |(mut coords, rem_var), i| {
                            let stride = out_shape[i + 1..].iter().product::<usize>() as u32;
                            let new_rem_var = scope.var(local(rem_var).load() % stride.into());
                            coords.push(local(rem_var).load() / stride.into());
                            (coords, new_rem_var)
                        },
                    );
                    coords.push(local(rem_var).load());
                    coords
                }
            };

            let result_coord = chain.iter().enumerate().fold(coord, |c, (i, &op)| {
                let in_shape = &shapes[i + 1];
                let out_shape_i = &shapes[i];
                match op {
                    MovOp::Reshape(_to) => {
                        let linear = coord_linearize(&c, out_shape_i);
                        let mut result = vec![];
                        let mut rem = linear;
                        for j in 0..in_shape.len() {
                            let stride = in_shape[j + 1..].iter().product::<usize>() as u32;
                            if j < in_shape.len() - 1 {
                                result.push(rem.clone() / stride.into());
                                rem = rem % stride.into();
                            } else {
                                result.push(rem.clone());
                            }
                        }
                        result
                    }
                    MovOp::Expand(_) => c
                        .into_iter()
                        .enumerate()
                        .map(|(j, coord_val)| {
                            if j < in_shape.len() && in_shape[j] == 1 {
                                0u32.into()
                            } else {
                                coord_val
                            }
                        })
                        .collect(),
                    MovOp::Permute(perm) => perm
                        .iter()
                        .enumerate()
                        .fold(vec![0usize; perm.len()], |mut acc, (j, &p)| {
                            acc[p] = j;
                            acc
                        })
                        .into_iter()
                        .map(|j| c[j].clone())
                        .collect(),
                    MovOp::Pad(amounts) => c
                        .into_iter()
                        .enumerate()
                        .map(|(j, coord_val)| {
                            if amounts[j].0 == 0 {
                                coord_val
                            } else {
                                coord_val - (amounts[j].0 as u32).into()
                            }
                        })
                        .collect(),
                    MovOp::Shrink(amounts) => c
                        .into_iter()
                        .enumerate()
                        .map(|(j, coord_val)| {
                            if amounts[j].0 == 0 {
                                coord_val
                            } else {
                                coord_val + (amounts[j].0 as u32).into()
                            }
                        })
                        .collect(),
                    MovOp::Flip(axis) => c
                        .into_iter()
                        .enumerate()
                        .map(|(j, coord_val)| {
                            if j == *axis {
                                LoweredAST::from((out_shape_i[*axis] - 1) as u32) - coord_val
                            } else {
                                coord_val
                            }
                        })
                        .collect(),
                }
            });

            let source_idx = coord_linearize(&result_coord, &base_shape);

            let loaded = LoweredAST::Load(VarRefType::Global(VarRef {
                id: binding_id,
                by: vec![Accessor::Index(Box::new(source_idx))],
            }));

            chain
                .iter()
                .enumerate()
                .find_map(|(i, op)| {
                    if let MovOp::Pad(amounts) = op {
                        let lo_total = amounts.iter().map(|(lo, _)| *lo).sum::<usize>() as u32;
                        let src_n = shapes[i + 1].iter().product::<usize>() as u32;
                        let in_bounds = thread_id
                            .clone()
                            .ge(lo_total.into())
                            .logical_and(thread_id.clone().lt((lo_total + src_n).into()));

                        let zero = match base.dt() {
                            DType::Basic(BasicTy::F32) => LoweredAST::Const(AstConst {
                                dt: DType::Basic(BasicTy::F32),
                                data: vec![ASTOrConst::Const(0.0f32.to_le_bytes().to_vec())],
                            }),
                            _ => LoweredAST::from(0u32),
                        };

                        Some(LoweredAST::FunctionCall {
                            ident: "select".to_string(),
                            args: vec![
                                Box::new(zero),
                                Box::new(loaded.clone()),
                                Box::new(in_bounds),
                            ],
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or(loaded)
        }
        _ => base_loaded,
    }
}
