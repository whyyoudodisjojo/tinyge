use std::collections::HashMap;

use crate::{
    asts::{
        jit::{JitAST, MovOp},
        lowered::{
            Accessor, LoweredAST, VarRef, VarRefType,
            scope::{Scope, entrypoint, local},
        },
    },
    dt::{BasicTy, DType, IntegerTy},
};

use super::super::pattern::RewriteRule;

pub fn coord_linearize(coords: &[LoweredAST], shape: &[usize]) -> LoweredAST {
    if coords.is_empty() || shape.is_empty() {
        return coords.first().cloned().unwrap_or(0u32.into());
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

pub struct PadCheck {
    pub pre_subtract_coords: Vec<LoweredAST>,
    pub amounts: Vec<(usize, usize)>,
    pub inner_shape: Vec<usize>,
}

pub fn apply_chain(
    coord: Vec<LoweredAST>,
    chain: &[&MovOp],
    shapes: &[Vec<usize>],
    _scope: &mut Scope,
) -> (Vec<LoweredAST>, Vec<PadCheck>) {
    let mut pad_checks = vec![];
    let result = chain.iter().enumerate().fold(coord, |c, (i, &op)| {
        let in_shape = &shapes[i + 1];
        let out_shape_i = &shapes[i];
        match op {
            MovOp::Reshape(_to) => {
                let linear = coord_linearize(&c, out_shape_i);
                if in_shape.is_empty() {
                    vec![linear]
                } else {
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
            }
            MovOp::Expand(_) => {
                let offset = c.len() - in_shape.len();
                c.into_iter()
                    .enumerate()
                    .filter_map(|(j, coord_val)| {
                        if j < offset {
                            None
                        } else {
                            let in_j = j - offset;
                            if in_shape[in_j] == 1 {
                                Some(0u32.into())
                            } else {
                                Some(coord_val)
                            }
                        }
                    })
                    .collect()
            }
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
            MovOp::Pad(amounts) => {
                pad_checks.push(PadCheck {
                    pre_subtract_coords: c.clone(),
                    amounts: amounts.clone(),
                    inner_shape: in_shape.clone(),
                });
                c.into_iter()
                    .enumerate()
                    .map(|(j, coord_val)| {
                        if amounts[j].0 == 0 {
                            coord_val
                        } else {
                            coord_val - (amounts[j].0 as u32).into()
                        }
                    })
                    .collect()
            }
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
    (result, pad_checks)
}

fn pad_bounds_check(
    loaded: LoweredAST,
    pad_checks: Vec<PadCheck>,
    base_dt: &DType,
    _scope: &mut Scope,
) -> LoweredAST {
    let zero = match base_dt.peel_all() {
        DType::Basic(BasicTy::F32) => LoweredAST::from(0.0f32),
        DType::Basic(BasicTy::Integer(IntegerTy::I32)) => LoweredAST::from(0i32),
        DType::Basic(BasicTy::Integer(IntegerTy::U32)) => LoweredAST::from(0u32),
        DType::Basic(BasicTy::Bool) => LoweredAST::from(false),
        _ => LoweredAST::from(0u32),
    };

    let mut result = loaded;
    for pc in pad_checks.into_iter().rev() {
        let mut dim_checks = vec![];
        for (j, coord_val) in pc.pre_subtract_coords.iter().enumerate() {
            let lo = pc.amounts[j].0 as u32;
            let hi = lo + pc.inner_shape[j] as u32;
            if lo > 0 || pc.amounts[j].1 > 0 {
                let in_dim = coord_val
                    .clone()
                    .ge(lo.into())
                    .logical_and(coord_val.clone().lt(hi.into()));
                dim_checks.push(in_dim);
            }
        }
        if !dim_checks.is_empty() {
            let in_bounds = dim_checks
                .into_iter()
                .reduce(|a, b| a.logical_and(b))
                .unwrap();
            result = LoweredAST::FunctionCall {
                ident: "select".to_string(),
                args: vec![
                    Box::new(zero.clone()),
                    Box::new(result),
                    Box::new(in_bounds),
                ],
            };
        }
    }
    result
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

            let (result_coord, pad_checks) = apply_chain(coord, &chain, &shapes, scope);
            let source_idx = coord_linearize(&result_coord, &base_shape);

            let loaded = LoweredAST::Load(VarRefType::Global(VarRef {
                id: binding_id,
                by: vec![Accessor::Index(Box::new(source_idx))],
            }));

            pad_bounds_check(loaded, pad_checks, &base.dt(), scope)
        }
        _ => base_loaded,
    }
}
