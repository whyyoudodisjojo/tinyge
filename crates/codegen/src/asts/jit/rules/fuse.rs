use std::collections::HashMap;

use crate::{
    asts::{
        ASTOrConst, AstConst,
        jit::{JitAST, ReduceOp},
        lowered::{
            Accessor, LoweredAST, VarRef, VarRefType,
            scope::{Scope, local},
        },
    },
    call,
};

use super::super::{
    pattern::RewriteRule,
    rules::{basic, movement},
};

pub fn fuse_reduce(
    matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let x = captured.get("x").unwrap().clone();

    let outer_op = match &matched {
        JitAST::AllReduce { op, .. } => *op,
        JitAST::Reduce { op, .. } => *op,
        _ => unreachable!(),
    };

    let needs_axis_decompose = match &matched {
        JitAST::AllReduce { operand, .. } => matches!(operand.as_ref(), JitAST::Reduce { .. }),
        _ => false,
    };

    if !needs_axis_decompose {
        return basic::lower_reduce(Box::new(x), outer_op, scope, var_producer, rules, None);
    }

    let (ax, inner_op) = match &matched {
        JitAST::AllReduce { operand, .. } => match operand.as_ref() {
            JitAST::Reduce { axis, op, .. } => (*axis, *op),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };

    let input_shape = x.shape();
    let axis_size = input_shape[ax] as u32;

    let intermediate_shape: Vec<usize> = input_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != ax)
        .map(|(_, &s)| s)
        .collect();

    let var_load = var_producer();
    let binding_id = match &var_load {
        LoweredAST::Load(VarRefType::Global(VarRef { id, .. })) => *id,
        _ => panic!("expected Global var load from var_producer"),
    };

    let (base, chain) = x.inner_movement_chain();
    let shapes: Vec<Vec<usize>> = std::iter::successors(Some(&x as &JitAST), |node| {
        if let JitAST::Movement { operand, .. } = node {
            Some(operand.as_ref())
        } else {
            None
        }
    })
    .map(|n| n.shape())
    .collect();
    let base_shape = base.shape();

    let result_dt = x.dt().peel_array();
    let scalar_dt = result_dt.peel_all();

    let outer_identity = crate::asts::jit::scalar_identity_bytes(&scalar_dt, outer_op);
    let outer_count = result_dt.element_count();
    let outer_init = LoweredAST::Const(AstConst {
        dt: result_dt.clone(),
        data: (0..outer_count)
            .map(|_| ASTOrConst::Const(outer_identity.clone()))
            .collect(),
    });
    let outer_acc = scope.mut_(outer_init);

    let intermediate_size: u32 = intermediate_shape.iter().product::<usize>() as u32;

    let outer_loop_var = scope.mut_(LoweredAST::from(0u32));
    let outer_loop = scope.for_loop(
        None,
        Some(
            local(outer_loop_var)
                .load()
                .lt(LoweredAST::from(intermediate_size)),
        ),
        Some(
            local(outer_loop_var)
                .store(local(outer_loop_var).load() + LoweredAST::from(1u32)),
        ),
        |outer_body: &mut Scope| {
            let outer_i = local(outer_loop_var).load();

            let inner_identity = crate::asts::jit::scalar_identity_bytes(&scalar_dt, inner_op);
            let inner_init = LoweredAST::Const(AstConst {
                dt: result_dt.clone(),
                data: (0..outer_count)
                    .map(|_| ASTOrConst::Const(inner_identity.clone()))
                    .collect(),
            });
            let inner_acc = outer_body.mut_(inner_init);

            let inner_loop_var = outer_body.mut_(LoweredAST::from(0u32));
            let inner_loop = outer_body.for_loop(
                None,
                Some(
                    local(inner_loop_var)
                        .load()
                        .lt(LoweredAST::from(axis_size)),
                ),
                Some(
                    local(inner_loop_var)
                        .store(local(inner_loop_var).load() + LoweredAST::from(1u32)),
                ),
                |inner_body: &mut Scope| {
                    let inner_i = local(inner_loop_var).load();

                    let intermediate_coord: Vec<LoweredAST> =
                        if intermediate_shape.len() <= 1 {
                            vec![outer_i.clone()]
                        } else {
                            let (mut coords, rem_var) =
                                (0..intermediate_shape.len() - 1).fold(
                                    (vec![], inner_body.var(outer_i.clone())),
                                    |(mut coords, rem_var), d| {
                                        let stride = intermediate_shape[d + 1..]
                                            .iter()
                                            .product::<usize>() as u32;
                                        let new_rem_var = inner_body.var(
                                            local(rem_var).load() % stride.into(),
                                        );
                                        coords
                                            .push(local(rem_var).load() / stride.into());
                                        (coords, new_rem_var)
                                    },
                                );
                            coords.push(local(rem_var).load());
                            coords
                        };

                    let mut input_coord = Vec::with_capacity(input_shape.len());
                    for d in 0..input_shape.len() {
                        if d == ax {
                            input_coord.push(inner_i.clone());
                        } else {
                            let out_d = if d < ax { d } else { d - 1 };
                            input_coord.push(intermediate_coord[out_d].clone());
                        }
                    }

                    let (result_coord, _pad_checks) =
                        movement::apply_chain(input_coord, &chain, &shapes, inner_body);
                    let source_idx =
                        movement::coord_linearize(&result_coord, &base_shape);

                    let var_load = LoweredAST::Load(VarRefType::Global(VarRef {
                        id: binding_id,
                        by: vec![Accessor::Index(Box::new(source_idx))],
                    }));

                    let inner_acc_val = match inner_op {
                        ReduceOp::Sum => local(inner_acc).load() + var_load,
                        ReduceOp::Prod => local(inner_acc).load() * var_load,
                        ReduceOp::Max => {
                            call!("max", local(inner_acc).load(), var_load)
                        }
                    };
                    inner_body.ast =
                        Some(local(inner_acc).store(inner_acc_val));
                },
            );

            let inner_result = LoweredAST::Load(local(inner_acc));
            let outer_acc_val = match outer_op {
                ReduceOp::Sum => local(outer_acc).load() + inner_result,
                ReduceOp::Prod => local(outer_acc).load() * inner_result,
                ReduceOp::Max => {
                    call!("max", local(outer_acc).load(), inner_result)
                }
            };
            outer_body.ast = Some(LoweredAST::Group(vec![
                inner_loop,
                local(outer_acc).store(outer_acc_val),
            ]));
        },
    );

    LoweredAST::Group(vec![outer_loop, LoweredAST::Load(local(outer_acc))])
}

pub fn fuse_cast_cast(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let JitAST::Cast {
        operand,
        dt: outer_dt,
    } = matched
    else {
        unreachable!()
    };
    let JitAST::Cast {
        operand: inner_operand,
        ..
    } = *operand
    else {
        unreachable!()
    };
    let fused = JitAST::Cast {
        operand: inner_operand,
        dt: outer_dt,
    };
    basic::cast(fused, HashMap::new(), scope, var_producer, rules)
}
