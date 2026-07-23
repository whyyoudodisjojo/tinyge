use std::collections::HashMap;

use crate::{
    asts::{
        ASTOrConst, AstConst,
        jit::{JitAST, JitBinOp, JitUnaryOp, MovOp, ReduceOp, TernaryOp},
        lowered::{
            BinOp, LoweredAST,
            scope::{Scope, local},
        },
    },
    call,
    dt::{BasicTy, DType, IntegerTy, VecTy},
};

use super::pattern::{PatJitAST, RewriteRule};

fn scalar_identity_bytes(dt: &DType, op: ReduceOp) -> Vec<u8> {
    match (dt, op) {
        (DType::Basic(BasicTy::F32), ReduceOp::Sum) => 0u32.to_le_bytes().to_vec(),
        (DType::Basic(BasicTy::F32), ReduceOp::Prod) => 1.0f32.to_le_bytes().to_vec(),
        (DType::Basic(BasicTy::F32), ReduceOp::Max) => f32::MIN.to_le_bytes().to_vec(),
        (DType::Basic(BasicTy::Integer(IntegerTy::U32)), ReduceOp::Sum) => {
            0u32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::U32)), ReduceOp::Prod) => {
            1u32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::U32)), ReduceOp::Max) => {
            0u32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::I32)), ReduceOp::Sum) => {
            0i32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::I32)), ReduceOp::Prod) => {
            1i32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::I32)), ReduceOp::Max) => {
            i32::MIN.to_le_bytes().to_vec()
        }
        _ => panic!("no identity for ({:?}, {:?})", dt, op),
    }
}

fn identity(operand: &JitAST, op: ReduceOp) -> LoweredAST {
    let result_dt = operand.dt().peel_array();
    let scalar_dt = result_dt.peel_all();
    let elem_bytes = scalar_identity_bytes(&scalar_dt, op);
    let count = result_dt.element_count();
    LoweredAST::Const(AstConst {
        dt: result_dt,
        data: (0..count)
            .map(|_| ASTOrConst::Const(elem_bytes.clone()))
            .collect(),
    })
}

fn lower_where_array(
    a_lowered: LoweredAST,
    b_lowered: LoweredAST,
    c_lowered: LoweredAST,
    a_dt: DType,
    b_dt: DType,
    c_dt: DType,
    n: u32,
    result_dt: DType,
    scope: &mut Scope,
) -> LoweredAST {
    let loop_idx = scope.var(LoweredAST::Const(AstConst {
        dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
        data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
    }));
    let result_init = LoweredAST::Const(AstConst {
        dt: result_dt.clone(),
        data: vec![],
    });
    let result_id = scope.mut_(result_init);
    let for_loop = scope.for_loop(
        Some(local(loop_idx).store(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
        }))),
        Some(local(loop_idx).load().lt(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(n.to_le_bytes().to_vec())],
        }))),
        Some(local(loop_idx).store(
            local(loop_idx).load()
                + LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const(1u32.to_le_bytes().to_vec())],
                }),
        )),
        |body_scope: &mut Scope| {
            let i_load = LoweredAST::Load(local(loop_idx));
            let i_load_for_idx = i_load.clone();
            let idx = |expr: &LoweredAST, operand_dt: &DType| -> LoweredAST {
                if matches!(operand_dt, DType::Vector(VecTy::Array(_, _))) {
                    match expr {
                        LoweredAST::Load(var) => {
                            LoweredAST::Load(var.clone().i(i_load_for_idx.clone()))
                        }
                        _ => panic!("cannot index into non-Load node"),
                    }
                } else {
                    expr.clone()
                }
            };
            body_scope.ast = Some(local(result_id).i(i_load).store(LoweredAST::FunctionCall {
                ident: "select".to_string(),
                args: vec![
                    Box::new(idx(&b_lowered, &b_dt)),
                    Box::new(idx(&a_lowered, &a_dt)),
                    Box::new(idx(&c_lowered, &c_dt)),
                ],
            }));
        },
    );
    LoweredAST::Group(vec![for_loop, LoweredAST::Load(local(result_id))])
}

fn lower_movement_flip(lowered: LoweredAST, dt: DType, n: u32, scope: &mut Scope) -> LoweredAST {
    let result = scope.mut_(LoweredAST::Const(AstConst {
        dt: dt.clone(),
        data: vec![],
    }));
    let i = scope.var(LoweredAST::Const(AstConst {
        dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
        data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
    }));
    let for_loop = scope.for_loop(
        Some(local(i).store(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
        }))),
        Some(local(i).load().lt(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(n.to_le_bytes().to_vec())],
        }))),
        Some(local(i).store(
            local(i).load()
                + LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const(1u32.to_le_bytes().to_vec())],
                }),
        )),
        |body| {
            let src = LoweredAST::Const(AstConst {
                dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                data: vec![ASTOrConst::Const((n - 1).to_le_bytes().to_vec())],
            }) - local(i).load();
            let val = match &lowered {
                LoweredAST::Load(var) => LoweredAST::Load(var.clone().i(src)),
                _ => panic!("cannot index into non-Load node"),
            };
            body.ast = Some(local(result).i(local(i).load()).store(val));
        },
    );
    LoweredAST::Group(vec![for_loop, LoweredAST::Load(local(result))])
}

fn lower_movement_pad(
    lowered: LoweredAST,
    dt: DType,
    amounts: &[(usize, usize)],
    scope: &mut Scope,
) -> LoweredAST {
    let DType::Vector(VecTy::Array(_, Some(n))) = &dt else {
        panic!("pad on non-array")
    };
    let n = *n;
    let total_lo = amounts.iter().map(|(lo, _)| *lo).sum::<usize>() as u32;
    let total_hi = amounts.iter().map(|(_, hi)| hi).sum::<usize>() as u32;
    let out_n = n + total_lo + total_hi;
    let result = scope.mut_(LoweredAST::Const(AstConst {
        dt: dt.clone(),
        data: vec![],
    }));
    let i = scope.var(LoweredAST::Const(AstConst {
        dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
        data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
    }));
    let for_loop = scope.for_loop(
        Some(local(i).store(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
        }))),
        Some(local(i).load().lt(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(out_n.to_le_bytes().to_vec())],
        }))),
        Some(local(i).store(
            local(i).load()
                + LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const(1u32.to_le_bytes().to_vec())],
                }),
        )),
        |body| {
            let ii = local(i).load();
            let in_bounds = ii
                .clone()
                .ge(LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const(total_lo.to_le_bytes().to_vec())],
                }))
                .logical_and(ii.clone().lt(LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const((n + total_lo).to_le_bytes().to_vec())],
                })));
            let src_idx = ii.clone()
                - LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const(total_lo.to_le_bytes().to_vec())],
                });
            let val = match &lowered {
                LoweredAST::Load(var) => LoweredAST::Load(var.clone().i(src_idx)),
                _ => panic!("cannot index into non-Load node"),
            };
            let selected = LoweredAST::FunctionCall {
                ident: "select".to_string(),
                args: vec![
                    Box::new(LoweredAST::Const(AstConst {
                        dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                        data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
                    })),
                    Box::new(val),
                    Box::new(in_bounds),
                ],
            };
            body.ast = Some(local(result).i(ii).store(selected));
        },
    );
    LoweredAST::Group(vec![for_loop, LoweredAST::Load(local(result))])
}

fn lower_movement_shrink(
    lowered: LoweredAST,
    dt: DType,
    amounts: &[(usize, usize)],
    scope: &mut Scope,
) -> LoweredAST {
    let DType::Vector(VecTy::Array(_, Some(n))) = &dt else {
        panic!("shrink on non-array")
    };
    let n = *n;
    let total_lo = amounts.iter().map(|(lo, _)| *lo).sum::<usize>() as u32;
    let total_rem = amounts.iter().map(|(lo, hi)| lo + hi).sum::<usize>() as u32;
    let out_n = n - total_rem;
    let result = scope.mut_(LoweredAST::Const(AstConst {
        dt: dt.clone(),
        data: vec![],
    }));
    let i = scope.var(LoweredAST::Const(AstConst {
        dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
        data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
    }));
    let for_loop = scope.for_loop(
        Some(local(i).store(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(0u32.to_le_bytes().to_vec())],
        }))),
        Some(local(i).load().lt(LoweredAST::Const(AstConst {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: vec![ASTOrConst::Const(out_n.to_le_bytes().to_vec())],
        }))),
        Some(local(i).store(
            local(i).load()
                + LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const(1u32.to_le_bytes().to_vec())],
                }),
        )),
        |body| {
            let src = local(i).load()
                + LoweredAST::Const(AstConst {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: vec![ASTOrConst::Const(total_lo.to_le_bytes().to_vec())],
                });
            let val = match &lowered {
                LoweredAST::Load(var) => LoweredAST::Load(var.clone().i(src)),
                _ => panic!("cannot index into non-Load node"),
            };
            body.ast = Some(local(result).i(local(i).load()).store(val));
        },
    );
    LoweredAST::Group(vec![for_loop, LoweredAST::Load(local(result))])
}

fn lower_reduce(
    operand: Box<JitAST>,
    op: ReduceOp,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let init = identity(&operand, op);
    let lowered = JitAST::graph_rewrite(*operand, scope, rules, var_producer);
    let acc = scope.mut_(init);
    let store = match op {
        ReduceOp::Sum => local(acc).store(local(acc).load() + lowered),
        ReduceOp::Prod => local(acc).store(local(acc).load() * lowered),
        ReduceOp::Max => local(acc).store(call!("max", local(acc).load(), lowered)),
    };
    LoweredAST::Group(vec![store, LoweredAST::Load(local(acc))])
}

// --- Rule transforms ---

pub fn cdiv(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "ceil_div".into(),
        args: vec![Box::new(l), Box::new(r)],
    }
}

pub fn binop_max(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "max".into(),
        args: vec![Box::new(l), Box::new(r)],
    }
}

pub fn cmod(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "ceil_mod".into(),
        args: vec![Box::new(l), Box::new(r)],
    }
}

pub fn fdiv(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "fdiv".into(),
        args: vec![Box::new(l), Box::new(r)],
    }
}

pub fn pow(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "pow".into(),
        args: vec![Box::new(l), Box::new(r)],
    }
}

pub fn floordiv(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "floor_div".into(),
        args: vec![Box::new(l), Box::new(r)],
    }
}

pub fn floormod(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "floor_mod".into(),
        args: vec![Box::new(l), Box::new(r)],
    }
}

pub fn threefry(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let l = JitAST::graph_rewrite(c.remove("lhs").unwrap(), scope, rules, var_producer);
    let r = JitAST::graph_rewrite(c.remove("rhs").unwrap(), scope, rules, var_producer);
    let key = LoweredAST::Const(AstConst {
        dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
        data: vec![ASTOrConst::Const(2654435761u32.to_le_bytes().to_vec())],
    });
    LoweredAST::BinaryOp {
        lhs: Box::new(LoweredAST::BinaryOp {
            lhs: Box::new(l),
            rhs: Box::new(key),
            op: BinOp::Mul,
        }),
        rhs: Box::new(r),
        op: BinOp::Add,
    }
}

pub fn exp2(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let o = JitAST::graph_rewrite(c.remove("x").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "exp2".into(),
        args: vec![Box::new(o)],
    }
}

pub fn log2(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let o = JitAST::graph_rewrite(c.remove("x").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "log2".into(),
        args: vec![Box::new(o)],
    }
}

pub fn sin(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let o = JitAST::graph_rewrite(c.remove("x").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "sin".into(),
        args: vec![Box::new(o)],
    }
}

pub fn sqrt(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let o = JitAST::graph_rewrite(c.remove("x").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "sqrt".into(),
        args: vec![Box::new(o)],
    }
}

pub fn reciprocal(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let o = JitAST::graph_rewrite(c.remove("x").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "reciprocal".into(),
        args: vec![Box::new(o)],
    }
}

pub fn trunc(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let o = JitAST::graph_rewrite(c.remove("x").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "trunc".into(),
        args: vec![Box::new(o)],
    }
}

pub fn bitcast(
    _matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let mut c = captured;
    let o = JitAST::graph_rewrite(c.remove("x").unwrap(), scope, rules, var_producer);
    LoweredAST::FunctionCall {
        ident: "bitcast".into(),
        args: vec![Box::new(o)],
    }
}

pub fn cast(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let JitAST::Cast { operand, dt } = matched else {
        unreachable!()
    };
    let src_dt = operand.dt();
    let src_count = src_dt.element_count();
    let dst_count = dt.element_count();
    let o = JitAST::graph_rewrite(*operand, scope, rules, var_producer);

    let data: Vec<ASTOrConst<LoweredAST>> = if src_count == dst_count {
        vec![ASTOrConst::AST(o)]
    } else if let LoweredAST::Load(var) = &o {
        let fields = ["x", "y", "z", "w"];
        (0..dst_count)
            .map(|i| {
                if i < src_count {
                    ASTOrConst::AST(LoweredAST::Load(var.clone().f(fields[i])))
                } else {
                    ASTOrConst::Const(0u32.to_le_bytes().to_vec())
                }
            })
            .collect()
    } else {
        let tmp = scope.var(o);
        let fields = ["x", "y", "z", "w"];
        (0..dst_count)
            .map(|i| {
                if i < src_count {
                    ASTOrConst::AST(LoweredAST::Load(local(tmp).f(fields[i])))
                } else {
                    ASTOrConst::Const(0u32.to_le_bytes().to_vec())
                }
            })
            .collect()
    };

    LoweredAST::Const(AstConst { dt, data })
}

pub fn ternary_where(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let JitAST::Ternary { a, b, c, op: _ } = matched else {
        unreachable!()
    };
    let a_dt = a.dt();
    let b_dt = b.dt();
    let c_dt = c.dt();
    let result_dt = a_dt.clone();
    let a_lowered = JitAST::graph_rewrite(*a, scope, rules, var_producer);
    let b_lowered = JitAST::graph_rewrite(*b, scope, rules, var_producer);
    let c_lowered = JitAST::graph_rewrite(*c, scope, rules, var_producer);
    match &result_dt {
        DType::Vector(VecTy::Array(_, Some(n))) => lower_where_array(
            a_lowered, b_lowered, c_lowered, a_dt, b_dt, c_dt, *n, result_dt, scope,
        ),
        _ => LoweredAST::FunctionCall {
            ident: "select".into(),
            args: vec![
                Box::new(b_lowered),
                Box::new(a_lowered),
                Box::new(c_lowered),
            ],
        },
    }
}

pub fn ternary_mulacc(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let JitAST::Ternary { a, b, c, op: _ } = matched else {
        unreachable!()
    };
    let a_lowered = JitAST::graph_rewrite(*a, scope, rules, var_producer);
    let b_lowered = JitAST::graph_rewrite(*b, scope, rules, var_producer);
    let c_lowered = JitAST::graph_rewrite(*c, scope, rules, var_producer);
    LoweredAST::BinaryOp {
        lhs: Box::new(LoweredAST::BinaryOp {
            lhs: Box::new(a_lowered),
            rhs: Box::new(b_lowered),
            op: BinOp::Mul,
        }),
        rhs: Box::new(c_lowered),
        op: BinOp::Add,
    }
}

pub fn movement(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let JitAST::Movement { operand, op } = matched else {
        unreachable!()
    };
    let dt = operand.dt();
    let lowered = JitAST::graph_rewrite(*operand, scope, rules, var_producer);
    match (op, dt.clone()) {
        (MovOp::Flip(_), DType::Vector(VecTy::Array(_, Some(n)))) => {
            lower_movement_flip(lowered, dt, n, scope)
        }
        (MovOp::Flip(_), DType::Vector(VecTy::Vec2(_))) => {
            let fields = ["x", "y"];
            let args: Vec<_> = fields
                .iter()
                .rev()
                .map(|f| {
                    Box::new(match &lowered {
                        LoweredAST::Load(var) => LoweredAST::Load(var.clone().f(f)),
                        _ => panic!("cannot swizzle non-Load node"),
                    })
                })
                .collect();
            LoweredAST::FunctionCall {
                ident: "vec2".into(),
                args,
            }
        }
        (MovOp::Flip(_), DType::Vector(VecTy::Vec3(_))) => {
            let fields = ["x", "y", "z"];
            let args: Vec<_> = fields
                .iter()
                .rev()
                .map(|f| {
                    Box::new(match &lowered {
                        LoweredAST::Load(var) => LoweredAST::Load(var.clone().f(f)),
                        _ => panic!("cannot swizzle non-Load node"),
                    })
                })
                .collect();
            LoweredAST::FunctionCall {
                ident: "vec3".into(),
                args,
            }
        }
        (MovOp::Flip(_), DType::Vector(VecTy::Vec4(_))) => {
            let fields = ["x", "y", "z", "w"];
            let args: Vec<_> = fields
                .iter()
                .rev()
                .map(|f| {
                    Box::new(match &lowered {
                        LoweredAST::Load(var) => LoweredAST::Load(var.clone().f(f)),
                        _ => panic!("cannot swizzle non-Load node"),
                    })
                })
                .collect();
            LoweredAST::FunctionCall {
                ident: "vec4".into(),
                args,
            }
        }
        (MovOp::Pad(amounts), DType::Vector(VecTy::Array(_, _))) => {
            lower_movement_pad(lowered, dt, &amounts, scope)
        }
        (MovOp::Shrink(amounts), DType::Vector(VecTy::Array(_, _))) => {
            lower_movement_shrink(lowered, dt, &amounts, scope)
        }
        _ => lowered,
    }
}

pub fn reduce(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let JitAST::Reduce {
        ref operand,
        axis: _,
        op,
    } = matched
    else {
        unreachable!()
    };
    lower_reduce(operand.clone(), op, scope, var_producer, rules)
}

pub fn all_reduce(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let JitAST::AllReduce { ref operand, op } = matched else {
        unreachable!()
    };
    lower_reduce(operand.clone(), op, scope, var_producer, rules)
}

pub fn builtin_rules() -> Vec<RewriteRule> {
    use PatJitAST::*;
    vec![
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Cdiv),
            },
            transform: cdiv,
        },
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Max),
            },
            transform: binop_max,
        },
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Cmod),
            },
            transform: cmod,
        },
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Fdiv),
            },
            transform: fdiv,
        },
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Pow),
            },
            transform: pow,
        },
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Floordiv),
            },
            transform: floordiv,
        },
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Floormod),
            },
            transform: floormod,
        },
        RewriteRule {
            pat: BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Threefry),
            },
            transform: threefry,
        },
        RewriteRule {
            pat: UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Exp2),
            },
            transform: exp2,
        },
        RewriteRule {
            pat: UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Log2),
            },
            transform: log2,
        },
        RewriteRule {
            pat: UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Sin),
            },
            transform: sin,
        },
        RewriteRule {
            pat: UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Sqrt),
            },
            transform: sqrt,
        },
        RewriteRule {
            pat: UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Reciprocal),
            },
            transform: reciprocal,
        },
        RewriteRule {
            pat: UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Trunc),
            },
            transform: trunc,
        },
        RewriteRule {
            pat: UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Bitcast),
            },
            transform: bitcast,
        },
        RewriteRule {
            pat: Cast {
                operand: Box::new(Var("x".into())),
                dt: None,
            },
            transform: cast,
        },
        RewriteRule {
            pat: Ternary {
                a: Box::new(Var("a".into())),
                b: Box::new(Var("b".into())),
                c: Box::new(Var("c".into())),
                op: Some(TernaryOp::Where),
            },
            transform: ternary_where,
        },
        RewriteRule {
            pat: Ternary {
                a: Box::new(Var("a".into())),
                b: Box::new(Var("b".into())),
                c: Box::new(Var("c".into())),
                op: Some(TernaryOp::Mulacc),
            },
            transform: ternary_mulacc,
        },
        RewriteRule {
            pat: Movement {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            transform: movement,
        },
        RewriteRule {
            pat: Reduce {
                operand: Box::new(Var("x".into())),
                axis: 0,
                op: None,
            },
            transform: reduce,
        },
        RewriteRule {
            pat: AllReduce {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            transform: all_reduce,
        },
    ]
}
