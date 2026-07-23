use std::collections::HashMap;

use crate::{
    asts::{
        ASTOrConst, AstConst,
        jit::{JitAST, JitBinOp, JitUnaryOp, ReduceOp},
        lowered::{
            BinOp, LoweredAST,
            scope::{Scope, local},
        },
    },
    call,
    dt::{BasicTy, DType, IntegerTy, VecTy},
};

use super::super::pattern::RewriteRule;

use super::simplify;

fn identity(operand: &JitAST, op: ReduceOp) -> LoweredAST {
    let result_dt = operand.dt().peel_array();
    let scalar_dt = result_dt.peel_all();
    let elem_bytes = crate::asts::jit::scalar_identity_bytes(&scalar_dt, op);
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

pub fn lower_reduce(
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

pub fn unaryop_basic(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let (operand, op) = match matched {
        JitAST::UnaryOp { operand, op } => (operand, op),
        _ => unreachable!(),
    };
    let o = JitAST::graph_rewrite(*operand, scope, rules, var_producer);
    match op {
        JitUnaryOp::Basic(basic) => LoweredAST::UnaryOp {
            operand: Box::new(o),
            op: basic,
        },
        _ => panic!("unaryop_basic called on non-basic UnaryOp"),
    }
}

pub fn binop_basic(
    matched: JitAST,
    _captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let (lhs, rhs, op) = match matched {
        JitAST::BinOp { lhs, rhs, op } => (lhs, rhs, op),
        _ => unreachable!(),
    };

    let lhs_simple = simplify::simplify_node(*lhs);
    let rhs_simple = simplify::simplify_node(*rhs);
    if let Some(simple) = simplify::simplify_binop(lhs_simple.clone(), rhs_simple.clone(), op) {
        return JitAST::graph_rewrite(simple, scope, rules, var_producer);
    }

    let l = JitAST::graph_rewrite(lhs_simple, scope, rules, var_producer);
    let r = JitAST::graph_rewrite(rhs_simple, scope, rules, var_producer);
    match op {
        JitBinOp::Basic(basic) => LoweredAST::BinaryOp {
            lhs: Box::new(l),
            rhs: Box::new(r),
            op: basic,
        },
        _ => panic!("binop_basic called on non-basic BinOp"),
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
