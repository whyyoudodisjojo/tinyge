use std::collections::HashMap;

use crate::{
    asts::{
        AstConst,
        jit::{JitAST, JitBinOp, JitUnaryOp, MovOp, ReduceOp, TernaryOp},
        lowered::{ASTOrConst, LoweredAST, scope::Scope},
    },
    dt::DType,
};

#[derive(Clone)]
pub enum PatJitAST {
    Var(String),

    Const(AstConst<Option<Self>, ()>),

    BinOp {
        lhs: Box<Self>,
        rhs: Box<Self>,
        op: Option<JitBinOp>,
    },
    UnaryOp {
        operand: Box<Self>,
        op: Option<JitUnaryOp>,
    },
    Cast {
        operand: Box<Self>,
        dt: Option<DType>,
    },
    Ternary {
        a: Box<Self>,
        b: Box<Self>,
        c: Box<Self>,
        op: Option<TernaryOp>,
    },
    Movement {
        operand: Box<Self>,
        op: Option<MovOp>,
    },
    Reduce {
        operand: Box<Self>,
        axis: usize,
        op: Option<ReduceOp>,
    },
    AllReduce {
        operand: Box<Self>,
        op: Option<ReduceOp>,
    },
}

impl PatJitAST {
    pub fn matches(&self, ast: &JitAST, ctx: &mut HashMap<String, JitAST>) -> bool {
        match (self, ast) {
            (PatJitAST::Var(n), _) => {
                ctx.insert(n.clone(), ast.clone());
                true
            }
            (PatJitAST::Const(c1), JitAST::Const(c2)) => {
                if c1.dt == c2.dt {
                    c1.data
                        .iter()
                        .zip(c2.data.iter())
                        .all(|(c, x)| match (c, x) {
                            (ASTOrConst::AST(a1), ASTOrConst::AST(a2)) => {
                                a1.as_ref().map(|a| a.matches(a2, ctx)).unwrap_or(true)
                            }
                            (ASTOrConst::Const(_), ASTOrConst::Const(_)) => true,
                            _ => false,
                        })
                } else {
                    false
                }
            }
            (
                PatJitAST::BinOp {
                    lhs: p_lhs,
                    rhs: p_rhs,
                    op: p_op,
                },
                JitAST::BinOp {
                    lhs: a_lhs,
                    rhs: a_rhs,
                    op: a_op,
                },
            ) => {
                if let Some(required_op) = p_op {
                    if required_op != a_op {
                        return false;
                    }
                }
                p_lhs.matches(a_lhs, ctx) && p_rhs.matches(a_rhs, ctx)
            }
            (
                PatJitAST::UnaryOp {
                    operand: p_operand,
                    op: p_op,
                },
                JitAST::UnaryOp {
                    operand: a_operand,
                    op: a_op,
                },
            ) => {
                if let Some(required_op) = p_op {
                    if required_op != a_op {
                        return false;
                    }
                }
                p_operand.matches(a_operand, ctx)
            }
            (
                PatJitAST::Cast {
                    operand: p_operand,
                    dt: p_dt,
                },
                JitAST::Cast {
                    operand: a_operand,
                    dt: a_dt,
                },
            ) => {
                if let Some(required_dt) = p_dt {
                    if *required_dt != *a_dt {
                        return false;
                    }
                }
                p_operand.matches(a_operand, ctx)
            }
            (
                PatJitAST::Ternary {
                    a: p_a,
                    b: p_b,
                    c: p_c,
                    op: p_op,
                },
                JitAST::Ternary {
                    a: a_a,
                    b: a_b,
                    c: a_c,
                    op: a_op,
                },
            ) => {
                if let Some(required_op) = p_op {
                    if required_op != a_op {
                        return false;
                    }
                }
                p_a.matches(a_a, ctx) && p_b.matches(a_b, ctx) && p_c.matches(a_c, ctx)
            }
            (
                PatJitAST::Movement {
                    operand: p_operand,
                    op: p_op,
                },
                JitAST::Movement {
                    operand: a_operand,
                    op: a_op,
                },
            ) => {
                if let Some(required_op) = p_op {
                    if required_op != a_op {
                        return false;
                    }
                }
                p_operand.matches(a_operand, ctx)
            }
            (
                PatJitAST::Reduce {
                    operand: p_operand,
                    axis: p_axis,
                    op: p_op,
                },
                JitAST::Reduce {
                    operand: a_operand,
                    axis: a_axis,
                    op: a_op,
                },
            ) => {
                if p_axis != a_axis {
                    return false;
                }
                if let Some(required_op) = p_op {
                    if required_op != a_op {
                        return false;
                    }
                }
                p_operand.matches(a_operand, ctx)
            }
            (
                PatJitAST::AllReduce {
                    operand: p_operand,
                    op: p_op,
                },
                JitAST::AllReduce {
                    operand: a_operand,
                    op: a_op,
                },
            ) => {
                if let Some(required_op) = p_op {
                    if required_op != a_op {
                        return false;
                    }
                }
                p_operand.matches(a_operand, ctx)
            }
            _ => false,
        }
    }
}

pub struct RewriteRule {
    pub pat: PatJitAST,
    pub transform: fn(
        JitAST,
        HashMap<String, JitAST>,
        &mut Scope,
        &mut dyn FnMut() -> LoweredAST,
        &[&RewriteRule],
    ) -> LoweredAST,
}

impl JitAST {
    pub fn graph_rewrite(
        ast: Self,
        scope: &mut Scope,
        rules: &[&RewriteRule],
        on_var: &mut dyn FnMut() -> LoweredAST,
    ) -> LoweredAST {
        for rule in rules {
            let mut ctx = HashMap::new();
            if rule.pat.matches(&ast, &mut ctx) {
                return (rule.transform)(ast, ctx, scope, on_var, rules);
            }
        }
        match ast {
            JitAST::Var { .. } => on_var(),
            JitAST::Const(c) => LoweredAST::Const(AstConst {
                dt: c.dt,
                data: c
                    .data
                    .into_iter()
                    .map(|d| match d {
                        ASTOrConst::AST(a) => {
                            ASTOrConst::AST(Self::graph_rewrite(a, scope, rules, on_var))
                        }
                        ASTOrConst::Const(c) => ASTOrConst::Const(c),
                    })
                    .collect(),
            }),
            JitAST::BinOp { lhs, rhs, op } => {
                let l = Self::graph_rewrite(*lhs, scope, rules, on_var);
                let r = Self::graph_rewrite(*rhs, scope, rules, on_var);
                match op {
                    JitBinOp::Basic(basic) => LoweredAST::BinaryOp {
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                        op: basic,
                    },
                    _ => panic!("non-basic JitBinOp must be handled by a rewrite rule"),
                }
            }
            JitAST::UnaryOp { operand, op } => {
                let o = Self::graph_rewrite(*operand, scope, rules, on_var);
                match op {
                    JitUnaryOp::Basic(basic) => LoweredAST::UnaryOp {
                        operand: Box::new(o),
                        op: basic,
                    },
                    _ => panic!("non-basic JitUnaryOp must be handled by a rewrite rule"),
                }
            }
            _ => panic!(
                "node must be handled by a rewrite rule: {:?}",
                std::mem::discriminant(&ast)
            ),
        }
    }
}
