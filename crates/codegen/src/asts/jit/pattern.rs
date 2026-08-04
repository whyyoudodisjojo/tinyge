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
        axis: Option<usize>,
        op: Option<ReduceOp>,
    },
    AllReduce {
        operand: Box<Self>,
        op: Option<ReduceOp>,
    },
}

impl PatJitAST {
    pub fn matches<I: Clone>(&self, ast: &JitAST<I>, ctx: &mut HashMap<String, JitAST<I>>) -> bool {
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
                if let Some(required_axis) = p_axis {
                    if required_axis != a_axis {
                        return false;
                    }
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

pub enum RewriteRule<I> {
    Pre {
        pat: PatJitAST,
        transform: fn(JitAST<I>, HashMap<String, JitAST<I>>) -> JitAST<I>,
    },
    Post {
        pat: PatJitAST,
        transform: fn(
            JitAST<I>,
            HashMap<String, JitAST<I>>,
            &mut Scope,
            &mut dyn FnMut(usize) -> LoweredAST,
            &[&RewriteRule<I>],
        ) -> LoweredAST,
    },
}

impl<I> RewriteRule<I> {
    pub fn pre(pat: PatJitAST, transform: fn(JitAST<I>, HashMap<String, JitAST<I>>) -> JitAST<I>) -> Self {
        Self::Pre { pat, transform }
    }

    pub fn post(
        pat: PatJitAST,
        transform: fn(
            JitAST<I>,
            HashMap<String, JitAST<I>>,
            &mut Scope,
            &mut dyn FnMut(usize) -> LoweredAST,
            &[&RewriteRule<I>],
        ) -> LoweredAST,
    ) -> Self {
        Self::Post { pat, transform }
    }
}

impl<I: Clone> JitAST<I> {
    fn pre_rewrite(ast: Self, rules: &[&RewriteRule<I>]) -> Self {
        let ast = match ast {
            JitAST::BinOp { lhs, rhs, op } => JitAST::BinOp {
                lhs: Box::new(Self::pre_rewrite(*lhs, rules)),
                rhs: Box::new(Self::pre_rewrite(*rhs, rules)),
                op,
            },
            JitAST::UnaryOp { operand, op } => JitAST::UnaryOp {
                operand: Box::new(Self::pre_rewrite(*operand, rules)),
                op,
            },
            JitAST::Cast { operand, dt } => JitAST::Cast {
                operand: Box::new(Self::pre_rewrite(*operand, rules)),
                dt,
            },
            JitAST::Ternary { a, b, c, op } => JitAST::Ternary {
                a: Box::new(Self::pre_rewrite(*a, rules)),
                b: Box::new(Self::pre_rewrite(*b, rules)),
                c: Box::new(Self::pre_rewrite(*c, rules)),
                op,
            },
            JitAST::Movement { operand, op } => JitAST::Movement {
                operand: Box::new(Self::pre_rewrite(*operand, rules)),
                op,
            },
            JitAST::Reduce { operand, axis, op } => JitAST::Reduce {
                operand: Box::new(Self::pre_rewrite(*operand, rules)),
                axis,
                op,
            },
            JitAST::AllReduce { operand, op } => JitAST::AllReduce {
                operand: Box::new(Self::pre_rewrite(*operand, rules)),
                op,
            },
            JitAST::Const(c) => JitAST::Const(AstConst {
                dt: c.dt,
                data: c
                    .data
                    .into_iter()
                    .map(|d| match d {
                        ASTOrConst::AST(a) => ASTOrConst::AST(Self::pre_rewrite(a, rules)),
                        ASTOrConst::Const(c) => ASTOrConst::Const(c),
                    })
                    .collect(),
            }),
            other => other,
        };

        for rule in rules {
            if let RewriteRule::Pre { pat, transform } = rule {
                let mut ctx = HashMap::new();
                if pat.matches(&ast, &mut ctx) {
                    return transform(ast, ctx);
                }
            }
        }

        ast
    }

    pub fn graph_rewrite_post(
        ast: Self,
        scope: &mut Scope,
        rules: &[&RewriteRule<I>],
        on_var: &mut dyn FnMut(usize) -> LoweredAST,
    ) -> LoweredAST {
        for rule in rules {
            if let RewriteRule::Post { pat, transform } = rule {
                let mut ctx = HashMap::new();
                if pat.matches(&ast, &mut ctx) {
                    return transform(ast, ctx, scope, on_var, rules);
                }
            }
        }
        match ast {
            JitAST::Var { id, .. } => return on_var(id),
            JitAST::Lowered { expr, .. } => return expr,
            JitAST::Const(c) => {
                return LoweredAST::Const(AstConst {
                    dt: c.dt,
                    data: c
                        .data
                        .into_iter()
                        .map(|d| match d {
                            ASTOrConst::AST(a) => {
                                ASTOrConst::AST(Self::graph_rewrite_post(a, scope, rules, on_var))
                            }
                            ASTOrConst::Const(c) => ASTOrConst::Const(c),
                        })
                        .collect(),
                });
            }
            _ => panic!(
                "node must be handled by a rewrite rule: {:?}",
                std::mem::discriminant(&ast)
            ),
        }
    }

    pub fn graph_rewrite(
        ast: Self,
        scope: &mut Scope,
        rules: &[&RewriteRule<I>],
        on_var: &mut dyn FnMut(usize) -> LoweredAST,
    ) -> LoweredAST {
        let ast = Self::pre_rewrite(ast, rules);
        Self::graph_rewrite_post(ast, scope, rules, on_var)
    }
}
