pub mod post_traversal;
pub mod pre_traversal;

use crate::asts::jit::{JitBinOp, JitUnaryOp, TernaryOp};
use PatJitAST::*;
use post_traversal::{basic, fuse, movement};
use pre_traversal::{fuse as pre_fuse, simplify};

use super::pattern::{PatJitAST, RewriteRule};

pub fn builtin_rules<I: Clone>() -> Vec<RewriteRule<I>> {
    vec![
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Cdiv),
            },
            basic::cdiv,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Max),
            },
            basic::binop_max,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Cmod),
            },
            basic::cmod,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Fdiv),
            },
            basic::fdiv,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Pow),
            },
            basic::pow,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Floordiv),
            },
            basic::floordiv,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Floormod),
            },
            basic::floormod,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Threefry),
            },
            basic::threefry,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Exp2),
            },
            basic::exp2,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Log2),
            },
            basic::log2,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Sin),
            },
            basic::sin,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Sqrt),
            },
            basic::sqrt,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Reciprocal),
            },
            basic::reciprocal,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Trunc),
            },
            basic::trunc,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Bitcast),
            },
            basic::bitcast,
        ),
        RewriteRule::pre(
            Cast {
                operand: Box::new(Cast {
                    operand: Box::new(Var("x".into())),
                    dt: None,
                }),
                dt: None,
            },
            pre_fuse::fuse_cast_cast,
        ),
        RewriteRule::post(
            Cast {
                operand: Box::new(Var("x".into())),
                dt: None,
            },
            basic::cast,
        ),
        RewriteRule::post(
            Ternary {
                a: Box::new(Var("a".into())),
                b: Box::new(Var("b".into())),
                c: Box::new(Var("c".into())),
                op: Some(TernaryOp::Where),
            },
            basic::ternary_where,
        ),
        RewriteRule::post(
            Ternary {
                a: Box::new(Var("a".into())),
                b: Box::new(Var("b".into())),
                c: Box::new(Var("c".into())),
                op: Some(TernaryOp::Mulacc),
            },
            basic::ternary_mulacc,
        ),
        RewriteRule::post(
            Movement {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            movement::movement,
        ),
        RewriteRule::pre(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: None,
            },
            simplify::simplify_binop_pre,
        ),
        RewriteRule::post(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: None,
            },
            basic::binop_basic,
        ),
        RewriteRule::post(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            basic::unaryop_basic,
        ),
        RewriteRule::post(
            AllReduce {
                operand: Box::new(Reduce {
                    operand: Box::new(Var("x".into())),
                    axis: None,
                    op: None,
                }),
                op: None,
            },
            fuse::fuse_reduce,
        ),
        RewriteRule::post(
            Reduce {
                operand: Box::new(AllReduce {
                    operand: Box::new(Var("x".into())),
                    op: None,
                }),
                axis: None,
                op: None,
            },
            fuse::fuse_reduce,
        ),
        RewriteRule::post(
            AllReduce {
                operand: Box::new(AllReduce {
                    operand: Box::new(Var("x".into())),
                    op: None,
                }),
                op: None,
            },
            fuse::fuse_reduce,
        ),
        RewriteRule::post(
            Reduce {
                operand: Box::new(Var("x".into())),
                axis: None,
                op: None,
            },
            basic::reduce,
        ),
        RewriteRule::post(
            AllReduce {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            basic::all_reduce,
        ),
    ]
}
