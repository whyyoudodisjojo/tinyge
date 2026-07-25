pub mod basic;
pub mod fuse;
pub mod movement;
pub mod simplify;

use crate::asts::jit::{JitBinOp, JitUnaryOp, TernaryOp};

use super::pattern::{PatJitAST, RewriteRule};

pub fn builtin_rules() -> Vec<RewriteRule> {
    use PatJitAST::*;
    vec![
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Cdiv),
            },
            basic::cdiv,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Max),
            },
            basic::binop_max,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Cmod),
            },
            basic::cmod,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Fdiv),
            },
            basic::fdiv,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Pow),
            },
            basic::pow,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Floordiv),
            },
            basic::floordiv,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Floormod),
            },
            basic::floormod,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: Some(JitBinOp::Threefry),
            },
            basic::threefry,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Exp2),
            },
            basic::exp2,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Log2),
            },
            basic::log2,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Sin),
            },
            basic::sin,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Sqrt),
            },
            basic::sqrt,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Reciprocal),
            },
            basic::reciprocal,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Trunc),
            },
            basic::trunc,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: Some(JitUnaryOp::Bitcast),
            },
            basic::bitcast,
        ),
        RewriteRule::new(
            Cast {
                operand: Box::new(Cast {
                    operand: Box::new(Var("x".into())),
                    dt: None,
                }),
                dt: None,
            },
            fuse::fuse_cast_cast,
        ),
        RewriteRule::new(
            Cast {
                operand: Box::new(Var("x".into())),
                dt: None,
            },
            basic::cast,
        ),
        RewriteRule::new(
            Ternary {
                a: Box::new(Var("a".into())),
                b: Box::new(Var("b".into())),
                c: Box::new(Var("c".into())),
                op: Some(TernaryOp::Where),
            },
            basic::ternary_where,
        ),
        RewriteRule::new(
            Ternary {
                a: Box::new(Var("a".into())),
                b: Box::new(Var("b".into())),
                c: Box::new(Var("c".into())),
                op: Some(TernaryOp::Mulacc),
            },
            basic::ternary_mulacc,
        ),
        RewriteRule::new(
            Movement {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            movement::movement,
        ),
        RewriteRule::new(
            BinOp {
                lhs: Box::new(Var("lhs".into())),
                rhs: Box::new(Var("rhs".into())),
                op: None,
            },
            basic::binop_basic,
        ),
        RewriteRule::new(
            UnaryOp {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            basic::unaryop_basic,
        ),
        RewriteRule::new(
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
        RewriteRule::new(
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
        RewriteRule::new(
            AllReduce {
                operand: Box::new(AllReduce {
                    operand: Box::new(Var("x".into())),
                    op: None,
                }),
                op: None,
            },
            fuse::fuse_reduce,
        ),
        RewriteRule::new(
            Reduce {
                operand: Box::new(Var("x".into())),
                axis: None,
                op: None,
            },
            basic::reduce,
        ),
        RewriteRule::new(
            AllReduce {
                operand: Box::new(Var("x".into())),
                op: None,
            },
            basic::all_reduce,
        ),
    ]
}
