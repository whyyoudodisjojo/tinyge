use std::collections::HashMap;

use crate::asts::{
    jit::JitAST,
    lowered::{LoweredAST, scope::Scope},
};

use super::super::pattern::RewriteRule;
use super::basic;

pub fn fuse_reduce(
    matched: JitAST,
    captured: HashMap<String, JitAST>,
    scope: &mut Scope,
    var_producer: &mut dyn FnMut() -> LoweredAST,
    rules: &[&RewriteRule],
) -> LoweredAST {
    let op = match matched {
        JitAST::AllReduce { op, .. } | JitAST::Reduce { op, .. } => op,
        _ => unreachable!(),
    };
    let x = captured.get("x").unwrap().clone();
    basic::lower_reduce(Box::new(x), op, scope, var_producer, rules, None)
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
