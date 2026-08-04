use std::collections::HashMap;

use crate::asts::jit::JitAST;

pub fn fuse_cast_cast<I>(
    matched: JitAST<I>,
    _captured: HashMap<String, JitAST<I>>,
) -> JitAST<I> {
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
    JitAST::Cast {
        operand: inner_operand,
        dt: outer_dt,
    }
}
