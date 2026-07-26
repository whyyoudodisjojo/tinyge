use std::collections::HashMap;

use crate::asts::{
    ASTOrConst,
    jit::{JitAST, JitBinOp},
    lowered::BinOp,
};
use crate::dt::{BasicTy, DType, IntegerTy};

enum Extracted {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl Extracted {
    fn from_jit(node: &JitAST) -> Option<Self> {
        match node {
            JitAST::Const(c) => {
                let bytes = match c.data.first()? {
                    ASTOrConst::Const(b) => b.as_slice(),
                    _ => return None,
                };
                match &c.dt {
                    DType::Basic(BasicTy::F32) => {
                        Some(Self::F32(f32::from_le_bytes(bytes[..4].try_into().ok()?)))
                    }
                    DType::Basic(BasicTy::Integer(IntegerTy::I32)) => {
                        Some(Self::I32(i32::from_le_bytes(bytes[..4].try_into().ok()?)))
                    }
                    DType::Basic(BasicTy::Integer(IntegerTy::U32)) => {
                        Some(Self::U32(u32::from_le_bytes(bytes[..4].try_into().ok()?)))
                    }
                    DType::Basic(BasicTy::Bool) => {
                        Some(Self::Bool(bytes.first().copied().unwrap_or(0) != 0))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::F32(x) => *x == 0.0,
            Self::I32(x) => *x == 0,
            Self::U32(x) => *x == 0,
            Self::Bool(x) => !x,
        }
    }

    fn is_one(&self) -> bool {
        match self {
            Self::F32(x) => *x == 1.0,
            Self::I32(x) => *x == 1,
            Self::U32(x) => *x == 1,
            Self::Bool(x) => *x,
        }
    }
}

fn const_zero(dt: &DType) -> JitAST {
    match dt {
        DType::Basic(BasicTy::F32) => JitAST::from(0.0f32),
        DType::Basic(BasicTy::Integer(IntegerTy::I32)) => JitAST::from(0i32),
        DType::Basic(BasicTy::Integer(IntegerTy::U32)) => JitAST::from(0u32),
        DType::Basic(BasicTy::Bool) => JitAST::from(false),
        _ => unreachable!(),
    }
}

pub fn simplify_binop_pre(matched: JitAST, captured: HashMap<String, JitAST>) -> JitAST {
    let lhs = captured.get("lhs").unwrap();
    let rhs = captured.get("rhs").unwrap();

    let JitAST::BinOp {
        op: JitBinOp::Basic(basic),
        ..
    } = &matched
    else {
        return matched;
    };

    let lv = Extracted::from_jit(lhs);
    let rv = Extracted::from_jit(rhs);

    match (&lv, &rv, basic) {
        (Some(Extracted::F32(a)), Some(Extracted::F32(b)), BinOp::Add) => JitAST::from(a + b),
        (Some(Extracted::F32(a)), Some(Extracted::F32(b)), BinOp::Mul) => JitAST::from(a * b),
        (Some(Extracted::F32(a)), Some(Extracted::F32(b)), BinOp::Sub) => JitAST::from(a - b),
        (Some(Extracted::F32(a)), Some(Extracted::F32(b)), BinOp::Div) => JitAST::from(a / b),
        (Some(Extracted::I32(a)), Some(Extracted::I32(b)), BinOp::Add) => JitAST::from(a + b),
        (Some(Extracted::I32(a)), Some(Extracted::I32(b)), BinOp::Mul) => JitAST::from(a * b),
        (Some(Extracted::I32(a)), Some(Extracted::I32(b)), BinOp::Sub) => JitAST::from(a - b),
        (Some(Extracted::I32(a)), Some(Extracted::I32(b)), BinOp::Div) => JitAST::from(a / b),
        (Some(Extracted::U32(a)), Some(Extracted::U32(b)), BinOp::Add) => JitAST::from(a + b),
        (Some(Extracted::U32(a)), Some(Extracted::U32(b)), BinOp::Mul) => JitAST::from(a * b),
        (Some(Extracted::U32(a)), Some(Extracted::U32(b)), BinOp::Sub) => JitAST::from(a - b),
        (Some(Extracted::U32(a)), Some(Extracted::U32(b)), BinOp::Div) => JitAST::from(a / b),
        (Some(lv), _, BinOp::Mul) if lv.is_zero() => const_zero(&lhs.dt()),
        (_, Some(rv), BinOp::Mul) if rv.is_zero() => const_zero(&rhs.dt()),
        (_, Some(rv), BinOp::Add) if rv.is_zero() => lhs.clone(),
        (Some(lv), _, BinOp::Add) if lv.is_zero() => rhs.clone(),
        (_, Some(rv), BinOp::Sub) if rv.is_zero() => lhs.clone(),
        (_, Some(rv), BinOp::Mul) if rv.is_one() => lhs.clone(),
        (Some(lv), _, BinOp::Mul) if lv.is_one() => rhs.clone(),
        (_, Some(rv), BinOp::Div) if rv.is_one() => lhs.clone(),
        _ => matched,
    }
}
