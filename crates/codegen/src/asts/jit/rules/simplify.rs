use crate::{
    asts::{
        ASTOrConst,
        jit::{JitAST, JitBinOp},
        lowered::BinOp,
    },
    dt::{BasicTy, DType},
};

pub fn simplify_binop(lhs: JitAST, rhs: JitAST, op: JitBinOp) -> Option<JitAST> {
    let basic = match op {
        JitBinOp::Basic(b) => b,
        _ => return None,
    };
    let lv = match &lhs {
        JitAST::Const(c) if c.dt == DType::Basic(BasicTy::F32) => {
            c.data.first().and_then(|d| match d {
                ASTOrConst::Const(bytes) => {
                    Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                }
                _ => None,
            })
        }
        _ => None,
    };
    let rv = match &rhs {
        JitAST::Const(c) if c.dt == DType::Basic(BasicTy::F32) => {
            c.data.first().and_then(|d| match d {
                ASTOrConst::Const(bytes) => {
                    Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                }
                _ => None,
            })
        }
        _ => None,
    };
    match (lv, rv, basic) {
        (Some(a), Some(b), BinOp::Add) => Some((a + b).into()),
        (Some(a), Some(b), BinOp::Mul) => Some((a * b).into()),
        (Some(a), Some(b), BinOp::Sub) => Some((a - b).into()),
        (Some(a), Some(b), BinOp::Div) => Some((a / b).into()),
        (_, Some(0.0), BinOp::Mul) | (Some(0.0), _, BinOp::Mul) => Some(0.0f32.into()),
        (_, Some(0.0), BinOp::Add) => Some(lhs),
        (Some(0.0), _, BinOp::Add) => Some(rhs),
        (_, Some(1.0), BinOp::Mul) => Some(lhs),
        (Some(1.0), _, BinOp::Mul) => Some(rhs),
        (_, Some(0.0), BinOp::Sub) => Some(lhs),
        (_, Some(1.0), BinOp::Div) => Some(lhs),
        _ => None,
    }
}

pub fn simplify_node(ast: JitAST) -> JitAST {
    match ast {
        JitAST::BinOp { lhs, rhs, op } => {
            let l = simplify_node(*lhs);
            let r = simplify_node(*rhs);
            simplify_binop(l.clone(), r.clone(), op).unwrap_or(JitAST::BinOp {
                lhs: Box::new(l),
                rhs: Box::new(r),
                op,
            })
        }
        _ => ast,
    }
}
