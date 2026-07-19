use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

use crate::asts::lowered::LoweredAST;

use super::{BinOp, UnaryOp};

impl Add for LoweredAST {
    type Output = LoweredAST;
    fn add(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Add,
        }
    }
}

impl Sub for LoweredAST {
    type Output = LoweredAST;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Sub,
        }
    }
}

impl Mul for LoweredAST {
    type Output = LoweredAST;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Mul,
        }
    }
}

impl Div for LoweredAST {
    type Output = LoweredAST;
    fn div(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Div,
        }
    }
}

impl Shr for LoweredAST {
    type Output = LoweredAST;
    fn shr(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Shr,
        }
    }
}

impl Shl for LoweredAST {
    type Output = LoweredAST;
    fn shl(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Shl,
        }
    }
}

impl BitAnd for LoweredAST {
    type Output = LoweredAST;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::BitwiseAnd,
        }
    }
}

impl BitOr for LoweredAST {
    type Output = LoweredAST;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::BitwiseOr,
        }
    }
}

impl BitXor for LoweredAST {
    type Output = LoweredAST;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::BitwiseXor,
        }
    }
}

impl Rem for LoweredAST {
    type Output = LoweredAST;
    fn rem(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Rem,
        }
    }
}

impl Not for LoweredAST {
    type Output = LoweredAST;
    fn not(self) -> Self::Output {
        Self::UnaryOp {
            operand: Box::new(self),
            op: UnaryOp::LogicalNot,
        }
    }
}

impl Neg for LoweredAST {
    type Output = LoweredAST;
    fn neg(self) -> Self::Output {
        Self::UnaryOp {
            operand: Box::new(self),
            op: UnaryOp::Neg,
        }
    }
}
