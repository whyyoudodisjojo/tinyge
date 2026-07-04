use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Shl, Shr, Sub};

use crate::asts::comptime::ComptimeAST;

use super::BinOp;

impl Add for ComptimeAST {
    type Output = ComptimeAST;
    fn add(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Add,
        }
    }
}

impl Sub for ComptimeAST {
    type Output = ComptimeAST;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Sub,
        }
    }
}

impl Mul for ComptimeAST {
    type Output = ComptimeAST;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Mul,
        }
    }
}

impl Div for ComptimeAST {
    type Output = ComptimeAST;
    fn div(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Div,
        }
    }
}

impl Shr for ComptimeAST {
    type Output = ComptimeAST;
    fn shr(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Shr,
        }
    }
}

impl Shl for ComptimeAST {
    type Output = ComptimeAST;
    fn shl(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Shl,
        }
    }
}

impl BitAnd for ComptimeAST {
    type Output = ComptimeAST;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::BitwiseAnd,
        }
    }
}

impl BitOr for ComptimeAST {
    type Output = ComptimeAST;
    fn bitor(self, rhs: Self) -> Self::Output {
        let neg_lhs = Self::UnaryOp {
            operand: Box::new(self),
            op: super::UnaryOp::BitwiseNot,
        };
        let neg_rhs = Self::UnaryOp {
            operand: Box::new(rhs),
            op: super::UnaryOp::BitwiseNot,
        };

        Self::BinaryOp {
            lhs: Box::new(neg_lhs),
            rhs: Box::new(neg_rhs),
            op: BinOp::BitwiseAnd,
        }
    }
}

impl BitXor for ComptimeAST {
    type Output = ComptimeAST;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let neg_lhs = Self::UnaryOp {
            operand: Box::new(self.clone()),
            op: super::UnaryOp::BitwiseNot,
        };
        let neg_rhs = Self::UnaryOp {
            operand: Box::new(rhs.clone()),
            op: super::UnaryOp::BitwiseNot,
        };

        let a = Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(neg_rhs),
            op: BinOp::BitwiseAnd,
        };
        let b = Self::BinaryOp {
            lhs: Box::new(neg_lhs),
            rhs: Box::new(rhs),
            op: BinOp::BitwiseAnd,
        };

        a | b
    }
}
