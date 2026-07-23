use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

use crate::asts::jit::{JitAST, JitBinOp, JitUnaryOp, MovOp, ReduceOp, TernaryOp};
use crate::dt::DType;

macro_rules! impl_binop_trait {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for JitAST {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                JitAST::BinOp {
                    lhs: Box::new(self),
                    rhs: Box::new(rhs),
                    op: $op,
                }
            }
        }
    };
}

impl_binop_trait!(Add, add, JitBinOp::Basic(crate::asts::lowered::BinOp::Add));
impl_binop_trait!(Sub, sub, JitBinOp::Basic(crate::asts::lowered::BinOp::Sub));
impl_binop_trait!(Mul, mul, JitBinOp::Basic(crate::asts::lowered::BinOp::Mul));
impl_binop_trait!(Div, div, JitBinOp::Basic(crate::asts::lowered::BinOp::Div));
impl_binop_trait!(Rem, rem, JitBinOp::Basic(crate::asts::lowered::BinOp::Rem));
impl_binop_trait!(
    BitAnd,
    bitand,
    JitBinOp::Basic(crate::asts::lowered::BinOp::BitwiseAnd)
);
impl_binop_trait!(
    BitOr,
    bitor,
    JitBinOp::Basic(crate::asts::lowered::BinOp::BitwiseOr)
);
impl_binop_trait!(
    BitXor,
    bitxor,
    JitBinOp::Basic(crate::asts::lowered::BinOp::BitwiseXor)
);
impl_binop_trait!(Shl, shl, JitBinOp::Basic(crate::asts::lowered::BinOp::Shl));
impl_binop_trait!(Shr, shr, JitBinOp::Basic(crate::asts::lowered::BinOp::Shr));

impl Not for JitAST {
    type Output = Self;
    fn not(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Basic(crate::asts::lowered::UnaryOp::BitwiseNot),
        }
    }
}

impl Neg for JitAST {
    type Output = Self;
    fn neg(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Basic(crate::asts::lowered::UnaryOp::Neg),
        }
    }
}

impl JitAST {
    pub fn eq(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Basic(crate::asts::lowered::BinOp::Eq),
        }
    }

    pub fn ne(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Basic(crate::asts::lowered::BinOp::Ne),
        }
    }

    pub fn gt(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Basic(crate::asts::lowered::BinOp::Gt),
        }
    }

    pub fn lt(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Basic(crate::asts::lowered::BinOp::Lt),
        }
    }

    pub fn ge(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Basic(crate::asts::lowered::BinOp::Ge),
        }
    }

    pub fn le(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Basic(crate::asts::lowered::BinOp::Le),
        }
    }

    pub fn logical_and(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Basic(crate::asts::lowered::BinOp::LogicalAnd),
        }
    }

    pub fn cdiv(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Cdiv,
        }
    }

    pub fn max(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Max,
        }
    }

    pub fn cmod(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Cmod,
        }
    }

    pub fn fdiv(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Fdiv,
        }
    }

    pub fn pow(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Pow,
        }
    }

    pub fn floordiv(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Floordiv,
        }
    }

    pub fn floormod(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Floormod,
        }
    }

    pub fn threefry(self, rhs: Self) -> Self {
        JitAST::BinOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: JitBinOp::Threefry,
        }
    }

    pub fn logical_not(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Basic(crate::asts::lowered::UnaryOp::LogicalNot),
        }
    }

    pub fn exp2(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Exp2,
        }
    }

    pub fn log2(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Log2,
        }
    }

    pub fn sin(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Sin,
        }
    }

    pub fn sqrt(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Sqrt,
        }
    }

    pub fn reciprocal(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Reciprocal,
        }
    }

    pub fn trunc(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Trunc,
        }
    }

    pub fn bitcast(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Bitcast,
        }
    }

    pub fn where_(cond: Self, true_: Self, false_: Self) -> Self {
        JitAST::Ternary {
            a: Box::new(cond),
            b: Box::new(true_),
            c: Box::new(false_),
            op: TernaryOp::Where,
        }
    }

    pub fn mulacc(a: Self, b: Self, c: Self) -> Self {
        JitAST::Ternary {
            a: Box::new(a),
            b: Box::new(b),
            c: Box::new(c),
            op: TernaryOp::Mulacc,
        }
    }

    pub fn sum(self, axis: usize) -> Self {
        JitAST::Reduce {
            operand: Box::new(self),
            axis,
            op: ReduceOp::Sum,
        }
    }

    pub fn prod(self, axis: usize) -> Self {
        JitAST::Reduce {
            operand: Box::new(self),
            axis,
            op: ReduceOp::Prod,
        }
    }

    pub fn reduce_max(self, axis: usize) -> Self {
        JitAST::Reduce {
            operand: Box::new(self),
            axis,
            op: ReduceOp::Max,
        }
    }

    pub fn sum_all(self) -> Self {
        JitAST::AllReduce {
            operand: Box::new(self),
            op: ReduceOp::Sum,
        }
    }

    pub fn prod_all(self) -> Self {
        JitAST::AllReduce {
            operand: Box::new(self),
            op: ReduceOp::Prod,
        }
    }

    pub fn max_all(self) -> Self {
        JitAST::AllReduce {
            operand: Box::new(self),
            op: ReduceOp::Max,
        }
    }

    pub fn reshape(self, shape: Vec<usize>) -> Self {
        JitAST::Movement {
            operand: Box::new(self),
            op: MovOp::Reshape(shape),
        }
    }

    pub fn expand(self, dims: Vec<usize>) -> Self {
        JitAST::Movement {
            operand: Box::new(self),
            op: MovOp::Expand(dims),
        }
    }

    pub fn permute(self, axes: Vec<usize>) -> Self {
        JitAST::Movement {
            operand: Box::new(self),
            op: MovOp::Permute(axes),
        }
    }

    pub fn pad(self, padding: Vec<(usize, usize)>) -> Self {
        JitAST::Movement {
            operand: Box::new(self),
            op: MovOp::Pad(padding),
        }
    }

    pub fn shrink(self, amount: Vec<(usize, usize)>) -> Self {
        JitAST::Movement {
            operand: Box::new(self),
            op: MovOp::Shrink(amount),
        }
    }

    pub fn flip(self, axis: usize) -> Self {
        JitAST::Movement {
            operand: Box::new(self),
            op: MovOp::Flip(axis),
        }
    }

    pub fn cast(self, dt: DType) -> Self {
        JitAST::Cast {
            operand: Box::new(self),
            dt,
        }
    }
}
