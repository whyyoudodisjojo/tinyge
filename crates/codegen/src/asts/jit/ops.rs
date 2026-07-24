use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};

use crate::asts::IntoWgslStruct;
use crate::asts::jit::{JitAST, JitBinOp, JitUnaryOp, MovOp, ReduceOp, ReduceTarget, TernaryOp};

macro_rules! impl_binop_trait {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<T> $trait for JitAST<T> {
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

impl<T> Not for JitAST<T> {
    type Output = Self;
    fn not(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Basic(crate::asts::lowered::UnaryOp::BitwiseNot),
        }
    }
}

impl<T> Neg for JitAST<T> {
    type Output = Self;
    fn neg(self) -> Self {
        JitAST::UnaryOp {
            operand: Box::new(self),
            op: JitUnaryOp::Basic(crate::asts::lowered::UnaryOp::Neg),
        }
    }
}

impl<T> JitAST<T> {
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

}

impl<T: ReduceTarget> JitAST<T> {
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
}

macro_rules! impl_flatten {
    ($trait:ident, [$($N:ident),+], $input:ty, $output:ty, [$($shape:expr),+]) => {
        pub trait $trait: Sized {
            type Output;
            fn flatten(self) -> JitAST<Self::Output>;
        }

        impl<E: IntoWgslStruct, $(const $N: usize),+> $trait for JitAST<$input>
        where [(); N0 * N1]:
        {
            type Output = $output;
            fn flatten(self) -> JitAST<$output> {
                use std::mem::transmute;
                let operand = unsafe { transmute::<Box<JitAST<$input>>, Box<JitAST<$output>>>(Box::new(self)) };
                JitAST::Movement { operand, op: MovOp::Reshape(vec![$($shape),+]) }
            }
        }
    };
}

macro_rules! impl_pad_shrink {
    ($pad_name:ident, $shrink_name:ident, [$($N:ident),+], [$($LO:ident, $HI:ident),+], $input:ty, $pad_output:ty, $shrink_output:ty) => {
        pub fn $pad_name<$(const $LO: usize, const $HI: usize),+>(self) -> JitAST<$pad_output> {
            use std::mem::transmute;
            let operand = unsafe { transmute::<Box<JitAST<$input>>, Box<JitAST<$pad_output>>>(Box::new(self)) };
            JitAST::Movement { operand, op: MovOp::Pad(vec![$(($LO, $HI)),+]) }
        }
        pub fn $shrink_name<$(const $LO: usize, const $HI: usize),+>(self) -> JitAST<$shrink_output> {
            use std::mem::transmute;
            let operand = unsafe { transmute::<Box<JitAST<$input>>, Box<JitAST<$shrink_output>>>(Box::new(self)) };
            JitAST::Movement { operand, op: MovOp::Shrink(vec![$(($LO, $HI)),+]) }
        }
    };
}

macro_rules! impl_reshape {
    ($name:ident, [$($M:ident),+], $output:ty, [$($shape:expr),+]) => {
        pub fn $name<$(const $M: usize),+>(self) -> JitAST<$output> {
            use std::mem::transmute;
            let operand = unsafe { transmute::<Box<JitAST<[E; N0]>>, Box<JitAST<$output>>>(Box::new(self)) };
            JitAST::Movement { operand, op: MovOp::Reshape(vec![$($shape),+]) }
        }
    };
}

macro_rules! impl_dim {
    (1) => {
        impl<E: IntoWgslStruct, const N0: usize> JitAST<[E; N0]> {
            impl_pad_shrink!(pad_1d, shrink_1d, [N0], [LO, HI], [E; N0], [E; N0 + LO + HI], [E; N0 - LO - HI]);
            impl_reshape!(reshape_2d, [M, P], [[E; P]; M], [M, P]);
            impl_reshape!(reshape_3d, [M, P, Q], [[[E; Q]; P]; M], [M, P, Q]);
            impl_reshape!(reshape_4d, [M, P, Q, R], [[[[E; R]; Q]; P]; M], [M, P, Q, R]);
        }
    };
    (2) => {
        impl_flatten!(Flatten2D, [N0, N1], [[E; N1]; N0], [E; N0 * N1], [N0 * N1]);
        impl<E: IntoWgslStruct, const N0: usize, const N1: usize> JitAST<[[E; N1]; N0]> {
            impl_pad_shrink!(pad_2d, shrink_2d, [N0, N1], [LO0, HI0, LO1, HI1],
                [[E; N1]; N0],
                [[E; N1 + LO1 + HI1]; N0 + LO0 + HI0],
                [[E; N1 - LO1 - HI1]; N0 - LO0 - HI0]);
        }
    };
    (3) => {
        impl_flatten!(Flatten3D, [N0, N1, N2], [[[E; N2]; N1]; N0], [[E; N2]; N0 * N1], [N0 * N1, N2]);
        impl<E: IntoWgslStruct, const N0: usize, const N1: usize, const N2: usize> JitAST<[[[E; N2]; N1]; N0]> {
            impl_pad_shrink!(pad_3d, shrink_3d, [N0, N1, N2], [LO0, HI0, LO1, HI1, LO2, HI2],
                [[[E; N2]; N1]; N0],
                [[[E; N2 + LO2 + HI2]; N1 + LO1 + HI1]; N0 + LO0 + HI0],
                [[[E; N2 - LO2 - HI2]; N1 - LO1 - HI1]; N0 - LO0 - HI0]);
        }
    };
    (4) => {
        impl_flatten!(Flatten4D, [N0, N1, N2, N3], [[[[E; N3]; N2]; N1]; N0], [[[E; N3]; N2]; N0 * N1], [N0 * N1, N2, N3]);
        impl<E: IntoWgslStruct, const N0: usize, const N1: usize, const N2: usize, const N3: usize> JitAST<[[[[E; N3]; N2]; N1]; N0]> {
            impl_pad_shrink!(pad_4d, shrink_4d, [N0, N1, N2, N3], [LO0, HI0, LO1, HI1, LO2, HI2, LO3, HI3],
                [[[[E; N3]; N2]; N1]; N0],
                [[[[E; N3 + LO3 + HI3]; N2 + LO2 + HI2]; N1 + LO1 + HI1]; N0 + LO0 + HI0],
                [[[[E; N3 - LO3 - HI3]; N2 - LO2 - HI2]; N1 - LO1 - HI1]; N0 - LO0 - HI0]);
        }
    };
}

impl_dim!(1);
impl_dim!(2);
impl_dim!(3);
impl_dim!(4);

impl<T> JitAST<T> {
    pub fn cast<I>(self) -> JitAST<I>
        where I: IntoWgslStruct
    {
        let operand: Box<JitAST<I>> = unsafe { std::mem::transmute(Box::new(self)) };
        JitAST::Cast {
            operand,
            dt: I::dt(),
        }
    }
}
