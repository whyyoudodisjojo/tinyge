pub mod ops;
pub mod pattern;
pub mod rules;
pub mod runner;

use wgpu::Buffer;

use crate::asts::AstConst;
use crate::asts::lowered::{ASTOrConst, BinOp, LoweredAST, UnaryOp, scope::Scope};
use crate::dt::{BasicTy, DType, IntegerTy, VecTy};

use self::pattern::RewriteRule;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MovOp {
    Reshape(Vec<usize>),
    Expand(Vec<usize>),
    Permute(Vec<usize>),
    Pad(Vec<(usize, usize)>),
    Shrink(Vec<(usize, usize)>),
    Flip(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitBinOp {
    Basic(BinOp),
    Cdiv,
    Max,
    Cmod,
    Fdiv,
    Pow,
    Floordiv,
    Floormod,
    Threefry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitUnaryOp {
    Basic(UnaryOp),
    Exp2,
    Log2,
    Sin,
    Sqrt,
    Reciprocal,
    Trunc,
    Bitcast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TernaryOp {
    Where,
    Mulacc,
}

#[derive(Clone)]
pub enum JitAST {
    Var {
        buffer: Buffer,
        dtype: DType,
    },

    Const(AstConst<JitAST>),

    BinOp {
        lhs: Box<JitAST>,
        rhs: Box<JitAST>,
        op: JitBinOp,
    },
    UnaryOp {
        operand: Box<JitAST>,
        op: JitUnaryOp,
    },
    Cast {
        operand: Box<JitAST>,
        dt: DType,
    },
    Ternary {
        a: Box<JitAST>,
        b: Box<JitAST>,
        c: Box<JitAST>,
        op: TernaryOp,
    },
    Movement {
        operand: Box<JitAST>,
        op: MovOp,
    },
    Reduce {
        operand: Box<JitAST>,
        axis: usize,
        op: ReduceOp,
    },
    AllReduce {
        operand: Box<JitAST>,
        op: ReduceOp,
    },
}

pub(crate) fn scalar_identity_bytes(dt: &DType, op: ReduceOp) -> Vec<u8> {
    match (dt, op) {
        (DType::Basic(BasicTy::F32), ReduceOp::Sum) => 0u32.to_le_bytes().to_vec(),
        (DType::Basic(BasicTy::F32), ReduceOp::Prod) => 1.0f32.to_le_bytes().to_vec(),
        (DType::Basic(BasicTy::F32), ReduceOp::Max) => f32::MIN.to_le_bytes().to_vec(),
        (DType::Basic(BasicTy::Integer(IntegerTy::U32)), ReduceOp::Sum) => {
            0u32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::U32)), ReduceOp::Prod) => {
            1u32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::U32)), ReduceOp::Max) => {
            0u32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::I32)), ReduceOp::Sum) => {
            0i32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::I32)), ReduceOp::Prod) => {
            1i32.to_le_bytes().to_vec()
        }
        (DType::Basic(BasicTy::Integer(IntegerTy::I32)), ReduceOp::Max) => {
            i32::MIN.to_le_bytes().to_vec()
        }
        _ => panic!("no identity for ({:?}, {:?})", dt, op),
    }
}

impl JitAST {
    pub fn collect_var_buffers<'a>(&'a self, out: &mut Vec<&'a Buffer>) {
        match self {
            JitAST::Var { buffer, .. } => out.push(buffer),
            JitAST::Const(_) => {}
            JitAST::BinOp { lhs, rhs, .. } => {
                lhs.collect_var_buffers(out);
                rhs.collect_var_buffers(out);
            }
            JitAST::UnaryOp { operand, .. }
            | JitAST::Cast { operand, .. }
            | JitAST::Movement { operand, .. }
            | JitAST::Reduce { operand, .. }
            | JitAST::AllReduce { operand, .. } => operand.collect_var_buffers(out),
            JitAST::Ternary { a, b, c, .. } => {
                a.collect_var_buffers(out);
                b.collect_var_buffers(out);
                c.collect_var_buffers(out);
            }
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        let from_dt = |dt: &DType| -> Vec<usize> {
            match dt {
                DType::Vector(VecTy::Vec2(_)) => vec![2],
                DType::Vector(VecTy::Vec3(_)) => vec![3],
                DType::Vector(VecTy::Vec4(_)) => vec![4],
                DType::Vector(VecTy::Array(_, Some(n))) => vec![*n as usize],
                _ => vec![],
            }
        };
        match self {
            JitAST::Var { dtype, .. } | JitAST::Cast { dt: dtype, .. } => from_dt(dtype),
            JitAST::Const(c) => from_dt(&c.dt),
            JitAST::BinOp { lhs, .. }
            | JitAST::UnaryOp { operand: lhs, .. }
            | JitAST::Ternary { a: lhs, .. } => lhs.shape(),
            JitAST::Movement { operand, op } => {
                let s = operand.shape();
                match op {
                    MovOp::Reshape(shape) => shape.clone(),
                    MovOp::Expand(dims) => dims.clone(),
                    MovOp::Permute(axes) => axes.iter().map(|&i| s[i]).collect(),
                    MovOp::Pad(amounts) => s
                        .iter()
                        .zip(amounts.iter())
                        .map(|(sz, (lo, hi))| sz + lo + hi)
                        .collect(),
                    MovOp::Shrink(amounts) => s
                        .iter()
                        .zip(amounts.iter())
                        .map(|(sz, (lo, hi))| sz - lo - hi)
                        .collect(),
                    MovOp::Flip(_) => s,
                }
            }
            JitAST::Reduce { operand, axis, .. } => {
                let mut s = operand.shape();
                if *axis < s.len() {
                    s.remove(*axis);
                }
                s
            }
            JitAST::AllReduce { .. } => vec![],
        }
    }

    pub fn collect_var_info(&self) -> (usize, Option<DType>) {
        match self {
            JitAST::Var { dtype, .. } => (1, Some(dtype.clone())),
            JitAST::Const(_) => (0, None),
            JitAST::BinOp { lhs, rhs, .. } => {
                let (lc, ld) = lhs.collect_var_info();
                let (rc, rd) = rhs.collect_var_info();
                (lc + rc, ld.or(rd))
            }
            JitAST::UnaryOp { operand, .. }
            | JitAST::Cast { operand, .. }
            | JitAST::Movement { operand, .. }
            | JitAST::Reduce { operand, .. }
            | JitAST::AllReduce { operand, .. } => operand.collect_var_info(),
            JitAST::Ternary { a, b, c, .. } => {
                let (ac, ad) = a.collect_var_info();
                let (bc, bd) = b.collect_var_info();
                let (cc, cd) = c.collect_var_info();
                (ac + bc + cc, ad.or(bd).or(cd))
            }
        }
    }

    pub fn dt(&self) -> DType {
        match self {
            JitAST::Var { dtype, .. } => dtype.peel_array(),
            JitAST::Const(c) => c.dt.clone(),
            JitAST::BinOp { lhs, .. } => lhs.dt(),
            JitAST::UnaryOp { operand, .. } => operand.dt(),
            JitAST::Cast { dt, .. } => dt.clone(),
            JitAST::Ternary { a, .. } => a.dt(),
            JitAST::Movement { operand, op } => match op {
                MovOp::Pad(amounts) => {
                    let inner = operand.dt();
                    match inner {
                        DType::Vector(VecTy::Array(inner_ty, Some(n))) => {
                            let total_pad =
                                amounts.iter().map(|(lo, hi)| lo + hi).sum::<usize>() as u32;
                            DType::Vector(VecTy::Array(inner_ty, Some(n + total_pad)))
                        }
                        other => other,
                    }
                }
                MovOp::Shrink(amounts) => {
                    let inner = operand.dt();
                    match inner {
                        DType::Vector(VecTy::Array(inner_ty, Some(n))) => {
                            let total_rem =
                                amounts.iter().map(|(lo, hi)| lo + hi).sum::<usize>() as u32;
                            DType::Vector(VecTy::Array(inner_ty, Some(n - total_rem)))
                        }
                        other => other,
                    }
                }
                _ => operand.dt(),
            },
            JitAST::Reduce { operand, .. } => operand.dt().peel_all(),
            JitAST::AllReduce { operand, .. } => operand.dt().peel_all(),
        }
    }

    pub fn lower_with_rewrite<F>(
        ast: Self,
        scope: &mut Scope,
        var_producer: &mut F,
        user_rules: &[RewriteRule],
    ) -> LoweredAST
    where
        F: FnMut() -> LoweredAST,
    {
        let builtins = rules::builtin_rules();
        let all_rules: Vec<_> = builtins.iter().chain(user_rules.iter()).collect();
        Self::graph_rewrite(ast, scope, &all_rules, var_producer)
    }
}
