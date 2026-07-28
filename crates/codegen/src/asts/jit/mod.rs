pub mod ops;
pub mod pattern;
pub mod rules;
pub mod runner;

use memory::buffers::BufferWithType;
use memory::socket::TinyBuffer;
use wgpu::{BufferUsages};

use crate::asts::lowered::{BinOp, LoweredAST, UnaryOp, scope::Scope};
use crate::asts::{AstConst, IntoWgslStruct};
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
        id: usize,
        buffer: TinyBuffer,
        dtype: DType,
    },

    Lowered {
        expr: LoweredAST,
        dt: DType,
    },

    Const(AstConst<Self>),

    BinOp {
        lhs: Box<Self>,
        rhs: Box<Self>,
        op: JitBinOp,
    },
    UnaryOp {
        operand: Box<Self>,
        op: JitUnaryOp,
    },
    Cast {
        operand: Box<Self>,
        dt: DType,
    },
    Ternary {
        a: Box<Self>,
        b: Box<Self>,
        c: Box<Self>,
        op: TernaryOp,
    },
    Movement {
        operand: Box<Self>,
        op: MovOp,
    },
    Reduce {
        operand: Box<Self>,
        axis: usize,
        op: ReduceOp,
    },
    AllReduce {
        operand: Box<Self>,
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

static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl<T> From<BufferWithType<T>> for JitAST
where
    T: IntoWgslStruct,
{
    fn from(value: BufferWithType<T>) -> Self {
        JitAST::Var {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            buffer: value.inner,
            dtype: T::dt(),
        }
    }
}

impl JitAST {
    pub fn new<T>() -> Self
    where
        T: IntoWgslStruct,
    {
        let buffer = TinyBuffer::new(
            T::wgsl_byte_size(),
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
        );

        JitAST::Var {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            buffer,
            dtype: T::dt(),
        }
    }

    pub fn collect_var_buffers<'a>(&'a self, out: &mut Vec<&'a TinyBuffer>) {
        let mut seen = std::collections::HashSet::new();
        self.collect_var_buffers_inner(&mut seen, out);
    }

    fn collect_var_buffers_inner<'a>(
        &'a self,
        seen: &mut std::collections::HashSet<usize>,
        out: &mut Vec<&'a TinyBuffer>,
    ) {
        match self {
            JitAST::Var { id, buffer, .. } => {
                if seen.insert(*id) {
                    out.push(buffer);
                }
            }
            JitAST::Lowered { .. } => {}
            JitAST::Const(c) => {
                for d in &c.data {
                    if let crate::asts::lowered::ASTOrConst::AST(a) = d {
                        a.collect_var_buffers_inner(seen, out);
                    }
                }
            }
            JitAST::BinOp { lhs, rhs, .. } => {
                lhs.collect_var_buffers_inner(seen, out);
                rhs.collect_var_buffers_inner(seen, out);
            }
            JitAST::UnaryOp { operand, .. }
            | JitAST::Cast { operand, .. }
            | JitAST::Movement { operand, .. }
            | JitAST::Reduce { operand, .. }
            | JitAST::AllReduce { operand, .. } => operand.collect_var_buffers_inner(seen, out),
            JitAST::Ternary { a, b, c, .. } => {
                a.collect_var_buffers_inner(seen, out);
                b.collect_var_buffers_inner(seen, out);
                c.collect_var_buffers_inner(seen, out);
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
            JitAST::Lowered { dt, .. } => from_dt(dt),
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
        let mut seen = std::collections::HashSet::new();
        self.collect_var_info_inner(&mut seen)
    }

    fn collect_var_info_inner(
        &self,
        seen: &mut std::collections::HashSet<usize>,
    ) -> (usize, Option<DType>) {
        match self {
            JitAST::Var { id, dtype, .. } => {
                if seen.insert(*id) {
                    (1, Some(dtype.clone()))
                } else {
                    (0, None)
                }
            }
            JitAST::Lowered { .. } => (0, None),
            JitAST::Const(c) => {
                let mut count = 0;
                let mut dt = None;
                for d in &c.data {
                    if let crate::asts::lowered::ASTOrConst::AST(a) = d {
                        let (cc, cd) = a.collect_var_info_inner(seen);
                        count += cc;
                        dt = dt.or(cd);
                    }
                }
                (count, dt)
            }
            JitAST::BinOp { lhs, rhs, .. } => {
                let (lc, ld) = lhs.collect_var_info_inner(seen);
                let (rc, rd) = rhs.collect_var_info_inner(seen);
                (lc + rc, ld.or(rd))
            }
            JitAST::UnaryOp { operand, .. }
            | JitAST::Cast { operand, .. }
            | JitAST::Movement { operand, .. }
            | JitAST::Reduce { operand, .. }
            | JitAST::AllReduce { operand, .. } => operand.collect_var_info_inner(seen),
            JitAST::Ternary { a, b, c, .. } => {
                let (ac, ad) = a.collect_var_info_inner(seen);
                let (bc, bd) = b.collect_var_info_inner(seen);
                let (cc, cd) = c.collect_var_info_inner(seen);
                (ac + bc + cc, ad.or(bd).or(cd))
            }
        }
    }

    pub fn dt(&self) -> DType {
        match self {
            JitAST::Var { dtype, .. } => dtype.peel_array(),
            JitAST::Lowered { dt, .. } => dt.clone(),
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

    pub fn inner_movement_chain(&self) -> (&JitAST, Vec<&MovOp>) {
        let mut ops = vec![];
        let mut current = self;
        while let JitAST::Movement { operand, op } = current {
            ops.push(op);
            current = operand.as_ref();
        }
        (current, ops)
    }

    pub fn lower_with_rewrite<F>(
        ast: Self,
        scope: &mut Scope,
        var_producer: &mut F,
        user_rules: &[RewriteRule],
    ) -> LoweredAST
    where
        F: FnMut(usize) -> LoweredAST,
    {
        let builtins = rules::builtin_rules();
        let all_rules: Vec<_> = builtins.iter().chain(user_rules.iter()).collect();
        Self::graph_rewrite(ast, scope, &all_rules, var_producer)
    }
}
