pub mod ops;
pub mod renderer;
pub mod scope;

use std::{collections::HashMap, fmt::Display};

use wgpu::BufferBindingType;

use scope::Scope;

use crate::dt::{BasicTy, BasicTyOrStructRef, DType, VecTy};

pub struct BindedBuffer {
    pub ident: String,
    pub ty: BufferBindingType,
    pub dt: DType,
}

pub struct Functions<'a> {
    pub args: HashMap<String, DType>,
    pub ret: Option<BasicTyOrStructRef>,
    pub ident: String,
    pub entrypoint_ty: Option<EntrypointData>,
    pub body: Scope<'a>,
}

pub enum EntrypointGlobals {
    GlobalInvocationId,
    LocalInvocationId,
}

impl Display for EntrypointGlobals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let x = match self {
            EntrypointGlobals::GlobalInvocationId => "g_id",
            EntrypointGlobals::LocalInvocationId => "l_id",
        };

        f.write_str(&x)
    }
}

pub enum EntrypointData {
    Compute { workgroup_sz: usize },
    Shader, // TODO
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub inner: Vec<(String, DType)>,
}

impl Struct {
    // sz, align
    fn wgsl_size_align(dt: &DType) -> (usize, usize) {
        match dt {
            DType::Basic(BasicTy::F32) | DType::Basic(BasicTy::Integer(_)) => (4, 4),
            DType::Atomic(_) => (4, 4),
            DType::Vector(VecTy::Vec2(_)) => (8, 8),
            DType::Vector(VecTy::Vec3(_)) => (12, 16),
            DType::Vector(VecTy::Array(_)) => (0, 0),
            DType::StructRef { ident: _ } => (0, 0),
            DType::Pad(bytes) => (*bytes, *bytes),
        }
    }

    pub fn required_padding(prev_field: &DType, next_field: &DType) -> usize {
        let (prev_size, prev_align) = Self::wgsl_size_align(prev_field);
        let (_, next_align) = Self::wgsl_size_align(next_field);

        if prev_align == 0 || next_align == 0 {
            return 0;
        }

        let prev_effective_size = if prev_size % prev_align == 0 {
            prev_size
        } else {
            prev_size + (prev_align - prev_size % prev_align)
        };

        let misalignment = prev_effective_size % next_align;
        if misalignment == 0 {
            0
        } else {
            next_align - misalignment
        }
    }
}

pub struct ShaderIR<'a> {
    pub structs: HashMap<String, Struct>,
    pub binded: Vec<BindedBuffer>,
    pub entrypoint_globals: Vec<EntrypointGlobals>,
    pub functions: Vec<Functions<'a>>,
}

#[derive(Clone)]
pub enum BinOp {
    Add,
    Mul,
    Sub,
    Div,
    BitwiseAnd,
    Shr,
    Shl,
    LogicalAnd,
    Eq,
    Gt,
}

#[derive(Clone)]
pub enum UnaryOp {
    BitwiseNot,
    LogicalNot,
    Neg,
}

#[derive(Clone)]
pub struct VarRef {
    pub id: usize,
    pub by_index: Vec<usize>,
}

#[derive(Clone)]
pub enum VarRefType {
    Local(VarRef),
    Global(VarRef),
    EntryPointGlobal(VarRef),
}

#[derive(Clone)]
pub struct ScopePtr(pub usize);

#[derive(Clone)]
pub enum LoweredAST {
    Store {
        var: VarRefType,
        val: Box<Self>,
    },
    Load(VarRefType),
    BinaryOp {
        lhs: Box<Self>,
        rhs: Box<Self>,
        op: BinOp,
    },
    UnaryOp {
        operand: Box<Self>,
        op: UnaryOp,
    },
    Conditional {
        cond: Box<Self>,
        true_block: ScopePtr,         // JUMP TO SCOPE
        else_block: Option<ScopePtr>, // JUMP TO SCOPE
    },
    ForLoop {
        init: Option<Box<Self>>,
        halt_cond: Option<Box<Self>>,
        increment: Option<Box<Self>>,
        body: ScopePtr, // JUMP TO SCOPE
    },
    WhileLoop {
        cond: Box<Self>,
        body: ScopePtr,
    },
    Const {
        dt: DType,
        data: Vec<u8>,
    },
    FunctionCall {
        ident: String,
        args: Vec<Box<Self>>,
    },
    Continue,
    Break,
    Return,
}
