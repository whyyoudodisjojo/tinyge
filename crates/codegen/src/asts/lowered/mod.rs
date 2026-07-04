pub mod ops;
pub mod renderer;
pub mod scope;

use std::{collections::HashMap, fmt::Display};

use wgpu::BufferBindingType;

use scope::Scope;

use crate::dt::{BasicTyOrStructRef, DType};

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
    pub inner: HashMap<String, DType>,
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
    Continue,
    Break,
    Return,
}
