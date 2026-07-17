pub mod ops;
pub mod renderer;
pub mod scope;

use std::{collections::HashMap, fmt::Display};

pub use scope::Scope;

use crate::dt::{BasicTy, BasicTyOrStructRef, DType, VecTy};

#[derive(Clone, Debug, darling::FromMeta)]
pub enum CustomBufferBindingType {
    Uniform,
    Storage { read_only: bool },
}

#[derive(Clone, Debug)]
pub struct BindingMeta {
    pub ident: String,
    pub ty: CustomBufferBindingType,
    pub struct_name: String,
}

use std::marker::PhantomData;

#[derive(Debug)]
pub struct BindedBuffer<T, const N: usize>(pub PhantomData<T>);

pub struct Functions {
    pub args: HashMap<String, DType>,
    pub ret: Option<BasicTyOrStructRef>,
    pub ident: String,
    pub entrypoint_ty: Option<EntrypointData>,
    pub body: Scope,
}

#[derive(Clone)]
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

#[derive(Debug, darling::FromMeta)]
pub enum EntrypointData {
    Compute { workgroup_sz: usize },
    Shader, // TODO
}

#[derive(Clone, Debug, PartialEq)]
pub struct Struct {
    pub name: String,
    pub inner: Vec<(String, DType)>,
}

impl Struct {
    // sz, align
    pub fn wgsl_size_align(struct_store: &HashMap<String, Self>, dt: &DType) -> (usize, usize) {
        match dt {
            DType::Basic(BasicTy::F32) | DType::Basic(BasicTy::Integer(_)) => (4, 4),
            DType::Atomic(_) => (4, 4),
            DType::Vector(VecTy::Vec2(_)) => (8, 8),
            DType::Vector(VecTy::Vec3(_)) => (12, 16),
            DType::Vector(VecTy::Vec4(_)) => (16, 16),
            DType::Vector(VecTy::Array(_)) => (0, 0),
            DType::StructRef { ident } => {
                let s = struct_store.get(ident).expect("struct not found in store");
                let mut offset = 0;
                let mut max_align = 0;
                for (_, field_dt) in &s.inner {
                    let (field_sz, field_al) = Self::wgsl_size_align(struct_store, field_dt);
                    max_align = max_align.max(field_al);
                    if max_align > 0 && offset % field_al != 0 {
                        offset += field_al - offset % field_al;
                    }
                    offset += field_sz;
                }
                if max_align > 0 && offset % max_align != 0 {
                    offset += max_align - offset % max_align;
                }
                (offset, max_align)
            }
            DType::Pad(bytes) => (*bytes, *bytes),
        }
    }

    pub fn required_padding(
        struct_store: &HashMap<String, Self>,
        prev_field: &DType,
        next_field: &DType,
    ) -> usize {
        let (prev_size, prev_align) = Self::wgsl_size_align(struct_store, prev_field);
        let (_, next_align) = Self::wgsl_size_align(struct_store, next_field);

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

    pub(crate) fn with_padding(self, struct_store: &HashMap<String, Self>) -> Self {
        let mut result = Vec::new();
        let mut prev_dtype: Option<DType> = None;
        let mut pad_counter = 0usize;

        for (name, dtype) in &self.inner {
            if let Some(ref prev) = prev_dtype {
                let padding_needed = Self::required_padding(struct_store, prev, dtype);
                if padding_needed > 0 {
                    result.push((format!("__pad_{}", pad_counter), DType::Pad(padding_needed)));
                    pad_counter += 1;
                }
            }
            result.push((name.clone(), dtype.clone()));
            prev_dtype = Some(dtype.clone());
        }

        Self {
            name: self.name,
            inner: result,
        }
    }
}

pub struct ShaderIR {
    pub structs: HashMap<String, Struct>,
    pub binded: Vec<BindingMeta>,
    pub entrypoint_globals: Vec<EntrypointGlobals>,
    pub functions: Vec<Functions>,
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

impl<T, const N: usize> BindedBuffer<T, N> {
    pub fn var_ref(&self) -> VarRefType {
        VarRefType::Global(VarRef {
            id: N,
            by_index: vec![],
        })
    }
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
