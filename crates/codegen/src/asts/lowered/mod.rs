pub mod ops;
pub mod renderer;
pub mod scope;

use std::{collections::HashMap, fmt::Display};

pub use scope::Scope;

use crate::dt::{BasicTy, BasicTyOrStructRef, DType, IntegerTy, VecTy};

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
            DType::Basic(BasicTy::F32)
            | DType::Basic(BasicTy::Bool)
            | DType::Basic(BasicTy::Integer(_)) => (4, 4),
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
    pub shared_vars: Vec<(String, DType)>,
    pub entrypoint_globals: Vec<EntrypointGlobals>,
    pub functions: Vec<Functions>,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub enum UnaryOp {
    BitwiseNot,
    LogicalNot,
    Neg,
}

#[derive(Debug, Clone)]
pub struct VarRef {
    pub id: usize,
    pub by: Vec<Accessor>,
}

#[derive(Debug, Clone)]
pub enum Accessor {
    Index(Box<LoweredAST>),
    Field(String),
}

#[derive(Clone, Debug)]
pub enum VarRefType {
    Local(VarRef),
    Global(VarRef),
    EntryPointGlobal(VarRef),
    Shared(VarRef),
}

impl VarRefType {
    fn into_var_ref(self) -> (VarRef, fn(VarRef) -> Self) {
        match self {
            VarRefType::Global(v) => (v, VarRefType::Global),
            VarRefType::Local(v) => (v, VarRefType::Local),
            VarRefType::Shared(v) => (v, VarRefType::Shared),
            VarRefType::EntryPointGlobal(v) => (v, VarRefType::EntryPointGlobal),
        }
    }
    pub fn index(self, idx: LoweredAST) -> Self {
        let (mut v, ctor) = self.into_var_ref();
        v.by.push(Accessor::Index(Box::new(idx)));
        ctor(v)
    }
    pub fn field(self, name: &str) -> Self {
        let (mut v, ctor) = self.into_var_ref();
        v.by.push(Accessor::Field(name.to_string()));
        ctor(v)
    }
}

impl<T, const N: usize> BindedBuffer<T, N> {
    pub fn var_ref(&self) -> VarRefType {
        VarRefType::Global(VarRef { id: N, by: vec![] })
    }
}

#[derive(Clone, Debug)]
pub struct ScopePtr(pub usize);

#[derive(Clone, Debug)]
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
    Group(Vec<Self>),
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

impl LoweredAST {
    pub fn dt(&self, ir: &ShaderIR, scope: &Scope) -> DType {
        match self {
            Self::Load(l) => match l {
                VarRefType::EntryPointGlobal(g) => {
                    DType::Vector(VecTy::Vec3(BasicTy::Integer(IntegerTy::U32)))
                        .apply_accessor(&g.by, ir, scope)
                }
                VarRefType::Local(l) => scope.local_vars[l.id]
                    .ast
                    .clone()
                    .dt(ir, scope)
                    .apply_accessor(&l.by, ir, scope),
                VarRefType::Shared(s) => ir.shared_vars[s.id]
                    .1
                    .clone()
                    .apply_accessor(&s.by, ir, scope),
                VarRefType::Global(g) => DType::StructRef {
                    ident: ir.binded[g.id].struct_name.clone(),
                }
                .apply_accessor(&g.by, ir, scope),
            },
            Self::BinaryOp {
                lhs: _,
                op: BinOp::Eq | BinOp::Gt | BinOp::LogicalAnd,
                ..
            } => DType::Basic(BasicTy::Bool),
            Self::BinaryOp { lhs, .. } => lhs.dt(ir, scope),
            Self::UnaryOp {
                operand: _,
                op: UnaryOp::LogicalNot,
                ..
            } => DType::Basic(BasicTy::Bool),
            Self::UnaryOp { operand, .. } => operand.dt(ir, scope),
            Self::Const { dt, .. } => dt.clone(),
            _ => panic!("cannot infer type from {:?}", self),
        }
    }
}
