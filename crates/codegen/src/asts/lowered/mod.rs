pub mod ops;
pub mod renderer;
pub mod scope;

use std::{collections::HashMap, fmt::Display};

pub use scope::Scope;

use crate::dt::{BasicTy, BasicTyOrStructRef, DType, IntegerTy, MaybeAtomic, VecTy};

#[derive(Clone, Debug, darling::FromMeta)]
pub enum CustomBufferBindingType {
    Uniform,
    Storage { read_only: bool },
}

#[derive(Clone, Debug)]
pub struct BindingMeta {
    pub ident: String,
    pub ty: CustomBufferBindingType,
    pub dtype: DType,
}

use std::marker::PhantomData;

#[derive(Debug)]
pub struct BindedBuffer<T, const N: usize>(pub PhantomData<T>);

#[derive(Debug)]
pub struct SharedData<T> {
    pub id: usize,
    _phantom: PhantomData<T>,
}

impl<T> SharedData<T> {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }

    pub fn var_ref(&self) -> VarRefType {
        VarRefType::Shared(VarRef {
            id: self.id,
            by: vec![],
        })
    }
}

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
            DType::Vector(VecTy::Array(_, None)) => (0, 0),
            DType::Vector(VecTy::Array(inner, Some(count))) => {
                let (elem_sz, elem_al) = Self::wgsl_size_align(
                    struct_store,
                    &match inner {
                        MaybeAtomic::Atomic(a) => DType::Atomic(a.clone()),
                        MaybeAtomic::Naked(n) => match n {
                            BasicTyOrStructRef::BasicTy(b) => DType::Basic(b.clone()),
                            BasicTyOrStructRef::StructRef { ident } => {
                                DType::StructRef { ident: ident.clone() }
                            }
                        },
                    },
                );
                ((elem_sz * *count as usize).max(16), elem_al)
            }
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
        current_offset: usize,
        next_field: &DType,
    ) -> usize {
        let (_, next_align) = Self::wgsl_size_align(struct_store, next_field);

        if next_align == 0 {
            return 0;
        }

        let misalignment = current_offset % next_align;
        if misalignment == 0 {
            0
        } else {
            next_align - misalignment
        }
    }

    pub(crate) fn with_padding(self, struct_store: &HashMap<String, Self>) -> Self {
        let mut result = Vec::new();
        let mut current_offset = 0usize;
        let mut pad_counter = 0usize;

        for (name, dtype) in &self.inner {
            let padding_needed = Self::required_padding(struct_store, current_offset, dtype);
            if padding_needed > 0 {
                result.push((format!("_pad_{}", pad_counter), DType::Pad(padding_needed)));
                pad_counter += 1;
                current_offset += padding_needed;
            }
            result.push((name.clone(), dtype.clone()));
            let (field_sz, field_al) = Self::wgsl_size_align(struct_store, dtype);
            if field_al > 0 && current_offset % field_al != 0 {
                current_offset += field_al - current_offset % field_al;
            }
            current_offset += field_sz;
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
    pub private_vars: Vec<(String, DType)>,
    pub entrypoint_globals: Vec<EntrypointGlobals>,
    pub functions: Vec<Functions>,
}

#[derive(Clone, Debug)]
pub enum BinOp {
    Add,
    Mul,
    Sub,
    Div,
    Rem,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    Shr,
    Shl,
    LogicalAnd,
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
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
    pub fn i(self, idx: LoweredAST) -> Self {
        let (mut v, ctor) = self.into_var_ref();
        v.by.push(Accessor::Index(Box::new(idx)));
        ctor(v)
    }
    pub fn f(self, name: &str) -> Self {
        let (mut v, ctor) = self.into_var_ref();
        v.by.push(Accessor::Field(name.to_string()));
        ctor(v)
    }
    pub fn load(self) -> LoweredAST {
        LoweredAST::Load(self)
    }
    pub fn store(self, val: LoweredAST) -> LoweredAST {
        LoweredAST::Store {
            var: self,
            val: Box::new(val),
        }
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
pub enum LoweredASTOrConst {
    LoweredAST(LoweredAST),
    Const(Vec<u8>),
}

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
        data: Vec<LoweredASTOrConst>,
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
                VarRefType::Global(g) => ir.binded[g.id]
                    .dtype
                    .clone()
                    .apply_accessor(&g.by, ir, scope),
            },
            Self::BinaryOp {
                lhs: _,
                op:
                    BinOp::Eq
                    | BinOp::Ne
                    | BinOp::Gt
                    | BinOp::Lt
                    | BinOp::Ge
                    | BinOp::Le
                    | BinOp::LogicalAnd,
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
            Self::FunctionCall { args, .. } => args.first().expect("function call with no args").dt(ir, scope),
            _ => panic!("cannot infer type from {:?}", self),
        }
    }

    pub fn eq(self, rhs: Self) -> Self {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Eq,
        }
    }
    pub fn ne(self, rhs: Self) -> Self {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Ne,
        }
    }
    pub fn gt(self, rhs: Self) -> Self {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Gt,
        }
    }
    pub fn lt(self, rhs: Self) -> Self {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Lt,
        }
    }
    pub fn ge(self, rhs: Self) -> Self {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Ge,
        }
    }
    pub fn le(self, rhs: Self) -> Self {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::Le,
        }
    }
    pub fn logical_and(self, rhs: Self) -> Self {
        Self::BinaryOp {
            lhs: Box::new(self),
            rhs: Box::new(rhs),
            op: BinOp::LogicalAnd,
        }
    }
    pub fn logical_or(self, rhs: Self) -> Self {
        !(!self).logical_and(!rhs)
    }

    pub fn store(self, val: Self) -> Self {
        match self {
            Self::Load(var) => Self::Store {
                var,
                val: Box::new(val),
            },
            _ => panic!("expected Load target for store, got {:?}", self),
        }
    }
}
