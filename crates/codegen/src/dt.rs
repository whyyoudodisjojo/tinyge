use crate::asts::lowered::{Accessor, Scope, ShaderIR, Struct};

#[derive(Clone, Debug, PartialEq)]
pub enum IntegerTy {
    U32,
    I32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BasicTy {
    F32,
    Bool,
    Integer(IntegerTy),
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaybeAtomic<A, N> {
    Atomic(A),
    Naked(N),
}

#[derive(Clone, Debug, PartialEq)]
pub enum IntegerTyOrStructRef {
    Integer(IntegerTy),
    StructRef { ident: String },
}

#[derive(Clone, Debug, PartialEq)]
pub enum VecTy {
    Vec4(BasicTy),
    Vec3(BasicTy),
    Vec2(BasicTy),
    Array(MaybeAtomic<IntegerTy, BasicTyOrStructRef>, Option<u32>),
}

#[derive(Clone, Debug, PartialEq)]
pub enum BasicTyOrStructRef {
    StructRef { ident: String },
    BasicTy(BasicTy),
}

#[derive(Clone, Debug, PartialEq)]
pub enum BasicTyOrStructDef {
    StructDef(Struct),
    BasicTy(BasicTy),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DType {
    Vector(VecTy),
    Atomic(IntegerTy),
    Basic(BasicTy),
    StructRef { ident: String },
}

impl DType {
    pub fn peel_array(&self) -> DType {
        match self {
            DType::Vector(VecTy::Array(inner, _)) => match inner {
                MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b)) => DType::Basic(b.clone()),
                MaybeAtomic::Naked(BasicTyOrStructRef::StructRef { ident }) => DType::StructRef {
                    ident: ident.clone(),
                },
                MaybeAtomic::Atomic(i) => DType::Atomic(i.clone()),
            },
            other => other.clone(),
        }
    }

    pub fn peel_all(&self) -> DType {
        match self {
            DType::Vector(VecTy::Vec2(b) | VecTy::Vec3(b) | VecTy::Vec4(b)) => {
                DType::Basic(b.clone())
            }
            DType::Vector(VecTy::Array(inner, _)) => match inner {
                MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b)) => DType::Basic(b.clone()),
                MaybeAtomic::Naked(BasicTyOrStructRef::StructRef { ident }) => DType::StructRef {
                    ident: ident.clone(),
                },
                MaybeAtomic::Atomic(i) => DType::Atomic(i.clone()),
            },
            other => other.clone(),
        }
    }

    pub fn element_count(&self) -> usize {
        match self {
            DType::Basic(_) | DType::Atomic(_) | DType::StructRef { .. } => 1,
            DType::Vector(VecTy::Vec2(_)) => 2,
            DType::Vector(VecTy::Vec3(_)) => 3,
            DType::Vector(VecTy::Vec4(_)) => 4,
            DType::Vector(VecTy::Array(_, Some(n))) => *n as usize,
            DType::Vector(VecTy::Array(_, None)) => 1,
        }
    }

    pub fn apply_accessor(self, acc: &[Accessor], ir: &ShaderIR, scope: &Scope) -> DType {
        if acc.is_empty() {
            return self;
        }
        match self {
            Self::Vector(ref v) => match v {
                VecTy::Array(inner, _) => {
                    let elem = match inner {
                        MaybeAtomic::Atomic(a) => DType::Atomic(a.clone()),
                        MaybeAtomic::Naked(n) => match n {
                            BasicTyOrStructRef::BasicTy(b) => DType::Basic(b.clone()),
                            BasicTyOrStructRef::StructRef { ident } => DType::StructRef {
                                ident: ident.clone(),
                            },
                        },
                    };
                    elem.apply_accessor(&acc[1..], ir, scope)
                }
                VecTy::Vec2(i) | VecTy::Vec3(i) | VecTy::Vec4(i) if acc.len() == 1 => match &acc[0]
                {
                    Accessor::Field(s) => match s.len() {
                        1 => DType::Basic(i.clone()),
                        2 => DType::Vector(VecTy::Vec2(i.clone())),
                        3 => DType::Vector(VecTy::Vec3(i.clone())),
                        4 => DType::Vector(VecTy::Vec4(i.clone())),
                        _ => self,
                    },
                    Accessor::Index(_) => DType::Basic(i.clone()),
                },
                _ => self,
            },
            Self::StructRef { ref ident } => {
                let s = ir.structs.get(ident).unwrap();
                match &acc[0] {
                    Accessor::Field(name) => {
                        let (_, field_dt) = s
                            .inner
                            .iter()
                            .find(|(n, _)| n == name)
                            .expect("field not found on struct");
                        field_dt.clone().apply_accessor(&acc[1..], ir, scope)
                    }
                    Accessor::Index(_) => self,
                }
            }
            Self::Basic(_) | Self::Atomic(_) => self,
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, DType::Atomic(_))
    }

    pub fn as_array_dtype(&self) -> DType {
        use crate::dt::{BasicTy, BasicTyOrStructRef, MaybeAtomic, VecTy};
        let inner = match self {
            DType::Basic(BasicTy::Integer(i)) => MaybeAtomic::Atomic(i.clone()),
            DType::Atomic(i) => MaybeAtomic::Atomic(i.clone()),
            DType::Basic(b) => MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b.clone())),
            DType::StructRef { ident } => MaybeAtomic::Naked(BasicTyOrStructRef::StructRef {
                ident: ident.clone(),
            }),
            DType::Vector(VecTy::Vec2(b)) => {
                MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b.clone()))
            }
            DType::Vector(VecTy::Vec3(b)) => {
                MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b.clone()))
            }
            DType::Vector(VecTy::Vec4(b)) => {
                MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b.clone()))
            }
            DType::Vector(VecTy::Array(_, _)) => panic!("cannot wrap array in array"),
        };
        DType::Vector(VecTy::Array(inner, None))
    }

    pub fn byte_size(&self) -> usize {
        match self {
            DType::Basic(BasicTy::F32) => 4,
            DType::Basic(BasicTy::Bool) => 4,
            DType::Basic(BasicTy::Integer(_)) => 4,
            DType::Atomic(_) => 4,
            DType::Vector(VecTy::Vec2(_)) => 8,
            DType::Vector(VecTy::Vec3(_)) => 12,
            DType::Vector(VecTy::Vec4(_)) => 16,
            DType::Vector(VecTy::Array(inner, Some(n))) => {
                let inner_sz = match inner {
                    MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b)) => {
                        DType::Basic(b.clone()).byte_size()
                    }
                    MaybeAtomic::Naked(BasicTyOrStructRef::StructRef { .. }) => {
                        panic!("struct byte size not supported")
                    }
                    MaybeAtomic::Atomic(i) => DType::Atomic(i.clone()).byte_size(),
                };
                inner_sz * *n as usize
            }
            DType::Vector(VecTy::Array(_, None)) => {
                panic!("runtime array byte size requires element count")
            }
            DType::StructRef { .. } => panic!("struct byte size not supported"),
        }
    }
}
