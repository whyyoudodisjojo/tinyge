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
    Pad(usize),
}

impl DType {
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
            Self::Basic(_) | Self::Atomic(_) | Self::Pad(_) => self,
        }
    }

    pub fn is_atomic(&self) -> bool {
        matches!(self, DType::Atomic(_))
    }
}
