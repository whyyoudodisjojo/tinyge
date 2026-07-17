use crate::asts::lowered::Struct;

#[derive(Clone, Debug, PartialEq)]
pub enum IntegerTy {
    U32,
    I32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BasicTy {
    F32,
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
    Array(MaybeAtomic<IntegerTyOrStructRef, BasicTyOrStructRef>),
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
