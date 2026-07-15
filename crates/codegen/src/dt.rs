use crate::asts::lowered::Struct;

#[derive(Clone, Debug)]
pub enum IntegerTy {
    U32,
    I32,
}

#[derive(Clone, Debug)]
pub enum BasicTy {
    F32,
    Integer(IntegerTy),
}

#[derive(Clone, Debug)]
pub enum MaybeAtomic<A> {
    Atomic(A),
    Naked(A),
}

#[derive(Clone, Debug)]
pub enum VecTy {
    Vec3(BasicTy),
    Vec2(BasicTy),
    Array(MaybeAtomic<BasicTy>),
}

#[derive(Clone, Debug)]
pub enum BasicTyOrStructRef {
    StructRef { ident: String },
    BasicTy(BasicTy),
}

#[derive(Clone, Debug)]
pub enum BasicTyOrStructDef {
    StructDef(Struct),
    BasicTy(BasicTy),
}

#[derive(Clone, Debug)]
pub enum DType {
    Vector(VecTy),
    Atomic(IntegerTy),
    Basic(BasicTy),
    StructRef { ident: String },
    Pad(usize),
}
