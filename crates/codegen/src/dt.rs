use crate::asts::comptime::Struct;

#[derive(Clone, Debug)]
pub enum BasicTy {
    F32,
    U32,
    I32,
}

#[derive(Clone, Debug)]
pub enum MaybeAtomic<A> {
    Atomic(A),
    Naked(A),
}

#[derive(Clone, Debug)]
pub enum VecTy<I> {
    Vec3(MaybeAtomic<I>),
    Vec2(MaybeAtomic<I>),
    Array(MaybeAtomic<I>),
}

#[derive(Clone, Debug)]
pub enum BasicTyOrStructRef {
    StructRef { id: usize },
    BasicTy(BasicTy),
}

#[derive(Clone, Debug)]
pub enum BasicTyOrStructDef {
    StructDef(Struct),
    BasicTy(BasicTy),
}

#[derive(Clone, Debug)]
pub enum BasicTyOrStructRefOrStructDef {
    StructRef { id: usize },
    StructDef(Struct),
    BasicTy(BasicTy),
}

#[derive(Clone, Debug)]
pub enum DType<T = BasicTyOrStructRef> {
    Vector(VecTy<T>), // Vectors cant be nested they can onkly have atomics with struct or atomics with basic ty or just basic tys or structs
    MaybeAtomic(MaybeAtomic<T>), // Naked struct or basic ty or atomic struct or basic ty
}
