use crate::asts::IntoWgslStruct;
use crate::dt::{BasicTy, BasicTyOrStructRef, DType, IntegerTy, MaybeAtomic, VecTy};

macro_rules! impl_primitive {
    ($ty:ty, $dt:expr) => {
        impl IntoWgslStruct for $ty {
            fn dt() -> DType {
                $dt
            }
        }
    };
}

impl_primitive!(f32, DType::Basic(BasicTy::F32));
impl_primitive!(bool, DType::Basic(BasicTy::Bool));
impl_primitive!(u32, DType::Basic(BasicTy::Integer(IntegerTy::U32)));
impl_primitive!(i32, DType::Basic(BasicTy::Integer(IntegerTy::I32)));

impl IntoWgslStruct for [f32; 2] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec2(BasicTy::F32))
    }
}
impl IntoWgslStruct for [f32; 3] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec3(BasicTy::F32))
    }
}
impl IntoWgslStruct for [u32; 2] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec2(BasicTy::Integer(IntegerTy::U32)))
    }
}
impl IntoWgslStruct for [u32; 3] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec3(BasicTy::Integer(IntegerTy::U32)))
    }
}
impl IntoWgslStruct for [i32; 2] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec2(BasicTy::Integer(IntegerTy::I32)))
    }
}
impl IntoWgslStruct for [i32; 3] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec3(BasicTy::Integer(IntegerTy::I32)))
    }
}

impl IntoWgslStruct for [f32; 4] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec4(BasicTy::F32))
    }
}
impl IntoWgslStruct for [u32; 4] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec4(BasicTy::Integer(IntegerTy::U32)))
    }
}
impl IntoWgslStruct for [i32; 4] {
    fn dt() -> DType {
        DType::Vector(VecTy::Vec4(BasicTy::Integer(IntegerTy::I32)))
    }
}

impl IntoWgslStruct for Vec<f32> {
    fn dt() -> DType {
        DType::Vector(VecTy::Array(MaybeAtomic::Naked(
            BasicTyOrStructRef::BasicTy(BasicTy::F32),
        )))
    }
}
impl IntoWgslStruct for Vec<u32> {
    fn dt() -> DType {
        DType::Vector(VecTy::Array(MaybeAtomic::Naked(
            BasicTyOrStructRef::BasicTy(BasicTy::Integer(IntegerTy::U32)),
        )))
    }
}
impl IntoWgslStruct for Vec<i32> {
    fn dt() -> DType {
        DType::Vector(VecTy::Array(MaybeAtomic::Naked(
            BasicTyOrStructRef::BasicTy(BasicTy::Integer(IntegerTy::I32)),
        )))
    }
}
