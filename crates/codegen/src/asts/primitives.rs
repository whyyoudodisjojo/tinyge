use crate::asts::IntoWgslStruct;
use crate::asts::lowered::LoweredASTOrConst;
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

macro_rules! impl_const_primitives {
    ($ty:ty, $val:ident => $dt:expr) => {
        impl From<$ty> for LoweredASTOrConst {
            fn from($val: $ty) -> Self {
                $dt
            }
        }
    };
}

impl_const_primitives!(f32, val => LoweredASTOrConst::Const(val.to_le_bytes().to_vec()));
impl_const_primitives!(i32, val => LoweredASTOrConst::Const(val.to_le_bytes().to_vec()));
impl_const_primitives!(u32, val => LoweredASTOrConst::Const(val.to_le_bytes().to_vec()));
impl_const_primitives!(bool, val => LoweredASTOrConst::Const(vec![if val { 1 } else { 0 }]));

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

impl<T> IntoWgslStruct for Vec<T>
where
    T: IntoWgslStruct,
{
    fn dt() -> DType {
        let dt = T::dt();

        let inner = match dt {
            DType::Basic(BasicTy::Integer(i)) => MaybeAtomic::Atomic(i),
            DType::Atomic(i) => MaybeAtomic::Atomic(i),
            DType::Basic(b) => MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b)),
            DType::StructRef { ident } => {
                MaybeAtomic::Naked(BasicTyOrStructRef::StructRef { ident })
            }
            _ => panic!("Cant get this brothr"),
        };

        DType::Vector(VecTy::Array(inner))
    }
}
