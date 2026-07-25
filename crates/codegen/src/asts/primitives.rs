use glam::{Vec2, Vec3, Vec3A, Vec4};

use crate::asts::jit::JitAST;
use crate::asts::lowered::{ASTOrConst, LoweredAST};
use crate::asts::{AstConst, IntoWgslStruct};
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
        impl From<$ty> for ASTOrConst<LoweredAST> {
            fn from($val: $ty) -> Self {
                $dt
            }
        }

        impl From<$ty> for LoweredAST {
            fn from(val: $ty) -> Self {
                LoweredAST::Const(<$ty as IntoWgslStruct>::into_const(vec![val.into()]))
            }
        }

        impl From<$ty> for ASTOrConst<JitAST> {
            fn from($val: $ty) -> Self {
                $dt
            }
        }

        impl From<$ty> for JitAST {
            fn from(val: $ty) -> Self {
                JitAST::Const(AstConst {
                    dt: <$ty as IntoWgslStruct>::dt(),
                    data: vec![val.into()],
                })
            }
        }
    };
}

impl_const_primitives!(f32, val => ASTOrConst::Const(val.to_le_bytes().to_vec()));
impl_const_primitives!(i32, val => ASTOrConst::Const(val.to_le_bytes().to_vec()));
impl_const_primitives!(u32, val => ASTOrConst::Const(val.to_le_bytes().to_vec()));
impl_const_primitives!(bool, val => ASTOrConst::Const(vec![if val { 1 } else { 0 }]));

impl_primitive!(f32, DType::Basic(BasicTy::F32));
impl_primitive!(bool, DType::Basic(BasicTy::Bool));
impl_primitive!(u32, DType::Basic(BasicTy::Integer(IntegerTy::U32)));
impl_primitive!(i32, DType::Basic(BasicTy::Integer(IntegerTy::I32)));

macro_rules! impl_const_array {
    ($n:literal) => {
        impl From<[f32; $n]> for ASTOrConst<LoweredAST> {
            fn from(val: [f32; $n]) -> Self {
                ASTOrConst::Const(val.iter().flat_map(|f| f.to_le_bytes()).collect())
            }
        }

        impl From<[f32; $n]> for LoweredAST {
            fn from(val: [f32; $n]) -> Self {
                LoweredAST::Const(<[f32; $n] as IntoWgslStruct>::into_const(
                    val.iter()
                        .map(|&f| ASTOrConst::Const(f.to_le_bytes().to_vec()))
                        .collect(),
                ))
            }
        }

        impl From<[f32; $n]> for ASTOrConst<JitAST> {
            fn from(val: [f32; $n]) -> Self {
                ASTOrConst::Const(val.iter().flat_map(|f| f.to_le_bytes()).collect())
            }
        }

        impl From<[f32; $n]> for JitAST {
            fn from(val: [f32; $n]) -> Self {
                JitAST::Const(AstConst {
                    dt: <[f32; $n] as IntoWgslStruct>::dt(),
                    data: val
                        .iter()
                        .map(|&f| ASTOrConst::Const(f.to_le_bytes().to_vec()))
                        .collect(),
                })
            }
        }
    };
}

impl_const_array!(2);
impl_const_array!(3);
impl_const_array!(4);

impl<T: IntoWgslStruct, const N: usize> IntoWgslStruct for [T; N] {
    fn dt() -> DType {
        match (T::dt(), N) {
            (DType::Basic(b), 2) => DType::Vector(VecTy::Vec2(b)),
            (DType::Basic(b), 3) => DType::Vector(VecTy::Vec3(b)),
            (DType::Basic(b), 4) => DType::Vector(VecTy::Vec4(b)),
            (inner_dt, _) => {
                let inner = match inner_dt {
                    DType::Atomic(i) => MaybeAtomic::Atomic(i),
                    DType::Basic(b) => MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b)),
                    DType::StructRef { ident } => {
                        MaybeAtomic::Naked(BasicTyOrStructRef::StructRef { ident })
                    }
                    DType::Vector(_vt @ VecTy::Array(_, None)) => {
                        panic!("unsized array inside a fixed-size array is not allowed")
                    }
                    DType::Vector(vt) => MaybeAtomic::Naked(BasicTyOrStructRef::Vec(Box::new(vt))),
                };
                DType::Vector(VecTy::Array(inner, Some(N as u32)))
            }
        }
    }

    fn wgsl_byte_size() -> u64 {
        match (T::dt(), N) {
            (DType::Basic(_), 2) => 8,
            (DType::Basic(_), 3) => 12,
            (DType::Basic(_), 4) => 16,
            _ => N as u64 * T::wgsl_byte_size(),
        }
    }
}

impl<T> IntoWgslStruct for Vec<T>
where
    T: IntoWgslStruct,
{
    fn dt() -> DType {
        let dt = T::dt();

        let inner = match dt {
            DType::Atomic(i) => MaybeAtomic::Atomic(i),
            DType::Basic(b) => MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(b)),
            DType::StructRef { ident } => {
                MaybeAtomic::Naked(BasicTyOrStructRef::StructRef { ident })
            }
            DType::Vector(_vt @ VecTy::Array(_, None)) => {
                panic!("unsized array inside another array is not allowed")
            }
            DType::Vector(vt) => MaybeAtomic::Naked(BasicTyOrStructRef::Vec(Box::new(vt))),
        };

        DType::Vector(VecTy::Array(inner, None))
    }

    fn wgsl_byte_size() -> u64 {
        T::wgsl_byte_size()
    }
}

macro_rules! copy_impl_from {
    ($ty1:ty, $ty2:ty) => {
        impl IntoWgslStruct for $ty1 {
            fn dt() -> DType {
                <$ty2>::dt()
            }

            fn into_const(data: Vec<ASTOrConst<LoweredAST>>) -> super::AstConst<LoweredAST> {
                <$ty2>::into_const(data)
            }
        }
    };
}

copy_impl_from!(Vec3A, [f32; 3]);
copy_impl_from!(Vec4, [f32; 4]);
copy_impl_from!(Vec2, [f32; 2]);
copy_impl_from!(Vec3, [f32; 3]);
