use std::collections::HashMap;

use crate::asts::lowered::{LoweredAST, LoweredASTOrConst, Struct};
use crate::dt::{BasicTy, DType};

pub mod lowered;
pub mod primitives;

pub trait IntoWgslStruct
where
    Self: Sized,
{
    fn dt() -> DType;

    fn into_const(data: Vec<LoweredASTOrConst>) -> LoweredAST {
        LoweredAST::Const {
            dt: Self::dt(),
            data,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Atomic<T>(pub T);

impl<T: IntoWgslStruct> IntoWgslStruct for Atomic<T> {
    fn dt() -> DType {
        match T::dt() {
            DType::Basic(BasicTy::Integer(ity)) => DType::Atomic(ity),
            _ => panic!("atomic only valid for integer types"),
        }
    }
}

pub struct WgslStructFactory {
    pub name: &'static str,
    pub make: fn() -> Struct,
}

inventory::collect!(WgslStructFactory);

pub fn build_struct_map() -> HashMap<String, Struct> {
    inventory::iter::<WgslStructFactory>
        .into_iter()
        .map(|f| (f.make)())
        .map(|s| (s.name.clone(), s))
        .collect()
}
