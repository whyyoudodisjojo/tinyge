use std::collections::HashMap;

use crate::asts::lowered::{ASTOrConst, LoweredAST, Struct};
use crate::dt::{BasicTy, DType};

pub mod jit;
pub mod lowered;
pub mod primitives;

#[derive(Clone, Debug)]
pub struct AstConst<T, C = Vec<u8>> {
    pub dt: DType,
    pub data: Vec<ASTOrConst<T, C>>,
}

impl<T: IntoWgslStruct + Into<ASTOrConst<LoweredAST>>> From<T> for AstConst<LoweredAST> {
    fn from(val: T) -> Self {
        T::into_const(vec![val.into()])
    }
}

pub trait IntoWgslStruct
where
    Self: Sized,
{
    fn dt() -> DType;

    fn into_const(data: Vec<ASTOrConst<LoweredAST>>) -> AstConst<LoweredAST> {
        AstConst {
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
