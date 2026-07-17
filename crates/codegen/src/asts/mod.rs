use std::collections::HashMap;

use crate::asts::lowered::Struct;
use crate::dt::{BasicTy, DType};

pub mod lowered;
pub mod primitives;

pub trait IntoWgslStruct {
    fn dt() -> DType;
}

pub fn atomic(dt: DType) -> DType {
    match dt {
        DType::Basic(BasicTy::Integer(ity)) => DType::Atomic(ity),
        _ => panic!("atomic only valid for integer types"),
    }
}

pub struct WgslStructFactory {
    pub name: &'static str,
    pub make: fn() -> Struct,
}

inventory::collect!(WgslStructFactory);

pub fn build_struct_map() -> HashMap<String, Struct> {
    let raw: Vec<Struct> = inventory::iter::<WgslStructFactory>
        .into_iter()
        .map(|f| (f.make)())
        .collect();

    let map: HashMap<String, Struct> = raw
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                Struct {
                    name: s.name.clone(),
                    inner: s.inner.clone(),
                },
            )
        })
        .collect();

    raw.into_iter()
        .map(|s| {
            let padded = s.with_padding(&map);
            (padded.name.clone(), padded)
        })
        .collect()
}
