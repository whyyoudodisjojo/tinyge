use std::collections::HashMap;

use crate::asts::lowered::Struct;
use crate::dt::DType;

pub mod lowered;

pub trait IntoWgslStruct {
    fn dt() -> (String, Struct);
}

pub struct WgslStructFactory {
    pub name: &'static str,
    pub make: fn() -> (String, Vec<(String, DType)>),
}

inventory::collect!(WgslStructFactory);

pub fn build_struct_map() -> HashMap<String, Struct> {
    let raw: Vec<(String, Vec<(String, DType)>)> = inventory::iter::<WgslStructFactory>
        .into_iter()
        .map(|f| (f.make)())
        .collect();

    let map: HashMap<String, Struct> = raw
        .iter()
        .map(|(name, fields)| {
            (
                name.clone(),
                Struct {
                    inner: fields.clone(),
                },
            )
        })
        .collect();

    raw.into_iter()
        .map(|(name, fields)| {
            let padded = Struct { inner: fields }.with_padding(&map);
            (name, padded)
        })
        .collect()
}
