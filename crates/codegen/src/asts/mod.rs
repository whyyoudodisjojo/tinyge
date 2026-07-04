use crate::asts::lowered::Struct;

pub mod lowered;

pub trait IntoWgslStruct: Into<(String, Struct)> {}
