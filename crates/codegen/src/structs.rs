#[macro_export]
macro_rules! shader_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $id:ident {
            $( $field_vis:vis $field_name:ident : $field_type:expr ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $id;

        impl From<$id> for $crate::asts::lowered::Struct{
            fn from(_item: $id) -> Self {
                let mut fields = std::collections::HashMap::new();
                $(
                    fields.insert(stringify!($field_name).to_string(), $field_type);
                )*
                $crate::asts::lowered::Struct{inner: fields, ident: stringify!($id).to_string()}
            }
        }

    };
}

#[cfg(test)]
mod test {
    use crate::{
        asts::lowered::Struct,
        dt::{BasicTy, BasicTyOrStructRef, DType, MaybeAtomic, VecTy},
    };
    shader_struct! {
        #[derive(Debug)]
        pub struct MyGpuParticle {
            position: DType::Vector(VecTy::Vec3(MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(BasicTy::F32)))),
            velocity: DType::Vector(VecTy::Vec3(MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(BasicTy::F32)))),
            id: DType::MaybeAtomic(MaybeAtomic::Atomic(BasicTyOrStructRef::BasicTy(BasicTy::U32))),
        }
    }

    #[test]
    fn test_macro() {
        let s = MyGpuParticle;

        let s: Struct = s.into();
        println!("{:#?}", s)
    }
}
