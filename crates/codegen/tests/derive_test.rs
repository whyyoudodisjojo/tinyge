use codegen::asts::lowered::Struct;
use codegen_macros::IntoWgslStruct;

#[derive(IntoWgslStruct)]
pub struct SimpleParticle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    #[codegen(atomic)]
    pub id: u32,
}

#[derive(IntoWgslStruct)]
pub struct BufferData {
    pub data: Vec<f32>,
    pub count: u32,
}

#[derive(IntoWgslStruct)]
pub struct BasicTypes {
    pub float_val: f32,
    pub uint_val: u32,
    pub int_val: i32,
    pub vec2_val: [f32; 2],
    pub vec3_val: [f32; 3],
}

#[derive(IntoWgslStruct)]
pub struct AtomicCounter {
    #[codegen(atomic)]
    pub count: u32,
}

#[derive(IntoWgslStruct, Clone, Copy)]
pub struct InnerStruct {
    pub value: f32,
}

#[derive(IntoWgslStruct)]
pub struct OuterStruct {
    pub inner: InnerStruct,
    pub count: u32,
}

#[test]
fn test_simple_particle() {
    let particle = SimpleParticle {
        position: [0.0; 3],
        velocity: [0.0; 3],
        id: 0,
    };

    let (name, s): (String, Struct) = particle.into();
    println!("SimpleParticle ({}): {:#?}", name, s);
}

#[test]
fn test_buffer_data() {
    let buffer = BufferData {
        data: vec![1.0, 2.0, 3.0],
        count: 3,
    };

    let (name, s): (String, Struct) = buffer.into();
    println!("BufferData ({}): {:#?}", name, s);
}

#[test]
fn test_basic_types() {
    let basic = BasicTypes {
        float_val: 1.0,
        uint_val: 1,
        int_val: -1,
        vec2_val: [0.0; 2],
        vec3_val: [0.0; 3],
    };

    let (name, s): (String, Struct) = basic.into();
    println!("BasicTypes ({}): {:#?}", name, s);
}

#[test]
fn test_atomic_counter() {
    let counter = AtomicCounter { count: 42 };

    let (name, s): (String, Struct) = counter.into();
    println!("AtomicCounter ({}): {:#?}", name, s);
}

#[test]
fn test_nested_struct() {
    let outer = OuterStruct {
        inner: InnerStruct { value: 42.0 },
        count: 5,
    };

    let (name, s): (String, Struct) = outer.into();
    println!("OuterStruct ({}): {:#?}", name, s);
}