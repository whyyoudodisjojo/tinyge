use codegen::asts::IntoWgslStruct;
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
    let (name, s): (String, Struct) = SimpleParticle::dt();
    println!("SimpleParticle ({}): {:#?}", name, s);
}

#[test]
fn test_buffer_data() {
    let (name, s): (String, Struct) = BufferData::dt();
    println!("BufferData ({}): {:#?}", name, s);
}

#[test]
fn test_basic_types() {
    let (name, s): (String, Struct) = BasicTypes::dt();
    println!("BasicTypes ({}): {:#?}", name, s);
}

#[test]
fn test_atomic_counter() {
    let (name, s): (String, Struct) = AtomicCounter::dt();
    println!("AtomicCounter ({}): {:#?}", name, s);
}

#[test]
fn test_nested_struct() {
    let (name, s): (String, Struct) = OuterStruct::dt();
    println!("OuterStruct ({}): {:#?}", name, s);

    let store = codegen::asts::build_struct_map();

    let inner_dt = codegen::dt::DType::StructRef {
        ident: "InnerStruct".to_string(),
    };
    assert_eq!(
        Struct::wgsl_size_align(&store, &inner_dt),
        (4, 4),
        "InnerStruct size/align"
    );

    let outer_dt = codegen::dt::DType::StructRef {
        ident: "OuterStruct".to_string(),
    };
    assert_eq!(
        Struct::wgsl_size_align(&store, &outer_dt),
        (8, 4),
        "OuterStruct size/align"
    );
}
