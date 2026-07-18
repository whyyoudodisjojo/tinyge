use codegen::asts::lowered::{BindedBuffer, LoweredAST, Scope, SharedData};
use codegen_macros::{IntoWgslStruct, shader};
use tinyge_graphics::shaders::ComputeShader;

#[derive(IntoWgslStruct)]
struct MyData {
    val: f32,
}

#[derive(IntoWgslStruct)]
struct SharedElem {
    val: f32,
}

#[shader(compute(workgroup_sz = 64))]
fn my_shader(#[binding(uniform)] _input: BindedBuffer<MyData, 0>) -> Scope {
    let mut scope = Scope::new();
    scope.ast = Some(LoweredAST::Return);
    scope
}

#[shader(compute(workgroup_sz = 256))]
fn shared_shader(
    #[binding(storage(read_only = true))] _input: BindedBuffer<MyData, 0>,
    _sdata: SharedData<SharedElem>,
) -> Scope {
    let mut scope = Scope::new();
    scope.ast = Some(_sdata.var_ref().store(LoweredAST::Const {
        dt: <SharedElem as codegen::asts::IntoWgslStruct>::dt(),
        data: vec![0u8; 4],
    }));
    scope
}

#[test]
fn test_shader_expands_and_runs() {
    let s = MyShader;
    assert_eq!(s.entry_point(), "my_shader");

    let wgsl = s.load_source_code();
    assert!(!wgsl.is_empty());
    println!("{wgsl}");
    assert!(wgsl.contains("struct MyData"));
    assert!(wgsl.contains("@compute @workgroup_size(64)"));
    assert!(wgsl.contains("fn my_shader"));
}

#[test]
fn test_shared_var() {
    let s = SharedShader;
    let wgsl = s.load_source_code();
    assert!(wgsl.contains("struct SharedElem"));
    println!("{wgsl}");
    assert!(wgsl.contains("var<workgroup>"));
    assert!(wgsl.contains("_sdata"));
    assert!(wgsl.contains("@compute @workgroup_size(256)"));
    assert!(wgsl.contains("fn shared_shader"));
}
