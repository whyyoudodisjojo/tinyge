use codegen::asts::lowered::{BindedBuffer, LoweredAST, Scope, SharedData};
use codegen::asts::{Atomic, IntoWgslStruct};
use codegen::dt::{DType, IntegerTy, MaybeAtomic, VecTy};
use codegen_macros::{IntoWgslStruct, shader};
use tinyge_graphics::shaders::ComputeShader;

#[repr(C)]
#[derive(IntoWgslStruct, Clone, Copy)]
struct MyData {
    val: f32,
}

#[repr(C)]
#[derive(IntoWgslStruct, Clone, Copy)]
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
    #[binding(storage(read_only = true))] _input: BindedBuffer<Vec<MyData>, 0>,
    _sdata: SharedData<SharedElem>,
) -> Scope {
    let mut scope = Scope::new();
    scope.ast = Some(_sdata.var_ref().store(LoweredAST::Const {
        dt: <SharedElem as codegen::asts::IntoWgslStruct>::dt(),
        data: vec![
            codegen::asts::lowered::LoweredASTOrConst::Const(0f32.to_le_bytes().to_vec());
            4
        ],
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
    let s = SharedShader {
        _input_elem_count: 0,
    };
    let wgsl = s.load_source_code();
    assert!(wgsl.contains("struct SharedElem"));
    println!("{wgsl}");
    assert!(wgsl.contains("var<workgroup>"));
    assert!(wgsl.contains("_sdata"));
    assert!(wgsl.contains("@compute @workgroup_size(256)"));
    assert!(wgsl.contains("fn shared_shader"));
}

#[test]
fn test_atomic_dt() {
    assert_eq!(
        <Atomic<u32> as IntoWgslStruct>::dt(),
        DType::Atomic(IntegerTy::U32),
    );
    assert_eq!(
        <Atomic<i32> as IntoWgslStruct>::dt(),
        DType::Atomic(IntegerTy::I32),
    );
}

#[test]
fn test_vec_atomic_dt() {
    assert_eq!(
        <Vec<Atomic<u32>> as IntoWgslStruct>::dt(),
        DType::Vector(VecTy::Array(MaybeAtomic::Atomic(IntegerTy::U32), None)),
    );
    assert_eq!(
        <Vec<Atomic<i32>> as IntoWgslStruct>::dt(),
        DType::Vector(VecTy::Array(MaybeAtomic::Atomic(IntegerTy::I32), None)),
    );
}

#[repr(C)]
#[derive(IntoWgslStruct, Clone, Copy)]
struct WithAtomic {
    counter: Atomic<u32>,
}

fn my_inject() -> LoweredAST {
    LoweredAST::FunctionCall {
        ident: "injected_helper".to_string(),
        args: vec![],
    }
}

#[shader(compute(workgroup_sz = 64))]
fn shader_with_extra(
    #[binding(uniform)] _input: BindedBuffer<MyData, 0>,
    inject: fn() -> LoweredAST,
) -> Scope {
    let mut scope = Scope::new();
    scope.ast = Some(LoweredAST::Group(vec![inject(), LoweredAST::Return]));
    scope
}

#[test]
fn test_extra_args() {
    let s = ShaderWithExtra { inject: my_inject };
    let wgsl = s.load_source_code();
    println!("{wgsl}");
    assert!(wgsl.contains("injected_helper"));
    assert!(wgsl.contains("@compute @workgroup_size(64)"));
    assert!(wgsl.contains("fn shader_with_extra"));
}

#[test]
fn test_derive_atomic_field() {
    let structs = codegen::asts::build_struct_map();
    let s = structs.get("WithAtomic").unwrap();
    assert_eq!(s.inner.len(), 1);
    assert_eq!(s.inner[0].0, "counter");
    assert_eq!(s.inner[0].1, DType::Atomic(IntegerTy::U32),);
}
