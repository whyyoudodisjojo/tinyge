use codegen::asts::IntoWgslStruct;
use codegen::asts::lowered::scope::*;
use codegen::asts::lowered::{BindedBuffer, LoweredAST, LoweredASTOrConst, Scope};
use codegen::{call, group};
use codegen_macros::{IntoWgslStruct, shader};
use tinyge_graphics::shaders::ComputeShader;

use super::compute_rects::RectangleBounds;

#[derive(IntoWgslStruct)]
struct Key {
    code: u32,
    idx: u32,
}

#[shader(compute(workgroup_sz = 256))]
fn generate_morton_keys(
    #[binding(storage(read_only = true))] in_rects: BindedBuffer<RectangleBounds, 0>,
    #[binding(storage(read_only = false))] out_keys: BindedBuffer<Vec<Key>, 1>,
    #[binding(uniform)] global_bounds: BindedBuffer<RectangleBounds, 2>,
    #[binding(uniform)] num_rects: BindedBuffer<u32, 3>,
) -> Scope {
    let mut scope = Scope::new();

    let idx = scope.var(entrypoint(0).f("x").load());

    let rect = scope.var(in_rects.var_ref().i(local(idx).load()).load());
    let centroid = scope.var(
        (local(rect).f("min").load() + local(rect).f("max").load())
            * cast::<f32>(vec![0.5f32.into()]),
    );
    let sz = scope
        .var(global_bounds.var_ref().f("max").load() - global_bounds.var_ref().f("min").load());

    let inv_sz = scope.mut_(cast::<[f32; 3]>(vec![
        0.0f32.into(),
        0.0f32.into(),
        0.0f32.into(),
    ]));
    let norm = scope.mut_(cast::<[f32; 3]>(vec![
        0.0f32.into(),
        0.0f32.into(),
        0.0f32.into(),
    ]));

    let quant = scope.mut_(cast::<[f32; 3]>(vec![
        0.0f32.into(),
        0.0f32.into(),
        0.0f32.into(),
    ]));
    let mx = scope.mut_(cast::<u32>(vec![0u32.into()]));
    let my = scope.mut_(cast::<u32>(vec![0u32.into()]));
    let mz = scope.mut_(cast::<u32>(vec![0u32.into()]));

    let if_x = scope.if_(
        local(sz).f("x").load().gt(cast::<f32>(vec![0.0f32.into()])),
        |_| {
            local(inv_sz)
                .f("x")
                .store(cast::<f32>(vec![1.0f32.into()]) / local(sz).f("x").load())
        },
    );
    let if_y = scope.if_(
        local(sz).f("y").load().gt(cast::<f32>(vec![0.0f32.into()])),
        |_| {
            local(inv_sz)
                .f("y")
                .store(cast::<f32>(vec![1.0f32.into()]) / local(sz).f("y").load())
        },
    );
    let if_z = scope.if_(
        local(sz).f("z").load().gt(cast::<f32>(vec![0.0f32.into()])),
        |_| {
            local(inv_sz)
                .f("z")
                .store(cast::<f32>(vec![1.0f32.into()]) / local(sz).f("z").load())
        },
    );

    let early_return = scope.if_(local(idx).load().ge(num_rects.var_ref().load()), |_| {
        LoweredAST::Return
    });

    let body = group!(
        early_return;
        if_x;
        if_y;
        if_z;
        local(norm).store(
            (local(centroid).load() - global_bounds.var_ref().f("min").load()) * local(inv_sz).load()
        );
        local(quant).store(
            call!("clamp", local(norm).load(), cast::<[f32; 3]>(vec![0.0f32.into(), 0.0f32.into(), 0.0f32.into()]), cast::<[f32; 3]>(vec![1.0f32.into(), 1.0f32.into(), 1.0f32.into()]))
                * cast::<f32>(vec![1023.0f32.into()])
        );
        local(mx).store(call!("u32", local(quant).f("x").load()) & cast::<u32>(vec![1023u32.into()]));
        local(my).store(call!("u32", local(quant).f("y").load()) & cast::<u32>(vec![1023u32.into()]));
        local(mz).store(call!("u32", local(quant).f("z").load()) & cast::<u32>(vec![1023u32.into()]));
        local(mx).store((local(mx).load() | (local(mx).load() << cast::<u32>(vec![16u32.into()]))) & cast::<u32>(vec![4278190335u32.into()]));
        local(mx).store((local(mx).load() | (local(mx).load() << cast::<u32>(vec![8u32.into()]))) & cast::<u32>(vec![50393103u32.into()]));
        local(mx).store((local(mx).load() | (local(mx).load() << cast::<u32>(vec![4u32.into()]))) & cast::<u32>(vec![51130563u32.into()]));
        local(mx).store((local(mx).load() | (local(mx).load() << cast::<u32>(vec![2u32.into()]))) & cast::<u32>(vec![153391689u32.into()]));
        local(my).store((local(my).load() | (local(my).load() << cast::<u32>(vec![16u32.into()]))) & cast::<u32>(vec![4278190335u32.into()]));
        local(my).store((local(my).load() | (local(my).load() << cast::<u32>(vec![8u32.into()]))) & cast::<u32>(vec![50393103u32.into()]));
        local(my).store((local(my).load() | (local(my).load() << cast::<u32>(vec![4u32.into()]))) & cast::<u32>(vec![51130563u32.into()]));
        local(my).store((local(my).load() | (local(my).load() << cast::<u32>(vec![2u32.into()]))) & cast::<u32>(vec![153391689u32.into()]));
        local(mz).store((local(mz).load() | (local(mz).load() << cast::<u32>(vec![16u32.into()]))) & cast::<u32>(vec![4278190335u32.into()]));
        local(mz).store((local(mz).load() | (local(mz).load() << cast::<u32>(vec![8u32.into()]))) & cast::<u32>(vec![50393103u32.into()]));
        local(mz).store((local(mz).load() | (local(mz).load() << cast::<u32>(vec![4u32.into()]))) & cast::<u32>(vec![511305637u32.into()]));
        local(mz).store((local(mz).load() | (local(mz).load() << cast::<u32>(vec![2u32.into()]))) & cast::<u32>(vec![153391689u32.into()]));
        out_keys.var_ref().i(local(idx).load()).store(Key::into_const(vec![
            LoweredASTOrConst::LoweredAST(
                ((local(mx).load() | (local(mx).load() >> cast::<u32>(vec![6u32.into()]))) << cast::<u32>(vec![2u32.into()]))
                | ((local(my).load() | (local(my).load() >> cast::<u32>(vec![6u32.into()]))) << cast::<u32>(vec![1u32.into()]))
                | (local(mz).load() | (local(mz).load() >> cast::<u32>(vec![6u32.into()])))
            ),
            LoweredASTOrConst::LoweredAST(local(idx).load()),
        ]));
    );

    scope.ast = Some(body);
    scope
}

#[test]
fn test_mortonize() {
    let s = GenerateMortonKeys;
    let wgsl = s.load_source_code();
    println!("{wgsl}");
    assert!(wgsl.contains("struct Key"));
    assert!(wgsl.contains("struct RectangleBounds"));
    assert!(wgsl.contains("@compute @workgroup_size(256)"));
    assert!(wgsl.contains("fn generate_morton_keys"));
    assert!(wgsl.contains("array<Key>"));
    assert!(wgsl.contains("var<uniform> global_bounds"));
    assert!(wgsl.contains("var<uniform> num_rects"));
}
