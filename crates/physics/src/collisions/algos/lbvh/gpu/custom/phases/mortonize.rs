use codegen::asts::lowered::{BindedBuffer, LoweredAST, Scope};
use codegen::asts::lowered::scope::*;
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
        (local(rect).f("min").load() + local(rect).f("max").load()) * f32(0.5),
    );
    let sz = scope.var(
        global_bounds.var_ref().f("max").load() - global_bounds.var_ref().f("min").load(),
    );

    let inv_sz = scope.mut_(vec3(0.0, 0.0, 0.0));
    let norm = scope.mut_(vec3(0.0, 0.0, 0.0));

    let quant = scope.mut_(vec3(0.0, 0.0, 0.0));
    let mx = scope.mut_(u32(0));
    let my = scope.mut_(u32(0));
    let mz = scope.mut_(u32(0));

    let if_x = scope.if_(local(sz).f("x").load().gt(f32(0.0)), |_| {
        local(inv_sz)
            .f("x")
            .store(f32(1.0) / local(sz).f("x").load())
    });
    let if_y = scope.if_(local(sz).f("y").load().gt(f32(0.0)), |_| {
        local(inv_sz)
            .f("y")
            .store(f32(1.0) / local(sz).f("y").load())
    });
    let if_z = scope.if_(local(sz).f("z").load().gt(f32(0.0)), |_| {
        local(inv_sz)
            .f("z")
            .store(f32(1.0) / local(sz).f("z").load())
    });

    let early_return = scope.if_(
        local(idx).load().ge(num_rects.var_ref().load()),
        |_| LoweredAST::Return,
    );

    let body = group!(
        early_return;
        if_x;
        if_y;
        if_z;
        local(norm).store(
            (local(centroid).load() - global_bounds.var_ref().f("min").load()) * local(inv_sz).load()
        );
        local(quant).store(
            call!("clamp", local(norm).load(), vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0))
                * f32(1023.0)
        );
        local(mx).store(call!("u32", local(quant).f("x").load()) & u32(1023));
        local(my).store(call!("u32", local(quant).f("y").load()) & u32(1023));
        local(mz).store(call!("u32", local(quant).f("z").load()) & u32(1023));
        local(mx).store((local(mx).load() | (local(mx).load() << u32(16))) & u32(4278190335));
        local(mx).store((local(mx).load() | (local(mx).load() << u32(8))) & u32(50393103));
        local(mx).store((local(mx).load() | (local(mx).load() << u32(4))) & u32(51130563));
        local(mx).store((local(mx).load() | (local(mx).load() << u32(2))) & u32(153391689));
        local(my).store((local(my).load() | (local(my).load() << u32(16))) & u32(4278190335));
        local(my).store((local(my).load() | (local(my).load() << u32(8))) & u32(50393103));
        local(my).store((local(my).load() | (local(my).load() << u32(4))) & u32(51130563));
        local(my).store((local(my).load() | (local(my).load() << u32(2))) & u32(153391689));
        local(mz).store((local(mz).load() | (local(mz).load() << u32(16))) & u32(4278190335));
        local(mz).store((local(mz).load() | (local(mz).load() << u32(8))) & u32(50393103));
        local(mz).store((local(mz).load() | (local(mz).load() << u32(4))) & u32(51130563));
        local(mz).store((local(mz).load() | (local(mz).load() << u32(2))) & u32(153391689));
        out_keys.var_ref().i(local(idx).load()).store(call!("Key",
            (local(mx).load() | (local(mx).load() >> u32(6))) << u32(2),
            (local(my).load() | (local(my).load() >> u32(6))) << u32(1),
            local(mz).load() | (local(mz).load() >> u32(6)),
        ));
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
