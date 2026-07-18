use codegen::asts::lowered::Scope;
use codegen::asts::lowered::scope::*;
use codegen::asts::lowered::{BindedBuffer, SharedData};
use codegen::{call, group};
use codegen_macros::{IntoWgslStruct, shader};
use tinyge_graphics::shaders::ComputeShader;

#[derive(IntoWgslStruct)]
pub struct Vertex {
    pos: [f32; 3],
}

#[derive(IntoWgslStruct)]
pub struct ModelInfo {
    offset: u32,
    stride: u32,
}

#[derive(IntoWgslStruct)]
pub struct RectangleBounds {
    min: [f32; 3],
    max: [f32; 3],
}

#[shader(compute(workgroup_sz = 256))]
fn compute_rects(
    #[binding(storage(read_only = true))] model_verts: BindedBuffer<Vertex, 0>,
    #[binding(storage(read_only = true))] model_infos: BindedBuffer<ModelInfo, 1>,
    #[binding(storage(read_only = false))] output_rect: BindedBuffer<RectangleBounds, 2>,
    sdata_min: SharedData<[f32; 3]>,
    sdata_max: SharedData<[f32; 3]>,
) -> Scope {
    let mut scope = Scope::new();

    let lid = scope.var("lid", entrypoint(1).f("x").load());
    let model_idx = scope.var("model_idx", entrypoint(0).f("x").load());
    let info = scope.var(
        "info",
        model_infos.var_ref().i(local(model_idx).load()).load(),
    );
    let model_offset = scope.var("model_offset", local(info).f("offset").load());
    let model_vertex_count = scope.var("model_vertex_count", local(info).f("stride").load());
    let local_min = scope.mut_(
        "local_min",
        vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY),
    );
    let local_max = scope.mut_(
        "local_max",
        vec3(-f32::INFINITY, -f32::INFINITY, -f32::INFINITY),
    );
    let i = scope.mut_("i", local(lid).load());
    let offset = scope.mut_("offset", u32(128));

    let body = group!(
        scope.while_loop(local(model_vertex_count).load().gt(local(i).load()), |b| {
            let v = b.var(
                "v",
                model_verts
                    .var_ref()
                    .i(local(model_offset).load() + local(i).load())
                    .f("pos")
                    .load(),
            );
            b.ast = Some(group!(
                local(local_min).store(call!("min", local(local_min).load(), local(v).load())),
                local(local_max).store(call!("max", local(local_max).load(), local(v).load())),
                local(i).store(local(i).load() + u32(256)),
            ));
        },),
        sdata_min
            .var_ref()
            .i(local(lid).load())
            .store(local(local_min).load()),
        sdata_max
            .var_ref()
            .i(local(lid).load())
            .store(local(local_max).load()),
        call!("workgroupBarrier"),
        scope.while_loop(local(offset).load().gt(u32(0)), |b| {
            let if_ast = b.cond(
                local(offset).load().gt(local(lid).load()),
                |ib| {
                    ib.ast = Some(group!(
                        sdata_min.var_ref().i(local(lid).load()).store(call!(
                            "min",
                            sdata_min.var_ref().i(local(lid).load()).load(),
                            sdata_min
                                .var_ref()
                                .i(local(lid).load() + local(offset).load())
                                .load(),
                        )),
                        sdata_max.var_ref().i(local(lid).load()).store(call!(
                            "max",
                            sdata_max.var_ref().i(local(lid).load()).load(),
                            sdata_max
                                .var_ref()
                                .i(local(lid).load() + local(offset).load())
                                .load(),
                        )),
                    ));
                },
                None::<fn(&mut Scope)>,
            );
            b.ast = Some(group!(
                if_ast,
                call!("workgroupBarrier"),
                local(offset).store(local(offset).load() >> u32(1)),
            ));
        }),
        scope.cond(
            local(lid).load().eq(u32(0)),
            |b| {
                b.ast = Some(group!(
                    output_rect
                        .var_ref()
                        .i(local(model_idx).load())
                        .f("min")
                        .store(sdata_min.var_ref().i(u32(0)).load()),
                    output_rect
                        .var_ref()
                        .i(local(model_idx).load())
                        .f("max")
                        .store(sdata_max.var_ref().i(u32(0)).load()),
                ));
            },
            None::<fn(&mut Scope)>
        ),
    );
    scope.ast = Some(body);

    scope
}

#[test]
fn test_compute_rects() {
    let s = ComputeRects;
    let wgsl = s.load_source_code();
    println!("{wgsl}");
    assert!(wgsl.contains("var<workgroup>"));
    assert!(wgsl.contains("@compute @workgroup_size(256)"));
    assert!(wgsl.contains("fn compute_rects"));
}
