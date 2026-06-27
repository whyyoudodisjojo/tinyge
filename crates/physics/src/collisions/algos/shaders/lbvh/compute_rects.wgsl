struct RectangleBounds {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct AtomicRectangleBounds {
    min_x: atomic<i32>,
    min_y: atomic<i32>,
    min_z: atomic<i32>,
    max_x: atomic<i32>,
    max_y: atomic<i32>,
    max_z: atomic<i32>,
}

struct ModelInfo {
    offset: u32,
    stride: u32,
}

@group(0) @binding(0) var<storage, read> model_verts: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read> model_infos: array<ModelInfo>;
@group(0) @binding(2) var<storage, read_write> output_rect_atomic: array<AtomicRectangleBounds>;
@group(0) @binding(3) var<storage, read_write> output_rect: array<RectangleBounds>;
@group(0) @binding(4) var<uniform> num_models: u32; 

const INF = 3.402823466e+38f;

var<workgroup> sdata_min: array<vec3<f32>, 256>;
var<workgroup> sdata_max: array<vec3<f32>, 256>;

fn f32toi32(f: f32) -> i32 {
    let i = bitcast<i32>(f);
    return select(i, i ^ 0x7FFFFFFF, i < 0);
}

fn i32tof32(i: i32) -> f32 {
    let decoded = select(i, i ^ 0x7FFFFFFF, i < 0);
    return bitcast<f32>(decoded);
}

@compute @workgroup_size(256)
fn compute_rects(
    @builtin(local_invocation_id) l_id: vec3<u32>,
    @builtin(workgroup_id) w_id: vec3<u32> 
) {
    let lid = l_id.x;
    let model_idx = w_id.x;

    let info = model_infos[model_idx];
    let model_offset = info.offset;
    let model_vertex_count = info.stride;

    var local_min = vec3<f32>(INF);
    var local_max = vec3<f32>(-INF);

    var i = lid;
    while (i < model_vertex_count) {
        let vert = model_verts[model_offset + i];
        local_min = min(local_min, vert);
        local_max = max(local_max, vert);
        i += 256u;
    }

    sdata_min[lid] = local_min;
    sdata_max[lid] = local_max;
    workgroupBarrier();

    for (var s = 256u / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            sdata_min[lid] = min(sdata_min[lid], sdata_min[lid + s]);
            sdata_max[lid] = max(sdata_max[lid], sdata_max[lid + s]);
        }
        workgroupBarrier();
    }    

    if (lid == 0u) {
        atomicMin(&output_rect_atomic[model_idx].min_x, f32toi32(sdata_min[0].x));
        atomicMin(&output_rect_atomic[model_idx].min_y, f32toi32(sdata_min[0].y));
        atomicMin(&output_rect_atomic[model_idx].min_z, f32toi32(sdata_min[0].z));

        atomicMax(&output_rect_atomic[model_idx].max_x, f32toi32(sdata_max[0].x));
        atomicMax(&output_rect_atomic[model_idx].max_y, f32toi32(sdata_max[0].y));
        atomicMax(&output_rect_atomic[model_idx].max_z, f32toi32(sdata_max[0].z));
    }
}

@compute @workgroup_size(64)
fn convert_atomic_bounds_to_bounds(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let model_idx = g_id.x;
    if (model_idx >= num_models) { return; }

    let min_x = atomicLoad(&output_rect_atomic[model_idx].min_x);
    let min_y = atomicLoad(&output_rect_atomic[model_idx].min_y);
    let min_z = atomicLoad(&output_rect_atomic[model_idx].min_z);

    let max_x = atomicLoad(&output_rect_atomic[model_idx].max_x);
    let max_y = atomicLoad(&output_rect_atomic[model_idx].max_y);
    let max_z = atomicLoad(&output_rect_atomic[model_idx].max_z);

    output_rect[model_idx].min = vec3<f32>(i32tof32(min_x), i32tof32(min_y), i32tof32(min_z));
    output_rect[model_idx].max = vec3<f32>(i32tof32(max_x), i32tof32(max_y), i32tof32(max_z));
}
