struct RectangleBounds {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct ModelInfo {
    offset: u32,
    stride: u32,
}

struct Vertex {
    pos: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> model_verts: array<Vertex>;
@group(0) @binding(1) var<storage, read> model_infos: array<ModelInfo>;
@group(0) @binding(2) var<storage, read_write> output_rect: array<RectangleBounds>;

const INF = 3.402823466e+38f;

var<workgroup> sdata_min: array<vec3<f32>, 256>;
var<workgroup> sdata_max: array<vec3<f32>, 256>;

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
        let vert = model_verts[model_offset + i].pos;
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
        output_rect[model_idx].min = sdata_min[0];
        output_rect[model_idx].max = sdata_max[0];
    }
}
