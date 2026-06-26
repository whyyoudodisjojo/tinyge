struct RectangleBounds{
    min: vec3<f32>,
    max: vec3<f32>
}

@group(0) @binding(0) var<storage, read>  in_rects: array<RectangleBounds>;
@group(0) @binding(1) var<storage, read_write> out_reduced: array<RectangleBounds>;
@group(0) @binding(2) var<uniform>        total_count: u32;

var<workgroup> shared_min: array<vec3<f32>, 256>;
var<workgroup> shared_max: array<vec3<f32>, 256>;

@compute @workgroup_size(256)
fn compute_global_bounds(@builtin(global_invocation_id) g_id: vec3<u32>, @builtin(local_invocation_id) l_id: vec3<u32>, @builtin(workgroup_id) w_id: vec3<u32>){
    let lid = l_id.x;
    let gid1 = g_id.x * 2u;
    let gid2 = gid1 + 1u;

    var local_min = vec3<f32>(1e30);
    var local_max = vec3<f32>(-1e30);

    if (gid1 < total_count){
        local_min = min(local_min, in_rects[gid1].min);
        local_max = max(local_max, in_rects[gid1].max);
    }

    if (gid2 < total_count){
        local_min = min(local_min, in_rects[gid2].min);
        local_max = max(local_max, in_rects[gid2].max);
    }

    shared_min[lid] = local_min;
    shared_max[lid] = local_max;
    workgroupBarrier();

    for(var s = 128u; s>0u; s>>= 1u){
        if (lid < s){
            shared_min[lid] = min(shared_min[lid], shared_min[lid + s]);
            shared_max[lid] = max(shared_max[lid], shared_max[lid+s]);
        }
        workgroupBarrier();
    }

    if (lid == 0u){
        out_reduced[w_id.x].min = vec3<f32>(shared_min[0]);
        out_reduced[w_id.x].max = vec3<f32>(shared_max[0]);
    }
}