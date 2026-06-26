struct Key {
    code: u32,
    idx: u32
}

struct Params {
    num_elems: u32,
    shift_bits: u32
}

@group(0) @binding(0) var<storage, read> in_keys: array<Key>;
@group(0) @binding(1) var<storage, read_write> global_counters: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> local_counters: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn count(
    @builtin(global_invocation_id) g_id: vec3<u32>, 
    @builtin(local_invocation_id) l_id: vec3<u32>, 
    @builtin(workgroup_id) w_id: vec3<u32>
) {
    let lid = l_id.x;
    let gid = g_id.x;

    if (lid < 16u) {
        atomicStore(&local_counters[lid], 0u);
    }
    workgroupBarrier();

    if (gid < params.num_elems) {
        let digit = (in_keys[gid].code >> params.shift_bits) & 0xFu;
        atomicAdd(&local_counters[digit], 1u);
    }
    workgroupBarrier();

    if (lid < 16u) {
        let num_workgroups = (params.num_elems + 255u) / 256u;
        let g_bucket_idx = lid * num_workgroups + w_id.x;

        global_counters[g_bucket_idx] = atomicLoad(&local_counters[lid]);
    }
}
