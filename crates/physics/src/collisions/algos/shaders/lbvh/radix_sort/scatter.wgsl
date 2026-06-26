struct Key{
    code: u32,
    idx: u32
}

struct Params{
    num_elems: u32,
    shift_bits: u32
}

@group(0) @binding(0) var<storage, read> in_keys: array<Key>;
@group(0) @binding(1) var<storage, read_write> out_keys: array<Key>;
@group(0) @binding(2) var<storage, read_write> global_counters: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> scatter_shared: array<u32, 256>;

@compute @workgroup_size(256)
fn radix_scatter(
    @builtin(global_invocation_id) g_id: vec3<u32>,
    @builtin(local_invocation_id) l_id: vec3<u32>,
    @builtin(workgroup_id) w_id: vec3<u32>
) {
    let lid = l_id.x;
    let gid = g_id.x;
    let num_workgroups = (params.num_elems+ 255u) / 256u;

    var key = Key(0u, 0u);
    var digit = 0u;

    if (gid < params.num_elems) {
        key = in_keys[gid];
        digit = (key.code >> params.shift_bits) & 0xFu;
    }

    var local_offset = 0u;
    for (var j = 0u; j < lid; j++) {
        if (gid < params.num_elems) {
            let other_digit = (in_keys[w_id.x * 256u + j].code >> params.shift_bits) & 0xFu;
            if (other_digit == digit) {
                local_offset++;
            }
        }
    }

    let g_bucket_idx = digit * num_workgroups + w_id.x;
    let g_base_offset = global_counters[g_bucket_idx];

    let target_index = g_base_offset + local_offset;

    if (gid < params.num_elems) {
        out_keys[target_index] = key;
    }
}