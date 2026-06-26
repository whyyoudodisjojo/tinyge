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
@group(0) @binding(2) var<storage, read> global_offsets: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn radix_scatter(
    @builtin(global_invocation_id) g_id: vec3<u32>,
    @builtin(local_invocation_id) l_id: vec3<u32>,
    @builtin(workgroup_id) w_id: vec3<u32>
) {
    let lid = l_id.x;
    let gid = g_id.x;
    let num_workgroups = (params.num_elems + 255u) / 256u;

    if (gid >= params.num_elems) {
        return;
    }

    let key = in_keys[gid];
    let digit = (key.code >> params.shift_bits) & 0xFu;

    var local_offset = 0u;
    for (var j = 0u; j < lid; j++) {
        let other_idx = w_id.x * 256u + j;
        if (other_idx < params.num_elems) {
            let other_digit = (in_keys[other_idx].code >> params.shift_bits) & 0xFu;
            if (other_digit == digit) {
                local_offset++;
            }
        }
    }

    let g_bucket_idx = digit * num_workgroups + w_id.x;
    let base_offset = global_offsets[g_bucket_idx];

    let target_index = base_offset + local_offset;

    out_keys[target_index] = key;
}
