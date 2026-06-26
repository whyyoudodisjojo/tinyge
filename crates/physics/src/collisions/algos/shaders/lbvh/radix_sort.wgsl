struct Params {
    shift: u32,
    num_elems: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> count_arr: array<u32, 16>;
@group(0) @binding(3) var<storage, read_write> output_arr: array<u32>;
@group(0) @binding(4) var<storage, read_write> global_offsets: array<atomic<u32>, 16>;

var<workgroup> local_counters: array<atomic<u32>, 16>;
var<workgroup> local_offsets: array<u32, 16>;
var<workgroup> shared_exchange: array<u32, 256>;

@compute @workgroup_size(256)
fn count(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let g_idx = global_id.x;
    let l_idx = local_id.x;

    if (l_idx < 16u) {
        atomicStore(&local_counters[l_idx], 0u);
    }
    workgroupBarrier();

    if (g_idx < params.num_elems) {
        let my_key = input_arr[g_idx];
        let my_digit = (my_key >> params.shift) & 15u;
        atomicAdd(&local_counters[my_digit], 1u);
    }
    workgroupBarrier();

    if (l_idx < 16u) {
        count_arr[l_idx] = atomicLoad(&local_counters[l_idx]);
    }
}

@compute @workgroup_size(1)
fn cumsum() {
    var sum = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let val = count_arr[i];
        count_arr[i] = sum;
        atomicStore(&global_offsets[i], sum);
        sum = sum + val;
    }
}

@compute @workgroup_size(256)
fn rearrange(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let g_idx = global_id.x;
    let l_idx = local_id.x;
    let wg_id = workgroup_id.x;

    if (l_idx < 16u) {
        atomicStore(&local_counters[l_idx], 0u);
    }
    workgroupBarrier();

    var my_key = 0u;
    var my_digit = 0u;
    if (g_idx < params.num_elems) {
        my_key = input_arr[g_idx];
        my_digit = (my_key >> params.shift) & 15u;
    }

    var local_rank = 0u;
    if (g_idx < params.num_elems) {
        local_rank = atomicAdd(&local_counters[my_digit], 1u);
    }
    workgroupBarrier();

    if (l_idx == 0u) {
        var sum = 0u;
        for (var i = 0u; i < 16u; i = i + 1u) {
            local_offsets[i] = sum;
            sum = sum + atomicLoad(&local_counters[i]);
        }
    }
    workgroupBarrier();

    if (g_idx < params.num_elems) {
        let wg_scatter_idx = local_offsets[my_digit] + local_rank;
        shared_exchange[wg_scatter_idx] = my_key;
    }
    workgroupBarrier();

    if (g_idx < params.num_elems) {
        let sorted_key = shared_exchange[l_idx];
        let digit = (sorted_key >> params.shift) & 15u;
        
        let global_pos = atomicAdd(&global_offsets[digit], 1u);
        output_arr[global_pos] = sorted_key;
    }
}
