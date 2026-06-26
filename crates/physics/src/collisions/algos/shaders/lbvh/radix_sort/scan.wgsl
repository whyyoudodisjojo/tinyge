struct Params {
    num_elems: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> counters: array<u32>;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn radix_scan(@builtin(local_invocation_id) l_id: vec3<u32>) {
    let lid = l_id.x;
    let num_workgroups = (params.num_elems + 255u) / 256u;
    let n = 16u * num_workgroups;
    
    let idx1 = lid * 2u;
    let idx2 = idx1 + 1u;
    
    if (idx1 < n) {
        temp[idx1] = counters[idx1];
    } else {
        temp[idx1] = 0u;
    }
    
    if (idx2 < n) {
        temp[idx2] = counters[idx2];
    } else {
        temp[idx2] = 0u;
    }
    workgroupBarrier();
    
    var offset = 1u;
    var d = n >> 1u;
    while (d > 0u) {
        if (lid < d) {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        offset *= 2u;
        d >>= 1u;
        workgroupBarrier();
    }
    
    if (lid == 0u) {
        temp[n - 1u] = 0u;
    }
    workgroupBarrier();
    
    d = 1u;
    while (d < n) {
        offset >>= 1u;
        if (lid < d) {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        d *= 2u;
        workgroupBarrier();
    }
    
    if (idx1 < n) {
        counters[idx1] = temp[idx1];
    }
    if (idx2 < n) {
        counters[idx2] = temp[idx2];
    }
}
