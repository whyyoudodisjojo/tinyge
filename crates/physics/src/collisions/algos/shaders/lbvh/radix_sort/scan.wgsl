struct Params {
    num_elems: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> counters: array<u32>;

var<workgroup> scan_shared: array<u32, 512>;

@compute @workgroup_size(256)
fn radix_scan(@builtin(local_invocation_id) l_id: vec3<u32>) {
    let lid = l_id.x;
    
    let num_workgroups = (params.num_elems + 255u) / 256u;
    let total_elements = 16u * num_workgroups;

    let idx1 = lid * 2u;
    let idx2 = idx1 + 1u;

    if (idx1 < total_elements) { 
        scan_shared[idx1] = counters[idx1]; 
    } else { 
        scan_shared[idx1] = 0u; 
    }
    
    if (idx2 < total_elements) { 
        scan_shared[idx2] = counters[idx2]; 
    } else { 
        scan_shared[idx2] = 0u; 
    }
    workgroupBarrier();

    var active_threads = 256u;
    var stride = 1u;
    
    while (active_threads > 0u) {
        if (lid < active_threads) {
            let index = (lid * 2u + 1u) * stride - 1u;
            let source = index - stride;
            scan_shared[index] += scan_shared[source];
        }
        stride *= 2u;
        active_threads >>= 1u;
        workgroupBarrier();
    }

    if (lid == 0u) { 
        scan_shared[511u] = 0u; 
    }
    workgroupBarrier();

    active_threads = 1u;
    stride = 256u; 
    
    while (active_threads <= 256u) {
        if (lid < active_threads) {
            let index = (lid * 2u + 1u) * stride - 1u;
            let source = index - stride;
            
            let tmp = scan_shared[source];
            scan_shared[source] = scan_shared[index];
            scan_shared[index] += tmp;
        }
        stride >>= 1u;
        active_threads <<= 1u;
        workgroupBarrier();
    }

    if (idx1 < total_elements) { 
        counters[idx1] = scan_shared[idx1]; 
    }
    if (idx2 < total_elements) { 
        counters[idx2] = scan_shared[idx2]; 
    }
}
