struct RectangleBounds {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct Key {
    code: u32, 
    idx: u32,  
}

struct BVHNode {
    min: vec3<f32>,
    max: vec3<f32>,
    parent: i32,
    left_child: i32,
    right_child: i32,
    node_type: u32, 
}

struct Params{
    num_leaves: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<Key>;
@group(0) @binding(1) var<storage, read> rects: array<RectangleBounds>;
@group(0) @binding(2) var<storage, read_write> nodes: array<BVHNode>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

fn merge(a_min: vec3<f32>, a_max: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> RectangleBounds {
    var res: RectangleBounds;
    res.min= min(a_min, b_min);
    res.max= max(a_max, b_max);
    return res;
}

fn delta(i: i32, j: i32) -> i32 {
    let num_leaves = i32(params.num_leaves);
    if (j < 0 || j >= num_leaves) { return -1; }
    
    let code_i = keys[i].code;
    let code_j = keys[j].code;
    
    if (code_i == code_j) {
        return i32(countLeadingZeros(code_i ^ code_j) + countLeadingZeros(u32(i ^ j)));
    }
    return i32(countLeadingZeros(code_i ^ code_j));
}

fn determine_range(idx: i32) -> vec2<i32> {
    let num_leaves = i32(params.num_leaves);
    
    var d = 1;
    if ((delta(idx, idx + 1) - delta(idx, idx - 1)) < 0) {
        d = -1;
    }
    
    let delta_min = delta(idx, idx - d);
    var l_max = 2;
    while (delta(idx, idx + l_max * d) > delta_min) {
        l_max *= 2;
    }
    
    var l = 0;
    var div = l_max / 2;
    while (div > 0) {
        if (delta(idx, idx + (l + div) * d) > delta_min) {
            l += div;
        }
        div /= 2;
    }
    let j = idx + l * d;
    
    return vec2<i32>(min(idx, j), max(idx, j));
}

fn find_split(start: i32, end: i32) -> i32 {
    let delta_node = delta(start, end);
    var split = start;
    var step = end - start;
    
    loop {
        step = (step + 1) >> 1;
        let new_split = split + step;
        if (new_split < end) {
            if (delta(start, new_split) > delta_node) {
                split = new_split;
            }
        }
        if (step <= 1) { break; }
    }
    return split;
}

@compute @workgroup_size(256)
fn build_leaves(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let gid = i32(g_id.x);
    if (gid >= i32(params.num_leaves)) { return; }

    let orig_idx = keys[gid].idx;
    let leaf_pos = gid; 

    nodes[leaf_pos].min = rects[orig_idx].min;
    nodes[leaf_pos].max = rects[orig_idx].max;
    nodes[leaf_pos].node_type = 0u;
    nodes[leaf_pos].left_child = -1;
    nodes[leaf_pos].right_child = -1;
    nodes[leaf_pos].parent = -1;
}

@compute @workgroup_size(256)
fn build_structure(@builtin(global_invocation_id) g_id: vec3<u32>) {
    let gid = i32(g_id.x);
    let num_leaves = i32(params.num_leaves);
    if (gid >= num_leaves - 1) { return; }

    let range = determine_range(gid);
    
    let split = find_split(range.x, range.y);

    var left_child = split;
    if (split != range.x) {
        left_child = num_leaves + split;
    }
    
    var right_child = split + 1;
    if (split + 1 != range.y) {
        right_child = num_leaves + (split + 1);
    }

    let internal_pos = num_leaves + gid;
    nodes[internal_pos].left_child = left_child;
    nodes[internal_pos].right_child = right_child;
    nodes[internal_pos].node_type = 1u;

    nodes[left_child].parent = internal_pos;
    nodes[right_child].parent = internal_pos;
}

@compute @workgroup_size(256)
fn compute_bounds(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = i32(global_id.x);
    let num_leaves = i32(params.num_leaves);
    if (gid >= num_leaves) { return; }

    var current = nodes[gid].parent;

    while (current != -1) {
        let count_idx = current - num_leaves; 
        
        let old_counter = atomicAdd(&counts[count_idx], 1u);
        
        if (old_counter == 0u) {
            return; 
        }

        let l = nodes[current].left_child;
        let r = nodes[current].right_child;
        
        let combined = merge(nodes[l].min, nodes[l].max, nodes[r].min, nodes[r].max);
        nodes[current].min = combined.min;
        nodes[current].max = combined.max;

        current = nodes[current].parent;
    }
}
