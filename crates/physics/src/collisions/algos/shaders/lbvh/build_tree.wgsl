struct Key{
    code: u32,
    idx: u32
}

struct RectangleBounds{
    min: vec3<f32>,
    max: vec3<f32>
}

struct BVHNode{
    rect: RectangleBounds,
    left_child: i32,
    right_child: i32,
    is_leaf: u32
}

@group(0) @binding(0) var<storage, read> in_keys: array<Key>;
@group(0) @binding(1) var<storage, read> in_rects: array<RectangleBounds>;
@group(0) @binding(2) var<storage, read_write> out_nodes: array<BVHNode>;
@group(0) @binding(3) var<storage, read_write> node_counters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> parent_indices: array<i32>;
@group(0) @binding(5) var<uniform> num_rects: u32;

fn leading_zeros_cmp(i: i32, j: i32, num_rects: i32) -> i32{
    if (j < 0 || j >= num_rects){
        return -1;
    }

    let c_i = in_keys[i];
    let c_j = in_keys[j];

    if (c_i.code == c_j.code && c_i.idx == c_j.idx){
        return 32 + i32(u32(i ^ j));
    }

    return i32(u32(i ^ j));
}

@compute @workgroup_size(256)
fn build_tree(@builtin(global_invocation_id) id: vec3<u32>){
    let idx = i32(id.x);
    let num_rects = i32(num_rects);
    let num_internal_nodes = num_rects - 1;

    if (idx >= num_internal_nodes){return;}

    var d: i32 = 1;
    if((leading_zeros_cmp(idx, idx+1, num_rects) - leading_zeros_cmp(idx, idx-1, num_rects)) < 0){d = -1;}

    let d_min = leading_zeros_cmp(idx, idx-d, num_rects);
    var l_max: i32 = 2;
    while (leading_zeros_cmp(idx, idx + l_max * d, num_rects) >  d_min){l_max *= 2;}

    var l: i32 = 0;
    var div: i32 = l_max/2;
    while (div > 0){
        if (leading_zeros_cmp(idx, idx + (l + div) * d, num_rects) > d_min){l += div;}
        div /= 2;
    }
    let j = idx + l * d;

    let d_node = leading_zeros_cmp(idx, j, num_rects);
    var s: i32 = 0;
    var div_s = (l + 1)/2;
    while (div_s > 0){
        if (leading_zeros_cmp(idx, idx + (s + div_s) * d, num_rects) > d_node){s += div_s;};
        if(div_s == 1){break;}
        div_s = (div_s + 1) / 2;
    }
    let split = idx + s * d + min(d, 0);

    var l_idx = split;
    if (min(idx, j) == split){l_idx = split + num_internal_nodes;}

    var r_idx = split + 1;
    if (max(idx, j) == split + 1){r_idx = split + 1 + num_internal_nodes;}

    out_nodes[idx].left_child = l_idx;
    out_nodes[idx].right_child = r_idx;
    out_nodes[idx].is_leaf = 0u;

    if (min(idx, j) == split){
        let leaf_rect_idx = in_keys[l_idx - num_internal_nodes].idx;
        out_nodes[l_idx].rect = in_rects[leaf_rect_idx];
        out_nodes[l_idx].right_child = i32(leaf_rect_idx);
        out_nodes[l_idx].is_leaf = 1u;
    }

    if (max(idx, j) == split + 1){
        let leaf_rect_idx = in_keys[r_idx - num_internal_nodes].idx;
        out_nodes[r_idx].rect = in_rects[leaf_rect_idx];
        out_nodes[r_idx].right_child = i32(leaf_rect_idx);
        out_nodes[r_idx].is_leaf = 1u;
    }
}

fn merge_rects(a: RectangleBounds, b: RectangleBounds) -> RectangleBounds{
    var res: RectangleBounds;
    res.min = min(a.min, b.min);
    res.max = min(a.max, b.max);
    return res;
}

@compute @workgroup_size(256)
fn compute_bounds(@builtin(global_invocation_id) id: vec3<u32>){
    let leaf_idx = i32(id.x);
    let num_rects = i32(num_rects);
    if (leaf_idx >= num_rects){return;}

    let num_internal_nodes = num_rects -1;
    var current_node = leaf_idx + num_internal_nodes;

    while (current_node != 0){
        let parent = parent_indices[current_node];
        let cnt = atomicAdd(&node_counters[parent], 1u); // TODO: Nuke atomics
        
        if (cnt == 0u){return;}

        let l = out_nodes[parent].left_child;
        let r = out_nodes[parent].right_child;

        out_nodes[parent].rect = merge_rects(out_nodes[l].rect, out_nodes[r].rect);

        current_node = parent;
    }
}

