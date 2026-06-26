struct Key{
    code: u32,
    idx: u32
}

struct RectangleBounds{
    min: vec3<f32>,
    max: vec3<f32>
}

@group(0) @binding(0) var<storage, read> in_rects: array<RectangleBounds>;
@group(0) @binding(1) var<storage, read_write> out_keys: array<Key>;
@group(0) @binding(2) var<uniform> global_bounds: RectangleBounds;
@group(0) @binding(3) var<uniform> num_rects: u32;

fn mortonize(val: u32) -> u32 {
    var x = val & 0x000003ffu;
    x = (x | (x << 16u)) & 0xff0000ffu;
    x = (x | (x << 8u))  & 0x0300f00fu;
    x = (x | (x << 4u))  & 0x030c30c3u;
    x = (x | (x << 2u))  & 0x09249249u;
    return x;
}

@compute @workgroup_size(256)
fn generate_morton_keys(@builtin(global_invocation_id) id: vec3<u32>){
    let idx = id.x;
    if (idx >= num_rects){return;}

    let rect = in_rects[idx];
    let centroid = (rect.min + rect.max) * 0.5;
    let sz = global_bounds.max - global_bounds.min;

    var inv_sz = vec3<f32>(0.0);
    if (sz.x > 0.0){inv_sz.x = 1.0/sz.x;}
    if (sz.y > 0.0){inv_sz.y = 1.0/sz.y;}
    if (sz.z > 0.0){inv_sz.z = 1.0/sz.z;}

    let norm = (centroid - global_bounds.min) * inv_sz;
    let quant = clamp(norm, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0;

    let u = vec3<u32>(quant);
    let code = (mortonize(u.x) << 2u) | (mortonize(u.y) << 1u) | mortonize(u.z);

    out_keys[idx] = Key(code, idx);
}