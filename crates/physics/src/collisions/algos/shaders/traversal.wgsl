struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
};

struct BVHNode {
    min: vec4<f32>,
    max: vec4<f32>,
    parent: i32,
    left_child: i32,
    right_child: i32,
    node_type: u32,
};

struct RayResult {
    hit_node_idx: i32,
    t_near: f32,
    padding: vec2<f32>,
};

@group(0) @binding(0)
var<storage, read> rays: array<Ray>;

@group(0) @binding(1)
var<storage, read> nodes: array<BVHNode>;

@group(0) @binding(2)
var<storage, read_write> results: array<RayResult>;

@group(0) @binding(3)
var<uniform> num_rays: u32;

@group(0) @binding(4)
var<uniform> root_idx: u32;

fn ray_intersects(ray: Ray, box_min: vec3<f32>, box_max: vec3<f32>, t_max: f32) -> f32 {
    var t_min_axis: vec3<f32>;
    var t_max_axis: vec3<f32>;
    
    if abs(ray.dir.x) < 1e-6 {
        if ray.origin.x < box_min.x || ray.origin.x > box_max.x {
            return -1.0;
        }
        t_min_axis.x = -1e30;
        t_max_axis.x = 1e30;
    } else {
        let tx1 = (box_min.x - ray.origin.x) * ray.inv_dir.x;
        let tx2 = (box_max.x - ray.origin.x) * ray.inv_dir.x;
        t_min_axis.x = min(tx1, tx2);
        t_max_axis.x = max(tx1, tx2);
    }
    
    if abs(ray.dir.y) < 1e-6 {
        if ray.origin.y < box_min.y || ray.origin.y > box_max.y {
            return -1.0;
        }
        t_min_axis.y = -1e30;
        t_max_axis.y = 1e30;
    } else {
        let ty1 = (box_min.y - ray.origin.y) * ray.inv_dir.y;
        let ty2 = (box_max.y - ray.origin.y) * ray.inv_dir.y;
        t_min_axis.y = min(ty1, ty2);
        t_max_axis.y = max(ty1, ty2);
    }
    
    if abs(ray.dir.z) < 1e-6 {
        if ray.origin.z < box_min.z || ray.origin.z > box_max.z {
            return -1.0;
        }
        t_min_axis.z = -1e30;
        t_max_axis.z = 1e30;
    } else {
        let tz1 = (box_min.z - ray.origin.z) * ray.inv_dir.z;
        let tz2 = (box_max.z - ray.origin.z) * ray.inv_dir.z;
        t_min_axis.z = min(tz1, tz2);
        t_max_axis.z = max(tz1, tz2);
    }
    
    let t_min = max(max(t_min_axis.x, t_min_axis.y), t_min_axis.z);
    let t_far = min(min(t_max_axis.x, t_max_axis.y), t_max_axis.z);
    
    if t_min <= t_far && t_far >= 0.0 && t_min < t_max {
        return max(t_min, 0.0);
    }
    return -1.0;
}

@compute @workgroup_size(256)
fn traverse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    if ray_idx >= num_rays {
        return;
    }
    
    let ray = rays[ray_idx];
    var stack: array<i32, 64>;
    var stack_ptr: i32 = 0;
    var t_max: f32 = 1e30;
    var hit_idx: i32 = -1;
    var hit_t: f32 = -1.0;
    
    stack[stack_ptr] = i32(root_idx);
    stack_ptr = stack_ptr + 1;
    
    while stack_ptr > 0 {
        stack_ptr = stack_ptr - 1;
        let node_idx = stack[stack_ptr];
        
        if node_idx < 0 || node_idx >= i32(arrayLength(&nodes)) {
            continue;
        }
        
        let node = nodes[node_idx];
        let t_near = ray_intersects(ray, node.min.xyz, node.max.xyz, t_max);
        
        if t_near < 0.0 {
            continue;
        }
        
        if node.node_type == 0u {
            if t_near < t_max {
                t_max = t_near;
                hit_idx = node_idx;
                hit_t = t_near;
            }
        } else {
            let left_t = ray_intersects(ray, nodes[node.left_child].min.xyz, nodes[node.left_child].max.xyz, t_max);
            let right_t = ray_intersects(ray, nodes[node.right_child].min.xyz, nodes[node.right_child].max.xyz, t_max);
            
            if left_t >= 0.0 && right_t >= 0.0 {
                if left_t < right_t {
                    if stack_ptr < 63 {
                        stack[stack_ptr] = node.right_child;
                        stack_ptr = stack_ptr + 1;
                        stack[stack_ptr] = node.left_child;
                        stack_ptr = stack_ptr + 1;
                    }
                } else {
                    if stack_ptr < 63 {
                        stack[stack_ptr] = node.left_child;
                        stack_ptr = stack_ptr + 1;
                        stack[stack_ptr] = node.right_child;
                        stack_ptr = stack_ptr + 1;
                    }
                }
            } else if left_t >= 0.0 {
                if stack_ptr < 64 {
                    stack[stack_ptr] = node.left_child;
                    stack_ptr = stack_ptr + 1;
                }
            } else if right_t >= 0.0 {
                if stack_ptr < 64 {
                    stack[stack_ptr] = node.right_child;
                    stack_ptr = stack_ptr + 1;
                }
            }
        }
    }
    
    results[ray_idx].hit_node_idx = hit_idx;
    results[ray_idx].t_near = hit_t;
}
