enable wgpu_ray_query;

struct RawCandidate {
    ray_idx: u32,
    instance_index: u32,
    primitive_index: u32,
    geometry_index: u32,
    barycentrics: vec2<f32>,
    t: f32,
};

struct GpuRay {
    origin: vec4<f32>,
    dir: vec4<f32>,
    inv_dir: vec4<f32>,
};

@group(0) @binding(0) var tlas: acceleration_structure;

@group(0) @binding(1)
var<storage, read> rays: array<GpuRay>;

@group(0) @binding(2)
var<storage, read_write> candidates: array<RawCandidate>;

@group(0) @binding(3)
var<storage, read_write> counter: atomic<u32>;

@group(0) @binding(4) var<uniform> num_rays: u32;

@group(0) @binding(5) var<uniform> max_candidates: u32;

@compute @workgroup_size(256)
fn traverse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_idx = global_id.x;
    if ray_idx >= num_rays {
        return;
    }

    let ray = rays[ray_idx];

    var desc: RayDesc;
    desc.flags = 0u;
    desc.cull_mask = 0xFFu;
    desc.tmin = 0.0;
    desc.tmax = 1e30;
    desc.origin = ray.origin.xyz;
    desc.dir = ray.dir.xyz;

    var q: ray_query;
    rayQueryInitialize(&q, tlas, desc);

    while (rayQueryProceed(&q)) {
        let cand = rayQueryGetCandidateIntersection(&q);
        rayQueryConfirmIntersection(&q);
    }

    let hit = rayQueryGetCommittedIntersection(&q);
    if (hit.kind > 0u) {
        let slot = atomicAdd(&counter, 1u);
        if slot < max_candidates {
            candidates[slot] = RawCandidate(
                ray_idx,
                hit.instance_index,
                hit.primitive_index,
                hit.geometry_index,
                hit.barycentrics,
                hit.t,
            );
        }
    }
}
