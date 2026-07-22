use codegen::asts::lowered::scope::*;
use codegen::asts::lowered::{BindedBuffer, LoweredAST, Scope};

use codegen::{call, group};
use codegen_macros::shader;
use tinyge_graphics::shaders::{ComputeShader, buffers::BufferWithType};

use crate::collisions::algos::{
    BVHNode, BVHTree, CpuBVHTraversal, CpuStorage, FlattenedBVHNode, GpuBVHTraversal, GpuStorage,
    Ray, RayResult, TraversalFlow,
};
use tinyge_graphics::shaders::ComputeShaderWrapper;

fn default_leaf_hit(
    s: &mut Scope,
    t_near: usize,
    t_max_var: usize,
    hit_idx: usize,
    hit_t: usize,
    node_idx: usize,
) -> LoweredAST {
    group!(
        s.if_(local(t_near).load().lt(local(t_max_var).load()), |_s| {
            group!(
                local(t_max_var).store(local(t_near).load());
                local(hit_idx).store(local(node_idx).load());
                local(hit_t).store(local(t_near).load());
            )
        })
    )
}

fn ray_intersects(
    s: &mut Scope,
    ray: usize,
    box_min: LoweredAST,
    box_max: LoweredAST,
    t_max: LoweredAST,
) -> (usize, LoweredAST) {
    let bmin = s.var(box_min);
    let bmax = s.var(box_max);
    let tx1 = s.var(
        (local(bmin).f("x").load() - local(ray).f("origin").f("x").load())
            * local(ray).f("inv_dir").f("x").load(),
    );
    let tx2 = s.var(
        (local(bmax).f("x").load() - local(ray).f("origin").f("x").load())
            * local(ray).f("inv_dir").f("x").load(),
    );
    let t_min_x = s.var(call!("min", local(tx1).load(), local(tx2).load()));
    let t_max_x = s.var(call!("max", local(tx1).load(), local(tx2).load()));
    let ty1 = s.var(
        (local(bmin).f("y").load() - local(ray).f("origin").f("y").load())
            * local(ray).f("inv_dir").f("y").load(),
    );
    let ty2 = s.var(
        (local(bmax).f("y").load() - local(ray).f("origin").f("y").load())
            * local(ray).f("inv_dir").f("y").load(),
    );
    let t_min_y = s.var(call!("min", local(ty1).load(), local(ty2).load()));
    let t_max_y = s.var(call!("max", local(ty1).load(), local(ty2).load()));
    let tz1 = s.var(
        (local(bmin).f("z").load() - local(ray).f("origin").f("z").load())
            * local(ray).f("inv_dir").f("z").load(),
    );
    let tz2 = s.var(
        (local(bmax).f("z").load() - local(ray).f("origin").f("z").load())
            * local(ray).f("inv_dir").f("z").load(),
    );
    let t_min_z = s.var(call!("min", local(tz1).load(), local(tz2).load()));
    let t_max_z = s.var(call!("max", local(tz1).load(), local(tz2).load()));
    let t_min = s.var(call!(
        "max",
        call!("max", local(t_min_x).load(), local(t_min_y).load()),
        local(t_min_z).load()
    ));
    let t_far = s.var(call!(
        "min",
        call!("min", local(t_max_x).load(), local(t_max_y).load()),
        local(t_max_z).load()
    ));
    let result = s.mut_((-1.0f32).into());
    let cond = s.if_(
        local(t_min)
            .load()
            .le(local(t_far).load())
            .logical_and(local(t_far).load().ge(0.0f32.into()))
            .logical_and(local(t_min).load().lt(t_max)),
        |_| local(result).store(call!("max", local(t_min).load(), 0.0f32.into())),
    );
    (result, cond)
}

impl CpuBVHTraversal for BVHTree<CpuStorage> {
    fn traverse_ray<F>(&self, ray: &Ray, mut callback: F)
    where
        F: FnMut(usize, f32, f32) -> TraversalFlow,
    {
        if self.storage.tree.is_empty() {
            return;
        }

        let mut stack = vec![self.storage.root_idx];
        let mut t_max_val = f32::INFINITY;

        while let Some(current_idx) = stack.pop() {
            let node = &self.storage.tree[current_idx];
            let Some(t_near) = ray.intersects_rect(node.rect(), t_max_val) else {
                continue;
            };

            match node {
                BVHNode::Leaf { idx, .. } => match callback(*idx, t_near, t_max_val) {
                    TraversalFlow::Break => return,
                    TraversalFlow::ContinueWithNewMax(new_max) => t_max_val = new_max,
                    TraversalFlow::Continue => {}
                },
                BVHNode::Internal {
                    left_child,
                    right_child,
                    ..
                } => {
                    let left_node = &self.storage.tree[*left_child];
                    let right_node = &self.storage.tree[*right_child];
                    let t_left = ray.intersects_rect(left_node.rect(), t_max_val);
                    let t_right = ray.intersects_rect(right_node.rect(), t_max_val);

                    match (t_left, t_right) {
                        (Some(tl), Some(tr)) => {
                            if tl < tr {
                                stack.push(*right_child);
                                stack.push(*left_child);
                            } else {
                                stack.push(*left_child);
                                stack.push(*right_child);
                            }
                        }
                        (Some(_), None) => stack.push(*left_child),
                        (None, Some(_)) => stack.push(*right_child),
                        (None, None) => {}
                    }
                }
            }
        }
    }
}

#[allow(unused_variables)]
#[shader(compute(workgroup_sz = 256))]
fn traverse(
    #[binding(storage(read_only = true))] rays: BindedBuffer<Vec<Ray>, 0>,
    #[binding(storage(read_only = true))] nodes: BindedBuffer<Vec<FlattenedBVHNode>, 1>,
    #[binding(storage(read_only = false))] results: BindedBuffer<Vec<RayResult>, 2>,
    #[binding(uniform)] num_rays: BindedBuffer<u32, 3>,
    #[binding(uniform)] root_idx: BindedBuffer<u32, 4>,
    leaf_hit: fn(&mut Scope, usize, usize, usize, usize, usize) -> LoweredAST,
    #[private] stack: [i32; 64],
) -> Scope {
    let mut scope = Scope::new();
    scope.add_local(
        "stack".into(),
        true,
        LoweredAST::Const {
            dt: <[i32; 64] as codegen::asts::IntoWgslStruct>::dt(),
            data: vec![],
        },
    );
    scope.num_inherited_locals = 1;

    let ray_idx = scope.var(entrypoint(0).f("x").load());
    let ray = scope.var(rays.var_ref().i(local(ray_idx).load()).load());

    let hit_idx = scope.mut_(0u32.into());
    let hit_t = scope.mut_((-1.0f32).into());
    let t_max_var = scope.mut_(1e30f32.into());
    let stack_ptr = scope.mut_(0u32.into());
    let node_idx = scope.mut_(0u32.into());

    let body = group!(
        scope.if_(local(ray_idx).load().ge(num_rays.var_ref().load()), |_| {
            LoweredAST::Return
        });
        local(stack_ptr).store(0u32.into());
        local(0).i(0u32.into()).store(call!("i32", root_idx.var_ref().load()));
        local(stack_ptr).store(local(stack_ptr).load() + 1u32.into());
        scope.while_loop(
            local(stack_ptr).load().gt(0u32.into()),
            |b| {
                let dec = local(stack_ptr).store(local(stack_ptr).load() - 1u32.into());
                let pop = local(node_idx).store(call!("u32", local(0).i(local(stack_ptr).load()).load()));
                let node = b.var(nodes.var_ref().i(local(node_idx).load()).load());
                let t_near = b.mut_((-1.0f32).into());
                let (ri, ri_cond) = ray_intersects(b, ray, local(node).f("min").f("xyz").load(), local(node).f("max").f("xyz").load(), local(t_max_var).load());
                let assign = local(t_near).store(local(ri).load());
                let miss = b.if_(local(t_near).load().lt(0.0f32.into()), |_| LoweredAST::Continue);
                let leaf = b.if_(local(node).f("node_type").load().eq(0u32.into()), |s| {
                    leaf_hit(s, t_near, t_max_var, hit_idx, hit_t, node_idx)
                });
                let lc = b.var(local(node).f("left_child").load());
                let rc = b.var(local(node).f("right_child").load());
                let lt = b.mut_((-1.0f32).into());
                let rt = b.mut_((-1.0f32).into());
                let (li, li_cond) = ray_intersects(b, ray, nodes.var_ref().i(local(lc).load()).f("min").f("xyz").load(), nodes.var_ref().i(local(lc).load()).f("max").f("xyz").load(), local(t_max_var).load());
                let al = local(lt).store(local(li).load());
                let (ri2, ri2_cond) = ray_intersects(b, ray, nodes.var_ref().i(local(rc).load()).f("min").f("xyz").load(), nodes.var_ref().i(local(rc).load()).f("max").f("xyz").load(), local(t_max_var).load());
                let ar = local(rt).store(local(ri2).load());
                let both = local(lt).load().ge(0.0f32.into()).logical_and(local(rt).load().ge(0.0f32.into()));
                let only_l = local(lt).load().ge(0.0f32.into()).logical_and(local(rt).load().lt(0.0f32.into()));
                let only_r = local(lt).load().lt(0.0f32.into()).logical_and(local(rt).load().ge(0.0f32.into()));

                group!(
                    dec;
                    pop;
                    ri_cond;
                    assign;
                    miss;
                    leaf;
                    li_cond;
                    al;
                    ri2_cond;
                    ar;
                    b.if_(both, |s| group!(
                        s.if_(local(lt).load().lt(local(rt).load()), |s| group!(
                            s.if_(local(stack_ptr).load().lt(63u32.into()), |_| group!(
                                local(0).i(local(stack_ptr).load()).store(local(rc).load());
                                local(stack_ptr).store(local(stack_ptr).load() + 1u32.into());
                                local(0).i(local(stack_ptr).load()).store(local(lc).load());
                                local(stack_ptr).store(local(stack_ptr).load() + 1u32.into());
                            ))
                        ));
                        s.if_(local(lt).load().ge(local(rt).load()), |s| group!(
                            s.if_(local(stack_ptr).load().lt(63u32.into()), |_| group!(
                                local(0).i(local(stack_ptr).load()).store(local(lc).load());
                                local(stack_ptr).store(local(stack_ptr).load() + 1u32.into());
                                local(0).i(local(stack_ptr).load()).store(local(rc).load());
                                local(stack_ptr).store(local(stack_ptr).load() + 1u32.into());
                            ))
                        ));
                    ));
                    b.if_(only_l, |s| group!(
                        s.if_(local(stack_ptr).load().lt(64u32.into()), |_| group!(
                            local(0).i(local(stack_ptr).load()).store(local(lc).load());
                            local(stack_ptr).store(local(stack_ptr).load() + 1u32.into());
                        ))
                    ));
                    b.if_(only_r, |s| group!(
                        s.if_(local(stack_ptr).load().lt(64u32.into()), |_| group!(
                            local(0).i(local(stack_ptr).load()).store(local(rc).load());
                            local(stack_ptr).store(local(stack_ptr).load() + 1u32.into());
                        ))
                    ));
                )
            },
        );
        results.var_ref().i(local(ray_idx).load()).f("hit_node_idx").store(call!("i32", local(hit_idx).load()));
        results.var_ref().i(local(ray_idx).load()).f("t_near").store(local(hit_t).load());
    );

    scope.ast = Some(body);
    scope
}

impl GpuBVHTraversal for BVHTree<GpuStorage> {
    fn traverse_gpu(
        &self,
        rays_buffer: &BufferWithType<Vec<Ray>>,
        num_rays: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> BufferWithType<RayResult> {
        use wgpu::{BufferDescriptor, BufferUsages};

        let shader = Traverse {
            leaf_hit: default_leaf_hit,
            stack: [0i32; 64],
            rays_elem_count: num_rays as u64,
            nodes_elem_count: self.storage.num_nodes as u64,
            results_elem_count: num_rays as u64,
        };

        let results_buffer: wgpu::Buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (num_rays as u64 * std::mem::size_of::<RayResult>() as u64).max(4),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let num_rays_buf: wgpu::Buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let root_idx_buf: wgpu::Buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        BufferWithType::<u32>::from(num_rays_buf.clone()).write(queue, &num_rays);
        BufferWithType::<u32>::from(root_idx_buf.clone())
            .write(queue, &(self.storage.root_idx as u32));

        let mut built_data = shader.build(device);
        let mut shader = shader;
        shader.dispatch(
            TraverseArgs {
                rays: rays_buffer.inner.clone().into(),
                nodes: self.storage.nodes_buffer.inner.clone().into(),
                results: results_buffer.clone().into(),
                num_rays: num_rays_buf.into(),
                root_idx: root_idx_buf.into(),
            },
            &mut built_data,
            device,
            queue,
        );

        BufferWithType::<RayResult>::from(results_buffer)
    }
}

#[test]
fn test_generated_wgsl() {
    let shader = Traverse {
        leaf_hit: default_leaf_hit,
        stack: [0i32; 64],
        rays_elem_count: 1024,
        nodes_elem_count: 1024,
        results_elem_count: 1024,
    };
    let source = shader.load_source_code();
    println!("{}", source);
    assert!(source.contains("var<private> stack"));
    assert!(source.contains("@compute @workgroup_size(256)"));
    assert!(source.contains("fn traverse"));
    assert!(source.contains("array<Ray>"));
    assert!(source.contains("array<FlattenedBVHNode>"));
    assert!(source.contains("array<RayResult>"));
}

#[test]
fn test_lbvh_traverse_gpu() {
    use crate::collisions::algos::test_utils::{read_buffer, setup_wgpu};
    use wgpu::util::DeviceExt;

    pollster::block_on(async {
        let (device, queue) = setup_wgpu().await;

        let num_models = 1u32;
        let num_verts = 3u32;

        let verts: Vec<[f32; 4]> = vec![
            [-1.0, -1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ];

        let model_verts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let model_infos: Vec<[u32; 2]> = vec![[0, 3]];

        let model_infos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&model_infos),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut builder = crate::collisions::algos::lbvh::gpu::custom::LBVHBuilder::new(
            num_models, num_verts, &device,
        );
        let bvh_tree: BVHTree<GpuStorage> = crate::collisions::algos::GpuCollisionAlgorithm::build(
            &mut builder,
            model_verts_buffer.into(),
            model_infos_buffer.into(),
            &device,
            &queue,
        );

        let rays: Vec<Ray> = vec![Ray::new(
            glam::Vec3A::new(0.0, 0.0, 1.0),
            glam::Vec3A::new(0.0, 0.0, -1.0),
        )];

        let rays_buffer: BufferWithType<Vec<Ray>> = device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&rays),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
            .into();

        let results_buffer = bvh_tree.traverse_gpu(&rays_buffer, 1, &device, &queue);

        let results: Vec<RayResult> = read_buffer(&device, &queue, &results_buffer.inner);

        println!(
            "Ray 0: hit_node_idx={}, t_near={}",
            results[0].hit_node_idx, results[0].t_near
        );
        assert!(
            results[0].hit_node_idx >= 0,
            "Expected a hit, got {}",
            results[0].hit_node_idx
        );
        assert!(
            (results[0].t_near - 1.0).abs() < 0.01,
            "Expected t_near ≈ 1.0, got {}",
            results[0].t_near
        );
    });
}
