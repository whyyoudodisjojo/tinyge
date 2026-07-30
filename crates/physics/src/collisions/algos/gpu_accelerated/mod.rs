use memory::{
    buffers::{AccelerationStructures, BufferWithType, ResourceType},
    socket::{TinyBlas, TinyTlas},
};
use tinyge_graphics::shaders::descriptors::{
    ResourceBinding, ResourceBindingType, ResourceGroupLayout,
};
use wgpu::{BufferUsages, ComputePassDescriptor, Device, ShaderStages};

pub struct AccelerationShader {
    pub num_rays: u32,
    pub max_candidates: u32,
    pub max_instances: u32,
    pub blas_vertex_count: u32,
    pub gpu_ray_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, codegen_macros::IntoWgslStruct)]
pub struct RawCandidate {
    pub ray_idx: u32,
    pub instance_index: u32,
    pub primitive_index: u32,
    pub geometry_index: u32,
    pub barycentrics: [f32; 2],
    pub t: f32,
    pub _pad: u32,
}

#[derive(codegen_macros::IntoBufferStruct)]
pub struct AccelerationArgs<'a> {
    #[resource(bindgroup_index = 0, resource_index = 0, ty = "acceleration_structure")]
    pub acc: AccelerationStructures<'a>,
    #[resource(bindgroup_index = 0, resource_index = 1, ty = "buffer")]
    pub rays_buffer: BufferWithType<Vec<crate::collisions::algos::GpuRay>>,
    #[resource(bindgroup_index = 0, resource_index = 2, ty = "buffer")]
    pub candidates_buffer: BufferWithType<Vec<RawCandidate>>,
    #[resource(bindgroup_index = 0, resource_index = 3, ty = "buffer")]
    pub counter_buffer: BufferWithType<u32>,
    #[resource(bindgroup_index = 0, resource_index = 4, ty = "buffer")]
    pub num_rays_buffer: BufferWithType<u32>,
    #[resource(bindgroup_index = 0, resource_index = 5, ty = "buffer")]
    pub max_candidates_buffer: BufferWithType<u32>,
}

impl<'a> tinyge_graphics::shaders::ComputeShader<'a> for AccelerationShader {
    type Args = AccelerationArgs<'a>;
    type Ret = ();

    fn resource_buffers_with_bind_group_layouts(
        &self,
    ) -> Vec<tinyge_graphics::shaders::descriptors::ResourceGroupLayout<'a>> {
        vec![ResourceGroupLayout {
            entries: vec![
                ResourceBinding {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::AccelerationStructure {
                        tlas_desc: wgpu::wgt::CreateTlasDescriptor {
                            label: None,
                            max_instances: self.max_instances,
                            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                        },
                        blas_desc: wgpu::wgt::CreateBlasDescriptor {
                            label: None,
                            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                        },
                        blas_geo_sz_desc: wgpu::BlasGeometrySizeDescriptors::Triangles {
                            descriptors: vec![wgpu::BlasTriangleGeometrySizeDescriptor {
                                vertex_format: wgpu::VertexFormat::Float32x3,
                                vertex_count: self.blas_vertex_count,
                                index_format: None,
                                index_count: None,
                                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                            }],
                        },
                        vertex_return: false,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: self.gpu_ray_size * self.num_rays as u64,
                        usages: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: (32 * self.max_candidates) as u64,
                        usages: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: 4,
                        usages: BufferUsages::STORAGE
                            | BufferUsages::COPY_DST
                            | BufferUsages::COPY_SRC,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: 4,
                        usages: BufferUsages::UNIFORM
                            | BufferUsages::STORAGE
                            | BufferUsages::COPY_DST,
                    },
                    count: None,
                },
                ResourceBinding {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: ResourceBindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        size: 4,
                        usages: BufferUsages::UNIFORM
                            | BufferUsages::STORAGE
                            | BufferUsages::COPY_DST,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn load_source_code(&self) -> String {
        include_str!("../shaders/acceleration.wgsl").to_string()
    }

    fn entry_point(&self) -> &'static str {
        "traverse"
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        build_data: &mut tinyge_graphics::shaders::ComputeShaderBuiltData<Self::Args>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        let bind_group_resources = vec![
            ResourceType::AccelerationStructure(args.acc.tlas),
            ResourceType::Buffer(args.rays_buffer.inner),
            ResourceType::Buffer(args.candidates_buffer.inner),
            ResourceType::Buffer(args.counter_buffer.inner),
            ResourceType::Buffer(args.num_rays_buffer.inner),
            ResourceType::Buffer(args.max_candidates_buffer.inner),
        ];

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_bind_group(
                0,
                build_data.bind_groups[0].get_or_create_bind_group(&bind_group_resources, device),
                &[],
            );
            pass.set_pipeline(&build_data.pipeline);
            pass.dispatch_workgroups((self.num_rays + 255) / 256, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use tinyge_graphics::shaders::ComputeShaderWrapper;
    use wgpu::util::DeviceExt;

    use super::*;

    async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
                required_limits: wgpu::Limits {
                    max_blas_primitive_count: 3,
                    max_blas_geometry_count: 1,
                    max_tlas_instance_count: 1,
                    max_acceleration_structures_per_shader_stage: 1,
                    ..wgpu::Limits::default()
                },
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device");

        (device, queue)
    }

    #[test]
    fn test_gpu_acceleration_dispatch() {
        use crate::collisions::algos::test_utils::read_buffer;

        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;

            let num_rays = 8u32;
            let max_candidates = 16u32;

            let mut shader = ComputeShaderWrapper::new(
                AccelerationShader {
                    num_rays,
                    max_candidates,
                    max_instances: 1,
                    blas_vertex_count: 3,
                    gpu_ray_size: 48,
                },
                &device,
            );

            let rays: Vec<crate::collisions::algos::GpuRay> = vec![
                crate::collisions::algos::GpuRay {
                    origin: [0.0, 0.0, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                crate::collisions::algos::GpuRay {
                    origin: [10.0, 10.0, 0.0, 0.0],
                    dir: [1.0, 0.0, 0.0, 0.0],
                    inv_dir: [1.0, 0.0, 0.0, 0.0],
                },
                crate::collisions::algos::GpuRay {
                    origin: [0.8, -0.8, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                crate::collisions::algos::GpuRay {
                    origin: [0.0, 0.0, 1.0, 0.0],
                    dir: [0.0, 0.0, 1.0, 0.0],
                    inv_dir: [0.0, 0.0, 1.0, 0.0],
                },
                crate::collisions::algos::GpuRay {
                    origin: [0.0, 0.0, 2.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                crate::collisions::algos::GpuRay {
                    origin: [0.0, 0.0, -1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                crate::collisions::algos::GpuRay {
                    origin: [0.0, -0.5, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                crate::collisions::algos::GpuRay {
                    origin: [5.0, 5.0, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
            ];
            queue.write_buffer(
                shader
                    .built_data
                    .as_ref()
                    .unwrap()
                    .buffers
                    .rays_buffer
                    .inner
                    .raw(),
                0,
                bytemuck::cast_slice(&rays),
            );

            let tri_verts: Vec<[f32; 3]> =
                vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&tri_verts),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::BLAS_INPUT,
            });

            let raw_blas = device.create_blas(
                &wgpu::wgt::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![wgpu::BlasTriangleGeometrySizeDescriptor {
                        vertex_format: wgpu::VertexFormat::Float32x3,
                        vertex_count: 3,
                        index_format: None,
                        index_count: None,
                        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                    }],
                },
            );

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: 3,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };

            let blas_entry = wgpu::BlasBuildEntry {
                blas: &raw_blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &size_desc,
                        vertex_buffer: &vertex_buffer,
                        first_vertex: 0,
                        vertex_stride: 12,
                        index_buffer: None,
                        first_index: None,
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            };

            shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .acc
                .tlas
                .build(&device);
            shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .acc
                .tlas
                .raw_mut()[0] = Some(wgpu::TlasInstance::new(
                &raw_blas,
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                0,
                0xFF,
            ));

            encoder.build_acceleration_structures(
                &[blas_entry],
                std::iter::once(shader.built_data.as_ref().unwrap().buffers.acc.tlas.raw()),
            );
            queue.submit(std::iter::once(encoder.finish()));

            queue.write_buffer(
                shader
                    .built_data
                    .as_ref()
                    .unwrap()
                    .buffers
                    .num_rays_buffer
                    .inner
                    .raw(),
                0,
                bytemuck::bytes_of(&num_rays),
            );
            queue.write_buffer(
                shader
                    .built_data
                    .as_ref()
                    .unwrap()
                    .buffers
                    .max_candidates_buffer
                    .inner
                    .raw(),
                0,
                bytemuck::bytes_of(&max_candidates),
            );
            queue.write_buffer(
                shader
                    .built_data
                    .as_ref()
                    .unwrap()
                    .buffers
                    .counter_buffer
                    .inner
                    .raw(),
                0,
                bytemuck::bytes_of(&0u32),
            );

            shader.dispatch(
                AccelerationArgs {
                    acc: shader.built_data.as_ref().unwrap().buffers.acc.clone(),
                    rays_buffer: shader
                        .built_data
                        .as_ref()
                        .unwrap()
                        .buffers
                        .rays_buffer
                        .clone(),
                    candidates_buffer: shader
                        .built_data
                        .as_ref()
                        .unwrap()
                        .buffers
                        .candidates_buffer
                        .clone(),
                    counter_buffer: shader
                        .built_data
                        .as_ref()
                        .unwrap()
                        .buffers
                        .counter_buffer
                        .clone(),
                    num_rays_buffer: shader
                        .built_data
                        .as_ref()
                        .unwrap()
                        .buffers
                        .num_rays_buffer
                        .clone(),
                    max_candidates_buffer: shader
                        .built_data
                        .as_ref()
                        .unwrap()
                        .buffers
                        .max_candidates_buffer
                        .clone(),
                },
                &device,
                &queue,
            );

            let counter: Vec<u32> = read_buffer(
                &device,
                &queue,
                shader
                    .built_data
                    .as_ref()
                    .unwrap()
                    .buffers
                    .counter_buffer
                    .inner
                    .raw(),
            );
            println!("Counter value: {}", counter[0]);
            assert!(
                counter[0] >= 4,
                "Expected at least 4 hits, got {}",
                counter[0]
            );

            let candidates: Vec<RawCandidate> = read_buffer(
                &device,
                &queue,
                shader
                    .built_data
                    .as_ref()
                    .unwrap()
                    .buffers
                    .candidates_buffer
                    .inner
                    .raw(),
            );

            for i in 0..counter[0].min(candidates.len() as u32) as usize {
                let hit = &candidates[i];
                println!(
                    "Hit {}: ray_idx={}, instance={}, primitive={}, geometry={}, barycentrics=({},{}), t={}",
                    i,
                    hit.ray_idx,
                    hit.instance_index,
                    hit.primitive_index,
                    hit.geometry_index,
                    hit.barycentrics[0],
                    hit.barycentrics[1],
                    hit.t
                );
            }

            let hit = &candidates[0];
            assert_eq!(hit.instance_index, 0, "Instance index should be 0");
            assert!(
                hit.t > 0.9 && hit.t < 2.1,
                "t should be ~1.0-2.0, got {}",
                hit.t
            );
        });
    }
}
