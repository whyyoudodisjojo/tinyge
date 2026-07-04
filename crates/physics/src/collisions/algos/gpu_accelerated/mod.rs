use tinyge_graphics::shaders::{
    buffers::ResourceType,
    descriptors::{ResourceBinding, ResourceBindingType, ResourceGroupLayout},
};
use wgpu::{BufferUsages, ShaderStages};

pub struct AccelerationShader {
    num_rays: u32,
    max_candidates: u32,
    max_instances: u32,
    blas_vertex_count: u32,
    gpu_ray_size: u64,
}

impl AccelerationShader {
    pub fn new(
        num_rays: u32,
        max_candidates: u32,
        max_instances: u32,
        blas_vertex_count: u32,
        gpu_ray_size: u64,
    ) -> Self {
        Self {
            num_rays,
            max_candidates,
            max_instances,
            blas_vertex_count,
            gpu_ray_size,
        }
    }
}

pub struct AccelerationArgs {
    pub tlas: wgpu::Tlas,
    pub rays_buffer: wgpu::Buffer,
    pub candidates_buffer: wgpu::Buffer,
    pub counter_buffer: wgpu::Buffer,
    pub num_rays_buffer: wgpu::Buffer,
    pub max_candidates_buffer: wgpu::Buffer,
}

impl<'a> tinyge_graphics::shaders::ComputeShader<'a> for AccelerationShader {
    type Args = AccelerationArgs;
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
                        usages: BufferUsages::STORAGE,
                        is_input: true,
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
                        usages: BufferUsages::STORAGE,
                        is_input: false,
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
                        usages: BufferUsages::STORAGE,
                        is_input: false,
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
                        usages: BufferUsages::UNIFORM | BufferUsages::STORAGE,
                        is_input: false,
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
                        usages: BufferUsages::UNIFORM | BufferUsages::STORAGE,
                        is_input: false,
                    },
                    count: None,
                },
            ],
        }]
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("../shaders/acceleration.wgsl")
    }

    fn entry_point(&self) -> &'static str {
        "traverse"
    }

    fn dispatch(
        &mut self,
        args: Self::Args,
        build_data: &mut tinyge_graphics::shaders::ComputeShaderBuiltData,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self::Ret {
        use wgpu::ComputePassDescriptor;

        let bind_group_resources = vec![
            ResourceType::AccelerationStructure(args.tlas),
            ResourceType::Buffer(args.rays_buffer),
            ResourceType::Buffer(args.candidates_buffer),
            ResourceType::Buffer(args.counter_buffer),
            ResourceType::Buffer(args.num_rays_buffer),
            ResourceType::Buffer(args.max_candidates_buffer),
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
    use tinyge_graphics::shaders::{ComputeShaderWrapper, buffers::Buffers};
    use wgpu::util::DeviceExt;

    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct RawCandidate {
        ray_idx: u32,
        instance_index: u32,
        primitive_index: u32,
        geometry_index: u32,
        barycentrics: [f32; 2],
        t: f32,
        _pad: u32,
    }

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

    fn read_buffer<T: bytemuck::Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<T> {
        use std::sync::mpsc;

        let size = buffer.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });

        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    #[test]
    fn test_gpu_acceleration_dispatch() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;

            let num_rays = 2u32;
            let max_candidates = 16u32;

            let mut shader = ComputeShaderWrapper::new(
                AccelerationShader::new(num_rays, max_candidates, 1, 3, 48),
                &device,
            );

            let input_buffers =
                Buffers::build(&device, &shader.buffer_build_spec.buffer_build_spec, true);

            let rays_buffer = input_buffers.resource_buffers[0].buffers[1]
                .clone()
                .unwrap();
            let astructs = &input_buffers.resource_buffers[0].acceleration_structures[0];
            let blas = &astructs.blas;
            let mut tlas = astructs.tlas.clone();

            // Write ray data (shared GpuRay: 48 bytes each)
            let rays: Vec<crate::collisions::algos::GpuRay> = vec![
                crate::collisions::algos::GpuRay {
                    origin: [0.0, 0.0, 1.0],
                    _pad1: 0.0,
                    dir: [0.0, 0.0, -1.0],
                    _pad2: 0.0,
                    inv_dir: [0.0, 0.0, -1.0],
                    _pad3: 0.0,
                },
                crate::collisions::algos::GpuRay {
                    origin: [10.0, 10.0, 0.0],
                    _pad1: 0.0,
                    dir: [1.0, 0.0, 0.0],
                    _pad2: 0.0,
                    inv_dir: [1.0, 0.0, 0.0],
                    _pad3: 0.0,
                },
            ];
            queue.write_buffer(&rays_buffer, 0, bytemuck::cast_slice(&rays));

            let tri_verts: Vec<[f32; 3]> =
                vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&tri_verts),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::BLAS_INPUT,
            });

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
                blas,
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

            tlas[0] = Some(wgpu::TlasInstance::new(
                blas,
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                0,
                0xFF,
            ));

            encoder.build_acceleration_structures(&[blas_entry], std::iter::once(&tlas));
            queue.submit(std::iter::once(encoder.finish()));

            let internal_bufs =
                Buffers::build(&device, &shader.buffer_build_spec.buffer_build_spec, false);
            let candidates_buffer = internal_bufs.resource_buffers[0].buffers[2]
                .clone()
                .unwrap();
            let counter_buffer = internal_bufs.resource_buffers[0].buffers[3]
                .clone()
                .unwrap();
            let num_rays_buffer = internal_bufs.resource_buffers[0].buffers[4]
                .clone()
                .unwrap();
            let max_candidates_buffer = internal_bufs.resource_buffers[0].buffers[5]
                .clone()
                .unwrap();

            queue.write_buffer(&num_rays_buffer, 0, bytemuck::bytes_of(&num_rays));
            queue.write_buffer(
                &max_candidates_buffer,
                0,
                bytemuck::bytes_of(&max_candidates),
            );
            queue.write_buffer(&counter_buffer, 0, bytemuck::bytes_of(&0u32));

            shader.dispatch(
                AccelerationArgs {
                    tlas,
                    rays_buffer,
                    candidates_buffer: candidates_buffer.clone(),
                    counter_buffer: counter_buffer.clone(),
                    num_rays_buffer,
                    max_candidates_buffer,
                },
                &device,
                &queue,
            );

            let counter: Vec<u32> = read_buffer(&device, &queue, &counter_buffer);
            println!("Counter value: {}", counter[0]);
            assert!(counter[0] > 0, "Ray should have hit the triangle");

            let candidates: Vec<RawCandidate> = read_buffer(&device, &queue, &candidates_buffer);
            let hit = &candidates[0];
            println!(
                "Hit: ray_idx={}, instance={}, primitive={}, geometry={}, barycentrics=({},{}), t={}",
                hit.ray_idx,
                hit.instance_index,
                hit.primitive_index,
                hit.geometry_index,
                hit.barycentrics[0],
                hit.barycentrics[1],
                hit.t
            );
            assert_eq!(hit.ray_idx, 0, "Ray 0 should be the hit");
            assert_eq!(hit.instance_index, 0, "Instance index should be 0");
            assert!(
                hit.t > 0.9 && hit.t < 1.1,
                "t should be ~1.0, got {}",
                hit.t
            );
        });
    }
}
