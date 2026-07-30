use tinyge_graphics::shaders::ComputeShaderWrapper;
use tinyge_physics::collisions::algos::GpuRay;
use tinyge_physics::collisions::algos::gpu_accelerated::{
    AccelerationArgs, AccelerationShader, RawCandidate,
};
use wgpu::{Blas, Device, Queue};

pub struct RayTracer {
    shader: ComputeShaderWrapper<AccelerationShader, AccelerationArgs<'static>>,
    device: Device,
    queue: Queue,
    num_rays: u32,
    max_candidates: u32,
}

impl RayTracer {
    pub fn new(shader_config: AccelerationShader, device: Device, queue: Queue) -> Self {
        let num_rays = shader_config.num_rays;
        let max_candidates = shader_config.max_candidates;
        let shader = ComputeShaderWrapper::new(shader_config, &device);

        Self {
            shader,
            device,
            queue,
            num_rays,
            max_candidates,
        }
    }

    pub fn build_tlas(&mut self, blases: &[&Blas]) {
        let args = &mut self.shader.built_data.as_mut().unwrap().buffers;
        args.acc.tlas.build(&self.device);

        for (i, blas) in blases.iter().enumerate() {
            args.acc.tlas.raw_mut()[i] = Some(wgpu::TlasInstance::new(
                blas,
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                0,
                0xFF,
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.build_acceleration_structures(&[], std::iter::once(args.acc.tlas.raw()));
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn set_rays(&self, rays: &[GpuRay]) {
        let buf = self
            .shader
            .built_data
            .as_ref()
            .unwrap()
            .buffers
            .rays_buffer
            .inner
            .raw();
        self.queue.write_buffer(buf, 0, bytemuck::cast_slice(rays));
    }

    pub fn dispatch(&mut self) {
        self.queue.write_buffer(
            self.shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .num_rays_buffer
                .inner
                .raw(),
            0,
            bytemuck::bytes_of(&self.num_rays),
        );
        self.queue.write_buffer(
            self.shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .max_candidates_buffer
                .inner
                .raw(),
            0,
            bytemuck::bytes_of(&self.max_candidates),
        );
        self.queue.write_buffer(
            self.shader
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

        let args = AccelerationArgs {
            acc: self.shader.built_data.as_ref().unwrap().buffers.acc.clone(),
            rays_buffer: self
                .shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .rays_buffer
                .clone(),
            candidates_buffer: self
                .shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .candidates_buffer
                .clone(),
            counter_buffer: self
                .shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .counter_buffer
                .clone(),
            num_rays_buffer: self
                .shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .num_rays_buffer
                .clone(),
            max_candidates_buffer: self
                .shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .max_candidates_buffer
                .clone(),
        };
        self.shader.dispatch(args, &self.device, &self.queue);
    }

    pub fn read_candidates(&self) -> (Vec<RawCandidate>, u32) {
        let counter: Vec<u32> = read_buffer(
            &self.device,
            &self.queue,
            self.shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .counter_buffer
                .inner
                .raw(),
        );
        let candidates: Vec<RawCandidate> = read_buffer(
            &self.device,
            &self.queue,
            self.shader
                .built_data
                .as_ref()
                .unwrap()
                .buffers
                .candidates_buffer
                .inner
                .raw(),
        );
        (candidates, counter[0])
    }

    pub fn num_rays(&self) -> u32 {
        self.num_rays
    }

    pub fn max_candidates(&self) -> u32 {
        self.max_candidates
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Features, util::DeviceExt};

    async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: Features::EXPERIMENTAL_RAY_QUERY,
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
    fn test_raytracer_dispatch() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;

            let num_rays = 8u32;
            let max_candidates = 16u32;
            let mut rt = RayTracer::new(
                AccelerationShader {
                    num_rays,
                    max_candidates,
                    max_instances: 1,
                    blas_vertex_count: 3,
                    gpu_ray_size: 48,
                },
                device,
                queue,
            );

            let tri_verts: Vec<[f32; 3]> =
                vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
            let vertex_buffer = rt
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&tri_verts),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::BLAS_INPUT,
                });

            let blas = rt.device.create_blas(
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

            let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: 3,
                index_format: None,
                index_count: None,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };

            let blas_entry = wgpu::BlasBuildEntry {
                blas: &blas,
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

            let mut encoder = rt
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.build_acceleration_structures(&[blas_entry], std::iter::empty::<&wgpu::Tlas>());
            rt.queue.submit(std::iter::once(encoder.finish()));

            rt.build_tlas(&[&blas]);

            let rays: Vec<GpuRay> = vec![
                GpuRay {
                    origin: [0.0, 0.0, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                GpuRay {
                    origin: [10.0, 10.0, 0.0, 0.0],
                    dir: [1.0, 0.0, 0.0, 0.0],
                    inv_dir: [1.0, 0.0, 0.0, 0.0],
                },
                GpuRay {
                    origin: [0.8, -0.8, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                GpuRay {
                    origin: [0.0, 0.0, 1.0, 0.0],
                    dir: [0.0, 0.0, 1.0, 0.0],
                    inv_dir: [0.0, 0.0, 1.0, 0.0],
                },
                GpuRay {
                    origin: [0.0, 0.0, 2.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                GpuRay {
                    origin: [0.0, 0.0, -1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                GpuRay {
                    origin: [0.0, -0.5, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
                GpuRay {
                    origin: [5.0, 5.0, 1.0, 0.0],
                    dir: [0.0, 0.0, -1.0, 0.0],
                    inv_dir: [0.0, 0.0, -1.0, 0.0],
                },
            ];
            rt.set_rays(&rays);
            rt.dispatch();

            let (candidates, counter) = rt.read_candidates();

            println!("Counter value: {}", counter);
            assert!(counter >= 4, "Expected at least 4 hits, got {}", counter);

            for i in 0..counter.min(candidates.len() as u32) as usize {
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
