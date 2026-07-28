use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use tinyge_graphics::{
    renderer::strategies::{
        RenderAble,
        single::{SinglePass, StateRenderSinglePass},
    },
    shaders::ShaderWrapper,
    state::{StateRender, StateUpdates},
};

use memory::{
    buffers::{BufferWithType, Buffers, ResourceType},
    socket::TinyBuffer,
};
use wgpu::{
    BufferDescriptor, BufferUsages, Color, Device, Operations, Queue, RenderPassColorAttachment,
    RenderPassDescriptor,
};
use winit::dpi::PhysicalSize;

use crate::{
    logic::UpdateEvents,
    shader::{
        Vertex,
        pentagon::{INDICES, Pentagon, VERTICES},
    },
};

pub struct State<'a> {
    pub buffers: Option<Buffers<'a>>,
    pub time_buffer: Option<BufferWithType<f32>>,
    pub sz: PhysicalSize<u32>,
    pub start_time: SystemTime,
    pub shaders: Shaders,
}

pub struct Shaders {
    pub pentagon: Arc<ShaderWrapper<Pentagon>>,
}

impl<'a> State<'a> {
    pub fn new(shader: Shaders) -> Self {
        Self {
            buffers: None,
            time_buffer: None,
            sz: PhysicalSize {
                width: 1920,
                height: 1080,
            },
            start_time: SystemTime::now(),
            shaders: shader,
        }
    }
}

impl<'a> StateUpdates<'a> for State<'a> {
    type UpdateEvent = UpdateEvents;

    fn init(&mut self, device: &Device, queue: &Queue) {
        let vertex_size = (std::mem::size_of::<Vertex>() * VERTICES.len()) as u64;
        let vertex_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: vertex_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&VERTICES));

        let index_size = (std::mem::size_of::<u16>() * INDICES.len()) as u64;
        let index_buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: index_size,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&index_buf, 0, bytemuck::cast_slice(INDICES));

        let time_buf_raw = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &time_buf_raw,
            0,
            bytemuck::bytes_of(&[SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as f32]),
        );

        self.time_buffer = Some(BufferWithType::from(TinyBuffer {
            inner: Some(time_buf_raw),
            size: 4,
            usages: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
        self.buffers = Some(Buffers {
            vertex_buffers: vec![TinyBuffer {
                inner: Some(vertex_buf),
                size: vertex_size,
                usages: BufferUsages::VERTEX,
            }],
            index_buffer: Some(TinyBuffer {
                inner: Some(index_buf),
                size: index_size,
                usages: BufferUsages::INDEX,
            }),
            resource_buffers: vec![],
        });
    }

    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&Queue>) {
        match update_event {
            UpdateEvents::Resize(sz) => self.sz = sz,
            UpdateEvents::TimeUpdate => {
                self.time_buffer.as_ref().zip(queue).map(|(t, q)| {
                    let time_val = SystemTime::now()
                        .duration_since(self.start_time)
                        .unwrap()
                        .as_secs_f32();
                    t.write(q, &[time_val])
                });
            }
        }
    }
}

impl<'a> StateRender for State<'a> {
    type RenderStrategy = SinglePass;

    fn render_height(&self) -> u32 {
        self.sz.height
    }

    fn render_width(&self) -> u32 {
        self.sz.width
    }
}

impl<'b> RenderAble for State<'b> {
    fn render_pass<'a>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: wgpu::LoadOp::Clear(Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        let buffers = self.buffers.as_ref().unwrap();
        let mut built_data = self.shaders.pentagon.shader.lock().unwrap();
        let sockets_ref = built_data.get_sockets().unwrap();
        render_pass.set_pipeline(&sockets_ref.pipeline);
        render_pass.set_vertex_buffer(0, buffers.vertex_buffers[0].raw().slice(..));
        render_pass.set_index_buffer(
            buffers.index_buffer.as_ref().unwrap().raw().slice(..),
            wgpu::IndexFormat::Uint16,
        );

        let resources: Vec<ResourceType> = vec![ResourceType::Buffer(
            self.time_buffer.as_ref().unwrap().inner.clone(),
        )];

        let built_data = built_data.get_sockets_mut().unwrap();
        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(&resources, device);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
    }
}

impl<'a> StateRenderSinglePass for State<'a> {}
