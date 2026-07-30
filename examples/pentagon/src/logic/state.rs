use std::time::{SystemTime, UNIX_EPOCH};

use tinyge_graphics::{
    renderer::strategies::{RenderAble, single::SinglePass},
    shaders::ShaderWrapper,
    state::{StateRender, StateUpdates},
};

use memory::buffers::ResourceType;
use wgpu::{
    Color, Device, Operations, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    TextureFormat,
};
use winit::dpi::PhysicalSize;

use crate::{
    logic::UpdateEvents,
    shader::pentagon::{INDICES, Pentagon, PentagonArgs, VERTICES},
};

pub struct State<'a> {
    pub queue: Option<Queue>,
    pub sz: PhysicalSize<u32>,
    pub start_time: SystemTime,
    pub shaders: Shaders,
    _phantom: std::marker::PhantomData<&'a ()>,
}

pub struct Shaders {
    pub pentagon: ShaderWrapper<Pentagon, PentagonArgs>,
}

impl<'a> State<'a> {
    pub fn new() -> Self {
        Self {
            queue: None,
            sz: PhysicalSize {
                width: 1920,
                height: 1080,
            },
            start_time: SystemTime::now(),
            shaders: Shaders {
                pentagon: ShaderWrapper::new(Pentagon::new()),
            },
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a> StateUpdates<'a> for State<'a> {
    type UpdateEvent = UpdateEvents;

    fn init(&mut self, _device: &Device, queue: &Queue) {
        self.queue = Some(queue.clone());
    }

    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&Queue>) {
        match update_event {
            UpdateEvents::Resize(sz) => self.sz = sz,
            UpdateEvents::TimeUpdate => {
                if let Some(built_data) = self.shaders.pentagon.built_data.as_ref() {
                    let time_val = SystemTime::now()
                        .duration_since(self.start_time)
                        .unwrap()
                        .as_secs_f32();
                    queue.map(|q| {
                        q.write_buffer(
                            built_data.buffers.time_buffer.raw(),
                            0,
                            bytemuck::bytes_of(&[time_val]),
                        )
                    });
                }
            }
        }
    }

    fn rebuild_shaders(&mut self, device: &Device, _texture_format: &TextureFormat) {
        self.shaders.pentagon.build(device, _texture_format, None);

        let built_data = self.shaders.pentagon.built_data.as_mut().unwrap();
        built_data.buffers.vertex_buffer.build(device);
        built_data.buffers.index_buffer.build(device);
        built_data.buffers.time_buffer.build(device);

        let queue = self.queue.as_ref().unwrap();
        queue.write_buffer(
            built_data.buffers.vertex_buffer.raw(),
            0,
            bytemuck::cast_slice(&VERTICES),
        );
        queue.write_buffer(
            built_data.buffers.index_buffer.raw(),
            0,
            bytemuck::cast_slice(INDICES),
        );
        queue.write_buffer(
            built_data.buffers.time_buffer.raw(),
            0,
            bytemuck::bytes_of(&[SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as f32]),
        );
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

        let built_data = self.shaders.pentagon.built_data.as_mut().unwrap();
        render_pass.set_pipeline(&built_data.pipeline);
        render_pass.set_vertex_buffer(0, built_data.buffers.vertex_buffer.raw().slice(..));
        render_pass.set_index_buffer(
            built_data.buffers.index_buffer.raw().slice(..),
            wgpu::IndexFormat::Uint16,
        );

        let resources: Vec<ResourceType> =
            vec![ResourceType::Buffer(built_data.buffers.time_buffer.clone())];

        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(&resources, device);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
    }
}
