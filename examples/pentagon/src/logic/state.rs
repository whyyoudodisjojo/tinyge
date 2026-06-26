use std::time::{SystemTime, UNIX_EPOCH};

use tinyge_graphics::{
    renderer::strategies::{
        RenderAble,
        single::{SinglePass, StateRenderSinglePass},
    },
    shaders::buffers::Buffers,
    state::{StateRender, StateUpdates},
};
use wgpu::{Color, Device, Operations, Queue, RenderPassColorAttachment, RenderPassDescriptor};
use winit::dpi::PhysicalSize;

use crate::{
    ShaderId,
    logic::UpdateEvents,
    shader::pentagon::{INDICES, VERTICES},
};

pub struct State {
    pub buffers: Option<Buffers>,
    pub sz: PhysicalSize<u32>,
    pub start_time: SystemTime,
}

impl State {
    pub fn new() -> Self {
        Self {
            buffers: None,
            sz: PhysicalSize {
                width: 1920,
                height: 1080,
            },
            start_time: SystemTime::now(),
        }
    }
}

impl StateUpdates for State {
    type K = ShaderId;
    type UpdateEvent = UpdateEvents;

    fn handle_shader_recompilation(
        &mut self,
        new_buffers: std::collections::HashMap<Self::K, Buffers>,
        queue: &Queue,
        device: &Device,
    ) {
        let new_buffer = new_buffers.into_iter().next().unwrap().1;
        match &mut self.buffers {
            Some(b) => b.copy_data_into(&new_buffer, device, queue),
            None => {
                queue.write_buffer(
                    &new_buffer.vertex_buffers[0],
                    0,
                    bytemuck::cast_slice(&VERTICES),
                );
                queue.write_buffer(
                    &new_buffer.index_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(INDICES),
                );
                queue.write_buffer(
                    &new_buffer.resource_buffers[0].buffers[0],
                    0,
                    bytemuck::cast_slice(&[SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as u32]),
                );
            }
        }

        self.buffers = Some(new_buffer);
    }

    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&wgpu::Queue>) {
        match update_event {
            UpdateEvents::Resize(sz) => self.sz = sz,
            UpdateEvents::TimeUpdate => {
                self.buffers.as_ref().zip(queue).map(|(b, q)| {
                    let time_val = SystemTime::now()
                        .duration_since(self.start_time)
                        .unwrap()
                        .as_secs_f32();
                    q.write_buffer(
                        &b.resource_buffers[0].buffers[0],
                        0,
                        bytemuck::cast_slice(&[time_val]),
                    )
                });
            }
        }
    }
}

impl StateRender for State {
    type RenderStrategy = SinglePass;

    fn render_height(&self) -> u32 {
        self.sz.height
    }

    fn render_width(&self) -> u32 {
        self.sz.width
    }
}

impl RenderAble<ShaderId> for State {
    fn render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &std::collections::HashMap<ShaderId, wgpu::RenderPipeline>,
        view: &wgpu::TextureView,
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

        render_pass.set_pipeline(pipeline_cache.get(&ShaderId(1)).unwrap());
        render_pass.set_vertex_buffer(
            0,
            self.buffers.as_ref().unwrap().vertex_buffers[0].slice(..),
        );
        render_pass.set_index_buffer(
            self.buffers
                .as_ref()
                .unwrap()
                .index_buffer
                .as_ref()
                .unwrap()
                .slice(..),
            wgpu::IndexFormat::Uint16,
        );
        render_pass.set_bind_group(
            0,
            self.buffers.as_ref().unwrap().resource_buffers[0]
                .bind_group
                .peek_last_bind_group()
                .unwrap(),
            &[],
        );
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
    }
}

impl StateRenderSinglePass<ShaderId> for State {}
