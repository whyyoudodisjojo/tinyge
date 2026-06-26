use std::time::{SystemTime, UNIX_EPOCH};

use image::DynamicImage;
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

use crate::{ShaderId, logic::UpdateEvents, shader::SpriteData};

pub struct State {
    pub buffers: Option<Buffers>,
    pub sz: PhysicalSize<u32>,
    pub start_time: SystemTime,
    pub texture_image: DynamicImage,
    pub sprites: Vec<SpriteData>,
}

impl State {
    pub fn new(texture_image: DynamicImage) -> Self {
        let sprites = vec![SpriteData::default()];

        Self {
            buffers: None,
            sz: PhysicalSize {
                width: 1920,
                height: 1080,
            },
            start_time: SystemTime::now(),
            texture_image,
            sprites,
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

        if let Some(resource_group) = new_buffer.resource_buffers.get(1) {
            if let Some(texture) = resource_group.textures.first() {
                texture.copy_image_data(self.texture_image.clone(), queue);
            }
        }

        match &mut self.buffers {
            Some(b) => {
                b.copy_data_into(&new_buffer, device, queue);
            }
            None => {
                let time_val = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f32();

                queue.write_buffer(
                    &new_buffer.resource_buffers[0].buffers[0].as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&[time_val]),
                );

                if !self.sprites.is_empty() {
                    queue.write_buffer(
                        &new_buffer.resource_buffers[0].buffers[1].as_ref().unwrap(),
                        0,
                        bytemuck::cast_slice(&self.sprites),
                    );
                }
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
                        &b.resource_buffers[0].buffers[0].as_ref().unwrap(),
                        0,
                        bytemuck::cast_slice(&[time_val]),
                    );
                });
            }
            UpdateEvents::SpriteUpdate(sprites) => {
                self.sprites = sprites;
                self.buffers.as_ref().zip(queue).map(|(b, q)| {
                    if !self.sprites.is_empty() {
                        q.write_buffer(
                            &b.resource_buffers[0].buffers[1].as_ref().unwrap(),
                            0,
                            bytemuck::cast_slice(&self.sprites),
                        )
                    }
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
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
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

        if let Some(buffers) = &self.buffers {
            render_pass.set_bind_group(
                0,
                buffers.resource_buffers[0]
                    .bind_group
                    .peek_last_bind_group()
                    .unwrap(),
                &[],
            );
            render_pass.set_bind_group(
                1,
                buffers.resource_buffers[1]
                    .bind_group
                    .peek_last_bind_group()
                    .unwrap(),
                &[],
            );
        }

        render_pass.draw(0..6, 0..self.sprites.len() as u32);
    }
}

impl StateRenderSinglePass<ShaderId> for State {}
