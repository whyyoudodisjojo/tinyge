use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use image::DynamicImage;
use tinyge_graphics::{
    renderer::strategies::{
        single::{SinglePass, StateRenderSinglePass},
        RenderAble,
    },
    shaders::buffers::{Buffers, ResourceType},
    state::{StateRender, StateUpdates},
};
use wgpu::{Color, Device, Operations, Queue, RenderPassColorAttachment, RenderPassDescriptor};
use winit::dpi::PhysicalSize;

use crate::{logic::UpdateEvents, shader::SpriteData, ShaderId};

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

    fn init<'a>(
        &mut self,
        shaders: &std::collections::HashMap<
            Self::K,
            tinyge_graphics::shaders::ShaderWrapper<
                'a,
                std::sync::Arc<dyn tinyge_graphics::shaders::Shader<'a>>,
            >,
        >,
        device: &Device,
        queue: &Queue,
    ) {
        use tinyge_graphics::shaders::buffers::Buffers;
        let shader_wrapper = shaders.get(&ShaderId(1)).unwrap();
        let spec = &shader_wrapper
            .buffer_build_spec
            .as_ref()
            .unwrap()
            .buffer_build_spec;
        let new_buffer = Buffers::build(device, spec, false);

        if let Some(resource_group) = new_buffer.resource_buffers.get(1) {
            if let Some(texture) = resource_group.textures.first() {
                texture.copy_image_data(self.texture_image.clone(), queue);
            }
        }

        let time_val = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f32();

        queue.write_buffer(
            new_buffer.resource_buffers[0].buffers[0].as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[time_val]),
        );

        if !self.sprites.is_empty() {
            queue.write_buffer(
                new_buffer.resource_buffers[0].buffers[1].as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&self.sprites),
            );
        }

        self.buffers = Some(new_buffer);
    }

    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&Queue>) {
        match update_event {
            UpdateEvents::Resize(sz) => self.sz = sz,
            UpdateEvents::TimeUpdate => {
                self.buffers.as_ref().zip(queue).map(|(b, q)| {
                    let time_val = SystemTime::now()
                        .duration_since(self.start_time)
                        .unwrap()
                        .as_secs_f32();
                    q.write_buffer(
                        b.resource_buffers[0].buffers[0].as_ref().unwrap(),
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
                            b.resource_buffers[0].buffers[1].as_ref().unwrap(),
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
    fn render_pass<'a>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &mut std::collections::HashMap<
            ShaderId,
            tinyge_graphics::shaders::ShaderWrapper<Arc<dyn tinyge_graphics::shaders::Shader<'a>>>,
        >,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
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

        let shader_wrapper = pipeline_cache.get_mut(&ShaderId(1)).unwrap();
        let built_data = shader_wrapper.buffer_build_spec.as_mut().unwrap();
        render_pass.set_pipeline(&built_data.pipeline);

        let buffers = self.buffers.as_ref().unwrap();
        let resources0: Vec<ResourceType> = buffers.resource_buffers[0]
            .buffers
            .iter()
            .filter_map(|b| b.as_ref().map(|buf| ResourceType::Buffer(buf.clone())))
            .collect();
        let bind_group0 = built_data.bind_groups[0].get_or_create_bind_group(&resources0, device);
        render_pass.set_bind_group(0, bind_group0, &[]);

        let resources1: Vec<ResourceType> = buffers.resource_buffers[1]
            .samplers
            .iter()
            .map(|s| ResourceType::Sampler(s.clone()))
            .chain(
                buffers.resource_buffers[1]
                    .textures
                    .iter()
                    .map(|t| ResourceType::Texture(t.clone())),
            )
            .collect();
        let bind_group1 = built_data.bind_groups[1].get_or_create_bind_group(&resources1, device);
        render_pass.set_bind_group(1, bind_group1, &[]);

        render_pass.draw(0..6, 0..self.sprites.len() as u32);
    }
}

impl StateRenderSinglePass<ShaderId> for State {}
