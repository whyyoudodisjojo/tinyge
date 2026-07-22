use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use tinyge_graphics::{
    renderer::strategies::{
        RenderAble,
        single::{SinglePass, StateRenderSinglePass},
    },
    shaders::buffers::{BufferWithType, Buffers, ResourceType},
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
    pub time_buffer: Option<BufferWithType<f32>>,
    pub sz: PhysicalSize<u32>,
    pub start_time: SystemTime,
}

impl State {
    pub fn new() -> Self {
        Self {
            buffers: None,
            time_buffer: None,
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
        let shader_wrapper = shaders.get(&ShaderId(1)).unwrap();
        let spec = &shader_wrapper
            .buffer_build_spec
            .as_ref()
            .unwrap()
            .buffer_build_spec;
        let new_buffer = Buffers::build(device, spec, false);
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
        let time_buf =
            BufferWithType::<f32>::from(new_buffer.resource_buffers[0].buffers[0].clone().unwrap());
        time_buf.write(
            queue,
            &[SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as f32],
        );
        self.time_buffer = Some(time_buf);
        self.buffers = Some(new_buffer);
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
            tinyge_graphics::shaders::ShaderWrapper<
                'a,
                Arc<dyn tinyge_graphics::shaders::Shader<'a>>,
            >,
        >,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        _queue: &Queue,
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
        let shader_wrapper = pipeline_cache.get_mut(&ShaderId(1)).unwrap();
        let built_data = shader_wrapper.buffer_build_spec.as_mut().unwrap();

        render_pass.set_pipeline(&built_data.pipeline);
        render_pass.set_vertex_buffer(0, buffers.vertex_buffers[0].slice(..));
        render_pass.set_index_buffer(
            buffers.index_buffer.as_ref().unwrap().slice(..),
            wgpu::IndexFormat::Uint16,
        );

        // Create bind group resources
        let resources: Vec<ResourceType> = vec![ResourceType::Buffer(
            self.time_buffer.as_ref().unwrap().inner.clone(),
        )];

        let bind_group = built_data.bind_groups[0].get_or_create_bind_group(&resources, device);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
    }
}

impl StateRenderSinglePass<ShaderId> for State {}
