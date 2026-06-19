use std::hash::Hash;

use wgpu::{CommandEncoderDescriptor, CurrentSurfaceTexture};

use crate::{
    renderer::{
        Renderer,
        styles::{RenderDispatcher, RenderPath},
    },
    state::{RendererAble, StateRender, StateUpdates},
};

pub trait StateRenderSinglePass<K>: StateRender + RendererAble<K> {}

pub trait SinglePassRenderer<K> {
    fn render<State>(&mut self, state: &mut State)
    where
        State: StateRenderSinglePass<K> + StateUpdates<K = K>;
}

pub struct SinglePass;

impl<'a, K> SinglePassRenderer<K> for Renderer<'a, K>
where
    K: Hash + Clone + PartialEq + Eq,
{
    fn render<State>(&mut self, state: &mut State)
    where
        State: StateRenderSinglePass<K> + StateUpdates<K = K>,
    {
        let Some(ctx) = &mut self.ctx else {
            return;
        };

        let render_width = state.render_width();
        let render_height = state.render_height();

        if ctx.surface_config.width != render_width || ctx.surface_config.height != render_height {
            ctx.surface_config.width = render_width;
            ctx.surface_config.height = render_height;

            let new_buffers = self.shader_manager.recompile_shaders(&ctx.device);
            state.handle_shader_recompilation(new_buffers);
        }

        let output = match ctx.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(s) => s,
            CurrentSurfaceTexture::Suboptimal(s) => {
                ctx.surface.configure(&ctx.device, &ctx.surface_config);
                s
            }
            CurrentSurfaceTexture::Timeout
            | CurrentSurfaceTexture::Occluded
            | CurrentSurfaceTexture::Validation => return,
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                return;
            }
        };

        let view = output
            .texture
            .create_view(&state.base_canvas_view_descriptor());

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        state.render_pass(&mut encoder, &self.shader_manager.pipeline_cache, &view);

        ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

impl<'b, S, K> RenderDispatcher<K> for RenderPath<'b, S, SinglePass>
where
    K: Clone + Hash + PartialEq + Eq,
    S: StateRenderSinglePass<K> + StateUpdates<K = K>,
{
    fn dispatch_render<'a>(&mut self, renderer: &mut Renderer<'a, K>) {
        SinglePassRenderer::render(renderer, self.inner);
    }
}
