use std::hash::Hash;

use wgpu::{CommandEncoderDescriptor, CurrentSurfaceTexture};

use crate::{
    renderer::{
        Renderer,
        strategies::{RenderDispatcher, RenderPath},
    },
    state::{RendererAble, StateRender, StateUpdates},
};

pub struct LayeredRenderPass<RenderPassState> {
    pub state: RenderPassState,
}

pub trait LayeredStateRender<K>: StateRender {
    fn get_render_layers<'a>(&'a self) -> &'a [LayeredRenderPass<&'a dyn RendererAble<K>>];
}

pub trait StateRenderedLayeredPass<K>: StateRender + LayeredStateRender<K> {}

pub struct LayeredPass;

pub trait LayeredPassRenderer<K> {
    fn render<State>(&mut self, state: &mut State)
    where
        State: StateRenderedLayeredPass<K> + StateUpdates<K = K>;
}

impl<'a, K> LayeredPassRenderer<K> for Renderer<'a, K>
where
    K: Hash + Clone + PartialEq + Eq,
{
    fn render<State>(&mut self, state: &mut State)
    where
        State: LayeredStateRender<K> + StateUpdates<K = K>,
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

        let layers = state.get_render_layers();

        layers.into_iter().for_each(|l| {
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });

            l.state
                .render_pass(&mut encoder, &self.shader_manager.pipeline_cache, &view);

            ctx.queue.submit(std::iter::once(encoder.finish()));
        });

        output.present();
    }
}

impl<'b, S, K> RenderDispatcher<K> for RenderPath<'b, S, LayeredPass>
where
    K: Clone + Hash + PartialEq + Eq,
    S: StateRenderedLayeredPass<K> + StateUpdates<K = K>,
{
    fn dispatch_render<'a>(&mut self, renderer: &mut Renderer<'a, K>) {
        LayeredPassRenderer::render(renderer, self.inner);
    }
}
