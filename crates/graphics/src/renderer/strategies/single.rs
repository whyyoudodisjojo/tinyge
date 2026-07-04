use std::hash::Hash;

use wgpu::{CommandEncoderDescriptor, CurrentSurfaceTexture};

use crate::{
    renderer::{
        Renderer,
        strategies::{RenderAble, RenderDispatcher, RenderPath},
    },
    state::{StateRender, StateUpdates},
};

pub trait StateRenderSinglePass<K>: StateRender + RenderAble<K> {}

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

        Self::prepare_surface(ctx, &mut self.shader_manager, state);

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

        state.render_pass(
            &mut encoder,
            &mut self.shader_manager.shaders,
            &view,
            &ctx.device,
            &ctx.queue,
        );

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
