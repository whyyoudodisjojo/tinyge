use wgpu::{CommandEncoderDescriptor, CurrentSurfaceTexture};

use crate::{
    renderer::{
        Renderer,
        strategies::{RenderAble, RenderDispatcher, RenderPath},
    },
    state::{StateRender, StateUpdates},
};

pub trait StateRenderSinglePass: StateRender + RenderAble {}

pub trait SinglePassRenderer<'a> {
    fn render<State>(&mut self, state: &mut State)
    where
        State: StateRenderSinglePass + StateUpdates<'a>;
}

pub struct SinglePass;

impl<'a> SinglePassRenderer<'a> for Renderer<'a>
{
    fn render<State>(&mut self, state: &mut State)
    where
        State: StateRenderSinglePass + StateUpdates<'a>,
    {
        let Some(ctx) = &mut self.ctx else {
            return;
        };

        Self::prepare_surface(&self.recompilation_manager, ctx, state);

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
            &view,
            &ctx.device,
            &ctx.queue,
        );

        ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

impl<'a, 'b, S> RenderDispatcher<'a> for RenderPath<'b, S, SinglePass>
where
    S: StateRenderSinglePass + StateUpdates<'a>,
{
    fn dispatch_render(&mut self, renderer: &mut Renderer<'a>) {
        SinglePassRenderer::render(renderer, self.inner);
    }
}
