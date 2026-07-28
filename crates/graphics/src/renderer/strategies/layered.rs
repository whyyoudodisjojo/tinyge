use wgpu::{CommandEncoderDescriptor, CurrentSurfaceTexture};

use crate::{
    renderer::{
        Renderer,
        strategies::{RenderAble, RenderDispatcher, RenderPath},
    },
    state::{StateRender, StateUpdates},
};

pub struct LayeredRenderPass<RenderPassState> {
    pub state: RenderPassState,
}

pub trait LayeredStateRender: StateRender {
    fn get_render_layers<'a>(
        &'a mut self,
    ) -> &'a mut [LayeredRenderPass<&'a mut dyn RenderAble>];
}

pub trait StateRenderedLayeredPass: StateRender + LayeredStateRender {}

pub struct LayeredPass;

pub trait LayeredPassRenderer<'a> {
    fn render<State>(&mut self, state: &mut State)
    where
        State: StateRenderedLayeredPass + StateUpdates<'a>;
}

impl<'a> LayeredPassRenderer<'a> for Renderer<'a>
{
    fn render<State>(&mut self, state: &mut State)
    where
        State: LayeredStateRender + StateUpdates<'a>,
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
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => return,
        };

        let view = output
            .texture
            .create_view(&state.base_canvas_view_descriptor());

        let layers = state.get_render_layers();

        layers.into_iter().for_each(|l| {
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });

            l.state.render_pass(
                &mut encoder,
                &view,
                &ctx.device,
                &ctx.queue,
            );

            ctx.queue.submit(std::iter::once(encoder.finish()));
        });

        output.present();
    }
}

impl<'a, 'b, S> RenderDispatcher<'a> for RenderPath<'b, S, LayeredPass>
where
    S: StateRenderedLayeredPass + StateUpdates<'a>,
{
    fn dispatch_render(&mut self, renderer: &mut Renderer<'a>) {
        LayeredPassRenderer::render(renderer, self.inner);
    }
}
