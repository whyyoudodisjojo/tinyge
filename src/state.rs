use std::collections::HashMap;

use wgpu::{Device, Queue, TextureViewDescriptor};

use crate::shaders::ShaderBuffers;

pub trait StateUpdates
where
    Self: Sized,
{
    type UpdateEvent;
    type K;

    fn handle_shader_recompilation(
        &mut self,
        new_buffers: HashMap<Self::K, ShaderBuffers>,
        queue: &Queue,
        device: &Device,
    );
    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&Queue>); // If background queue might not be present so the state must be updated but nothing will retrigger; ideally shouldnt hit this this is aexecuted during a redraw but ye
}

pub trait StateRender {
    type RenderStrategy;
    fn base_canvas_view_descriptor(&self) -> TextureViewDescriptor<'static> {
        TextureViewDescriptor::default()
    }

    fn render_width(&self) -> u32;
    fn render_height(&self) -> u32;
}
