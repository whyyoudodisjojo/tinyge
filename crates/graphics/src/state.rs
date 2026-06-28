use std::collections::HashMap;

use wgpu::{Device, Queue, TextureViewDescriptor};

use crate::shaders::{Shader, ShaderWrapper};

pub trait StateUpdates
where
    Self: Sized,
{
    type UpdateEvent;
    type K;

    fn init<'a>(
        &mut self,
        shaders: &HashMap<Self::K, ShaderWrapper<'a, std::sync::Arc<dyn Shader<'a>>>>,
        device: &Device,
        queue: &Queue,
    );
    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&Queue>);
}

pub trait StateRender {
    type RenderStrategy;
    fn base_canvas_view_descriptor(&self) -> TextureViewDescriptor<'static> {
        TextureViewDescriptor::default()
    }

    fn render_width(&self) -> u32;
    fn render_height(&self) -> u32;
}
