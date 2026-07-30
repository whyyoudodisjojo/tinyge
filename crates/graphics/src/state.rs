use wgpu::{Device, Queue, TextureFormat, TextureViewDescriptor};

pub trait StateUpdates<'a>
where
    Self: Sized,
{
    type UpdateEvent;

    fn init(&mut self, _device: &Device, _queue: &Queue) {}
    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&Queue>);
    fn rebuild_shaders(
        &mut self,
        _device: &Device,
        _texture_format: &TextureFormat,
        _queue: &Queue,
    ) {
    }
}

pub trait StateRender {
    type RenderStrategy;
    fn base_canvas_view_descriptor(&self) -> TextureViewDescriptor<'static> {
        TextureViewDescriptor::default()
    }

    fn render_width(&self) -> u32;
    fn render_height(&self) -> u32;
}
