use std::{collections::HashMap, hash::Hash};

use wgpu::{CommandEncoder, Queue, RenderPipeline, TextureView, TextureViewDescriptor};

use crate::shaders::ShaderBuffers;

pub trait StateUpdates
where
    Self: Sized,
{
    type UpdateEvent;

    fn handle_shader_recompilation<K>(&mut self, new_buffers: HashMap<K, ShaderBuffers>);
    fn update(&mut self, update_event: Self::UpdateEvent, queue: Option<&Queue>); // If background queue might not be present so the state must be updated but nothing will retrigger; ideally shouldnt hit this this is aexecuted during a redraw but ye
}

pub trait StateRender
where
    Self::Key: Hash + PartialEq + Eq + Clone,
{
    type Key;

    fn base_canvas_view_descriptor(&self) -> TextureViewDescriptor<'static> {
        TextureViewDescriptor::default()
    }

    fn render_width(&self) -> u32;
    fn render_height(&self) -> u32;

    fn render_pass(
        &self,
        encoder: &mut CommandEncoder,
        pipeline_cache: &HashMap<Self::Key, RenderPipeline>,
        view: &TextureView,
    );
}
