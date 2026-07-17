use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use wgpu::{CommandEncoder, TextureView};

use crate::{
    renderer::Renderer,
    shaders::{Shader, ShaderWrapper},
};

pub mod layered;
pub mod single;

pub struct RenderPath<'a, S, Style> {
    pub inner: &'a mut S,
    _phantom: PhantomData<Style>,
}

impl<'a, S, Style> RenderPath<'a, S, Style> {
    pub fn new(s: &'a mut S) -> Self {
        Self {
            inner: s,
            _phantom: PhantomData,
        }
    }
}

pub trait RenderDispatcher<K> {
    fn dispatch_render<'a>(&mut self, renderer: &mut Renderer<'a, K>);
}

pub trait RenderAble<K> {
    fn render_pass<'a>(
        &mut self,
        encoder: &mut CommandEncoder,
        shaders: &mut HashMap<K, ShaderWrapper<'a, Arc<dyn Shader<'a>>>>,
        view: &TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );
}
