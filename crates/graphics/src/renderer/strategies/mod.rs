use std::{marker::PhantomData};

use wgpu::{CommandEncoder, TextureView};

use crate::{
    renderer::Renderer,
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

pub trait RenderDispatcher<'a> {
    fn dispatch_render(&mut self, renderer: &mut Renderer<'a>);
}

pub trait RenderAble {
    fn render_pass<'a>(
        &mut self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    );
}
