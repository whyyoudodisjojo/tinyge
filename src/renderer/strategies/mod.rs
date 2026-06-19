use std::marker::PhantomData;

use crate::renderer::Renderer;

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
