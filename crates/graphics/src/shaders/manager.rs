use std::{
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
    thread,
};

use wgpu::{Device, PipelineCache, PipelineCacheDescriptor, TextureFormat};

use crate::shaders::{RecompilationData, Shader, ShaderWrapper};

pub struct RecompilationManager{
    shader_recompilation_handles: Vec<Sender<RecompilationData>>,
    cache: Option<PipelineCache>,
    curr_texture_format: Option<TextureFormat>
}

impl RecompilationManager{
    pub fn new() -> Self{
        Self { shader_recompilation_handles: vec![], cache: None, curr_texture_format: None}
    }

    pub fn new_with_cache(device: &Device, cache_desc: PipelineCacheDescriptor) -> Self{
        Self{
            shader_recompilation_handles: vec![],
            cache: unsafe {Some(device.create_pipeline_cache(&cache_desc))},
            curr_texture_format: None
        }
    }

    pub fn recompile_all(&self, device: &Device){
        self.shader_recompilation_handles.iter().for_each(|b| b.send(RecompilationData { device: device.clone() , texture_format: self.curr_texture_format.unwrap(), cache: self.cache.clone() }).unwrap())
    }

    pub fn update_texture_format(&mut self, texture_format: TextureFormat){
        self.curr_texture_format = Some(texture_format)
    }

    pub fn register_shader<S>(&mut self, shader: Arc<ShaderWrapper<S>>, rx: Receiver<RecompilationData>)
        where S: for<'a> Shader<'a> + Send + 'static
    {
        self.shader_recompilation_handles.push(shader.get_sender_tx());
        thread::spawn(move || {
            ShaderWrapper::<S>::watch(rx, shader.shader.clone());
        });
    }
}