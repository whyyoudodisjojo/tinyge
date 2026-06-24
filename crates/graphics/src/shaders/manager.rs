use std::{collections::HashMap, hash::Hash, mem};

use wgpu::{Device, PipelineCache, PipelineCacheDescriptor, RenderPipeline, TextureFormat};

use crate::shaders::{Buffers, Shader};

pub struct ShaderManager<'a, K> {
    pub compilation_cache: Option<PipelineCache>,
    pub pipeline_cache: HashMap<K, RenderPipeline>,
    pub shaders: HashMap<K, &'a dyn Shader>,
    pub compilation_pending_shaders: HashMap<K, &'a dyn Shader>,
    pub texture_format: Option<TextureFormat>,
}

impl<'a, K> ShaderManager<'a, K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            compilation_cache: None,
            pipeline_cache: HashMap::new(),
            shaders: HashMap::new(),
            compilation_pending_shaders: HashMap::new(),
            texture_format: None,
        }
    }

    pub fn new_with_compilation_cache(
        device: &Device,
        cache_descriptor: PipelineCacheDescriptor,
    ) -> Self {
        Self {
            compilation_cache: Some(unsafe { device.create_pipeline_cache(&cache_descriptor) }),
            pipeline_cache: HashMap::new(),
            shaders: HashMap::new(),
            compilation_pending_shaders: HashMap::new(),
            texture_format: None,
        }
    }

    pub fn update_texture_format(&mut self, texture_format: TextureFormat) {
        self.texture_format = Some(texture_format);
    }

    pub fn register_shader<S>(&mut self, key: K, shader: S)
    where
        S: Shader + Sized + 'static,
    {
        let shader: &'static dyn Shader = Box::leak(Box::new(shader));

        self.compilation_pending_shaders.insert(key, shader);
    }

    pub fn recompile_shaders(&mut self, device: &Device) -> Option<HashMap<K, Buffers>> {
        self.pipeline_cache.clear();

        let pending_shaders = mem::take(&mut self.compilation_pending_shaders);

        self.shaders.extend(pending_shaders);

        Some(
            self.shaders
                .iter()
                .map(|(k, s)| {
                    let build_data = s.build(
                        device,
                        &self.texture_format.unwrap(),
                        self.compilation_cache.as_ref(),
                    );

                    self.pipeline_cache.insert(k.clone(), build_data.pipeline);

                    (k.clone(), build_data.buffers)
                })
                .collect(),
        )
    }
}
