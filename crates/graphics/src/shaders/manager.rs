use std::{collections::HashMap, hash::Hash, mem, sync::Arc};

use wgpu::{Device, PipelineCache, PipelineCacheDescriptor, TextureFormat};

use crate::shaders::{Shader, ShaderWrapper};

pub struct ShaderManager<'a, K> {
    pub compilation_cache: Option<PipelineCache>,
    pub shaders: HashMap<K, ShaderWrapper<'a, Arc<dyn Shader<'a>>>>,
    pub compilation_pending_shaders: HashMap<K, Arc<dyn Shader<'a>>>,
    pub texture_format: Option<TextureFormat>,
}

impl<'a, K> ShaderManager<'a, K>
where
    K: Eq + PartialEq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            compilation_cache: None,
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
        S: Shader<'a> + Sized + 'static,
    {
        self.compilation_pending_shaders
            .insert(key, Arc::new(shader));
    }

    pub fn recompile_shaders(&mut self, device: &Device) {
        let pending_shaders = mem::take(&mut self.compilation_pending_shaders);

        self.shaders.extend(
            pending_shaders
                .into_iter()
                .map(|(k, s)| {
                    (
                        k,
                        ShaderWrapper::new(
                            s,
                            device,
                            &self.texture_format.unwrap(),
                            self.compilation_cache.as_ref(),
                        ),
                    )
                })
                .collect::<Vec<_>>(),
        );
        let texture_format = self.texture_format.unwrap();
        let compilation_cache = self.compilation_cache.as_ref();
        self.shaders.values_mut().for_each(|s| {
            s.recompile(device, &texture_format, compilation_cache);
        });
    }
}
