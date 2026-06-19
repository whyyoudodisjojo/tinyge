pub mod strategies;
use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Weak},
};

use wgpu::*;
use winit::window::Window;

use crate::{
    shaders::{ShaderBuffers, ShaderManager},
    state::{StateRender, StateUpdates},
};

pub struct RendererCtx<'a> {
    pub instance: Instance,
    pub surface: Surface<'a>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_config: SurfaceConfiguration,
    pub window: Arc<Window>,
}

pub struct RendererDescriptor<'a> {
    pub instance_descriptor: RendererInstanceDescriptor,
    pub adapter_descriptor: AdapterDescriptor,
    pub device_descriptor: DeviceDescriptor<'a>,
}

#[derive(Clone)]
pub struct RendererInstanceDescriptor {
    pub backends: Backends,
    pub flags: InstanceFlags,
    pub memory_budget_thresholds: MemoryBudgetThresholds,
    pub backend_options: BackendOptions,
}

impl From<RendererInstanceDescriptor> for InstanceDescriptor {
    fn from(value: RendererInstanceDescriptor) -> Self {
        Self {
            backends: value.backends,
            flags: value.flags,
            memory_budget_thresholds: value.memory_budget_thresholds,
            backend_options: value.backend_options,
            display: None,
        }
    }
}
pub struct AdapterDescriptor {
    pub power_preference: PowerPreference,
    pub force_fallback_adapter: bool,
}

pub struct Renderer<'a, K> {
    pub ctx: Option<RendererCtx<'a>>,
    pub descriptor: RendererDescriptor<'a>,
    pub shader_manager: ShaderManager<'a, K>,
}

impl<'a, K> Renderer<'a, K>
where
    K: Hash + Eq + PartialEq + Clone,
{
    pub fn new(descriptor: RendererDescriptor<'a>, shader_manager: ShaderManager<'a, K>) -> Self {
        Self {
            ctx: None,
            descriptor,
            shader_manager,
        }
    }

    pub async fn init(&mut self, window: Arc<Window>) -> Option<HashMap<K, ShaderBuffers>> {
        let instance = wgpu::Instance::new(self.descriptor.instance_descriptor.clone().into());

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: self.descriptor.adapter_descriptor.power_preference,
                force_fallback_adapter: self.descriptor.adapter_descriptor.force_fallback_adapter,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&self.descriptor.device_descriptor)
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .cloned()
            .unwrap_or_else(|| surface_caps.formats[0]);

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        self.shader_manager.update_texture_format(format);
        let new_buffers = self.shader_manager.recompile_shaders(&device);

        surface.configure(&device, &surface_config);

        self.ctx = Some(RendererCtx {
            instance,
            surface,
            adapter,
            device,
            queue,
            surface_config,
            window,
        });

        new_buffers
    }

    pub fn window(&self) -> Option<Weak<Window>> {
        self.ctx.as_ref().map(|c| Arc::downgrade(&c.window))
    }

    pub fn prepare_surface<State>(
        ctx: &mut RendererCtx,
        shader_manager: &mut ShaderManager<K>,
        state: &mut State,
    ) where
        State: StateRender + StateUpdates<K = K>,
    {
        let render_width = state.render_width();
        let render_height = state.render_height();

        if ctx.surface_config.width != render_width || ctx.surface_config.height != render_height {
            ctx.surface_config.width = render_width;
            ctx.surface_config.height = render_height;

            let new_buffers = shader_manager.recompile_shaders(&ctx.device);
            new_buffers.map(|n| state.handle_shader_recompilation(n, &ctx.queue, &ctx.device));
        }
    }
}
