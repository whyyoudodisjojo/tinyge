mod logic;
mod shader;

use std::sync::{Arc, Mutex};

use tinyge_graphics::{
    game_loop::GameLoop, renderer::{AdapterDescriptor, Renderer, RendererDescriptor, RendererInstanceDescriptor}, shaders::{ShaderWrapper, manager::RecompilationManager},
};
use wgpu::{Backends, wgt::DeviceDescriptor};
use winit::event_loop::EventLoop;

use crate::{
    logic::{executor::Executor, state::{Shaders, State}}, shader::pentagon::Pentagon,
};

fn main() {
    let mut recompilation_manager = RecompilationManager::new();
    let (shader, rx) = ShaderWrapper::new(Arc::new(Mutex::new(Pentagon::new())));
    recompilation_manager.register_shader(shader.clone(), rx);

    let renderer = Renderer::new(
        RendererDescriptor {
            instance_descriptor: RendererInstanceDescriptor {
                backends: Backends::PRIMARY,
                flags: Default::default(),
                memory_budget_thresholds: Default::default(),
                backend_options: Default::default(),
            },
            adapter_descriptor: AdapterDescriptor {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
            },
            device_descriptor: DeviceDescriptor {
                label: None,
                required_features: Default::default(),
                required_limits: Default::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            },
        },
        recompilation_manager,
    );

    // TODO: Have GameLoop struct not allow creation without a vlaid render strategy impl
    let mut game_loop = GameLoop::new(State::new(Shaders{pentagon: shader}), Executor, renderer);

    let event_loop = EventLoop::with_user_event().build().unwrap();

    event_loop.run_app(&mut game_loop).unwrap();
}
