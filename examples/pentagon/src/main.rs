mod logic;
mod shader;

use tinyge_core::{
    game_loop::GameLoop,
    renderer::{AdapterDescriptor, Renderer, RendererDescriptor, RendererInstanceDescriptor},
    shaders::manager::ShaderManager,
};
use wgpu::{Backends, wgt::DeviceDescriptor};
use winit::event_loop::EventLoop;

use crate::{
    logic::{executor::Executor, state::State},
    shader::{ShaderId, pentagon::Pentagon},
};

fn main() {
    let mut shader_manager: ShaderManager<ShaderId> = ShaderManager::new();

    let shader = Pentagon;
    shader_manager.register_shader(ShaderId(1), shader);

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
        shader_manager,
    );

    // TODO: Have GameLoop struct not allow creation without a vlaid render strategy impl
    let mut game_loop = GameLoop::new(State::new(), Executor, renderer);

    let event_loop = EventLoop::with_user_event().build().unwrap();

    event_loop.run_app(&mut game_loop).unwrap();
}
