mod logic;
mod shader;

use image::{DynamicImage, Rgba};
use tinyge_core::{
    game_loop::GameLoop,
    renderer::{AdapterDescriptor, Renderer, RendererDescriptor, RendererInstanceDescriptor},
    shaders::manager::ShaderManager,
};
use wgpu::{wgt::DeviceDescriptor, Backends, Extent3d};
use winit::event_loop::EventLoop;

use crate::{
    logic::{executor::Executor, state::State},
    shader::{sprites::Sprites, ShaderId},
};

fn generate_texture() -> DynamicImage{
    let size = 64u32;
    let mut img = image::RgbaImage::new(size, size);
    let center = (size as f32) / 2.0;
    let radius = (size as f32) / 2.0 - 2.0;

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let dx = x as f32 - center;
        let dy = y as f32 - center;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist <= radius {
            let alpha = if dist > radius - 4.0 {
                ((radius - dist) / 4.0 * 255.0) as u8
            } else {
                255
            };
            *pixel = Rgba([255, 255, 255, alpha]);
        } else {
            *pixel = Rgba([0, 0, 0, 0]);
        }
    }

    DynamicImage::ImageRgba8(img)
}

fn main() {
    let image = generate_texture();

    let sprites = Sprites {
        texture_size: Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        },
    };

    let mut shader_manager: ShaderManager<ShaderId> = ShaderManager::new();
    shader_manager.register_shader(ShaderId(1), sprites);

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

    let mut game_loop = GameLoop::new(State::new(image), Executor, renderer);

    let event_loop = EventLoop::with_user_event().build().unwrap();

    event_loop.run_app(&mut game_loop).unwrap();
}
