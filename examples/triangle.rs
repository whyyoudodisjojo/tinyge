use tinyge::{
    game_loop::{
        GameLoop,
        events::{BaseEvent, EventsExecutor},
    },
    renderer::{AdapterDescriptor, Renderer, RendererDescriptor, RendererInstanceDescriptor},
    shaders::{
        ColorTargetStateData, Shader, ShaderBuffers, ShaderManager, ShaderMeshBufferLayouts,
        ShaderPipelineDescriptor, ShaderVertexBufferLayout,
    },
    state::StateUpdates,
};
use wgpu::{
    Backends, BlendComponent, BlendState, ColorWrites, Device, MultisampleState, PrimitiveState,
    Queue, VertexAttribute, VertexBufferLayout, VertexFormat, wgt::DeviceDescriptor,
};
use winit::{dpi::PhysicalSize, event::WindowEvent};

#[derive(Hash, Clone, PartialEq, Eq)]
pub struct ShaderId(u32);

pub struct State {
    buffers: Option<ShaderBuffers>,
}

struct Triangle;

impl Shader for Triangle {
    fn mesh_buffers_layouts(&self) -> tinyge::shaders::ShaderMeshBufferLayouts<'static> {
        let vertex_sz = (3 * 4) + (3 * 4);
        let vertex_buffer_sz = vertex_sz * 3;

        let layout = VertexBufferLayout {
            array_stride: vertex_sz,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                },
            ],
        };

        ShaderMeshBufferLayouts {
            vertex_buffer_layouts: vec![ShaderVertexBufferLayout {
                vertex_buffer: layout,
                vertex_buffer_size: vertex_buffer_sz,
            }],
            index_buffer_size: 6,
        }
    }

    fn resource_buffers_bind_group_layouts(
        &self,
    ) -> Vec<tinyge::shaders::ResourceBufferBindGroupLayoutWithUsages> {
        vec![]
    }

    fn load_source_code(&self) -> &'static str {
        include_str!("./shaders/triangle.wgsl")
    }

    fn shader_pipeline_desc(&self) -> tinyge::shaders::ShaderPipelineDescriptor<'static> {
        ShaderPipelineDescriptor {
            vertex_entry_point: Some("vs_main"),
            vertex_compilation_options: Default::default(),
            primitive_state: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                conservative: false,
                polygon_mode: wgpu::PolygonMode::Fill,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment_targets: &[Some(ColorTargetStateData {
                blend: Some(BlendState {
                    color: BlendComponent::REPLACE,
                    alpha: BlendComponent::REPLACE,
                }),
                write_mask: ColorWrites::ALL,
            })],
            fragment_compilation_options: Default::default(),
            fragment_entry_point: Some("fs_main"),
            multiview_mask: None,
        }
    }
}

impl StateUpdates for State {
    type K = ShaderId;
    type UpdateEvent = PhysicalSize<u32>;

    fn handle_shader_recompilation(
        &mut self,
        new_buffers: std::collections::HashMap<Self::K, tinyge::shaders::ShaderBuffers>,
        queue: &Queue,
        device: &Device,
    ) {
        let new_buffer = new_buffers.into_iter().next().unwrap().1;

        self.buffers
            .as_ref()
            .map(|b| b.copy_data_into(&new_buffer, device, queue));

        self.buffers = Some(new_buffer);
    }

    fn update(&mut self, _update_event: Self::UpdateEvent, _queue: Option<&wgpu::Queue>) {}
}

#[derive(Clone)]
pub struct Executor;
impl EventsExecutor<State> for Executor {
    type CustomEvent = ();
    type UpdateEvent = PhysicalSize<u32>;

    fn handle_event(
        &mut self,
        event: tinyge::game_loop::events::BaseEvent<Self::CustomEvent>,
        mut tx: tinyge::game_loop::events::RenderEventHandle<
            tinyge::game_loop::events::UpdateEventOrTimedEvent<
                Self::UpdateEvent,
                Self::CustomEvent,
            >,
        >,
        state: &State,
    ) {
        match event {
            BaseEvent::Resumed => {
                tx.force_redraw_async();
            }
            BaseEvent::WindowEvent(WindowEvent::Resized(sz)) => tx
                .send(tinyge::game_loop::events::UpdateEventOrTimedEvent::UpdateEvent(sz))
                .unwrap(),
            _ => return,
        }
    }
}

fn main() {
    let mut shader_manager: ShaderManager<ShaderId> = ShaderManager::new();

    let shader = Triangle;
    shader_manager.register_shader(ShaderId(1), &shader);

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

    let game_loop = GameLoop::new(State { buffers: None }, Executor, renderer);
}
