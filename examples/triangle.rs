use tinyge::{
    game_loop::{
        GameLoop,
        events::{BaseEvent, EventsExecutor},
    },
    renderer::{
        AdapterDescriptor, Renderer, RendererDescriptor, RendererInstanceDescriptor,
        strategies::{
            RenderAble,
            single::{SinglePass, StateRenderSinglePass},
        },
    },
    shaders::{
        ColorTargetStateData, Shader, ShaderBuffers, ShaderManager, ShaderMeshBufferLayouts,
        ShaderPipelineDescriptor, ShaderVertexBufferLayout,
    },
    state::{StateRender, StateUpdates},
};
use wgpu::{
    Backends, BlendComponent, BlendState, Color, ColorWrites, Device, MultisampleState, Operations,
    PrimitiveState, Queue, RenderPassColorAttachment, RenderPassDescriptor, VertexAttribute,
    VertexBufferLayout, VertexFormat, wgt::DeviceDescriptor,
};
use winit::{dpi::PhysicalSize, event::WindowEvent, event_loop::EventLoop};

#[derive(Hash, Clone, PartialEq, Eq)]
pub struct ShaderId(u32);

pub struct State {
    buffers: Option<ShaderBuffers>,
    sz: PhysicalSize<u32>,
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
            index_buffer_size: 20,
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

const TRIANGE_GEO: [Vertex; 3] = [
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    }, // Top (Red)
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    }, // Bottom Left (Green)
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    }, // Bottom Right (Blue)
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4, /* padding */ 0];

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
        match &mut self.buffers {
            Some(b) => b.copy_data_into(&new_buffer, device, queue),
            None => {
                queue.write_buffer(
                    &new_buffer.vertex_buffers[0],
                    0,
                    bytemuck::cast_slice(&TRIANGE_GEO),
                );
                queue.write_buffer(&new_buffer.index_buffer, 0, bytemuck::cast_slice(INDICES));
            }
        }

        self.buffers = Some(new_buffer);
    }

    fn update(&mut self, update_event: Self::UpdateEvent, _queue: Option<&wgpu::Queue>) {
        self.sz = update_event;
    }
}

impl StateRender for State {
    type RenderStrategy = SinglePass;

    fn render_height(&self) -> u32 {
        self.sz.height
    }

    fn render_width(&self) -> u32 {
        self.sz.width
    }
}

impl RenderAble<ShaderId> for State {
    fn render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &std::collections::HashMap<ShaderId, wgpu::RenderPipeline>,
        view: &wgpu::TextureView,
    ) {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: wgpu::LoadOp::Clear(Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(pipeline_cache.get(&ShaderId(1)).unwrap());
        render_pass.set_vertex_buffer(
            0,
            self.buffers.as_ref().unwrap().vertex_buffers[0].slice(..),
        );
        render_pass.set_index_buffer(
            self.buffers.as_ref().unwrap().index_buffer.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        render_pass.draw_indexed(0..3, 0, 0..1);
    }
}

impl StateRenderSinglePass<ShaderId> for State {}

// TODO: Investigate Clone requirement
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
        _state: &State,
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

// TODO: Move to Arc in shader manager
const SHADER: Triangle = Triangle;

fn main() {
    let mut shader_manager: ShaderManager<ShaderId> = ShaderManager::new();

    shader_manager.register_shader(ShaderId(1), &SHADER);

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
    let mut game_loop = GameLoop::new(
        State {
            buffers: None,
            sz: PhysicalSize {
                width: 400,
                height: 600,
            },
        },
        Executor,
        renderer,
    );

    let event_loop = EventLoop::with_user_event().build().unwrap();

    event_loop.run_app(&mut game_loop).unwrap();
}
