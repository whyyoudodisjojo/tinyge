use std::time::Duration;

use rand::random;
use tinyge_core::game_loop::events::{
    BaseEvent, EventSchedule, EventsExecutor, RenderEventHandle, UpdateEventOrTimedEvent,
};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::logic::{state::State, CustomEvents, UpdateEvents};
use crate::shader::SpriteData;

pub struct Executor;

impl Executor {
    pub fn new() -> Self {
        Self
    }
}

const SPAWN_INTERVAL_SECS: f32 = 2.0;

impl EventsExecutor<State> for Executor {
    type CustomEvent = CustomEvents;
    type UpdateEvent = UpdateEvents;

    fn handle_event(
        &mut self,
        event: BaseEvent<Self::CustomEvent>,
        mut tx: RenderEventHandle<UpdateEventOrTimedEvent<Self::UpdateEvent, Self::CustomEvent>>,
        state: &State,
    ) {
        match event {
            BaseEvent::Resumed => {
                tx.force_redraw_async();
                self.emit_event(
                    CustomEvents::TimeTick,
                    EventSchedule::In(Duration::from_millis(16)),
                    tx,
                );
            }
            BaseEvent::CustomEvent(custom_event) => match custom_event {
                CustomEvents::TimeTick => {
                    tx.send(UpdateEventOrTimedEvent::UpdateEvent(
                        UpdateEvents::TimeUpdate,
                    ))
                    .unwrap();

                    // Spawn a sprite every SPAWN_INTERVAL_SECS seconds
                    let elapsed = state.start_time.elapsed().unwrap().as_secs_f32();
                    if elapsed % SPAWN_INTERVAL_SECS < 0.02 {
                        tx.send(UpdateEventOrTimedEvent::TimedEvent(
                            CustomEvents::SpawnSprite,
                        ))
                        .unwrap();
                    }

                    tx.force_redraw_async();

                    self.emit_event(
                        CustomEvents::TimeTick,
                        EventSchedule::In(Duration::from_millis(16)),
                        tx,
                    );
                }
                CustomEvents::SpawnSprite => {
                    let mut sprites = state.sprites.clone();
                    let new_sprite = SpriteData {
                        pos: [random::<f32>() * 2.0 - 1.0, random::<f32>() * 2.0 - 1.0],
                        scale: [0.1, 0.1],
                        uv_offset: [0.0, 0.0],
                        uv_scale: [1.0, 1.0],
                    };
                    sprites.push(new_sprite);

                    tx.send(UpdateEventOrTimedEvent::UpdateEvent(
                        UpdateEvents::SpriteUpdate(sprites),
                    ))
                    .unwrap();
                }
                CustomEvents::MovePlayer { dx, dy } => {
                    if !state.sprites.is_empty() {
                        let mut sprites = state.sprites.clone();
                        sprites[0].pos[0] += dx;
                        sprites[0].pos[1] += dy;

                        tx.send(UpdateEventOrTimedEvent::UpdateEvent(
                            UpdateEvents::SpriteUpdate(sprites),
                        ))
                        .unwrap();
                    }
                }
            },
            BaseEvent::WindowEvent(WindowEvent::Resized(sz)) => {
                tx.send(UpdateEventOrTimedEvent::UpdateEvent(UpdateEvents::Resize(
                    sz,
                )))
                .unwrap();
            }
            BaseEvent::WindowEvent(WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(key),
                        ..
                    },
                ..
            }) => {
                let move_speed = 0.05;
                match key {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        tx.send(UpdateEventOrTimedEvent::TimedEvent(
                            CustomEvents::MovePlayer {
                                dx: 0.0,
                                dy: move_speed,
                            },
                        ))
                        .unwrap();
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        tx.send(UpdateEventOrTimedEvent::TimedEvent(
                            CustomEvents::MovePlayer {
                                dx: 0.0,
                                dy: -move_speed,
                            },
                        ))
                        .unwrap();
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        tx.send(UpdateEventOrTimedEvent::TimedEvent(
                            CustomEvents::MovePlayer {
                                dx: -move_speed,
                                dy: 0.0,
                            },
                        ))
                        .unwrap();
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        tx.send(UpdateEventOrTimedEvent::TimedEvent(
                            CustomEvents::MovePlayer {
                                dx: move_speed,
                                dy: 0.0,
                            },
                        ))
                        .unwrap();
                    }
                    _ => {}
                }
            }
            _ => return,
        }
    }
}
