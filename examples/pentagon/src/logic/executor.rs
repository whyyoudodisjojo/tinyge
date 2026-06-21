use std::time::Duration;

use tinyge_core::game_loop::events::{BaseEvent, EventsExecutor};
use winit::event::WindowEvent;

use crate::logic::{TimedEvent, UpdateEvents, state::State};

pub struct Executor;

impl EventsExecutor<State> for Executor {
    type CustomEvent = TimedEvent;
    type UpdateEvent = UpdateEvents;

    fn handle_event(
        &mut self,
        event: tinyge_core::game_loop::events::BaseEvent<Self::CustomEvent>,
        mut tx: tinyge_core::game_loop::events::RenderEventHandle<
            tinyge_core::game_loop::events::UpdateEventOrTimedEvent<
                Self::UpdateEvent,
                Self::CustomEvent,
            >,
        >,
        _state: &State,
    ) {
        match event {
            BaseEvent::Resumed => {
                tx.force_redraw_async();
                self.emit_event(
                    TimedEvent,
                    tinyge_core::game_loop::events::EventSchedule::In(Duration::from_millis(7)),
                    tx,
                );
            }
            BaseEvent::CustomEvent(_) => {
                tx.send(
                    tinyge_core::game_loop::events::UpdateEventOrTimedEvent::UpdateEvent(
                        UpdateEvents::TimeUpdate,
                    ),
                )
                .unwrap();

                tx.force_redraw_async();

                self.emit_event(
                    TimedEvent,
                    tinyge_core::game_loop::events::EventSchedule::In(Duration::from_millis(7)),
                    tx,
                );
            }
            BaseEvent::WindowEvent(WindowEvent::Resized(sz)) => tx
                .send(
                    tinyge_core::game_loop::events::UpdateEventOrTimedEvent::UpdateEvent(
                        UpdateEvents::Resize(sz),
                    ),
                )
                .unwrap(),
            _ => return,
        }
    }
}
