use std::{
    sync::{
        Weak,
        mpsc::{SendError, Sender},
    },
    thread::sleep,
    time::Duration,
};

use rayon::spawn;
use winit::{
    event::{DeviceEvent, StartCause, WindowEvent},
    window::Window,
};

pub enum EventSchedule {
    Now,
    In(Duration),
}

pub enum BaseEvent<CustomEvent> {
    Resumed,
    AboutToWait,
    DeviceEvent(winit::event::DeviceId, DeviceEvent),
    MemoryWarning,
    Exiting,
    NewEvents(StartCause),
    Suspended,
    UserEvent,
    WindowEvent(WindowEvent),
    CustomEvent(CustomEvent),
}

pub struct EventHandler<Executor> {
    pub executor: Executor,
}

pub enum UpdateEventOrTimedEvent<UpdateEvent, CustomEvent> {
    UpdateEvent(UpdateEvent),
    TimedEvent(CustomEvent),
}

#[derive(Clone)]
pub struct RenderEventHandle<T> {
    tx: Sender<T>,
    redraw_required: bool,
    window: Option<Weak<Window>>, // Background tasks dont have window handle so it cant trigger a redraw
}

impl<T> RenderEventHandle<T> {
    pub fn new(tx: Sender<T>, window: Option<Weak<Window>>) -> Self {
        Self {
            tx,
            redraw_required: false,
            window,
        }
    }

    pub fn force_redraw_async(&mut self) {
        self.redraw_required = true;
    }

    pub fn send(&mut self, t: T) -> Result<(), SendError<T>> {
        self.tx.send(t).inspect(|_| self.redraw_required = true)
    }
}

impl<T> Drop for RenderEventHandle<T> {
    fn drop(&mut self) {
        if self.redraw_required {
            self.window
                .as_ref()
                .and_then(|a| a.upgrade())
                .map(|a| a.request_redraw());
        }
    }
}

pub trait EventsExecutor<State>
where
    Self::CustomEvent: Send + Sync + 'static,
    Self::UpdateEvent: Send + Sync + 'static,
{
    type CustomEvent;
    type UpdateEvent;
    fn handle_event(
        &mut self,
        event: BaseEvent<Self::CustomEvent>,
        tx: RenderEventHandle<UpdateEventOrTimedEvent<Self::UpdateEvent, Self::CustomEvent>>,
        state: &State,
    );
    fn emit_event(
        &self,
        event: Self::CustomEvent,
        when: EventSchedule,
        mut tx: RenderEventHandle<UpdateEventOrTimedEvent<Self::UpdateEvent, Self::CustomEvent>>,
    ) {
        match when {
            EventSchedule::Now => tx.send(UpdateEventOrTimedEvent::TimedEvent(event)).unwrap(),
            EventSchedule::In(duration) => {
                spawn(move || {
                    sleep(duration);
                    tx.send(UpdateEventOrTimedEvent::TimedEvent(event)).unwrap();
                });
            }
        }
    }
}
