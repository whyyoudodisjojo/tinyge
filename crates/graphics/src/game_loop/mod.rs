pub mod events;

use std::{
    hash::Hash,
    sync::{
        Arc,
        mpsc::{self, Receiver, Sender},
    },
};

use winit::{application::ApplicationHandler, event::WindowEvent, window::Window};

use crate::{
    game_loop::events::{BaseEvent, EventsExecutor, RenderEventHandle, UpdateEventOrTimedEvent},
    renderer::{
        Renderer,
        strategies::{RenderDispatcher, RenderPath, single::StateRenderSinglePass},
    },
    state::StateUpdates,
};

pub struct GameLoop<State, Executor>
where
    Executor: EventsExecutor<State>,
    State: Send + Sync + 'static + StateUpdates,
{
    pub state: State,
    pub executor: Executor,
    pub renderer: Renderer<'static, State::K>,
    pub rx: Receiver<UpdateEventOrTimedEvent<Executor::UpdateEvent, Executor::CustomEvent>>,
    pub tx: Sender<UpdateEventOrTimedEvent<Executor::UpdateEvent, Executor::CustomEvent>>,
}

impl<State, Executor> GameLoop<State, Executor>
where
    Executor: EventsExecutor<State>,
    State: Send + Sync + 'static + StateUpdates,
{
    pub fn new(state: State, executor: Executor, renderer: Renderer<'static, State::K>) -> Self {
        let (tx, rx) = mpsc::channel();

        Self {
            state,
            executor,
            renderer,
            rx,
            tx,
        }
    }
}

impl<State, Executor> ApplicationHandler<()> for GameLoop<State, Executor>
where
    Executor: EventsExecutor<State>,
    State: Send
        + Sync
        + 'static
        + StateUpdates<UpdateEvent = <Executor as EventsExecutor<State>>::UpdateEvent>
        + StateRenderSinglePass<State::K>,
    State::K: Eq + PartialEq + Hash + Clone,
    for<'b> RenderPath<'b, State, State::RenderStrategy>: RenderDispatcher<State::K>,
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attrs = Window::default_attributes();

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        pollster::block_on(self.renderer.init(window));

        let (queue, device) = self
            .renderer
            .ctx
            .as_ref()
            .map(|c| (&c.queue, &c.device))
            .unwrap();

        self.state
            .init(&self.renderer.shader_manager.shaders, device, queue);

        self.executor.handle_event(
            BaseEvent::Resumed,
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &mut self.state,
        );
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.executor.handle_event(
            BaseEvent::AboutToWait,
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        self.executor.handle_event(
            BaseEvent::DeviceEvent(device_id, event),
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.executor.handle_event(
            BaseEvent::Exiting,
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );
    }

    fn memory_warning(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.executor.handle_event(
            BaseEvent::MemoryWarning,
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );
    }

    fn new_events(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        self.executor.handle_event(
            BaseEvent::NewEvents(cause),
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );
    }

    fn suspended(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.executor.handle_event(
            BaseEvent::Suspended,
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, _event: ()) {
        self.executor.handle_event(
            BaseEvent::UserEvent,
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        self.executor.handle_event(
            BaseEvent::WindowEvent(event.clone()),
            RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
            &self.state,
        );

        if event == WindowEvent::RedrawRequested {
            while let Ok(artifact_or_timed_event) = self.rx.try_recv() {
                match artifact_or_timed_event {
                    UpdateEventOrTimedEvent::UpdateEvent(r) => self
                        .state
                        .update(r, self.renderer.ctx.as_ref().map(|c| &c.queue)),
                    UpdateEventOrTimedEvent::TimedEvent(e) => self.executor.handle_event(
                        BaseEvent::CustomEvent(e),
                        RenderEventHandle::new(self.tx.clone(), self.renderer.window()),
                        &self.state,
                    ),
                }
            }

            RenderPath::new(&mut self.state).dispatch_render(&mut self.renderer);
        }
    }
}
