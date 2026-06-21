use winit::dpi::PhysicalSize;

pub mod executor;
pub mod state;

pub enum UpdateEvents {
    Resize(PhysicalSize<u32>),
    TimeUpdate,
}

pub struct TimedEvent;
