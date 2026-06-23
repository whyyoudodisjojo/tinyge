use winit::dpi::PhysicalSize;

use crate::shader::SpriteData;

pub mod executor;
pub mod state;

pub enum UpdateEvents {
    Resize(PhysicalSize<u32>),
    TimeUpdate,
    SpriteUpdate(Vec<SpriteData>),
}

pub enum CustomEvents {
    TimeTick,
    SpawnSprite,
    MovePlayer { dx: f32, dy: f32 },
}
