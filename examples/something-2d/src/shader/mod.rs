pub mod sprites;

#[derive(Hash, Clone, PartialEq, Eq)]
pub struct ShaderId(pub u32);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteData {
    pub pos: [f32; 2],
    pub scale: [f32; 2],
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],
}

impl Default for SpriteData {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0],
            scale: [0.05, 0.05],
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
        }
    }
}
