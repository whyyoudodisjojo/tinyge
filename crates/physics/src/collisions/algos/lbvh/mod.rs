use bytemuck::{Pod, Zeroable};
use glam::Vec3A;

pub mod cpu;
pub mod gpu;

#[repr(C)]
#[derive(Hash, Pod, Zeroable, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Key {
    pub code: u32,
    pub idx: u32,
}

impl Key {
    pub fn mortonize(mut x: u32) -> u32 {
        x &= 0x000003ff;
        x = (x | (x << 16)) & 0xff0000ff;
        x = (x | (x << 8)) & 0x0300f00f;
        x = (x | (x << 4)) & 0x030c30c3;
        x = (x | (x << 2)) & 0x09249249;
        x
    }

    pub fn new(centroid: Vec3A, global_min: Vec3A, global_max: Vec3A, idx: usize) -> Self {
        let sz = global_max - global_min;
        let mask = sz.cmpgt(Vec3A::ZERO);
        let inv_sz = Vec3A::select(mask, Vec3A::ONE / sz, Vec3A::ZERO);
        let norm = (centroid - global_min) * inv_sz;
        let quant = norm.clamp(Vec3A::ZERO, Vec3A::ONE) * 1023.0;
        let u = quant.as_uvec3();
        let code = (Self::mortonize(u.x) << 2) | (Self::mortonize(u.y) << 1) | Self::mortonize(u.z);
        Self {
            code,
            idx: idx as u32,
        }
    }
}
