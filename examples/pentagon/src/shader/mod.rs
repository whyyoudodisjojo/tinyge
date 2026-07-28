pub mod pentagon;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}