use codegen::asts::jit::JitAST;
use pollster::block_on;
use wgpu::{BufferDescriptor, BufferUsages};

pub fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("no adapter");
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("no device")
    })
}

pub fn make_input_buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let b = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (data.len() * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    b
}

pub fn run_ast(ast: JitAST, device: &wgpu::Device, queue: &wgpu::Queue, n: u32) -> Vec<f32> {
    let JitAST::Var { buffer, .. } = ast.realize(device, queue, n) else {
        panic!("expected Var");
    };
    read_buffer(device, queue, &buffer)
}

pub fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    let size = buffer.size();
    let staging = device.create_buffer(&BufferDescriptor {
        label: None,
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}
