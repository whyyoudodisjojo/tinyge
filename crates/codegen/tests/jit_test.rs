use codegen::asts::jit::{JitAST, JitBinOp};
use codegen::asts::lowered::BinOp;
use codegen::dt::{BasicTy, BasicTyOrStructRef, DType, MaybeAtomic, VecTy};
use pollster::block_on;
use wgpu::{BufferDescriptor, BufferUsages};

fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
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

fn make_input_buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> wgpu::Buffer {
    let b = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (data.len() * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    b
}

fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
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

#[test]
fn elementwise_add() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;

    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);

    let arr_dt = DType::Vector(VecTy::Array(
        MaybeAtomic::Naked(BasicTyOrStructRef::BasicTy(BasicTy::F32)),
        None,
    ));

    let ast = JitAST::BinOp {
        lhs: Box::new(JitAST::Var {
            buffer: a_buf,
            dtype: arr_dt.clone(),
        }),
        rhs: Box::new(JitAST::Var {
            buffer: b_buf,
            dtype: arr_dt.clone(),
        }),
        op: JitBinOp::Basic(BinOp::Add),
    };

    let JitAST::Var {
        buffer: result_buf, ..
    } = ast.realize(&device, &queue, n)
    else {
        panic!("expected Var");
    };
    let result = read_buffer(&device, &queue, &result_buf);

    for i in 0..n as usize {
        assert_eq!(result[i], a_data[i] + b_data[i]);
    }
}
