use super::helpers::*;
use codegen::asts::jit::JitAST;

#[test]
fn elementwise_add() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf) + JitAST::new::<[f32; 64]>(b_buf);
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert_eq!(result[i], a_data[i] + b_data[i]);
    }
}

#[test]
fn elementwise_sub() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf) - JitAST::new::<[f32; 64]>(b_buf);
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert_eq!(result[i], a_data[i] - b_data[i]);
    }
}

#[test]
fn elementwise_mul() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf) * JitAST::new::<[f32; 64]>(b_buf);
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert_eq!(result[i], a_data[i] * b_data[i]);
    }
}

#[test]
fn elementwise_div() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).rev().collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf) / JitAST::new::<[f32; 64]>(b_buf);
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert!((result[i] - a_data[i] / b_data[i]).abs() < 1e-5);
    }
}

#[test]
fn max() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf).max(JitAST::new::<[f32; 64]>(b_buf));
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert_eq!(result[i], a_data[i].max(b_data[i]));
    }
}

#[test]
fn pow() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| (i % 10 + 1) as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| ((i % 3) + 1) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf).pow(JitAST::new::<[f32; 64]>(b_buf));
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert!((result[i] - a_data[i].powf(b_data[i])).abs() < 1e-2);
    }
}

#[test]
fn exp2() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| ((i as i32 % 10) - 5) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf).exp2();
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert!((result[i] - 2.0f32.powf(a_data[i])).abs() < 1e-5);
    }
}

#[test]
fn sin() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n)
        .map(|i| (i as f32) * std::f32::consts::TAU / n as f32)
        .collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf).sin();
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert!((result[i] - a_data[i].sin()).abs() < 1e-6);
    }
}

#[test]
fn sqrt() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf).sqrt();
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert!((result[i] - a_data[i].sqrt()).abs() < 1e-6);
    }
}

#[test]
fn reciprocal() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf).reciprocal();
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert!((result[i] - (1.0 / a_data[i])).abs() < 1e-6);
    }
}

#[test]
fn neg() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = -JitAST::new::<[f32; 64]>(a_buf);
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert_eq!(result[i], -a_data[i]);
    }
}

#[test]
fn where_() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let c_data: Vec<f32> = (0..n).map(|_| -1.0).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);
    let c_buf = make_input_buffer(&device, &queue, &c_data);

    let ast = JitAST::where_(
        JitAST::new::<[f32; 64]>(a_buf).eq(0.0f32.into()),
        JitAST::new::<[f32; 64]>(b_buf),
        JitAST::new::<[f32; 64]>(c_buf),
    );
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        let expected = if (i as u32) % 2 == 0 { i as f32 } else { -1.0 };
        assert_eq!(result[i], expected);
    }
}

#[test]
fn mulacc() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let c_data: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);
    let b_buf = make_input_buffer(&device, &queue, &b_data);
    let c_buf = make_input_buffer(&device, &queue, &c_data);

    let ast = JitAST::mulacc(
        JitAST::new::<[f32; 64]>(a_buf),
        JitAST::new::<[f32; 64]>(b_buf),
        JitAST::new::<[f32; 64]>(c_buf),
    );
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert_eq!(result[i], a_data[i] * b_data[i] + c_data[i]);
    }
}

#[test]
fn add_with_const() {
    let (device, queue) = setup_wgpu();
    let n = 64u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 64]>(a_buf) + 10.0f32.into();
    let result = run_ast(ast, &device, &queue, n);

    for i in 0..n as usize {
        assert_eq!(result[i], a_data[i] + 10.0);
    }
}
