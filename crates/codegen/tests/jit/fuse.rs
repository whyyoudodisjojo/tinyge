use super::helpers::*;
use codegen::asts::jit::JitAST;

#[test]
fn sum_axis_0() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<Vec<f32>>(a_buf).reshape(vec![2, 3]).sum(0);
    let result = run_ast(ast, &device, &queue, 3);

    assert_eq!(result, vec![3.0, 5.0, 7.0]);
}

#[test]
fn max_all() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<Vec<f32>>(a_buf).reshape(vec![2, 3]).max_all();
    let result = run_ast(ast, &device, &queue, 1);

    assert_eq!(result, vec![5.0]);
}

#[test]
fn prod_all() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (1..=5).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<Vec<f32>>(a_buf).reshape(vec![5]).prod_all();
    let result = run_ast(ast, &device, &queue, 1);

    assert!((result[0] - 120.0).abs() < 1e-4);
}
