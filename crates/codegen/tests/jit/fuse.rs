use super::helpers::*;
use codegen::asts::jit::JitAST;

#[test]
fn chain_movement_reduce() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 6]>(a_buf)
        .reshape(vec![2, 3])
        .flip(1)
        .sum(0)
        .prod_all();
    let result = run_ast(ast, &device, &queue, 1);

    assert!((result[0] - 105.0).abs() < 1e-4);
}
