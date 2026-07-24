use super::helpers::*;
use codegen::asts::jit::JitAST;

#[test]
fn reshape_2x4_to_4x2() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 8]>(a_buf)
        .reshape(vec![4, 2])
        .reshape(vec![8]);
    let result = run_ast(ast, &device, &queue, 8);

    for i in 0..8 {
        assert_eq!(result[i], a_data[i]);
    }
}

#[test]
fn permute_transpose() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 6]>(a_buf)
        .reshape(vec![2, 3])
        .permute(vec![1, 0])
        .reshape(vec![6]);
    let result = run_ast(ast, &device, &queue, 6);

    let expected: Vec<f32> = vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
    for i in 0..6 {
        assert_eq!(result[i], expected[i]);
    }
}

#[test]
fn pad_2d() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 4]>(a_buf)
        .reshape(vec![2, 2])
        .pad(vec![(1, 1), (1, 1)])
        .reshape(vec![16]);
    let result = run_ast(ast, &device, &queue, 16);

    let mut expected = [0.0f32; 16];
    expected[5] = 0.0;
    expected[6] = 1.0;
    expected[9] = 2.0;
    expected[10] = 3.0;
    for i in 0..16 {
        assert_eq!(result[i], expected[i], "at index {i}");
    }
}

#[test]
fn flip_axis_1() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 6]>(a_buf)
        .reshape(vec![2, 3])
        .flip(1)
        .reshape(vec![6]);
    let result = run_ast(ast, &device, &queue, 6);

    let expected: Vec<f32> = vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0];
    for i in 0..6 {
        assert_eq!(result[i], expected[i]);
    }
}

#[test]
fn shrink_2d() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 16]>(a_buf)
        .reshape(vec![4, 4])
        .shrink(vec![(1, 1), (1, 1)])
        .reshape(vec![4]);
    let result = run_ast(ast, &device, &queue, 4);

    let expected: Vec<f32> = vec![5.0, 6.0, 9.0, 10.0];
    for i in 0..4 {
        assert_eq!(result[i], expected[i]);
    }
}

#[test]
fn expand_broadcast() {
    let (device, queue) = setup_wgpu();
    let a_data: Vec<f32> = (0..4).map(|i| i as f32).collect();
    let a_buf = make_input_buffer(&device, &queue, &a_data);

    let ast = JitAST::new::<[f32; 4]>(a_buf)
        .reshape(vec![4])
        .expand(vec![2, 4])
        .reshape(vec![8]);
    let result = run_ast(ast, &device, &queue, 8);

    let expected: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0];
    for i in 0..8 {
        assert_eq!(result[i], expected[i]);
    }
}
