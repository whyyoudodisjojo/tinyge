pub mod phases;
pub mod radix_sort;

#[cfg(test)]
mod tests {
    use wgpu::util::DeviceExt;

    use crate::collisions::algos::lbvh::{FlattenedBVHNode, gpu::phases::LBVHBuilder};

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ModelInfo {
        offset: u32,
        stride: u32,
    }

    async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create device");

        (device, queue)
    }

    #[test]
    fn test_gpu_lbvh_build() {
        pollster::block_on(async {
            let (device, queue) = setup_wgpu().await;

            let vertices: Vec<[f32; 3]> = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [2.0, 2.0, 2.0],
                [3.0, 2.0, 2.0],
                [2.5, 3.0, 2.0],
                [-1.0, -1.0, -1.0],
                [0.0, -1.0, -1.0],
                [-0.5, 0.0, -1.0],
                [1.5, 1.5, 1.5],
                [2.5, 1.5, 1.5],
                [2.0, 2.5, 1.5],
            ];

            let model_infos: Vec<ModelInfo> = vec![
                ModelInfo {
                    offset: 0,
                    stride: 3,
                },
                ModelInfo {
                    offset: 3,
                    stride: 3,
                },
                ModelInfo {
                    offset: 6,
                    stride: 3,
                },
                ModelInfo {
                    offset: 9,
                    stride: 3,
                },
            ];

            let num_models = model_infos.len() as u32;
            let num_verts = vertices.len() as u32;

            let model_verts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let model_infos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&model_infos),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            let mut builder = LBVHBuilder::new(num_models, num_verts, &device);
            let nodes_buffer = builder.run(model_verts_buffer, model_infos_buffer, &device, &queue);

            let nodes = FlattenedBVHNode::read_buffer(&device, &queue, &nodes_buffer);

            for i in 0..num_models as usize {
                assert_eq!(nodes[i].node_type, 0, "Node {} should be a leaf", i);
                assert_eq!(
                    nodes[i].left_child, -1,
                    "Leaf node {} should have left_child=-1",
                    i
                );
                assert_eq!(
                    nodes[i].right_child, -1,
                    "Leaf node {} should have right_child=-1",
                    i
                );
                if nodes.len() > 1 {
                    assert!(
                        nodes[i].parent >= num_models as i32,
                        "Leaf node {} should have internal node as parent",
                        i
                    );
                }
            }

            for i in num_models as usize..nodes.len() {
                assert_eq!(
                    nodes[i].node_type, 1,
                    "Node {} should be internal (type=1), got type={}",
                    i, nodes[i].node_type
                );
                assert!(
                    nodes[i].left_child >= 0,
                    "Internal node {} should have left child, got {}",
                    i,
                    nodes[i].left_child
                );
                assert!(
                    nodes[i].right_child >= 0,
                    "Internal node {} should have right child, got {}",
                    i,
                    nodes[i].right_child
                );
            }

            let root_idx = nodes.len() - 1;
            assert!(
                nodes[root_idx].left_child >= 0 || nodes[root_idx].right_child >= 0,
                "Root should have children"
            );

            assert_eq!(nodes.len(), (2 * num_models - 1) as usize);
        });
    }
}
