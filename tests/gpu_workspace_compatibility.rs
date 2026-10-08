#![cfg(feature = "gpu")]

use arkan::gpu::{GpuNetwork, GpuWorkspace, WgpuBackend, WgpuOptions};
use arkan::{KanConfig, KanNetwork};

fn model(hidden: usize) -> KanNetwork {
    KanNetwork::new(KanConfig {
        input_dim: 2,
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        output_dim: 2,
        hidden_dims: vec![hidden],
        grid_size: 3,
        spline_order: 2,
        init_seed: Some(21),
        ..KanConfig::default()
    })
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
    }
}

#[test]
#[ignore = "Requires GPU adapter"]
fn lazy_workspaces_support_all_forward_entries_and_reuse() {
    let backend = WgpuBackend::init(WgpuOptions::default()).unwrap();
    let cpu = model(3);
    let mut gpu = GpuNetwork::from_cpu(&backend, &cpu).unwrap();
    for limited in [false, true] {
        for entry in 0..5 {
            let mut workspace = if limited {
                GpuWorkspace::empty_with_limit(2, 2, backend.max_vram_alloc())
            } else {
                GpuWorkspace::empty(2, 2)
            };
            let input = [0.25, -0.4];
            let mut expected = [0.0; 2];
            cpu.forward_single(&input, &mut expected, &mut cpu.create_workspace(1));
            let output = match entry {
                0 => gpu.forward_single(&input, &mut workspace).unwrap(),
                1 => gpu.forward_batch(&input, 1, &mut workspace).unwrap(),
                2 => gpu
                    .forward_batch_async(&input, 1, &mut workspace)
                    .unwrap()
                    .wait()
                    .unwrap(),
                3 => gpu
                    .forward_batch_training(&input, 1, &mut workspace)
                    .unwrap(),
                _ => {
                    let output = gpu
                        .forward_batch_softmax(&input, 1, &mut workspace)
                        .unwrap();
                    assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                    continue;
                }
            };
            close(&output, &expected);
            for batch in [4, 1] {
                let input = input.repeat(batch);
                close(
                    &gpu.forward_batch_training(&input, batch, &mut workspace)
                        .unwrap(),
                    &expected.repeat(batch),
                );
            }
        }
    }
}

#[test]
#[ignore = "Requires GPU adapter"]
fn softmax_rejects_changed_geometry_and_incompatible_workspace() {
    let backend = WgpuBackend::init(WgpuOptions::default()).unwrap();
    let mut gpu = GpuNetwork::from_cpu(&backend, &model(3)).unwrap();
    let mut workspace = gpu.create_workspace(1).unwrap();
    gpu.layers[0].uniforms.grid_min += 0.25;
    assert!(gpu
        .forward_batch_softmax(&[0.25, -0.4], 1, &mut workspace)
        .is_err());
    let mut gpu = GpuNetwork::from_cpu(&backend, &model(3)).unwrap();
    let mut wrong = GpuWorkspace::empty(3, 2);
    assert!(gpu
        .forward_batch_softmax(&[0.25, -0.4], 1, &mut wrong)
        .is_err());
    assert!(wrong.input.is_none());
}

#[test]
#[ignore = "Requires GPU adapter"]
fn malformed_existing_tensor_metadata_returns_errors_before_preparation() {
    let backend = WgpuBackend::init(WgpuOptions::default()).unwrap();
    let mut gpu = GpuNetwork::from_cpu(&backend, &model(3)).unwrap();
    for field in 0..6 {
        let mut workspace = gpu.create_workspace(1).unwrap();
        gpu.forward_batch_training(&[0.25, -0.4], 1, &mut workspace)
            .unwrap();
        match field {
            0 => workspace.intermediates[0].shape.clear(),
            1 => workspace.z_values[0].shape.clear(),
            2 => workspace.grad_output.as_mut().unwrap().shape.clear(),
            3 => workspace.grad_input.as_mut().unwrap().shape.clear(),
            4 => workspace.input.as_mut().unwrap().shape[0] = 100,
            _ => workspace.max_batch = 100,
        }
        let batch = if field == 5 { 2 } else { 1 };
        assert!(
            gpu.forward_batch_training(&[0.25, -0.4].repeat(batch), batch, &mut workspace)
                .is_err(),
            "field {field}"
        );
    }
    let mut workspace = gpu.create_workspace(1).unwrap();
    gpu.forward_batch(&[0.25, -0.4], 1, &mut workspace).unwrap();
    workspace.intermediates[0].shape.clear();
    assert!(gpu.forward_batch(&[0.25, -0.4], 1, &mut workspace).is_err());
}

#[test]
#[ignore = "Requires GPU adapter"]
fn workspace_reallocates_for_another_models_hidden_width() {
    let backend = WgpuBackend::init(WgpuOptions::default()).unwrap();
    let mut workspace = GpuWorkspace::empty(2, 2);
    for hidden in [3, 4, 3] {
        let cpu = model(hidden);
        let mut gpu = GpuNetwork::from_cpu(&backend, &cpu).unwrap();
        for batch in [2, 1] {
            let input = [0.25, -0.4].repeat(batch);
            let mut expected = vec![0.0; input.len()];
            cpu.forward_batch(&input, &mut expected, &mut cpu.create_workspace(batch));
            close(
                &gpu.forward_batch_training(&input, batch, &mut workspace)
                    .unwrap(),
                &expected,
            );
            let mut weights = vec![];
            let mut biases = vec![];
            gpu.backward_batch(
                &vec![1.0; expected.len()],
                batch,
                &mut workspace,
                &mut weights,
                &mut biases,
            )
            .unwrap();
            assert_eq!(weights.len(), cpu.layers.len());
        }
    }
}
