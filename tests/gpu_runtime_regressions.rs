#![cfg(feature = "gpu")]
use arkan::gpu::{
    GpuAdam, GpuAdamConfig, GpuNetwork, GpuSgd, GpuSgdConfig, GpuTensor, GpuWorkspace, WgpuBackend,
    WgpuOptions,
};
use arkan::{KanConfigBuilder, KanNetwork};

fn backend() -> WgpuBackend {
    let backend = WgpuBackend::init(WgpuOptions::default()).unwrap();
    eprintln!("GPU adapter: {:?}", backend.adapter_info());
    backend
}
fn network(order: usize) -> KanNetwork {
    KanNetwork::new(
        KanConfigBuilder::new()
            .input_dim(2)
            .output_dim(1)
            .hidden_dims(vec![3])
            .grid_size(5)
            .spline_order(order)
            .grid_range(-1., 1.)
            .normalization(vec![0.5, -0.2], vec![0.3, 0.7])
            .seed(42)
            .build()
            .unwrap(),
    )
}

#[test]
#[ignore = "Requires GPU adapter"]
fn model_snapshot_rejects_invalid_cpu_conversion_and_sync() {
    let b = backend();
    let mut cpu = network(3);
    let gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let before = cpu.layers[0].weights.clone();
    cpu.config.input_dim = 0;
    assert!(GpuNetwork::from_cpu(&b, &cpu).is_err());
    assert!(gpu.sync_weights_to_cpu(&mut cpu).is_err());
    assert_eq!(cpu.layers[0].weights, before);
}

#[test]
#[ignore = "Requires GPU adapter"]
fn model_snapshot_rejects_mutated_gpu_geometry_before_execution() {
    let b = backend();
    let cpu = network(3);
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut workspace = gpu.create_workspace(1).unwrap();
    gpu.layers[0].in_dim += 1;
    assert!(gpu.forward_batch(&[0.5, -0.2], 1, &mut workspace).is_err());
}

#[test]
#[ignore = "Requires GPU adapter"]
fn native_clipping_preserves_large_finite_gradient_direction() {
    let b = backend();
    let mut cpu = network(3);
    for layer in &mut cpu.layers {
        layer.weights.fill(0.0);
        layer.bias.fill(0.0);
    }
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut workspace = gpu.create_workspace(1).unwrap();
    let mut optimizer = GpuAdam::new(
        b.device.clone(),
        b.queue.clone(),
        &gpu.layer_param_sizes(),
        GpuAdamConfig::with_lr(0.01),
    );
    gpu.train_step_gpu_native_with_options(
        &[0.5, -0.2],
        &[-1e20],
        1,
        None,
        &mut workspace,
        &mut optimizer,
        &arkan::TrainOptions {
            max_grad_norm: Some(1.0),
            weight_decay: 0.0,
        },
    )
    .unwrap();
    let mut sum = 0.0f64;
    for tensor in workspace.grad_weights.iter().chain(&workspace.grad_bias) {
        for value in tensor.download(&b.device, &b.queue).unwrap() {
            sum += (value as f64).powi(2);
        }
    }
    assert!(
        (sum.sqrt() - 1.0).abs() < 1e-5,
        "clipped norm {}",
        sum.sqrt()
    );
    gpu.sync_weights_to_cpu(&mut cpu).unwrap();
    assert!(cpu.layers.last().unwrap().bias[0] < 0.0);
}

#[test]
#[ignore = "Requires GPU adapter"]
fn native_training_rejects_invalid_clip_threshold_before_updates() {
    let b = backend();
    let mut cpu = network(3);
    let before = cpu.layers[0].weights.clone();
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut workspace = gpu.create_workspace(1).unwrap();
    let mut optimizer = GpuAdam::new(
        b.device.clone(),
        b.queue.clone(),
        &gpu.layer_param_sizes(),
        GpuAdamConfig::with_lr(0.01),
    );
    for threshold in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(gpu
            .train_step_gpu_native_with_options(
                &[0.5, -0.2],
                &[1.0],
                1,
                None,
                &mut workspace,
                &mut optimizer,
                &arkan::TrainOptions {
                    max_grad_norm: Some(threshold),
                    weight_decay: 0.0
                },
            )
            .is_err());
    }
    gpu.sync_weights_to_cpu(&mut cpu).unwrap();
    assert_eq!(cpu.layers[0].weights, before);
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 2e-4, "index {i}: {a} != {b}");
    }
}
#[test]
#[ignore = "Requires GPU adapter"]
fn normalized_forward_and_backward_match_cpu() {
    let b = backend();
    for order in 2..=5 {
        let mut cpu = network(order);
        cpu.layers[1].set_normalization(&[0.1, -0.1, 0.2], &[0.4, 0.6, 0.8]);
        let input = [0.5, -0.2, 10., -10., 0.7, 0.1];
        let mut cpu_ws = cpu.create_workspace(3);
        let mut expected = vec![0.; 3];
        cpu.forward_batch(&input, &mut expected, &mut cpu_ws);
        let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
        let mut ws = gpu.create_workspace(3).unwrap();
        close(&gpu.forward_batch(&input, 3, &mut ws).unwrap(), &expected);
        close(
            &gpu.forward_batch_training(&input, 3, &mut ws).unwrap(),
            &expected,
        );
        let mut gw = vec![];
        let mut gb = vec![];
        let gi = gpu
            .backward_batch(&[1.; 3], 3, &mut ws, &mut gw, &mut gb)
            .unwrap();
        // Derive gradients with the CPU training contract: loss gradient = 1.
        let targets: Vec<_> = expected.iter().map(|v| v - 1.5).collect();
        cpu.train_step(&input, &targets, None, 0., &mut cpu_ws);
        for i in 0..gw.len() {
            close(&gw[i], &cpu_ws.weight_grads[i]);
            close(&gb[i], &cpu_ws.bias_grads[i]);
        }
        assert_eq!(gi[2], 0.);
        assert_eq!(gi[3], 0.);
        // Check unsaturated input derivative independently by finite differences.
        for i in [0, 1, 4, 5] {
            let mut plus = input;
            let mut minus = input;
            plus[i] += 0.001;
            minus[i] -= 0.001;
            let mut p = vec![0.; 3];
            let mut m = vec![0.; 3];
            cpu.forward_batch(&plus, &mut p, &mut cpu_ws);
            cpu.forward_batch(&minus, &mut m, &mut cpu_ws);
            let numerical = (p.iter().sum::<f32>() - m.iter().sum::<f32>()) / 0.002;
            assert!(
                (gi[i] - numerical).abs() < 0.003,
                "input {i}: {} != {numerical}",
                gi[i]
            );
        }
    }
}
#[test]
#[ignore = "Requires GPU adapter"]
fn training_workspace_growth_matches_fresh_workspace() {
    let b = backend();
    let cpu = network(3);
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut reused = gpu.create_workspace(1).unwrap();
    for batch in [1, 4, 2, 16] {
        let input = vec![0.5; batch * 2];
        let grad = vec![1.; batch];
        let mut fresh = gpu.create_workspace(batch).unwrap();
        let mut a = vec![];
        let mut ab = vec![];
        let mut c = vec![];
        let mut cb = vec![];
        gpu.forward_batch_training(&input, batch, &mut reused)
            .unwrap();
        let ai = gpu
            .backward_batch(&grad, batch, &mut reused, &mut a, &mut ab)
            .unwrap();
        gpu.forward_batch_training(&input, batch, &mut fresh)
            .unwrap();
        let ci = gpu
            .backward_batch(&grad, batch, &mut fresh, &mut c, &mut cb)
            .unwrap();
        close(&ai, &ci);
        for i in 0..a.len() {
            close(&a[i], &c[i]);
            close(&ab[i], &cb[i]);
        }
    }
}
#[test]
#[ignore = "Requires GPU adapter"]
fn optimizer_rejects_missing_and_misshaped_gradients() {
    let b = backend();
    let p = GpuTensor::storage_rw(&b.device, &[1.; 4], vec![4]).unwrap();
    let bias = GpuTensor::storage_rw(&b.device, &[1.], vec![1]).unwrap();
    let short = GpuTensor::storage_rw(&b.device, &[1.], vec![1]).unwrap();
    let params = [(&p.buffer, &bias.buffer)];
    let mut adam = GpuAdam::new(
        b.device.clone(),
        b.queue.clone(),
        &[(4, 1)],
        GpuAdamConfig::default(),
    );
    let mut sgd = GpuSgd::new(
        b.device.clone(),
        b.queue.clone(),
        &[(4, 1)],
        GpuSgdConfig::default(),
    );
    assert!(adam.step(&params, &[]).is_err());
    assert_eq!(adam.t, 0);
    assert!(sgd.step(&params, &[]).is_err());
    assert!(adam
        .step(&params, &[(&short.buffer, &bias.buffer)])
        .is_err());
    assert!(sgd.step(&params, &[(&short.buffer, &bias.buffer)]).is_err());
    close(&p.download(&b.device, &b.queue).unwrap(), &[1.; 4]);
}
#[test]
#[ignore = "Requires GPU adapter"]
fn workspace_custom_cap_covers_all_training_storage() {
    let b = backend();
    let mut ws = GpuWorkspace::new_with_limit(&b.device, 1, 1, 1, 32).unwrap();
    assert!(ws.ensure_intermediates(&b.device, &[1, 64, 1], 1).is_err());
    assert!(ws.prepare_training(&b.device, &[64, 1], 1).is_err());
    assert!(ws.prepare_grad_buffers(&b.device, &[(1, 1, 64)]).is_err());
}

#[test]
#[ignore = "Requires GPU adapter"]
fn checked_allocations_reject_overflow_and_device_limits() {
    let b = backend();
    assert!(GpuTensor::uninit_with_limit(
        &b.device,
        vec![usize::MAX, 2],
        wgpu::BufferUsages::empty(),
        None
    )
    .is_err());
    let beyond = b.device.limits().max_storage_buffer_binding_size as usize / 4 + 1;
    assert!(GpuTensor::uninit_with_limit(
        &b.device,
        vec![beyond],
        wgpu::BufferUsages::empty(),
        None
    )
    .is_err());
    assert!(GpuAdam::new_with_limit(
        b.device.clone(),
        b.queue.clone(),
        &[(9, 1)],
        GpuAdamConfig::default(),
        32
    )
    .is_err());
    assert!(GpuSgd::new_with_limit(
        b.device.clone(),
        b.queue.clone(),
        &[(9, 1)],
        GpuSgdConfig::default(),
        32
    )
    .is_err());
}
#[test]
#[ignore = "Requires GPU adapter"]
fn layout_changes_resize_saved_inputs_and_bias_gradients() {
    let b = backend();
    let mut ws = GpuWorkspace::new(&b.device, 1, 2, 1).unwrap();
    ws.prepare_training(&b.device, &[2, 3, 1], 1).unwrap();
    ws.prepare_training(&b.device, &[2, 9, 1], 1).unwrap();
    assert_eq!(ws.z_values[1].shape, [1, 9]);
    assert!(ws.grad_input.as_ref().unwrap().shape[1] >= 9);
    ws.prepare_grad_buffers(&b.device, &[(9, 1, 4)]).unwrap();
    ws.prepare_grad_buffers(&b.device, &[(1, 9, 4)]).unwrap();
    assert_eq!(ws.grad_bias[0].num_elements(), 9);
    ws.ensure_intermediates(&b.device, &[2, 9, 1], 1).unwrap();
    assert_eq!(ws.intermediates.len(), 1);
}
#[test]
#[ignore = "Requires GPU adapter"]
fn async_submissions_and_softmax_preserve_queued_results() {
    let b = backend();
    let mut cpu = network(3);
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut ws = gpu.create_workspace(1).unwrap();
    let a = gpu.forward_batch_async(&[0.5, -0.2], 1, &mut ws).unwrap();
    let input = [0.7, 0.1, 1., -1.];
    let c = gpu.forward_batch_async(&input, 2, &mut ws).unwrap();
    let mut cpu_ws = cpu.create_workspace(2);
    let mut expected = vec![0.; 2];
    cpu.forward_batch(&input, &mut expected, &mut cpu_ws);
    close(&c.wait().unwrap(), &expected);
    let mut first = [0.];
    cpu.forward_batch(&[0.5, -0.2], &mut first, &mut cpu_ws);
    close(&a.wait().unwrap(), &first);
    close(
        &gpu.forward_batch_softmax(&input, 2, &mut ws).unwrap(),
        &[1., 1.],
    );
    cpu.layers[0].set_normalization(&[0., 0.], &[1., 1.]);
    gpu.sync_weights(&cpu).unwrap();
    cpu.forward_batch(&input, &mut expected, &mut cpu_ws);
    close(&gpu.forward_batch(&input, 2, &mut ws).unwrap(), &expected);
}

#[test]
#[ignore = "Requires GPU adapter"]
fn imported_small_std_matches_cpu_or_is_rejected_before_execution() {
    let b = backend();
    let mut cpu = network(3);
    cpu.layers[0].mean = [0., 0.].to_vec();
    cpu.layers[0].std = [1e-8, 1e-8].to_vec();
    let input = [0., 0., 5e-9, -5e-9];
    let mut cpu_ws = cpu.create_workspace(2);
    let mut expected = vec![0.; 2];
    cpu.forward_batch(&input, &mut expected, &mut cpu_ws);
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut ws = gpu.create_workspace(2).unwrap();
    close(
        &gpu.forward_batch_training(&input, 2, &mut ws).unwrap(),
        &expected,
    );
    cpu.layers[0].std[0] = 1e-40;
    assert!(
        GpuNetwork::from_cpu(&b, &cpu).is_err(),
        "GPU subnormal std must be rejected before execution"
    );
}

#[test]
#[ignore = "Requires GPU adapter"]
fn backward_rejects_batch_without_matching_saved_forward() {
    let b = backend();
    let cpu = network(3);
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut ws = gpu.create_workspace(4).unwrap();
    gpu.forward_batch_training(&[0.5; 8], 4, &mut ws).unwrap();
    gpu.forward_batch_training(&[0.5; 2], 1, &mut ws).unwrap();
    let mut w = vec![];
    let mut bias = vec![];
    assert!(gpu
        .backward_batch(&[1.; 4], 4, &mut ws, &mut w, &mut bias)
        .is_err());
}

#[test]
#[ignore = "Requires GPU adapter"]
fn review_saved_training_batch_survives_other_workspace_forward() {
    let b = backend();
    let cpu = network(3);
    let mut gpu = GpuNetwork::from_cpu(&b, &cpu).unwrap();
    let mut reference = gpu.create_workspace(4).unwrap();
    let mut saved = gpu.create_workspace(4).unwrap();
    let mut other = gpu.create_workspace(1).unwrap();
    let input = [0.5, -0.2, 0.6, -0.1, 0.4, -0.3, 0.7, 0.];
    let gradient = [1., 2., 3., 4.];
    let mut expected_w = vec![];
    let mut expected_b = vec![];
    gpu.forward_batch_training(&input, 4, &mut reference)
        .unwrap();
    let expected_i = gpu
        .backward_batch(
            &gradient,
            4,
            &mut reference,
            &mut expected_w,
            &mut expected_b,
        )
        .unwrap();
    for training in [false, true] {
        gpu.forward_batch_training(&input, 4, &mut saved).unwrap();
        if training {
            gpu.forward_batch_training(&[0.5, -0.2], 1, &mut other)
                .unwrap();
        } else {
            gpu.forward_batch(&[0.5, -0.2], 1, &mut other).unwrap();
        }
        let mut actual_w = vec![];
        let mut actual_b = vec![];
        let actual_i = gpu
            .backward_batch(&gradient, 4, &mut saved, &mut actual_w, &mut actual_b)
            .unwrap();
        close(&actual_i, &expected_i);
        for layer in 0..expected_w.len() {
            close(&actual_w[layer], &expected_w[layer]);
            close(&actual_b[layer], &expected_b[layer]);
        }
    }
}

#[test]
#[ignore = "Requires GPU adapter"]
fn review_gpu_cpu_snapshot_preserves_actual_layer_normalization() {
    let b = backend();
    let mut source = network(3);
    source.layers[0].mean = vec![0.25, 0.];
    source.layers[0].std = vec![0.5, 1e-8];
    source.layers[1].set_normalization(&[0.1, 0.2, -0.1], &[0.4, 0.6, 0.8]);
    let mut gpu = GpuNetwork::from_cpu(&b, &source).unwrap();
    // Exercise updated GPU normalization, not just the original conversion.
    source.layers[1].set_normalization(&[-0.1, 0.1, 0.3], &[0.8, 0.5, 0.4]);
    gpu.sync_weights_from_cpu(&source).unwrap();
    let mut destination = network(3);
    for layer in &mut destination.layers {
        layer.set_normalization(&vec![0.; layer.in_dim], &vec![1.; layer.in_dim]);
        layer.weights.fill(0.);
        layer.bias.fill(-99.);
    }
    gpu.sync_weights_to_cpu(&mut destination).unwrap();
    for (actual, expected) in destination.layers.iter().zip(&source.layers) {
        assert_eq!(actual.mean, expected.mean);
        assert_eq!(actual.std, expected.std);
        assert_eq!(actual.weights, expected.weights);
        assert_eq!(actual.bias, expected.bias);
    }
    let input = [0.25, 0., 0.5, 5e-9, -0.25, -5e-9];
    let mut gpu_ws = gpu.create_workspace(3).unwrap();
    let gpu_output = gpu.forward_batch(&input, 3, &mut gpu_ws).unwrap();
    let mut cpu_ws = destination.create_workspace(3);
    let mut cpu_output = vec![0.; 3];
    destination.forward_batch(&input, &mut cpu_output, &mut cpu_ws);
    close(&cpu_output, &gpu_output);
}

#[test]
#[ignore = "Requires GPU adapter"]
fn review_gpu_cpu_snapshot_rejects_invalid_late_layer_without_mutation() {
    let b = backend();
    let source = network(3);
    let gpu = GpuNetwork::from_cpu(&b, &source).unwrap();
    let mut destination = network(3);
    destination.layers[0].weights.fill(0.);
    destination.layers[0].bias.fill(-99.);
    destination.layers[1].mean.clear();
    let before = destination.clone();
    assert!(gpu.sync_weights_to_cpu(&mut destination).is_err());
    for (actual, expected) in destination.layers.iter().zip(&before.layers) {
        assert_eq!(actual.weights, expected.weights);
        assert_eq!(actual.bias, expected.bias);
        assert_eq!(actual.mean, expected.mean);
        assert_eq!(actual.std, expected.std);
    }
}
