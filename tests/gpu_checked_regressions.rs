#![cfg(feature = "gpu")]

use arkan::gpu::{
    GpuAdam, GpuAdamConfig, GpuNetwork, GpuSgd, GpuSgdConfig, GpuTensor, GpuWorkspace, WgpuBackend,
    WgpuOptions,
};
use arkan::{ArkanError, KanConfigBuilder, KanNetwork};

fn backend() -> WgpuBackend {
    WgpuBackend::init(WgpuOptions::default()).unwrap()
}
fn model(input: usize, range: (f32, f32), grid: usize, order: usize) -> KanNetwork {
    KanNetwork::new(
        KanConfigBuilder::new()
            .input_dim(input)
            .output_dim(1)
            .hidden_dims(vec![])
            .grid_size(grid)
            .spline_order(order)
            .grid_range(range.0, range.1)
            .normalization(vec![0.; input], vec![1.; input])
            .seed(2)
            .build()
            .unwrap(),
    )
}
fn limited_backend() -> WgpuBackend {
    let limits = wgpu::Limits {
        max_compute_workgroups_per_dimension: 1,
        ..wgpu::Limits::default()
    };
    WgpuBackend::init(WgpuOptions::with_limits(limits)).unwrap()
}
#[test]
#[ignore = "Requires GPU adapter"]
fn shifted_grid_is_supported_faithfully_or_rejected() {
    let backend = backend();
    let mut cpu = model(1, (1_000_000., 1_000_001.), 5, 3);
    cpu.layers[0].weights.fill(0.);
    cpu.layers[0].weights[2] = 1.;
    cpu.layers[0].bias.fill(0.);
    match GpuNetwork::from_cpu(&backend, &cpu) {
        Err(ArkanError::Validation(_)) => (),
        Err(error) => panic!("unexpected error: {error}"),
        Ok(mut gpu) => {
            let mut ws = gpu.create_workspace(1).unwrap();
            let actual = gpu.forward_batch(&[1_000_000.5], 1, &mut ws).unwrap();
            assert!((actual[0] - 0.02857143).abs() < 1e-6, "{actual:?}");
        }
    }
}
#[test]
#[ignore = "Requires GPU adapter"]
fn small_stored_std_derivative_matches_forward_in_construction_and_sync() {
    let backend = backend();
    for sync in [false, true] {
        let mut cpu = model(1, (-1., 1.), 3, 2);
        for (i, w) in cpu.layers[0].weights.iter_mut().enumerate() {
            *w = i as f32;
        }
        cpu.layers[0].std[0] = if sync { 1. } else { 1e-7 };
        let mut gpu = GpuNetwork::from_cpu(&backend, &cpu).unwrap();
        cpu.layers[0].std[0] = 1e-7;
        if sync {
            gpu.sync_weights(&cpu).unwrap();
        }
        let mut ws = gpu.create_workspace(1).unwrap();
        gpu.forward_batch_training(&[0.], 1, &mut ws).unwrap();
        let derivative = gpu
            .backward_batch(&[1.], 1, &mut ws, &mut vec![], &mut vec![])
            .unwrap()[0];
        let hi = gpu.forward_batch(&[1e-9], 1, &mut ws).unwrap()[0];
        let lo = gpu.forward_batch(&[-1e-9], 1, &mut ws).unwrap()[0];
        let finite_difference = (hi - lo) / 2e-9;
        assert!(
            (derivative / finite_difference - 1.).abs() < 1e-3,
            "{derivative} vs {finite_difference}"
        );
    }
}
#[test]
#[ignore = "Requires GPU adapter"]
fn direct_softmax_rejects_foreign_workspace() {
    let backend = backend();
    let mut gpu = GpuNetwork::from_cpu(&backend, &model(1, (-1., 1.), 3, 2)).unwrap();
    let mut ws = GpuWorkspace::new(&backend.device, 1, 1, 3).unwrap();
    assert!(gpu.apply_softmax(1, &mut ws).is_err());
}
#[test]
#[ignore = "Requires GPU adapter"]
fn dispatch_limits_reject_forward_training_and_softmax_before_upload() {
    let backend = limited_backend();
    let mut gpu = GpuNetwork::from_cpu(&backend, &model(1, (-1., 1.), 3, 2)).unwrap();
    let mut ws = gpu.create_workspace(65).unwrap();
    ws.input.as_ref().unwrap().update(&backend.queue, &[9.; 65]);
    for training in [false, true] {
        let result = if training {
            gpu.forward_batch_training(&[0.; 65], 65, &mut ws)
        } else {
            gpu.forward_batch(&[0.; 65], 65, &mut ws)
        };
        assert!(matches!(result, Err(ArkanError::UnsupportedLimits(_))));
    }
    assert!(matches!(
        gpu.forward_batch_async(&[0.; 65], 65, &mut ws),
        Err(ArkanError::UnsupportedLimits(_))
    ));
    assert!(matches!(
        gpu.forward_batch_softmax(&[0.; 65], 65, &mut ws),
        Err(ArkanError::UnsupportedLimits(_))
    ));
    assert_eq!(
        ws.input
            .as_ref()
            .unwrap()
            .download(&backend.device, &backend.queue)
            .unwrap(),
        vec![9.; 65]
    );
    assert!(matches!(
        gpu.apply_softmax(65, &mut ws),
        Err(ArkanError::UnsupportedLimits(_))
    ));
}
#[test]
#[ignore = "Requires GPU adapter"]
fn dispatch_limits_reject_backward_weights_and_inputs() {
    let backend = limited_backend();
    for (input, batch) in [(9, 1), (2, 33)] {
        let mut gpu = GpuNetwork::from_cpu(&backend, &model(input, (-1., 1.), 3, 2)).unwrap();
        let mut ws = gpu.create_workspace(batch).unwrap();
        gpu.forward_batch_training(&vec![0.; input * batch], batch, &mut ws)
            .unwrap();
        assert!(matches!(
            gpu.backward_batch(&vec![1.; batch], batch, &mut ws, &mut vec![], &mut vec![]),
            Err(ArkanError::UnsupportedLimits(_))
        ));
    }
}
#[test]
#[ignore = "Requires GPU adapter"]
fn dispatch_limits_reject_optimizers_before_adam_time() {
    let backend = limited_backend();
    let weights =
        GpuTensor::upload(&backend.device, &backend.queue, &[1.; 257], vec![257]).unwrap();
    let bias = GpuTensor::upload(&backend.device, &backend.queue, &[1.], vec![1]).unwrap();
    let gradients =
        GpuTensor::upload(&backend.device, &backend.queue, &[1.; 257], vec![257]).unwrap();
    let params = [(&weights.buffer, &bias.buffer)];
    let grads = [(&gradients.buffer, &bias.buffer)];
    let mut adam = GpuAdam::new(
        backend.device.clone(),
        backend.queue.clone(),
        &[(257, 1)],
        GpuAdamConfig::default(),
    );
    let mut sgd = GpuSgd::new(
        backend.device.clone(),
        backend.queue.clone(),
        &[(257, 1)],
        GpuSgdConfig::default(),
    );
    assert!(matches!(
        adam.step(&params, &grads),
        Err(ArkanError::UnsupportedLimits(_))
    ));
    assert_eq!(adam.t, 0);
    assert!(matches!(
        sgd.step(&params, &grads),
        Err(ArkanError::UnsupportedLimits(_))
    ));
}

#[test]
#[ignore = "Requires GPU adapter"]
fn ordinary_grids_stay_supported_and_grid_sync_rejection_is_atomic() {
    let backend = backend();
    for order in 2..=5 {
        for (range, grid) in [
            ((-1., 1.), 5),
            ((-2., 3.), 16),
            ((0., 1.), 64),
            ((-0.001, 0.001), 5),
        ] {
            let cpu = model(1, range, grid, order);
            let mut gpu = GpuNetwork::from_cpu(&backend, &cpu).unwrap();
            gpu.sync_weights(&cpu).unwrap();
        }
    }
    let cpu = model(1, (-1., 1.), 5, 3);
    let mut gpu = GpuNetwork::from_cpu(&backend, &cpu).unwrap();
    let mut shifted = model(1, (1_000_000., 1_000_001.), 5, 3);
    shifted.layers[0].weights.fill(99.);
    assert!(gpu.sync_weights(&shifted).is_err());
    let mut got = cpu.clone();
    gpu.sync_weights_to_cpu(&mut got).unwrap();
    assert_eq!(got.layers[0].weights, cpu.layers[0].weights);
}
