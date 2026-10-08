#![cfg(feature = "gpu")]

use arkan::gpu::{
    GpuAdam, GpuAdamConfig, GpuNetwork, GpuSgd, GpuSgdConfig, GpuWorkspace, WgpuBackend,
    WgpuOptions,
};
use arkan::optimizer::{Adam, AdamConfig, Optimizer, SGDConfig, SafetyConfig, SGD};
use arkan::{ArkanResult, KanConfigBuilder, KanNetwork, TrainOptions};

fn backend() -> WgpuBackend {
    let backend = WgpuBackend::init(WgpuOptions::default()).unwrap();
    eprintln!("GPU adapter: {:?}", backend.adapter_info());
    backend
}

fn model(range: f32) -> KanNetwork {
    let mut cpu = KanNetwork::new(
        KanConfigBuilder::new()
            .input_dim(1)
            .output_dim(1)
            .hidden_dims(vec![])
            .grid_size(3)
            .spline_order(2)
            .grid_range(-range, range)
            .normalization(vec![0.0], vec![1.0])
            .seed(2)
            .build()
            .unwrap(),
    );
    cpu.layers[0].weights.fill(1.0);
    cpu.layers[0].bias.fill(0.0);
    cpu
}

struct Training {
    cpu: KanNetwork,
    gpu: GpuNetwork,
    workspace: GpuWorkspace,
    adam: Adam,
    sgd: SGD,
    gpu_adam: GpuAdam,
    gpu_sgd: GpuSgd,
}

impl Training {
    fn new(backend: &WgpuBackend) -> Self {
        let cpu = model(1.0);
        let gpu = GpuNetwork::from_cpu(backend, &cpu).unwrap();
        let workspace = gpu.create_workspace(1).unwrap();
        let adam = Adam::new(&cpu, AdamConfig::with_lr(0.1));
        let sgd = SGD::new(
            &cpu,
            SGDConfig {
                lr: 0.1,
                momentum: 0.8,
                ..SGDConfig::default()
            },
        );
        let gpu_adam = GpuAdam::new(
            backend.device.clone(),
            backend.queue.clone(),
            &gpu.layer_param_sizes(),
            GpuAdamConfig::with_lr(0.1),
        );
        let gpu_sgd = GpuSgd::new(
            backend.device.clone(),
            backend.queue.clone(),
            &gpu.layer_param_sizes(),
            GpuSgdConfig {
                lr: 0.1,
                momentum: 0.8,
                ..GpuSgdConfig::default()
            },
        );
        Self {
            cpu,
            gpu,
            workspace,
            adam,
            sgd,
            gpu_adam,
            gpu_sgd,
        }
    }

    // Exercise every public trainer, including the MSE delegator.
    fn step(&mut self, entry: usize, target: f32, options: &TrainOptions) -> ArkanResult<f32> {
        let input = &[0.0];
        let target = &[target];
        match entry {
            0 => self.gpu.train_step_mse(
                input,
                target,
                1,
                &mut self.workspace,
                &mut self.adam,
                &mut self.cpu,
            ),
            1 => self.gpu.train_step_cross_entropy(
                input,
                target,
                1,
                &mut self.workspace,
                &mut self.adam,
                &mut self.cpu,
            ),
            2 => self.gpu.train_step_with_options(
                input,
                target,
                None,
                1,
                &mut self.workspace,
                &mut self.adam,
                &mut self.cpu,
                options,
            ),
            3 => self.gpu.train_step_sgd(
                input,
                target,
                1,
                &mut self.workspace,
                &mut self.sgd,
                &mut self.cpu,
            ),
            4 => self.gpu.train_step_sgd_with_options(
                input,
                target,
                None,
                1,
                &mut self.workspace,
                &mut self.sgd,
                &mut self.cpu,
                options,
            ),
            5 => self.gpu.train_step_gpu_native(
                input,
                target,
                1,
                &mut self.workspace,
                &mut self.gpu_adam,
            ),
            6 => self.gpu.train_step_gpu_native_sgd(
                input,
                target,
                1,
                &mut self.workspace,
                &mut self.gpu_sgd,
            ),
            7 => self.gpu.train_step_gpu_native_with_options(
                input,
                target,
                1,
                None,
                &mut self.workspace,
                &mut self.gpu_adam,
                options,
            ),
            _ => unreachable!(),
        }
    }
}

fn assert_cpu_unchanged(actual: &KanNetwork, before: &KanNetwork) {
    assert_eq!(actual.layers.len(), before.layers.len());
    for (actual, before) in actual.layers.iter().zip(&before.layers) {
        assert_eq!(actual.weights, before.weights, "CPU weights changed");
        assert_eq!(actual.bias, before.bias, "CPU bias changed");
        assert_eq!(actual.mean, before.mean);
        assert_eq!(actual.std, before.std);
    }
}

fn assert_adam_unchanged(actual: &Adam, before: &Adam) {
    assert_eq!(actual.get_state_version(), before.get_state_version());
    assert_eq!(actual.layer_states.len(), before.layer_states.len());
    for (actual, before) in actual.layer_states.iter().zip(&before.layer_states) {
        for (actual, before) in [
            (&actual.weights, &before.weights),
            (&actual.bias, &before.bias),
        ] {
            assert_eq!(actual.m.as_slice(), before.m.as_slice());
            assert_eq!(actual.v.as_slice(), before.v.as_slice());
            assert_eq!(actual.t, before.t);
        }
    }
}

fn assert_sgd_unchanged(actual: &SGD, before: &SGD) {
    assert_eq!(actual.get_state_version(), before.get_state_version());
    assert_eq!(actual.velocities.len(), before.velocities.len());
    for ((actual_w, actual_b), (before_w, before_b)) in
        actual.velocities.iter().zip(&before.velocities)
    {
        assert_eq!(actual_w.as_slice(), before_w.as_slice());
        assert_eq!(actual_b.as_slice(), before_b.as_slice());
    }
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_trainers_reject_incompatible_cpu_snapshot_before_updates() {
    let backend = backend();
    for entry in 0..5 {
        for invalid_normalization in [false, true] {
            eprintln!("entry {entry}, invalid normalization {invalid_normalization}");
            let mut training = Training::new(&backend);
            training.step(entry, 0.0, &TrainOptions::default()).unwrap();
            if invalid_normalization {
                training.cpu.layers[0].std[0] = f32::MIN_POSITIVE / 2.0;
            } else {
                training.cpu = model(2.0);
            }
            training.cpu.try_parameters_mut().unwrap();
            let before = training.cpu.clone();
            let adam_before = training.adam.clone();
            let sgd_before = training.sgd.clone();
            assert!(training.step(entry, 0.0, &TrainOptions::default()).is_err());
            assert_adam_unchanged(&training.adam, &adam_before);
            assert_sgd_unchanged(&training.sgd, &sgd_before);
            assert_cpu_unchanged(&training.cpu, &before);
        }
    }
}

#[test]
#[ignore = "Requires GPU adapter"]
fn rejected_optimizer_step_rolls_back_additional_hybrid_decay() {
    let backend = backend();
    for entry in [2, 4] {
        for invalid_config in [true, false] {
            eprintln!("entry {entry}, invalid config {invalid_config}");
            let mut training = Training::new(&backend);
            training.step(entry, 0.0, &TrainOptions::default()).unwrap();
            let target = if invalid_config {
                training.adam.config.beta1 = 2.0;
                training.sgd.config.momentum = 2.0;
                0.0
            } else {
                training.adam.config.safety = SafetyConfig::strict();
                training.sgd.config.safety = SafetyConfig::strict();
                f32::NAN
            };
            let before = training.cpu.clone();
            let adam_before = training.adam.clone();
            let sgd_before = training.sgd.clone();
            let mut gpu_before = model(1.0);
            training.gpu.sync_weights_to_cpu(&mut gpu_before).unwrap();
            let options = TrainOptions {
                weight_decay: 0.5,
                max_grad_norm: None,
            };
            assert!(training.step(entry, target, &options).is_err());
            assert_adam_unchanged(&training.adam, &adam_before);
            assert_sgd_unchanged(&training.sgd, &sgd_before);
            assert_cpu_unchanged(&training.cpu, &before);
            let mut downloaded = model(1.0);
            training.gpu.sync_weights_to_cpu(&mut downloaded).unwrap();
            assert_cpu_unchanged(&downloaded, &gpu_before);
        }
    }
}

#[test]
#[ignore = "Requires GPU adapter"]
fn all_trainers_reject_zeroed_public_output_before_shape_arithmetic() {
    let backend = backend();
    for entry in 0..8 {
        eprintln!("entry {entry}");
        let mut training = Training::new(&backend);
        let mut control = Training::new(&backend);
        let options = TrainOptions::default();
        training.step(entry, 0.0, &options).unwrap();
        control.step(entry, 0.0, &options).unwrap();
        let before = training.cpu.clone();
        let adam_before = training.adam.clone();
        let sgd_before = training.sgd.clone();
        let gpu_t_before = training.gpu_adam.t;
        training.gpu.output_dim = 0;
        assert!(training.step(entry, 0.0, &options).is_err());
        assert_cpu_unchanged(&training.cpu, &before);
        assert_adam_unchanged(&training.adam, &adam_before);
        assert_sgd_unchanged(&training.sgd, &sgd_before);
        assert_eq!(training.gpu_adam.t, gpu_t_before);
        training.gpu.output_dim = 1;
        let mut downloaded = model(1.0);
        let mut expected = model(1.0);
        training.gpu.sync_weights_to_cpu(&mut downloaded).unwrap();
        control.gpu.sync_weights_to_cpu(&mut expected).unwrap();
        assert_cpu_unchanged(&downloaded, &expected);
        // A subsequent real update exposes changes to private native momentum/moments.
        training.step(entry, 0.0, &options).unwrap();
        control.step(entry, 0.0, &options).unwrap();
        training.gpu.sync_weights_to_cpu(&mut downloaded).unwrap();
        control.gpu.sync_weights_to_cpu(&mut expected).unwrap();
        assert_cpu_unchanged(&downloaded, &expected);
    }
}

#[test]
#[ignore = "Requires GPU adapter"]
fn successful_hybrid_options_preserve_both_decay_factors() {
    let backend = backend();
    for (entry, target, weights, bias) in [
        (2, 1.0, [0.931; 5], 0.0),
        (4, 1.0, [0.931; 5], 0.0),
        (2, 0.0, [0.931, 0.831, 0.831, 0.831, 0.931], -0.1),
        (4, 0.0, [0.931, 0.906, 0.781, 0.906, 0.931], -0.2),
    ] {
        let mut training = Training::new(&backend);
        training.adam.config.weight_decay = 0.2;
        training.sgd.config.weight_decay = 0.2;
        let options = TrainOptions {
            weight_decay: 0.5,
            max_grad_norm: None,
        };
        // Output 1, basis [0, 1/8, 3/4, 1/8, 0]. Decays give 0.95 * 0.98
        // before subtracting the first Adam or SGD update; bias has no extra decay.
        let loss = training.step(entry, target, &options).unwrap();
        assert_eq!(loss, if target == 1.0 { 0.0 } else { 1.0 });
        for (actual, expected) in training.cpu.layers[0].weights.iter().zip(weights) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "expected {expected}, got {actual}"
            );
        }
        assert!((training.cpu.layers[0].bias[0] - bias).abs() < 1e-6);
        let mut downloaded = model(1.0);
        training.gpu.sync_weights_to_cpu(&mut downloaded).unwrap();
        assert_cpu_unchanged(&downloaded, &training.cpu);
    }
}

fn skipped_hybrid_update_preserves_transaction(entry: usize, reason: &str) {
    let backend = backend();
    let mut training = Training::new(&backend);
    training.adam.bump_version();
    training.sgd.bump_version();
    // Warm every moment/velocity element, including inactive spline coefficients.
    let wg = vec![vec![1.0; training.cpu.layers[0].weights.len()]];
    let bg = vec![vec![1.0; training.cpu.layers[0].bias.len()]];
    training
        .adam
        .step(&mut training.cpu, &wg, &bg, None)
        .unwrap();
    training
        .sgd
        .step(&mut training.cpu, &wg, &bg, None)
        .unwrap();
    training.gpu.sync_weights(&training.cpu).unwrap();
    let mut gpu_before = model(1.0);
    training.gpu.sync_weights_to_cpu(&mut gpu_before).unwrap();
    // A distinct valid CPU snapshot exposes an unwanted upload even after rollback.
    training.cpu.layers[0]
        .weights
        .iter_mut()
        .for_each(|w| *w += 0.125);
    training.cpu.layers[0].mean[0] = 0.05;
    training.cpu.layers[0].std[0] = 1.25;
    let mut options = TrainOptions {
        weight_decay: 0.5,
        max_grad_norm: None,
    };
    let target = match reason {
        "raw gradient" => f32::NAN,
        "unscale" => {
            training.adam.config.safety = SafetyConfig::with_amp(1e-320);
            training.sgd.config.safety = SafetyConfig::with_amp(1e-320);
            0.0
        }
        "prospective update" => {
            // Raw gradients and extra decay stay finite; Adam's variance and
            // SGD's parameter update overflow only during optimizer preview.
            training.sgd.config.lr = f32::MAX;
            options.weight_decay = if entry == 4 { 1e-40 } else { 0.5 };
            -1e30
        }
        "extra decay" => {
            training.adam.config.lr = f32::MAX;
            training.sgd.config.lr = f32::MAX;
            training.cpu.layers[0].weights.fill(4.0);
            0.0
        }
        _ => unreachable!(),
    };
    let before = training.cpu.clone();
    let adam_before = training.adam.clone();
    let sgd_before = training.sgd.clone();
    eprintln!("entry {entry}, skip {reason}");
    training.step(entry, target, &options).unwrap();
    assert_adam_unchanged(&training.adam, &adam_before);
    assert_sgd_unchanged(&training.sgd, &sgd_before);
    assert_cpu_unchanged(&training.cpu, &before);
    let mut downloaded = model(1.0);
    training.gpu.sync_weights_to_cpu(&mut downloaded).unwrap();
    assert_cpu_unchanged(&downloaded, &gpu_before);
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_adam_raw_gradient_skip_rolls_back_additional_decay() {
    skipped_hybrid_update_preserves_transaction(2, "raw gradient");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_sgd_raw_gradient_skip_rolls_back_additional_decay() {
    skipped_hybrid_update_preserves_transaction(4, "raw gradient");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_adam_unscale_skip_rolls_back_additional_decay() {
    skipped_hybrid_update_preserves_transaction(2, "unscale");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_sgd_unscale_skip_rolls_back_additional_decay() {
    skipped_hybrid_update_preserves_transaction(4, "unscale");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_adam_prospective_update_skip_rolls_back_additional_decay() {
    skipped_hybrid_update_preserves_transaction(2, "prospective update");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_sgd_prospective_update_skip_rolls_back_additional_decay() {
    skipped_hybrid_update_preserves_transaction(4, "prospective update");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_adam_nonfinite_extra_decay_is_rolled_back() {
    skipped_hybrid_update_preserves_transaction(2, "extra decay");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn hybrid_sgd_nonfinite_extra_decay_is_rolled_back() {
    skipped_hybrid_update_preserves_transaction(4, "extra decay");
}

#[test]
#[ignore = "Requires GPU adapter"]
fn successful_zero_lr_hybrid_step_advances_history_and_syncs() {
    let backend = backend();
    for entry in [2, 4] {
        let mut training = Training::new(&backend);
        training.step(entry, 0.0, &TrainOptions::default()).unwrap();
        training.adam.config.lr = 0.0;
        training.sgd.config.lr = 0.0;
        training.cpu.layers[0]
            .weights
            .iter_mut()
            .for_each(|w| *w += 0.125);
        let before = training.cpu.clone();
        let adam_t = training.adam.layer_states[0].weights.t;
        let velocity = training.sgd.velocities[0].1.as_slice()[0];
        training
            .step(
                entry,
                -1.0,
                &TrainOptions {
                    weight_decay: 0.5,
                    max_grad_norm: None,
                },
            )
            .unwrap();
        assert_cpu_unchanged(&training.cpu, &before);
        if entry == 2 {
            assert_eq!(training.adam.layer_states[0].weights.t, adam_t + 1);
        } else {
            assert_ne!(training.sgd.velocities[0].1.as_slice()[0], velocity);
        }
        let mut downloaded = model(1.0);
        training.gpu.sync_weights_to_cpu(&mut downloaded).unwrap();
        assert_cpu_unchanged(&downloaded, &before);
    }
}
