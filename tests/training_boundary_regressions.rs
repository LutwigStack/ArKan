use arkan::{
    Adam, AdamConfig, KanConfig, KanNetwork, Optimizer, SGDConfig, SafetyConfig, TrainOptions, SGD,
};

fn network() -> KanNetwork {
    KanNetwork::new(KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![2],
        grid_size: 3,
        spline_order: 2,
        input_mean: vec![0.25, -0.5],
        input_std: vec![0.5, 2.0],
        init_seed: Some(11),
        ..KanConfig::default()
    })
}

fn check_unscale_before_clip<O: Optimizer>(make_optimizer: impl Fn(&KanNetwork) -> O) {
    let mut actual = network();
    let mut expected = actual.clone();
    let mut actual_optimizer = make_optimizer(&actual);
    let mut expected_optimizer = make_optimizer(&expected);
    let mut actual_workspace = actual.create_workspace(2);
    let mut expected_workspace = expected.create_workspace(2);
    let input = [0.3, -0.4, 0.8, 0.1];
    let targets = [-4.0, -4.0];
    let options = TrainOptions {
        max_grad_norm: Some(0.1),
        weight_decay: 0.0,
    };
    // Obtain the original unscaled-loss gradients independently of optimizer dispatch.
    expected.train_step(&input, &targets, None, 0.0, &mut expected_workspace);
    expected_optimizer
        .step(
            &mut expected,
            &expected_workspace.weight_grads,
            &expected_workspace.bias_grads,
            options.max_grad_norm,
        )
        .unwrap();
    actual
        .train_step_with_optimizer(
            &input,
            &targets,
            None,
            &mut actual_workspace,
            &mut actual_optimizer,
            &options,
        )
        .unwrap();
    for (got, want) in actual.layers.iter().zip(&expected.layers) {
        for (&got, &want) in got
            .weights
            .iter()
            .chain(&got.bias)
            .zip(want.weights.iter().chain(&want.bias))
        {
            assert!(
                (got - want).abs() < 1e-6,
                "optimizer preprocessing differs: {got} versus {want}"
            );
        }
    }
}

#[test]
fn sgd_training_unscales_before_global_clipping() {
    check_unscale_before_clip(|net| {
        SGD::new(
            net,
            SGDConfig::with_lr(0.25).with_safety(SafetyConfig::with_amp(16.0)),
        )
    });
}

#[test]
fn adam_training_unscales_before_global_clipping() {
    check_unscale_before_clip(|net| {
        Adam::new(
            net,
            AdamConfig {
                lr: 0.25,
                epsilon: 0.01,
                safety: SafetyConfig::with_amp(16.0),
                ..AdamConfig::default()
            },
        )
    });
}

#[test]
fn custom_losses_use_the_production_reverse_pass() {
    use arkan::loss::{masked_bce_with_logits, masked_huber};

    let mut net = network();
    net.layers[1].set_normalization(&[0.1, -0.2], &[0.7, 1.4]);
    let inputs = [0.4, -0.3, 100.0, -100.0];
    let targets = [0.0, 1.0];
    let mut workspace = net.create_workspace(2);
    for huber in [false, true] {
        let loss = |predictions: &[f32]| {
            if huber {
                masked_huber(predictions, &targets, 0.4, None)
            } else {
                masked_bce_with_logits(predictions, &targets, None)
            }
        };
        let mut predictions = [0.0; 2];
        let pass = net
            .try_forward_for_backward(&inputs, &mut predictions, &mut workspace)
            .unwrap();
        let (_, output_gradient) = loss(&predictions);
        let gradients = pass.backward(&output_gradient).unwrap();
        let weight_gradients = gradients.weights.to_vec();
        let bias_gradients = gradients.biases.to_vec();
        // Independently perturb every parameter, including unused/clamped spans.
        let epsilon = 0.001;
        for layer_index in 0..net.layers.len() {
            for bias in [false, true] {
                let expected = if bias {
                    &bias_gradients[layer_index]
                } else {
                    &weight_gradients[layer_index]
                };
                for (parameter_index, &analytic) in expected.iter().enumerate() {
                    let mut losses = [0.0; 2];
                    for (side, direction) in [-1.0, 1.0].iter().enumerate() {
                        let mut perturbed = net.clone();
                        let layer = &mut perturbed.layers[layer_index];
                        let parameters = if bias {
                            &mut layer.bias
                        } else {
                            &mut layer.weights
                        };
                        parameters[parameter_index] += direction * epsilon;
                        perturbed.forward_batch(&inputs, &mut predictions, &mut workspace);
                        losses[side] = loss(&predictions).0;
                    }
                    let numerical = (losses[1] - losses[0]) / (2.0 * epsilon);
                    assert!((analytic - numerical).abs() < 2e-4,
                        "huber={huber}, layer={layer_index}, bias={bias}, parameter={parameter_index}: {analytic} != {numerical}");
                }
            }
        }
    }
}

#[test]
fn rejected_gradient_is_recoverable_and_backward_releases_model_borrow() {
    let mut net = network();
    let mut workspace = net.create_workspace(1);
    net.train_step(&[0.3, -0.4], &[0.0], None, 0.0, &mut workspace);
    let before = workspace.weight_grads.clone();
    let mut output = [0.0];
    let pass = net
        .try_forward_for_backward(&[0.3, -0.4], &mut output, &mut workspace)
        .unwrap();
    assert!(pass.backward(&[]).is_err());
    assert_eq!(workspace.weight_grads, before);

    let mut optimizer = SGD::new(&net, SGDConfig::with_lr(0.01));
    let pass = net
        .try_forward_for_backward(&[0.3, -0.4], &mut output, &mut workspace)
        .unwrap();
    let gradients = pass.backward(&[1.0]).unwrap();
    optimizer
        .step(&mut net, gradients.weights, gradients.biases, None)
        .unwrap();
}

#[test]
fn backward_workspace_reuses_growing_shrinking_and_empty_batches_across_models() {
    let net = network();
    let other = KanNetwork::new(KanConfig {
        input_dim: 3,
        input_mean: vec![0.0; 3],
        input_std: vec![1.0; 3],
        hidden_dims: vec![4],
        output_dim: 2,
        init_seed: Some(2),
        ..KanConfig::default()
    });
    let mut workspace = net.create_workspace(1);
    for model in [&net, &other, &net] {
        for batch in [4, 1, 0, 2] {
            let input = vec![0.2; batch * model.config.input_dim];
            let mut output = vec![0.0; batch * model.config.output_dim];
            let gradient = vec![1.0; output.len()];
            let pass = model
                .try_forward_for_backward(&input, &mut output, &mut workspace)
                .unwrap();
            let actual = pass.backward(&gradient).unwrap();
            let mut fresh = model.create_workspace(batch);
            let pass = model
                .try_forward_for_backward(&input, &mut output, &mut fresh)
                .unwrap();
            let expected = pass.backward(&gradient).unwrap();
            assert_eq!(actual.weights, expected.weights);
            assert_eq!(actual.biases, expected.biases);
            assert_eq!(actual.weights.len(), model.layers.len());
            if batch == 0 {
                assert!(actual
                    .weights
                    .iter()
                    .chain(actual.biases)
                    .flatten()
                    .all(|&x| x == 0.0));
            }
        }
    }
}

#[test]
fn mse_into_checks_shapes_before_writes_and_clears_masked_slots() {
    let mut gradients = [99.0; 3];
    assert!(arkan::loss::masked_mse_into(&[1.0; 3], &[0.0; 2], None, &mut gradients).is_err());
    assert_eq!(gradients, [99.0; 3]);
    assert!(
        arkan::loss::masked_mse_into(&[1.0; 3], &[0.0; 3], Some(&[1.0; 2]), &mut gradients)
            .is_err()
    );
    assert_eq!(gradients, [99.0; 3]);
    let loss = arkan::loss::masked_mse_into(
        &[1.0, 20.0, 3.0],
        &[0.0; 3],
        Some(&[1.0, 0.0, 2.0]),
        &mut gradients,
    )
    .unwrap();
    assert_eq!(loss, 19.0 / 3.0);
    assert_eq!(gradients, [2.0 / 3.0, 0.0, 4.0]);
}

fn zero_rate_network(multithreading_threshold: usize) -> KanNetwork {
    let mut net = KanNetwork::new(KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: 3,
        spline_order: 3,
        input_mean: vec![0.0],
        input_std: vec![1.0],
        multithreading_threshold,
        init_seed: Some(42),
        ..KanConfig::default()
    });
    net.layers[0].weights.fill(0.0);
    net.layers[0].weights[0] = -0.0;
    net.layers[0].bias[0] = -0.0;
    net
}

fn parameter_bits(net: &KanNetwork) -> Vec<u32> {
    net.layers
        .iter()
        .flat_map(|layer| layer.weights.iter().chain(&layer.bias))
        .map(|value| value.to_bits())
        .collect()
}

// Compare NaNs and signed zeros exactly, including reusable buffer capacities.
fn workspace_bits(workspace: &arkan::Workspace) -> Vec<(Vec<u32>, usize)> {
    let mut result: Vec<_> = [
        &workspace.z_buffer,
        &workspace.basis_values,
        &workspace.basis_derivs,
        &workspace.layer_output,
        &workspace.layer_input,
        &workspace.layer_grads,
        &workspace.staging_buffer,
        &workspace.predictions_buffer,
        &workspace.grad_output,
    ]
    .into_iter()
    .chain(&workspace.layers_inputs)
    .map(|buffer| {
        (
            buffer
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            buffer.capacity(),
        )
    })
    .collect();
    for buffer in workspace.weight_grads.iter().chain(&workspace.bias_grads) {
        result.push((
            buffer.iter().map(|value| value.to_bits()).collect(),
            buffer.capacity(),
        ));
    }
    result.push((
        workspace.grid_indices.clone(),
        workspace.grid_indices.capacity(),
    ));
    for buffer in &workspace.layers_grid_indices {
        result.push((buffer.clone(), buffer.capacity()));
    }
    for capacity in [
        workspace.layers_inputs.capacity(),
        workspace.layers_grid_indices.capacity(),
        workspace.weight_grads.capacity(),
        workspace.bias_grads.capacity(),
        workspace.batch_capacity(),
        workspace.history_batch_size(),
    ] {
        result.push((vec![], capacity));
    }
    result
}

fn check_zero_rate_nonfinite_gradients(threshold: usize) {
    for learning_rate in [0.0, -0.0] {
        for input in [0.0, 1.0] {
            for target in [3e38_f32, -3e38_f32] {
                for options in [
                    TrainOptions::default(),
                    TrainOptions {
                        max_grad_norm: Some(1.0),
                        weight_decay: 0.1,
                    },
                ] {
                    let mut actual = zero_rate_network(threshold);
                    let mut reference = actual.clone();
                    let original = parameter_bits(&actual);
                    let mut workspace = actual.create_workspace(1);
                    let mut reference_workspace = reference.create_workspace(1);
                    assert_eq!(
                        workspace_bits(&workspace),
                        workspace_bits(&reference_workspace)
                    );
                    let loss = actual
                        .try_train_step_with_options(
                            &[input],
                            &[target],
                            None,
                            learning_rate,
                            &mut workspace,
                            &options,
                        )
                        .unwrap();
                    // A nonzero update uses the same loss/backward/clipping path.
                    let reference_loss = reference
                        .try_train_step_with_options(
                            &[input],
                            &[target],
                            None,
                            0.125,
                            &mut reference_workspace,
                            &options,
                        )
                        .unwrap();
                    assert_eq!(loss.to_bits(), f32::INFINITY.to_bits());
                    assert_eq!(loss.to_bits(), reference_loss.to_bits());
                    assert_eq!(workspace.predictions_buffer[0].to_bits(), 0.0_f32.to_bits());
                    let gradient = if target > 0.0 {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    };
                    assert_eq!(workspace.grad_output[0].to_bits(), gradient.to_bits());
                    if options.max_grad_norm.is_none() {
                        assert_eq!(workspace.bias_grads[0][0].to_bits(), gradient.to_bits());
                    }
                    assert_eq!(workspace.history_batch_size(), 1);
                    assert_eq!(
                        workspace_bits(&workspace),
                        workspace_bits(&reference_workspace)
                    );
                    assert_eq!(
                        parameter_bits(&actual), original,
                        "zero-rate parameter mutation: lr={learning_rate:?}, input={input}, target={target}, options={options:?}, threshold={threshold}",
                    );
                }
            }
        }
    }
}

#[test]
fn zero_learning_rate_preserves_parameter_bits_serial() {
    check_zero_rate_nonfinite_gradients(usize::MAX);
}

#[cfg(feature = "parallel")]
#[test]
fn zero_learning_rate_preserves_parameter_bits_parallel() {
    check_zero_rate_nonfinite_gradients(1);
}

#[test]
fn zero_learning_rate_still_clips_finite_gradients() {
    let thresholds: &[usize] = if cfg!(feature = "parallel") {
        &[usize::MAX, 1]
    } else {
        &[usize::MAX]
    };
    for &threshold in thresholds {
        for learning_rate in [0.0, -0.0] {
            let mut actual = zero_rate_network(threshold);
            let mut reference = actual.clone();
            let original = parameter_bits(&actual);
            let mut workspace = actual.create_workspace(1);
            let mut reference_workspace = reference.create_workspace(1);
            let options = TrainOptions {
                max_grad_norm: Some(0.25),
                weight_decay: 0.1,
            };
            let loss = actual
                .try_train_step_with_options(
                    &[0.0],
                    &[4.0],
                    None,
                    learning_rate,
                    &mut workspace,
                    &options,
                )
                .unwrap();
            let reference_loss = reference
                .try_train_step_with_options(
                    &[0.0],
                    &[4.0],
                    None,
                    0.125,
                    &mut reference_workspace,
                    &options,
                )
                .unwrap();
            assert_eq!(loss.to_bits(), 16.0_f32.to_bits());
            assert_eq!(loss.to_bits(), reference_loss.to_bits());
            let norm = workspace
                .weight_grads
                .iter()
                .chain(&workspace.bias_grads)
                .flatten()
                .map(|&g| f64::from(g).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 0.25).abs() < 1e-6,
                "finite gradients were not clipped: {norm}"
            );
            assert_eq!(
                workspace_bits(&workspace),
                workspace_bits(&reference_workspace)
            );
            assert_eq!(parameter_bits(&actual), original);
        }
    }
}

fn decay_order_fixture() -> KanNetwork {
    let mut model = KanNetwork::new(KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: 1,
        spline_order: 1,
        grid_range: (0.0, 1.0),
        input_mean: vec![0.0],
        input_std: vec![1.0],
        multithreading_threshold: usize::MAX,
        init_seed: Some(42),
        ..KanConfig::default()
    });
    model.layers[0].weights.fill(1.0 / 16.0);
    model.layers[0].bias.fill(1.0 / 8.0);
    model
}

#[test]
fn direct_sgd_decays_before_indexed_gradient_updates_and_never_decays_biases() {
    let cases = [
        (
            0.25f32,
            [-0.75, -0.25],
            [31.0 / 128.0, 15.0 / 128.0],
            [1.0 / 4.0, 1.0 / 8.0],
        ),
        (
            0.5,
            [-0.5, -0.5],
            [23.0 / 128.0, 23.0 / 128.0],
            [3.0 / 16.0, 3.0 / 16.0],
        ),
        (
            0.75,
            [-0.25, -0.75],
            [15.0 / 128.0, 31.0 / 128.0],
            [1.0 / 8.0, 1.0 / 4.0],
        ),
    ];
    for (input, gradient, decayed, undecayed) in cases {
        for decay in [0.0, -0.0, 0.5] {
            let mut model = decay_order_fixture();
            let mut no_update = model.clone();
            let mut workspace = model.create_workspace(1);
            let mut no_update_workspace = no_update.create_workspace(1);
            let capacities = (
                model.layers[0].weights.capacity(),
                model.layers[0].bias.capacity(),
            );
            let options = TrainOptions {
                max_grad_norm: None,
                weight_decay: decay,
            };
            let reference_loss = no_update
                .try_train_step_with_options(
                    &[input],
                    &[11.0 / 16.0],
                    None,
                    0.0,
                    &mut no_update_workspace,
                    &options,
                )
                .unwrap();
            let loss = model
                .try_train_step_with_options(
                    &[input],
                    &[11.0 / 16.0],
                    None,
                    1.0 / 4.0,
                    &mut workspace,
                    &options,
                )
                .unwrap();
            assert_eq!(loss.to_bits(), (1.0 / 4.0_f32).to_bits());
            assert_eq!(loss.to_bits(), reference_loss.to_bits());
            assert_eq!(
                workspace.weight_grads[0]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                gradient.map(f32::to_bits)
            );
            assert_eq!(workspace.bias_grads[0][0].to_bits(), (-1.0f32).to_bits());
            let expected = if decay > 0.0 { decayed } else { undecayed };
            assert_eq!(
                parameter_bits(&model),
                [expected[0], expected[1], 3.0 / 8.0].map(f32::to_bits)
            );
            assert_eq!(
                workspace_bits(&workspace),
                workspace_bits(&no_update_workspace)
            );
            assert_eq!(
                (
                    model.layers[0].weights.capacity(),
                    model.layers[0].bias.capacity()
                ),
                capacities
            );
        }
    }
}

#[test]
fn negative_direct_sgd_decay_is_rejected_before_warmed_state_changes() {
    for decay in [-0.5, -f32::MIN_POSITIVE] {
        let mut model = decay_order_fixture();
        let mut workspace = model.create_workspace(1);
        model
            .try_train_step_with_options(
                &[0.5],
                &[11.0 / 16.0],
                None,
                0.0,
                &mut workspace,
                &TrainOptions::default(),
            )
            .unwrap();
        let parameters = parameter_bits(&model);
        let buffers = workspace_bits(&workspace);
        let capacities = (
            model.layers[0].weights.capacity(),
            model.layers[0].bias.capacity(),
        );
        let result = model.try_train_step_with_options(
            &[0.25],
            &[0.0],
            None,
            1.0 / 4.0,
            &mut workspace,
            &TrainOptions {
                max_grad_norm: None,
                weight_decay: decay,
            },
        );
        assert!(matches!(result, Err(arkan::ArkanError::Cpu(_))));
        assert_eq!(parameter_bits(&model), parameters);
        assert_eq!(workspace_bits(&workspace), buffers);
        assert_eq!(
            (
                model.layers[0].weights.capacity(),
                model.layers[0].bias.capacity()
            ),
            capacities
        );
    }
}
