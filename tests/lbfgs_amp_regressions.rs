use arkan::optimizer::{LBFGSConfig, LineSearchMethod, SafetyConfig, LBFGS};
use arkan::{ArkanError, KanConfig, KanNetwork};

fn network() -> KanNetwork {
    let mut network = KanNetwork::new(KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![],
        input_mean: vec![0.0],
        input_std: vec![1.0],
        init_seed: Some(42),
        ..Default::default()
    });
    network.layers[0].weights[0] = 1.0;
    network
}

fn objective(network: &KanNetwork, factor: f64) -> (f64, Vec<f32>) {
    let params = LBFGS::flatten_params(network);
    let x = f64::from(params[0]);
    let mut gradient = vec![0.0; params.len()];
    gradient[0] = (factor * x) as f32;
    (0.5 * x * x, gradient)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn lbfgs_amp_matches_unscaled_objective_trials_history_and_convergence() {
    for method in [
        LineSearchMethod::NoLineSearch,
        LineSearchMethod::Backtracking,
        LineSearchMethod::StrongWolfe,
    ] {
        for legacy_flag in [true, false] {
            let mut plain = network();
            let mut scaled = plain.clone();
            let config = LBFGSConfig {
                lr: 0.25,
                max_iter: 1,
                line_search_fn: method,
                ..Default::default()
            };
            let mut control = LBFGS::new(&plain, config);
            let mut amp = LBFGS::new(
                &scaled,
                LBFGSConfig {
                    safety: SafetyConfig {
                        unscale_before_step: legacy_flag,
                        ..SafetyConfig::with_amp(4.0)
                    },
                    ..config
                },
            );
            for _ in 0..3 {
                let mut plain_trials = Vec::new();
                let mut scaled_trials = Vec::new();
                let expected = control
                    .step_lbfgs(&mut plain, |model| {
                        plain_trials.push(model.layers[0].weights[0].to_bits());
                        Ok(objective(model, 1.0))
                    })
                    .unwrap();
                let actual = amp
                    .step_lbfgs(&mut scaled, |model| {
                        scaled_trials.push(model.layers[0].weights[0].to_bits());
                        Ok(objective(model, 4.0))
                    })
                    .unwrap();
                assert_eq!(actual.to_bits(), expected.to_bits());
                assert_eq!(scaled_trials, plain_trials);
                assert_eq!(
                    bits(&LBFGS::flatten_params(&scaled)),
                    bits(&LBFGS::flatten_params(&plain))
                );
                assert_eq!(
                    bits(&amp.two_loop_recursion(&vec![1.0; LBFGS::flatten_params(&scaled).len()])),
                    bits(
                        &control
                            .two_loop_recursion(&vec![1.0; LBFGS::flatten_params(&plain).len()])
                    )
                );
                assert_eq!(amp.num_evals(), control.num_evals());
            }
        }
    }
}

#[test]
fn lbfgs_amp_fixed_step_keeps_loss_unscaled() {
    let mut model = network();
    let mut optimizer = LBFGS::new(
        &model,
        LBFGSConfig {
            lr: 0.25,
            max_iter: 1,
            line_search_fn: LineSearchMethod::NoLineSearch,
            safety: SafetyConfig::with_amp(4.0),
            ..Default::default()
        },
    );
    let loss = optimizer
        .step_lbfgs(&mut model, |model| Ok(objective(model, 4.0)))
        .unwrap();
    assert_eq!(model.layers[0].weights[0], 0.75);
    assert_eq!(loss, 0.28125);
}

#[test]
fn lbfgs_amp_checks_unscaled_convergence() {
    let mut model = network();
    let mut optimizer = LBFGS::new(
        &model,
        LBFGSConfig {
            tolerance_grad: 1.5,
            safety: SafetyConfig::with_amp(4.0),
            ..Default::default()
        },
    );
    let loss = optimizer
        .step_lbfgs(&mut model, |model| Ok(objective(model, 4.0)))
        .unwrap();
    assert_eq!(model.layers[0].weights[0], 1.0);
    assert_eq!(loss, 0.5);
    assert_eq!(optimizer.num_evals(), 1);
}

#[test]
fn lbfgs_tiny_amp_overflow_rolls_back_warmed_model_and_history() {
    for skip in [false, true] {
        for fail_at in [1, 2, 3] {
            let mut model = network();
            let mut optimizer = LBFGS::new(
                &model,
                LBFGSConfig {
                    lr: 0.25,
                    max_iter: 1,
                    line_search_fn: LineSearchMethod::NoLineSearch,
                    ..Default::default()
                },
            );
            optimizer
                .step_lbfgs(&mut model, |model| Ok(objective(model, 1.0)))
                .unwrap();
            optimizer.config.max_iter = 2;
            optimizer.config.safety = SafetyConfig {
                skip_step_on_nan: skip,
                grad_scaling_factor: Some(1e-40),
                ..SafetyConfig::strict()
            };
            let before = bits(&LBFGS::flatten_params(&model));
            let probe = vec![1.0; before.len()];
            let history_before = bits(&optimizer.two_loop_recursion(&probe));
            let evaluations = optimizer.num_evals();
            #[cfg(feature = "serde")]
            let serialized_before = bincode::serialize(&optimizer).unwrap();
            let initial_loss = objective(&model, 1.0).0;
            let mut calls = 0;
            let result = optimizer.step_lbfgs(&mut model, |model| {
                calls += 1;
                let (loss, mut gradient) = objective(model, 1e-40);
                if calls == fail_at {
                    gradient[0] = 1.0;
                }
                Ok((loss, gradient))
            });
            assert_eq!(calls, fail_at);
            if skip {
                assert_eq!(result.unwrap(), initial_loss);
            } else {
                assert!(matches!(result, Err(ArkanError::NaNEncountered { .. })));
            }
            assert_eq!(optimizer.num_evals(), evaluations + calls);
            assert_eq!(bits(&LBFGS::flatten_params(&model)), before);
            assert_eq!(bits(&optimizer.two_loop_recursion(&probe)), history_before);
            #[cfg(feature = "serde")]
            {
                type SerializedState = (
                    LBFGSConfig,
                    Vec<Vec<f32>>,
                    Vec<Vec<f32>>,
                    Vec<f64>,
                    Option<Vec<f32>>,
                    Option<Vec<f32>>,
                    u64,
                    usize,
                );
                let mut expected: SerializedState =
                    bincode::deserialize(&serialized_before).unwrap();
                expected.7 += calls;
                assert_eq!(
                    bincode::serialize(&optimizer).unwrap(),
                    bincode::serialize(&expected).unwrap()
                );
            }
        }
    }
}
