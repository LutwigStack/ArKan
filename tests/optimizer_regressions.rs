use arkan::optimizer::{
    Adam, AdamConfig, LBFGSConfig, LineSearchMethod, Optimizer, SGDConfig, SafetyConfig, LBFGS, SGD,
};
use arkan::{ArkanError, KanConfig, KanNetwork};

fn network(hidden: bool) -> KanNetwork {
    KanNetwork::new(KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: if hidden { vec![1] } else { vec![] },
        grid_size: 3,
        spline_order: 3,
        input_mean: vec![0.0],
        input_std: vec![1.0],
        init_seed: Some(42),
        ..Default::default()
    })
}
fn gradients(net: &KanNetwork) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    (
        net.layers
            .iter()
            .map(|l| vec![0.0; l.weights.len()])
            .collect(),
        net.layers.iter().map(|l| vec![0.0; l.bias.len()]).collect(),
    )
}
fn quadratic(net: &KanNetwork, curvature: f64) -> Result<(f64, Vec<f32>), ArkanError> {
    let params = LBFGS::flatten_params(net);
    let mut grad = vec![0.0; params.len()];
    grad[0] = (curvature * params[0] as f64) as f32;
    Ok((0.5 * curvature * (params[0] as f64).powi(2), grad))
}
#[test]
fn malformed_late_gradient_is_rejected_before_adam_or_sgd_mutation() {
    for adam in [true, false] {
        let mut net = network(true);
        let before = LBFGS::flatten_params(&net);
        let (mut wg, mut bg) = gradients(&net);
        wg[0].fill(1.0);
        bg[1].clear();
        if adam {
            let mut opt = Adam::new(&net, AdamConfig::default());
            assert!(opt.step(&mut net, &wg, &bg, None).is_err());
            assert_eq!(opt.layer_states[0].weights.t, 0);
        } else {
            let mut opt = SGD::new(&net, SGDConfig::default());
            assert!(opt.step(&mut net, &wg, &bg, None).is_err());
            assert!(opt.velocities[0].0.as_slice().iter().all(|&v| v == 0.0));
        }
        assert_eq!(LBFGS::flatten_params(&net), before);
    }
}
#[test]
fn malformed_optimizer_state_is_rejected_before_mutation() {
    let mut net = network(true);
    let before = LBFGS::flatten_params(&net);
    let (wg, bg) = gradients(&net);
    let mut adam = Adam::new(&net, AdamConfig::default());
    adam.layer_states[1].bias.v.resize(0);
    assert!(adam.step(&mut net, &wg, &bg, None).is_err());
    let mut sgd = SGD::new(&net, SGDConfig::default());
    sgd.velocities.pop();
    assert!(sgd.step(&mut net, &wg, &bg, None).is_err());
    assert_eq!(LBFGS::flatten_params(&net), before);
}
#[test]
fn sgd_clips_globally_and_preserves_large_finite_gradient_direction() {
    for large in [false, true] {
        let mut net = network(true);
        for l in &mut net.layers {
            l.bias.fill(0.0);
        }
        let (wg, mut bg) = gradients(&net);
        for g in &mut bg {
            g[0] = if large { 1e20 } else { 1.0 };
        }
        SGD::new(&net, SGDConfig::with_lr(1.0))
            .step(&mut net, &wg, &bg, Some(1.0))
            .unwrap();
        for l in &net.layers {
            assert!((l.bias[0] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        }
    }
}
#[test]
fn invalid_configuration_and_transformed_nonfinite_gradients_are_atomic() {
    let mut net = network(false);
    let before = LBFGS::flatten_params(&net);
    let (mut wg, bg) = gradients(&net);
    wg[0].fill(1.0);
    for scale in [0.0, f64::NAN, 1e-320] {
        let safety = SafetyConfig {
            grad_scaling_factor: Some(scale),
            ..SafetyConfig::strict()
        };
        assert!(SGD::new(&net, SGDConfig::default().with_safety(safety))
            .step(&mut net, &wg, &bg, None)
            .is_err());
    }
    let mut adam = Adam::new(
        &net,
        AdamConfig {
            beta1: 1.0,
            ..Default::default()
        },
    );
    assert!(adam.step(&mut net, &wg, &bg, None).is_err());
    let mut sgd = SGD::new(&net, SGDConfig::default());
    assert!(sgd.step(&mut net, &wg, &bg, Some(-1.0)).is_err());
    assert_eq!(LBFGS::flatten_params(&net), before);
}
#[test]
fn lbfgs_exhausted_wolfe_search_rolls_back_and_obeys_evaluation_budget() {
    let mut net = network(false);
    net.layers[0].weights[0] = 1.0;
    let before = LBFGS::flatten_params(&net);
    let mut opt = LBFGS::new(&net, LBFGSConfig::default());
    let mut calls = 0;
    let result = opt.step_lbfgs(&mut net, |net| {
        calls += 1;
        quadratic(net, 1e10)
    });
    assert!(result.is_err());
    assert!(calls <= 25);
    assert_eq!(opt.num_evals(), calls);
    assert_eq!(LBFGS::flatten_params(&net), before);
}
#[test]
fn lbfgs_closure_error_preserves_original_error_and_rolls_back() {
    for method in [
        LineSearchMethod::NoLineSearch,
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        let mut net = network(false);
        net.layers[0].weights[0] = 1.0;
        let before = LBFGS::flatten_params(&net);
        let mut opt = LBFGS::new(
            &net,
            LBFGSConfig {
                line_search_fn: method,
                ..Default::default()
            },
        );
        let mut calls = 0;
        let error = opt
            .step_lbfgs(&mut net, |net| {
                calls += 1;
                if calls == 2 {
                    Err(ArkanError::optimizer("objective failure"))
                } else {
                    quadratic(net, 1.0)
                }
            })
            .unwrap_err();
        assert!(error.to_string().contains("objective failure"));
        assert_eq!(LBFGS::flatten_params(&net), before);
    }
}
#[test]
fn lbfgs_loss_matches_accepted_parameters_and_iteration_budget() {
    for iterations in [1, 3] {
        let mut net = network(false);
        net.layers[0].weights[0] = 1.0;
        let mut opt = LBFGS::new(
            &net,
            LBFGSConfig {
                lr: 0.25,
                max_iter: iterations,
                line_search_fn: LineSearchMethod::NoLineSearch,
                tolerance_change: 0.0,
                ..Default::default()
            },
        );
        let loss = opt.step_lbfgs(&mut net, |net| quadratic(net, 1.0)).unwrap();
        assert_eq!(opt.num_evals(), iterations + 1);
        assert!((net.layers[0].weights[0] - 0.75_f32.powi(iterations as i32)).abs() < 1e-6);
        assert!((loss - quadratic(&net, 1.0).unwrap().0).abs() < 1e-12);
    }
}
#[test]
fn lbfgs_invalid_config_and_gradient_layout_are_rejected() {
    let mut net = network(false);
    let before = LBFGS::flatten_params(&net);
    for config in [
        LBFGSConfig {
            history_size: 0,
            ..Default::default()
        },
        LBFGSConfig {
            max_iter: 0,
            ..Default::default()
        },
        LBFGSConfig {
            max_eval: Some(1),
            ..Default::default()
        },
    ] {
        assert!(LBFGS::new(&net, config)
            .step_lbfgs(&mut net, |net| quadratic(net, 1.0))
            .is_err());
    }
    for gradient in [vec![], vec![f32::NAN; before.len()]] {
        assert!(LBFGS::new(
            &net,
            LBFGSConfig {
                safety: SafetyConfig::strict(),
                ..Default::default()
            }
        )
        .step_lbfgs(&mut net, |_| Ok((1.0, gradient.clone())))
        .is_err());
    }
    assert_eq!(LBFGS::flatten_params(&net), before);
}
#[test]
fn strict_numerical_update_failure_does_not_mutate_earlier_tensors_or_state() {
    for adam in [true, false] {
        let mut net = network(true);
        let (mut wg, bg) = gradients(&net);
        wg[0].fill(1.0);
        wg[1].fill(f32::MAX);
        let before = LBFGS::flatten_params(&net);
        if adam {
            let mut opt = Adam::new(
                &net,
                AdamConfig::default().with_safety(SafetyConfig::strict()),
            );
            assert!(opt.step(&mut net, &wg, &bg, None).is_err());
            assert_eq!(opt.layer_states[0].weights.t, 0);
            assert!(opt.layer_states[0]
                .weights
                .m
                .as_slice()
                .iter()
                .all(|&v| v == 0.0));
        } else {
            let mut opt = SGD::new(
                &net,
                SGDConfig::with_lr(2.0).with_safety(SafetyConfig::strict()),
            );
            assert!(opt.step(&mut net, &wg, &bg, None).is_err());
            assert!(opt.velocities[0].0.as_slice().iter().all(|&v| v == 0.0));
        }
        assert_eq!(LBFGS::flatten_params(&net), before);
    }
}
#[test]
fn lbfgs_line_search_accepts_consistent_points_and_stops_at_total_budget() {
    for method in [
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
        LineSearchMethod::NoLineSearch,
    ] {
        let mut net = network(false);
        net.layers[0].weights[0] = 1.0;
        let mut opt = LBFGS::new(
            &net,
            LBFGSConfig {
                lr: 0.25,
                max_eval: Some(2),
                line_search_fn: method,
                ..Default::default()
            },
        );
        let loss = opt.step_lbfgs(&mut net, |net| quadratic(net, 1.0)).unwrap();
        assert_eq!(opt.num_evals(), 2);
        assert!((net.layers[0].weights[0] - 0.75).abs() < 1e-6);
        assert_eq!(loss, quadratic(&net, 1.0).unwrap().0);
    }
}
#[test]
fn lbfgs_failure_after_an_accepted_iteration_restores_parameters_and_history() {
    let mut net = network(false);
    net.layers[0].weights[0] = 1.0;
    let before = LBFGS::flatten_params(&net);
    let mut opt = LBFGS::new(
        &net,
        LBFGSConfig {
            lr: 0.25,
            max_iter: 3,
            line_search_fn: LineSearchMethod::NoLineSearch,
            ..Default::default()
        },
    );
    #[cfg(feature = "serde")]
    let mut original_state: L3State =
        bincode::deserialize(&bincode::serialize(&opt).unwrap()).unwrap();
    let mut evaluations = 0;
    assert!(opt
        .step_lbfgs(&mut net, |net| {
            evaluations += 1;
            if evaluations == 3 {
                Err(ArkanError::optimizer("third evaluation failed"))
            } else {
                quadratic(net, 2.0)
            }
        })
        .is_err());
    assert_eq!(LBFGS::flatten_params(&net), before);
    let gradient: Vec<f32> = (0..before.len())
        .map(|i| if i == 0 { 2.0 } else { 0.0 })
        .collect();
    assert_eq!(opt.two_loop_recursion(&gradient)[0], -2.0);
    assert_eq!(opt.num_evals(), 3);
    #[cfg(feature = "serde")]
    {
        original_state.7 = 3;
        assert_eq!(
            bincode::serialize(&opt).unwrap(),
            bincode::serialize(&original_state).unwrap()
        );
    }
}
#[test]
fn default_skip_policy_preserves_parameters_for_unscale_overflow() {
    let mut net = network(false);
    let before = LBFGS::flatten_params(&net);
    let (mut wg, bg) = gradients(&net);
    wg[0].fill(1.0);
    let safety = SafetyConfig::with_amp(1e-320);
    let mut adam = Adam::new(&net, AdamConfig::default().with_safety(safety));
    adam.step(&mut net, &wg, &bg, None).unwrap();
    assert_eq!(adam.layer_states[0].weights.t, 0);
    SGD::new(&net, SGDConfig::default().with_safety(safety))
        .step(&mut net, &wg, &bg, None)
        .unwrap();
    assert_eq!(LBFGS::flatten_params(&net), before);
}
#[cfg(feature = "serde")]
#[test]
fn deserialized_invalid_adam_state_is_rejected_when_bound_to_network() {
    let mut net = network(false);
    let before = LBFGS::flatten_params(&net);
    let mut value = serde_json::to_value(Adam::new(&net, AdamConfig::default())).unwrap();
    value["layer_states"] = serde_json::json!([]);
    let mut adam: Adam = serde_json::from_value(value).unwrap();
    let (wg, bg) = gradients(&net);
    assert!(adam.step(&mut net, &wg, &bg, None).is_err());
    assert_eq!(LBFGS::flatten_params(&net), before);
}

#[test]
fn lbfgs_line_search_recovers_from_nonfinite_trials_before_accepting() {
    for method in [
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        for safety in [SafetyConfig::default(), SafetyConfig::strict()] {
            let mut net = network(false);
            net.layers[0].weights[0] = 10.0;
            let mut optimizer = LBFGS::new(
                &net,
                LBFGSConfig {
                    max_iter: 1,
                    line_search_fn: method,
                    safety,
                    ..Default::default()
                },
            );
            let mut rejected = 0;
            let loss = optimizer
                .step_lbfgs(&mut net, |net| {
                    let params = LBFGS::flatten_params(net);
                    let x = params[0] as f64;
                    let loss = x.exp() + (-x).exp();
                    let mut gradient = vec![0.0; params.len()];
                    gradient[0] = (x.exp() - (-x).exp()) as f32;
                    if !loss.is_finite() || !gradient[0].is_finite() {
                        rejected += 1;
                    }
                    Ok((loss, gradient))
                })
                .unwrap();
            assert!(rejected > 0);
            assert!(loss < 3.0, "{method:?}: loss={loss}");
            let x = net.layers[0].weights[0] as f64;
            assert_eq!(loss, x.exp() + (-x).exp());
            assert!(optimizer.num_evals() > 2 && optimizer.num_evals() <= 25);
            assert!(optimizer
                .two_loop_recursion(&vec![1.0; LBFGS::flatten_params(&net).len()])
                .iter()
                .all(|g| g.is_finite()));
        }
    }
}

#[test]
fn lbfgs_closure_error_after_numerical_rejection_is_preserved() {
    for method in [
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        let mut net = network(false);
        net.layers[0].weights[0] = 1.0;
        let before = LBFGS::flatten_params(&net);
        let mut optimizer = LBFGS::new(
            &net,
            LBFGSConfig {
                line_search_fn: method,
                ..Default::default()
            },
        );
        let mut calls = 0;
        let error = optimizer
            .step_lbfgs(&mut net, |net| {
                calls += 1;
                match calls {
                    1 => quadratic(net, 1.0),
                    2 => Ok((f64::INFINITY, vec![0.0; before.len()])),
                    _ => Err(ArkanError::optimizer(
                        "objective error after rejected trial",
                    )),
                }
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("objective error after rejected trial"));
        assert_eq!(LBFGS::flatten_params(&net), before);
        assert_eq!(calls, 3);
    }
}

#[test]
fn lbfgs_malformed_trial_gradient_is_rejected_without_backtracking() {
    for method in [
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        let mut net = network(false);
        net.layers[0].weights[0] = 1.0;
        let before = LBFGS::flatten_params(&net);
        let mut optimizer = LBFGS::new(
            &net,
            LBFGSConfig {
                line_search_fn: method,
                ..Default::default()
            },
        );
        let mut calls = 0;
        assert!(optimizer
            .step_lbfgs(&mut net, |net| {
                calls += 1;
                if calls == 1 {
                    quadratic(net, 1.0)
                } else {
                    Ok((f64::INFINITY, vec![]))
                }
            })
            .is_err());
        assert_eq!(calls, 2);
        assert_eq!(LBFGS::flatten_params(&net), before);
    }
}

#[test]
fn lbfgs_rejects_nonfinite_trial_gradients_even_when_trial_loss_decreases() {
    for method in [
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        let mut net = network(false);
        net.layers[0].weights[0] = 1.0;
        let mut optimizer = LBFGS::new(
            &net,
            LBFGSConfig {
                max_iter: 1,
                line_search_fn: method,
                safety: SafetyConfig::strict(),
                ..Default::default()
            },
        );
        let mut calls = 0;
        let loss = optimizer
            .step_lbfgs(&mut net, |net| {
                calls += 1;
                let (loss, mut gradient) = quadratic(net, 1.0)?;
                if calls == 2 {
                    gradient[0] = f32::NAN;
                }
                Ok((loss, gradient))
            })
            .unwrap();
        assert_eq!(calls, 3);
        assert_eq!(loss, 0.125);
        assert_eq!(net.layers[0].weights[0], 0.5);
    }
}

#[test]
fn lbfgs_numerical_trial_exhaustion_applies_safety_policy_and_rolls_back() {
    for method in [
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        for safety in [SafetyConfig::default(), SafetyConfig::strict()] {
            let mut net = network(false);
            net.layers[0].weights[0] = 1.0;
            let before = LBFGS::flatten_params(&net);
            let mut optimizer = LBFGS::new(
                &net,
                LBFGSConfig {
                    max_eval: Some(2),
                    line_search_fn: method,
                    safety,
                    ..Default::default()
                },
            );
            let mut calls = 0;
            let result = optimizer.step_lbfgs(&mut net, |net| {
                calls += 1;
                if calls == 1 {
                    quadratic(net, 1.0)
                } else {
                    Ok((f64::INFINITY, vec![0.0; before.len()]))
                }
            });
            if safety.skip_step_on_nan {
                assert_eq!(result.unwrap(), 0.5);
            } else {
                assert!(result.is_err());
            }
            assert_eq!(LBFGS::flatten_params(&net), before);
            assert_eq!(calls, 2);
            assert_eq!(optimizer.num_evals(), 2);
        }
    }
}

#[test]
fn lbfgs_invalid_initial_objective_never_enters_line_search() {
    for safety in [SafetyConfig::default(), SafetyConfig::strict()] {
        let mut net = network(false);
        let before = LBFGS::flatten_params(&net);
        let mut optimizer = LBFGS::new(
            &net,
            LBFGSConfig {
                safety,
                ..Default::default()
            },
        );
        let mut calls = 0;
        let result = optimizer.step_lbfgs(&mut net, |_| {
            calls += 1;
            Ok((f64::INFINITY, vec![0.0; before.len()]))
        });
        if safety.skip_step_on_nan {
            assert_eq!(result.unwrap(), f64::INFINITY);
        } else {
            assert!(result.is_err());
        }
        assert_eq!(calls, 1);
        assert_eq!(LBFGS::flatten_params(&net), before);
    }
}

#[test]
fn lbfgs_shared_objective_can_mutate_captured_gradient_and_state() {
    let mut net = network(false);
    net.layers[0].weights[0] = 1.0;
    let mut optimizer = LBFGS::new(
        &net,
        LBFGSConfig {
            lr: 0.25,
            max_iter: 1,
            history_size: 1,
            line_search_fn: LineSearchMethod::NoLineSearch,
            ..Default::default()
        },
    );
    let mut gradient = vec![0.0; LBFGS::flatten_params(&net).len()];
    let mut observed = Vec::new();
    let loss = optimizer
        .step_lbfgs(&mut net, |net: &KanNetwork| {
            let x = net.layers[0].weights[0];
            observed.push(x);
            gradient[0] = 2.0 * x;
            Ok((f64::from(x).powi(2), gradient.clone()))
        })
        .unwrap();
    assert_eq!(observed, [1.0, 0.5]);
    assert_eq!(gradient[0], 1.0);
    assert_eq!(loss, 0.25);
    assert_eq!(net.layers[0].weights[0], 0.5);
    assert_eq!(optimizer.num_evals(), 2);
    assert_eq!(optimizer.two_loop_recursion(&gradient)[0], -0.5);
}

#[test]
fn lbfgs_nonfinite_parameters_are_rejected_before_objective_calls() {
    for layer in 0..2 {
        for bias in [false, true] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut net = network(true);
                if bias {
                    net.layers[layer].bias[0] = value;
                } else {
                    net.layers[layer].weights[0] = value;
                }
                let before: Vec<_> = LBFGS::flatten_params(&net)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                let mut optimizer = LBFGS::new(
                    &net,
                    LBFGSConfig {
                        safety: SafetyConfig::strict(),
                        ..Default::default()
                    },
                );
                let mut calls = 0;
                let error = optimizer
                    .step_lbfgs(&mut net, |net| {
                        calls += 1;
                        quadratic(net, 1.0)
                    })
                    .unwrap_err();
                assert!(matches!(error, ArkanError::NaNEncountered { .. }));
                assert_eq!(calls, 0);
                assert_eq!(optimizer.num_evals(), 0);
                let after: Vec<_> = LBFGS::flatten_params(&net)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                assert_eq!(after, before);
            }
        }
    }
}

#[test]
fn lbfgs_nonfinite_trial_parameters_do_not_consume_objective_budget() {
    for method in [
        LineSearchMethod::NoLineSearch,
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        for safety in [SafetyConfig::default(), SafetyConfig::strict()] {
            let mut net = network(false);
            let before = LBFGS::flatten_params(&net);
            let mut optimizer = LBFGS::new(
                &net,
                LBFGSConfig {
                    lr: f32::MAX,
                    max_iter: 1,
                    max_eval: Some(2),
                    line_search_fn: method,
                    safety,
                    ..Default::default()
                },
            );
            let mut calls = 0;
            let result = optimizer.step_lbfgs(&mut net, |_| {
                calls += 1;
                let mut gradient = vec![0.0; before.len()];
                gradient[0] = f32::MAX;
                Ok((1.0, gradient))
            });
            if safety.skip_step_on_nan {
                assert_eq!(result.unwrap(), 1.0);
            } else {
                assert!(matches!(
                    result.unwrap_err(),
                    ArkanError::NaNEncountered { .. }
                ));
            }
            assert_eq!(calls, 1);
            assert_eq!(optimizer.num_evals(), 1);
            assert_eq!(LBFGS::flatten_params(&net), before);
        }
    }
}

#[test]
fn lbfgs_cancellation_preserves_warmed_history_and_original_error() {
    for method in [
        LineSearchMethod::NoLineSearch,
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        let mut net = network(false);
        net.layers[0].weights[0] = 1.0;
        let mut optimizer = LBFGS::new(
            &net,
            LBFGSConfig {
                lr: 0.25,
                max_iter: 1,
                history_size: 1,
                line_search_fn: method,
                ..Default::default()
            },
        );
        assert_eq!(
            optimizer
                .step_lbfgs(&mut net, |net| quadratic(net, 2.0))
                .unwrap(),
            0.25
        );
        let before = LBFGS::flatten_params(&net);
        let probe = vec![2.0; before.len()];
        let direction = optimizer.two_loop_recursion(&probe);
        assert_eq!(direction, vec![-1.0; before.len()]);
        #[cfg(feature = "serde")]
        let mut original_state: L3State =
            bincode::deserialize(&bincode::serialize(&optimizer).unwrap()).unwrap();
        let mut calls = 0;
        let error = optimizer
            .step_lbfgs(&mut net, |net| {
                calls += 1;
                if calls == 2 {
                    Err(ArkanError::optimizer("cancel warmed objective"))
                } else {
                    quadratic(net, 2.0)
                }
            })
            .unwrap_err();
        assert!(
            matches!(error, ArkanError::Optimizer(message) if message == "cancel warmed objective")
        );
        assert_eq!(calls, 2);
        assert_eq!(optimizer.num_evals(), 4);
        assert_eq!(LBFGS::flatten_params(&net), before);
        assert_eq!(optimizer.two_loop_recursion(&probe), direction);
        #[cfg(feature = "serde")]
        {
            original_state.7 = 4;
            assert_eq!(
                bincode::serialize(&optimizer).unwrap(),
                bincode::serialize(&original_state).unwrap()
            );
        }
    }
}

// Literal public warm-up fixtures for L3; no second two-loop implementation.
fn l3_warmed_fixture(width: usize, history: &str) -> (KanNetwork, LBFGS) {
    let mut net = KanNetwork::new(KanConfig {
        input_dim: width,
        output_dim: width,
        hidden_dims: vec![width],
        grid_size: 3,
        spline_order: 3,
        input_mean: vec![0.0; width],
        input_std: vec![1.0; width],
        init_seed: Some(42),
        ..Default::default()
    });
    for layer in &mut net.layers {
        layer.weights.fill(0.0);
        layer.bias.fill(0.0);
    }
    net.layers[0].weights[0] = 1.0;
    let mut opt = LBFGS::new(
        &net,
        LBFGSConfig {
            lr: 0.25,
            max_iter: 1,
            max_eval: Some(2),
            tolerance_grad: 0.0,
            tolerance_change: 0.0,
            history_size: if history == "E1" { 1 } else { 3 },
            line_search_fn: LineSearchMethod::NoLineSearch,
            safety: SafetyConfig::strict(),
        },
    );
    let steps = match history {
        "H0" => 0,
        "H1" => 1,
        "H3" | "E1" => 3,
        _ => unreachable!(),
    };
    let n = 12 * width * width + 2 * width;
    for step in 0..steps {
        if step > 0 {
            opt.config.lr = 0.5;
        }
        let loss = opt
            .step_lbfgs(&mut net, |model| {
                let x = model.layers[0].weights[0];
                let mut g = vec![0.0; n];
                g[0] = 2.0 * x;
                Ok((f64::from(x).powi(2), g))
            })
            .unwrap();
        let x = [0.5f32, 0.25, 0.125][step];
        assert_eq!(net.layers[0].weights[0].to_bits(), x.to_bits());
        assert_eq!(loss.to_bits(), f64::from(x).powi(2).to_bits());
    }
    let p = LBFGS::flatten_params(&net);
    assert_eq!(p.len(), n);
    assert!(p[1..].iter().all(|v| v.to_bits() == 0));
    assert_eq!(opt.num_evals(), 2 * steps);
    (net, opt)
}
fn l3_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn l3_mixed(n: usize) -> Vec<f32> {
    let pattern = [
        2.0,
        -3.0,
        0.25,
        -0.5,
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::MAX,
    ];
    (0..n).map(|i| pattern[i % pattern.len()]).collect()
}

#[test]
fn lbfgs_warmed_public_recursion_preserves_literal_direction_and_legacy_lengths() {
    for (width, history, lengths) in [
        (1, "H1", vec![0, 1, 13, 14, 17]),
        (1, "H3", vec![0, 1, 13, 14, 17]),
        (8, "H1", vec![784]),
        (8, "H3", vec![784]),
        (1, "E1", vec![14]),
    ] {
        let (net, opt) = l3_warmed_fixture(width, history);
        let parameters = l3_bits(&LBFGS::flatten_params(&net));
        let evaluations = opt.num_evals();
        let version = opt.get_state_version();
        let config = format!("{:?}", opt.config);
        for n in lengths {
            let g = vec![2.0; n];
            let direction = opt.two_loop_recursion(&g);
            assert_eq!(direction.len(), n);
            assert!(direction.iter().all(|v| v.to_bits() == (-1.0f32).to_bits()));
            let mixed = l3_mixed(n);
            let direction = opt.two_loop_recursion(&mixed);
            assert_eq!(direction.len(), n);
            for (g, d) in mixed.iter().zip(&direction) {
                if *g != 0.0 {
                    assert_eq!(d.to_bits(), ((-0.5f64 * f64::from(*g)) as f32).to_bits());
                }
            }
            assert_eq!(opt.num_evals(), evaluations);
            assert_eq!(opt.get_state_version(), version);
            assert_eq!(format!("{:?}", opt.config), config);
            assert_eq!(l3_bits(&LBFGS::flatten_params(&net)), parameters);
        }
    }
    for (width, n) in [(1, 0), (1, 14), (8, 784)] {
        let (_, opt) = l3_warmed_fixture(width, "H0");
        let g = l3_mixed(n);
        let direction = opt.two_loop_recursion(&g);
        assert_eq!(
            l3_bits(&direction),
            g.iter().map(|v| (-v).to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(opt.num_evals(), 0);
    }
}

#[cfg(feature = "serde")]
type L3State = (
    LBFGSConfig,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<f64>,
    Option<Vec<f32>>,
    Option<Vec<f32>>,
    u64,
    usize,
);
#[cfg(feature = "serde")]
fn l3_literal_state(net: &KanNetwork, opt: &LBFGS, history: &str) -> L3State {
    let parameters = LBFGS::flatten_params(net);
    let count = if history == "H1" { 1 } else { 3 };
    let entries = if history == "E1" { 2..3 } else { 0..count };
    let mut s = Vec::new();
    let mut y = Vec::new();
    let mut rho = Vec::new();
    for i in entries {
        let mut a = vec![0.0; parameters.len()];
        let mut b = a.clone();
        a[0] = [-0.5, -0.25, -0.125][i];
        b[0] = [-1.0, -0.5, -0.25][i];
        s.push(a);
        y.push(b);
        rho.push([2.0, 8.0, 32.0][i]);
    }
    let mut g = vec![0.0; parameters.len()];
    g[0] = 2.0 * parameters[0];
    let tuple = (
        opt.config,
        s,
        y,
        rho,
        Some(parameters),
        Some(g),
        0,
        2 * count,
    );
    assert_eq!(
        bincode::serialize(&tuple).unwrap(),
        bincode::serialize(opt).unwrap(),
        "literal normal tuple field order and state"
    );
    tuple
}

#[cfg(feature = "serde")]
#[test]
fn lbfgs_serde_warmed_recursion_preserves_all_state_bits_and_exceptional_lengths() {
    for (width, history, lengths) in [
        (1, "H1", vec![0, 1, 13, 14, 17]),
        (1, "H3", vec![0, 1, 13, 14, 17]),
        (8, "H1", vec![784]),
        (8, "H3", vec![784]),
        (1, "E1", vec![14]),
    ] {
        let (net, opt) = l3_warmed_fixture(width, history);
        l3_literal_state(&net, &opt, history);
        let before = bincode::serialize(&opt).unwrap();
        let restored: LBFGS = bincode::deserialize(&before).unwrap();
        assert_eq!(bincode::serialize(&restored).unwrap(), before);
        for n in lengths {
            let g = l3_mixed(n);
            let direction = opt.two_loop_recursion(&g);
            assert_eq!(
                l3_bits(&restored.two_loop_recursion(&g)),
                l3_bits(&direction)
            );
            assert_eq!(bincode::serialize(&opt).unwrap(), before);
        }
        if width == 1 && history != "E1" {
            for index in [0, 7] {
                for raw in [0x7f800000, 0xff800000, 0x7fc00011, 0xffc00022] {
                    let mut g = l3_mixed(14);
                    g[index] = f32::from_bits(raw);
                    let actual = opt.two_loop_recursion(&g);
                    let copy = restored.two_loop_recursion(&g);
                    assert_eq!(actual.len(), 14);
                    assert_eq!(l3_bits(&actual), l3_bits(&copy));
                    assert_eq!(bincode::serialize(&opt).unwrap(), before);
                }
            }
        }
    }
}

#[cfg(feature = "serde")]
fn l3_forged_optimizer(kind: &str) -> LBFGS {
    let (net, opt) = l3_warmed_fixture(1, "H1");
    l3_literal_state(&net, &opt, "H1"); // Prove the ordered tuple before forging.
    let mut s = vec![0.0; 14];
    let mut y = s.clone();
    let rho = match kind {
        "fallback" => {
            s[0] = 1.0;
            y[0] = 2.0f32.powi(-20);
            2.0f64.powi(20)
        }
        "nan" => {
            s[0] = f32::from_bits(0x7fc00011);
            y[0] = 1.0;
            1.0
        }
        "length" => {
            s.resize(13, 0.0);
            s[0] = 1.0;
            y[0] = 1.0;
            1.0
        }
        "cardinality" => {
            s[0] = 1.0;
            y[0] = 1.0;
            1.0
        }
        "rho" => {
            s[0] = 1.0;
            y[0] = 1.0;
            -1.0
        }
        _ => unreachable!(),
    };
    let tuple: L3State = (
        opt.config,
        vec![s],
        if kind == "cardinality" {
            vec![]
        } else {
            vec![y]
        },
        vec![rho],
        None,
        None,
        0,
        0,
    );
    bincode::deserialize(&bincode::serialize(&tuple).unwrap()).unwrap()
}

#[cfg(feature = "serde")]
#[test]
fn lbfgs_serde_gamma_fallback_and_unchecked_nan_direction_keep_original_entry_errors() {
    let (mut net, _) = l3_warmed_fixture(1, "H0");
    let model = l3_bits(&LBFGS::flatten_params(&net));
    let opt = l3_forged_optimizer("fallback");
    let before = bincode::serialize(&opt).unwrap();
    let probe = vec![2.0; 14];
    let direction = opt.two_loop_recursion(&probe);
    assert_eq!(direction[0].to_bits(), (-2.0f32.powi(21)).to_bits());
    assert!(direction[1..]
        .iter()
        .all(|v| v.to_bits() == (-2.0f32).to_bits()));
    assert_eq!(bincode::serialize(&opt).unwrap(), before);
    for kind in ["nan", "length", "cardinality", "rho"] {
        let mut opt = l3_forged_optimizer(kind);
        let before = bincode::serialize(&opt).unwrap();
        if kind == "nan" {
            let mut g = l3_mixed(14);
            g[0] = f32::from_bits(0x7fc00022);
            let direction = opt.two_loop_recursion(&g);
            assert_eq!(direction.len(), 14);
            assert!(direction.iter().all(|v| v.is_nan()));
            assert_eq!(bincode::serialize(&opt).unwrap(), before);
        }
        let mut calls = 0;
        let error = opt
            .step_lbfgs(&mut net, |_| {
                calls += 1;
                unreachable!("entry rejected before callback")
            })
            .unwrap_err();
        match (kind, error) {
            ("nan", ArkanError::Optimizer(message)) => {
                assert_eq!(message, "LBFGS history must be finite")
            }
            ("rho", ArkanError::Optimizer(message)) => {
                assert_eq!(message, "LBFGS curvature state must be finite and positive")
            }
            (
                "length",
                ArkanError::TensorShapeMismatch {
                    param_shape,
                    grad_shape,
                },
            ) => {
                assert_eq!(param_shape, vec![14]);
                assert_eq!(grad_shape, vec![13]);
            }
            (
                "cardinality",
                ArkanError::TensorShapeMismatch {
                    param_shape,
                    grad_shape,
                },
            ) => {
                assert_eq!(param_shape, vec![1]);
                assert_eq!(grad_shape, vec![0]);
            }
            (_, error) => panic!("wrong entry error: {error:?}"),
        }
        assert_eq!(calls, 0);
        assert_eq!(opt.num_evals(), 0);
        assert_eq!(bincode::serialize(&opt).unwrap(), before);
        assert_eq!(l3_bits(&LBFGS::flatten_params(&net)), model);
    }
}

#[cfg(feature = "serde")]
#[test]
fn lbfgs_entry_config_and_topology_errors_precede_forged_history_without_state_changes() {
    for clear_topology in [false, true] {
        let (mut net, opt) = l3_warmed_fixture(1, "H1");
        let mut tuple = l3_literal_state(&net, &opt, "H1");
        tuple.0.history_size = 0;
        tuple.1[0].resize(13, 0.0);
        let mut opt: LBFGS = bincode::deserialize(&bincode::serialize(&tuple).unwrap()).unwrap();
        if clear_topology {
            net.layers.clear();
        }
        let before = bincode::serialize(&opt).unwrap();
        let model = l3_bits(&LBFGS::flatten_params(&net));
        let mut calls = 0;
        let error = opt
            .step_lbfgs(&mut net, |_| {
                calls += 1;
                unreachable!("entry rejected before callback")
            })
            .unwrap_err();
        match error {
            ArkanError::Cpu(message) if clear_topology => {
                assert_eq!(message, "Network topology no longer matches its layout")
            }
            ArkanError::Optimizer(message) if !clear_topology => assert_eq!(
                message,
                "LBFGS requires positive lr, max_iter and history_size, and max_eval >= 2"
            ),
            error => panic!("wrong precedence: {error:?}"),
        }
        assert_eq!(calls, 0);
        assert_eq!(opt.num_evals(), 2);
        assert_eq!(bincode::serialize(&opt).unwrap(), before);
        assert_eq!(l3_bits(&LBFGS::flatten_params(&net)), model);
    }
}

#[test]
fn lbfgs_wolfe_zoom_returns_the_current_callback_point() {
    for policy in [
        LineSearchMethod::StrongWolfe,
        LineSearchMethod::Backtracking,
    ] {
        let (mut net, mut opt) = l3_warmed_fixture(1, "H1");
        opt.config.line_search_fn = policy;
        opt.config.lr = 4.0;
        opt.config.max_iter = 1;
        opt.config.max_eval = Some(4);
        opt.config.tolerance_grad = 0.0;
        opt.config.tolerance_change = 0.0;
        let mut points = Vec::new();
        let loss = opt
            .step_lbfgs(&mut net, |net| {
                let p = LBFGS::flatten_params(net);
                let x = p[0];
                points.push(p);
                let mut g = vec![0.0; 14];
                g[0] = 2.0 * x;
                Ok((f64::from(x).powi(2), g))
            })
            .unwrap();
        assert_eq!(
            points.iter().map(|p| p[0].to_bits()).collect::<Vec<_>>(),
            [0.5f32, -1.5, -0.5, 0.0].map(f32::to_bits)
        );
        assert!(points
            .iter()
            .all(|p| p[1..].iter().all(|v| v.to_bits() == 0)));
        assert_eq!(
            l3_bits(&LBFGS::flatten_params(&net)),
            l3_bits(points.last().unwrap())
        );
        assert_eq!(loss.to_bits(), 0.0f64.to_bits());
        assert_eq!(opt.num_evals(), 6);
    }
}

#[test]
fn lbfgs_accepted_point_commits_previous_state_when_curvature_is_rejected() {
    let (mut net, mut opt) = l3_warmed_fixture(1, "H0");
    let mut points = Vec::new();
    let loss = opt
        .step_lbfgs(&mut net, |model| {
            let x = model.layers[0].weights[0];
            points.push(x.to_bits());
            let mut gradient = vec![0.0; 14];
            gradient[0] = 2.0;
            Ok((2.0 * f64::from(x), gradient))
        })
        .unwrap();
    assert_eq!(points, [1.0f32, 0.5].map(f32::to_bits));
    assert_eq!(loss.to_bits(), 1.0f64.to_bits());
    assert_eq!(opt.num_evals(), 2);
    let parameters = LBFGS::flatten_params(&net);
    assert_eq!(parameters[0].to_bits(), 0.5f32.to_bits());
    assert!(parameters[1..].iter().all(|v| v.to_bits() == 0));
    assert_eq!(
        l3_bits(&opt.two_loop_recursion(&[1.0; 14])),
        vec![(-1.0f32).to_bits(); 14]
    );
    #[cfg(feature = "serde")]
    {
        let mut gradient = vec![0.0; 14];
        gradient[0] = 2.0;
        let expected: L3State = (
            opt.config,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Some(parameters),
            Some(gradient),
            0,
            2,
        );
        assert_eq!(
            bincode::serialize(&opt).unwrap(),
            bincode::serialize(&expected).unwrap()
        );
    }
}

#[test]
fn lbfgs_shape_error_and_numerical_skip_after_acceptance_restore_warmed_state() {
    for malformed in [true, false] {
        let (mut net, mut opt) = l3_warmed_fixture(1, "H1");
        opt.config.max_iter = 2;
        opt.config.max_eval = Some(3);
        opt.config.safety.skip_step_on_nan = !malformed;
        let model = l3_bits(&LBFGS::flatten_params(&net));
        let direction = l3_bits(&opt.two_loop_recursion(&[2.0; 14]));
        let evaluations = opt.num_evals();
        #[cfg(feature = "serde")]
        let before = bincode::serialize(&opt).unwrap();
        let mut points = Vec::new();
        let result = opt.step_lbfgs(&mut net, |network| {
            let x = network.layers[0].weights[0];
            points.push(x.to_bits());
            if malformed && points.len() == 3 {
                return Ok((f64::from(x).powi(2), Vec::new()));
            }
            let mut gradient = vec![0.0; 14];
            gradient[0] = if points.len() == 3 { f32::NAN } else { 2.0 * x };
            Ok((f64::from(x).powi(2), gradient))
        });
        assert_eq!(points, [0.5f32, 0.375, 0.28125].map(f32::to_bits));
        if malformed {
            assert!(
                matches!(result, Err(ArkanError::TensorShapeMismatch { param_shape, grad_shape })
                if param_shape == [14] && grad_shape == [0])
            );
        } else {
            assert_eq!(result.unwrap().to_bits(), 0.25f64.to_bits());
        }
        assert_eq!(opt.num_evals(), evaluations + 3);
        assert_eq!(l3_bits(&LBFGS::flatten_params(&net)), model);
        assert_eq!(l3_bits(&opt.two_loop_recursion(&[2.0; 14])), direction);
        #[cfg(feature = "serde")]
        {
            let mut expected: L3State = bincode::deserialize(&before).unwrap();
            expected.7 += 3;
            assert_eq!(
                bincode::serialize(&opt).unwrap(),
                bincode::serialize(&expected).unwrap()
            );
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn lbfgs_stationary_step_saturates_evaluation_counter() {
    let (mut net, mut opt) = l3_warmed_fixture(1, "H1");
    let model = l3_bits(&LBFGS::flatten_params(&net));
    let mut state: L3State = bincode::deserialize(&bincode::serialize(&opt).unwrap()).unwrap();
    state.7 = usize::MAX;
    opt = bincode::deserialize(&bincode::serialize(&state).unwrap()).unwrap();
    let before = bincode::serialize(&opt).unwrap();
    assert_eq!(opt.num_evals(), usize::MAX);
    assert_eq!(before, bincode::serialize(&state).unwrap());
    let mut calls = 0;
    let loss = opt
        .step_lbfgs(&mut net, |_| {
            calls += 1;
            Ok((7.0, vec![0.0; model.len()]))
        })
        .unwrap();
    assert_eq!(loss.to_bits(), 7.0f64.to_bits());
    assert_eq!(calls, 1);
    assert_eq!(opt.num_evals(), usize::MAX);
    assert_eq!(l3_bits(&LBFGS::flatten_params(&net)), model);
    assert_eq!(bincode::serialize(&opt).unwrap(), before);
}

// Snapshots use public numerical state and capacities, rather than private Cow variants.
type AmpSnapshot = Vec<(Vec<u32>, usize)>;
type AmpOptimizerSnapshot = (AmpSnapshot, (Vec<u32>, Option<u64>));

fn amp_gradient_snapshot(weights: &[Vec<f32>], biases: &[Vec<f32>]) -> AmpSnapshot {
    weights
        .iter()
        .chain(biases)
        .map(|tensor| (l3_bits(tensor), tensor.capacity()))
        .collect()
}

fn amp_model_snapshot(model: &KanNetwork) -> AmpSnapshot {
    model
        .layers
        .iter()
        .flat_map(|layer| [&layer.weights, &layer.bias, &layer.mean, &layer.std])
        .map(|tensor| (l3_bits(tensor), tensor.capacity()))
        .collect()
}

fn amp_adam_snapshot(optimizer: &Adam) -> AmpOptimizerSnapshot {
    let mut result = vec![(vec![], optimizer.layer_states.capacity())];
    for state in &optimizer.layer_states {
        for tensor in [&state.weights, &state.bias] {
            result.push((l3_bits(tensor.m.as_slice()), tensor.m.capacity()));
            result.push((l3_bits(tensor.v.as_slice()), tensor.v.capacity()));
            result.push((vec![], tensor.t));
        }
    }
    let config = optimizer.config;
    (
        result,
        (
            vec![
                config.lr.to_bits(),
                config.beta1.to_bits(),
                config.beta2.to_bits(),
                config.epsilon.to_bits(),
                config.weight_decay.to_bits(),
                u32::from(config.safety.fail_on_nan),
                u32::from(config.safety.skip_step_on_nan),
                u32::from(config.safety.unscale_before_step),
            ],
            config.safety.grad_scaling_factor.map(f64::to_bits),
        ),
    )
}

fn amp_sgd_snapshot(optimizer: &SGD) -> AmpOptimizerSnapshot {
    let mut result = vec![(vec![], optimizer.velocities.capacity())];
    for (weights, biases) in &optimizer.velocities {
        result.push((l3_bits(weights.as_slice()), weights.capacity()));
        result.push((l3_bits(biases.as_slice()), biases.capacity()));
    }
    let config = optimizer.config;
    (
        result,
        (
            vec![
                config.lr.to_bits(),
                config.momentum.to_bits(),
                config.weight_decay.to_bits(),
                u32::from(config.nesterov),
                u32::from(config.safety.fail_on_nan),
                u32::from(config.safety.skip_step_on_nan),
                u32::from(config.safety.unscale_before_step),
            ],
            config.safety.grad_scaling_factor.map(f64::to_bits),
        ),
    )
}

fn amp_check_finite_identity<O: Optimizer>(
    make: impl Fn(&KanNetwork, SafetyConfig) -> O,
    snapshot: impl Fn(&O) -> AmpOptimizerSnapshot,
) {
    let values = [
        0.125,
        -0.5,
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        -f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        2.0,
        -4.0,
    ];
    for hidden in [false, true] {
        for (fail_on_nan, skip_step_on_nan) in
            [(true, false), (false, true), (true, true), (false, false)]
        {
            for clipping in [None, Some(100.0), Some(0.25)] {
                let mut plain = network(hidden);
                let mut identity = plain.clone();
                let (mut weights, mut biases) = gradients(&plain);
                let mut element = 0;
                for tensor in weights.iter_mut().chain(&mut biases) {
                    tensor.reserve(8);
                    for value in tensor.iter_mut() {
                        *value = values[element % values.len()];
                        element += 1;
                    }
                }
                let norm = weights
                    .iter()
                    .chain(&biases)
                    .flatten()
                    .map(|&value| f64::from(value).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(norm > 0.25 && norm < 100.0, "clipping fixture norm={norm}");
                let inputs = amp_gradient_snapshot(&weights, &biases);
                let outer_capacities = (weights.capacity(), biases.capacity());
                let safety = SafetyConfig {
                    fail_on_nan,
                    skip_step_on_nan,
                    grad_scaling_factor: None,
                    unscale_before_step: false,
                };
                let mut plain_optimizer = make(&plain, safety);
                let mut identity_optimizer = make(
                    &identity,
                    SafetyConfig {
                        grad_scaling_factor: Some(1.0),
                        ..safety
                    },
                );
                let plain_config = snapshot(&plain_optimizer).1;
                let identity_config = snapshot(&identity_optimizer).1;
                assert_eq!(plain_config.1, None);
                assert_eq!(identity_config.1, Some(1.0f64.to_bits()));
                for _ in 0..3 {
                    plain_optimizer
                        .step(&mut plain, &weights, &biases, clipping)
                        .unwrap();
                    assert_eq!(amp_gradient_snapshot(&weights, &biases), inputs);
                    identity_optimizer
                        .step(&mut identity, &weights, &biases, clipping)
                        .unwrap();
                    assert_eq!(amp_model_snapshot(&identity), amp_model_snapshot(&plain));
                    let plain_state = snapshot(&plain_optimizer);
                    let mut identity_state = snapshot(&identity_optimizer);
                    assert_eq!(plain_state.1, plain_config);
                    assert_eq!(identity_state.1, identity_config);
                    identity_state.1 .1 = None;
                    assert_eq!(identity_state, plain_state);
                    assert_eq!(
                        identity_optimizer.get_state_version(),
                        plain_optimizer.get_state_version()
                    );
                    assert_eq!(amp_gradient_snapshot(&weights, &biases), inputs);
                    assert_eq!((weights.capacity(), biases.capacity()), outer_capacities);
                }
            }
        }
    }
}

#[test]
fn amp_identity_preserves_finite_parameter_state_and_input_bits() {
    amp_check_finite_identity(
        |model, safety| {
            Adam::new(
                model,
                AdamConfig {
                    lr: 0.125,
                    weight_decay: 0.05,
                    safety,
                    ..AdamConfig::default()
                },
            )
        },
        amp_adam_snapshot,
    );
    amp_check_finite_identity(
        |model, safety| {
            SGD::new(
                model,
                SGDConfig {
                    lr: 0.125,
                    momentum: 0.5,
                    weight_decay: 0.05,
                    nesterov: true,
                    safety,
                },
            )
        },
        amp_sgd_snapshot,
    );
}

fn amp_check_late_rejection<O: Optimizer>(
    make: impl Fn(&KanNetwork, SafetyConfig) -> O,
    snapshot: impl Fn(&O) -> AmpOptimizerSnapshot,
    update_context: &str,
) {
    for (fail_on_nan, skip_step_on_nan) in [(true, false), (false, true), (true, true)] {
        for bad_gradient in [
            f32::from_bits(0x7fc0_1234),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
        ] {
            let mut model = network(true);
            let (weights, mut biases) = gradients(&model);
            biases[0][0] = 0.125;
            let safety = SafetyConfig {
                fail_on_nan,
                skip_step_on_nan,
                grad_scaling_factor: Some(1.0),
                unscale_before_step: true,
            };
            let mut optimizer = make(&model, safety);
            optimizer.step(&mut model, &weights, &biases, None).unwrap();
            biases.last_mut().unwrap()[0] = bad_gradient;
            let parameters = amp_model_snapshot(&model);
            let state = snapshot(&optimizer);
            let version = optimizer.get_state_version();
            let inputs = amp_gradient_snapshot(&weights, &biases);
            let outer_capacities = (weights.capacity(), biases.capacity());
            let result = optimizer.step(&mut model, &weights, &biases, None);
            if skip_step_on_nan {
                assert!(
                    result.is_ok(),
                    "skip must take precedence over strict failure"
                );
            } else {
                match result {
                    Err(ArkanError::NaNEncountered {
                        param_index,
                        context,
                    }) => {
                        assert_eq!(param_index, usize::from(bad_gradient.is_finite()));
                        assert_eq!(
                            context,
                            if bad_gradient.is_finite() {
                                update_context
                            } else {
                                "gradient"
                            }
                        );
                    }
                    other => panic!("expected numerical rejection, got {other:?}"),
                }
            }
            assert_eq!(amp_model_snapshot(&model), parameters);
            assert_eq!(snapshot(&optimizer), state);
            assert_eq!(optimizer.get_state_version(), version);
            assert_eq!(amp_gradient_snapshot(&weights, &biases), inputs);
            assert_eq!((weights.capacity(), biases.capacity()), outer_capacities);
        }
    }
}

#[test]
fn amp_identity_late_rejection_preserves_warmed_parameters_state_and_inputs() {
    amp_check_late_rejection(
        |model, safety| Adam::new(model, AdamConfig::default().with_safety(safety)),
        amp_adam_snapshot,
        "Adam update or state",
    );
    amp_check_late_rejection(
        |model, safety| SGD::new(model, SGDConfig::with_lr(2.0).with_safety(safety)),
        amp_sgd_snapshot,
        "SGD update or state",
    );
}
