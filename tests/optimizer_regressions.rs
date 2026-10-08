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
    }
}
