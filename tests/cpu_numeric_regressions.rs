use arkan::config::{ConfigError, EPSILON};
use arkan::{KanConfig, KanNetwork};

fn config(order: usize, hidden: Vec<usize>, threshold: usize) -> KanConfig {
    KanConfig::builder()
        .input_dim(1)
        .output_dim(1)
        .hidden_dims(hidden)
        .grid_size(5)
        .spline_order(order)
        .grid_range(-1.0, 1.0)
        .multithreading_threshold(threshold)
        .seed(1)
        .build()
        .unwrap()
}

fn normalization_gradient(std: f32, coefficient_step: f32, threshold: usize, batch: usize) {
    let mut model = KanNetwork::new(config(1, vec![1], threshold));
    model.layers[0].weights.fill(0.0);
    model.layers[0].bias[0] = 0.0;
    model.layers[1].std[0] = std;
    for (k, weight) in model.layers[1].weights.iter_mut().enumerate() {
        *weight = k as f32 * coefficient_step;
    }
    let mut workspace = model.create_workspace(batch);
    let mut output = vec![0.0; batch];
    let pass = model
        .try_forward_for_backward(&vec![0.0; batch], &mut output, &mut workspace)
        .unwrap();
    let gradients = pass.backward(&vec![1.0; batch]).unwrap();
    // Linear coefficients increase by coefficient_step every interval of width 0.4.
    let expected = (2.5_f64 * coefficient_step as f64 / std as f64 * batch as f64) as f32;
    let actual = gradients.biases[0][0];
    assert!(
        actual.is_finite() && (actual - expected).abs() <= expected.abs() * 1e-6,
        "std={std:e}: upstream derivative {actual:e}, expected {expected:e}"
    );
}

#[test]
fn backward_uses_actual_positive_stored_std() {
    normalization_gradient(1e-7, 1.0, usize::MAX, 2);

    let mut model = KanNetwork::new(config(1, vec![1], usize::MAX));
    model.layers[0].weights.fill(0.0);
    model.layers[0].bias[0] = 5e-8;
    model.layers[1].std[0] = 1e-7;
    for (k, weight) in model.layers[1].weights.iter_mut().enumerate() {
        *weight = k as f32;
    }
    let mut workspace = model.create_workspace(1);
    let mut output = [0.0];
    let pass = model
        .try_forward_for_backward(&[0.0], &mut output, &mut workspace)
        .unwrap();
    let actual = pass.backward(&[1.0]).unwrap().biases[0][0];
    assert!((actual - 2.5e7).abs() < 10.0);
    let step = 1e-10;
    model.layers[0].bias[0] = 5e-8 + step;
    let mut plus = [0.0];
    model
        .try_forward_single(&[0.0], &mut plus, &mut workspace)
        .unwrap();
    model.layers[0].bias[0] = 5e-8 - step;
    let mut minus = [0.0];
    model
        .try_forward_single(&[0.0], &mut minus, &mut workspace)
        .unwrap();
    let finite_difference = (plus[0] - minus[0]) / (2.0 * step);
    assert!((actual - finite_difference).abs() / actual < 1e-3);
}

#[test]
fn overflowing_std_reciprocal_keeps_representable_derivative() {
    let std = f32::from_bits(1);
    normalization_gradient(std, std, usize::MAX, 2);
}

fn cancelling_subnormal_derivative(threshold: usize, batch: usize) {
    let mut model = KanNetwork::new(config(1, vec![1], threshold));
    model.layers[0].weights.fill(0.0);
    model.layers[0].bias[0] = 0.0;
    model.layers[1].std[0] = f32::from_bits(1);
    model.layers[1].weights.fill(f32::MAX);
    let mut workspace = model.create_workspace(batch);
    let mut output = vec![0.0; batch];
    let pass = model
        .try_forward_for_backward(&vec![0.0; batch], &mut output, &mut workspace)
        .unwrap();
    let gradients = pass.backward(&vec![1.0; batch]).unwrap();
    // Equal linear coefficients give a constant output, even at enormous scale.
    assert!(output
        .iter()
        .all(|&y| y.is_finite() && (y / f32::MAX - 1.0).abs() < 1e-6));
    assert_eq!(gradients.biases[0], [0.0]);
    assert!(gradients.weights[0].iter().all(|&g| g == 0.0));
    assert!(gradients.weights[1].iter().all(|g| g.is_finite()));
}

#[test]
fn subnormal_std_preserves_cancelling_coefficient_derivatives() {
    cancelling_subnormal_derivative(usize::MAX, 2);
}

fn saturated_gradient(threshold: usize, batch: usize) {
    let mut model = KanNetwork::new(config(1, vec![1], threshold));
    model.layers[0].weights.fill(0.0);
    model.layers[0].bias[0] = 2.0;
    model.layers[1].weights.fill(0.0);
    model.layers[1].weights[5] = f32::MAX;
    let mut workspace = model.create_workspace(batch);
    let mut output = vec![0.0; batch];
    let pass = model
        .try_forward_for_backward(&vec![0.0; batch], &mut output, &mut workspace)
        .unwrap();
    let gradients = pass.backward(&vec![1.0; batch]).unwrap();
    assert_eq!(output, vec![f32::MAX; batch]);
    assert_eq!(gradients.biases[0], [0.0]);
    assert!(gradients.weights[0].iter().all(|&g| g == 0.0));
    assert_eq!(gradients.biases[1], [batch as f32]);
    assert_eq!(gradients.weights[1][5], batch as f32);
    assert!(gradients.weights[1][..5].iter().all(|&g| g == 0.0));
}

#[test]
fn clamped_extreme_coefficient_has_zero_upstream_gradient() {
    saturated_gradient(usize::MAX, 2);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_backward_uses_actual_std_and_safe_subnormal_scaling() {
    normalization_gradient(1e-7, 1.0, 1, 64);
    let std = f32::from_bits(1);
    normalization_gradient(std, std, 1, 64);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_clamped_extreme_coefficient_has_zero_upstream_gradient() {
    saturated_gradient(1, 64);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_subnormal_std_preserves_cancelling_coefficient_derivatives() {
    cancelling_subnormal_derivative(1, 64);
}

#[test]
fn subnormal_grid_preserves_finite_partition_of_unity() {
    let mut cfg = config(3, vec![], usize::MAX);
    cfg.grid_range = (0.0, 1e-40);
    cfg.validate().unwrap();
    let mut model = KanNetwork::try_new(cfg).unwrap();
    model.layers[0].weights.fill(1.0);
    model.layers[0].bias.fill(0.0);
    let input = [0.0, 5e-41, 1e-40];
    let mut workspace = model.create_workspace(input.len());
    for x in input {
        let mut output = [0.0];
        model
            .try_forward_single(&[x], &mut output, &mut workspace)
            .unwrap();
        assert!((output[0] - 1.0).abs() < 1e-5, "x={x:e}: {output:?}");
    }
    let mut output = [0.0; 3];
    let pass = model
        .try_forward_for_backward(&input, &mut output, &mut workspace)
        .unwrap();
    let gradients = pass.backward(&[1.0; 3]).unwrap();
    assert!(output.iter().all(|&y| (y - 1.0).abs() < 1e-5));
    assert!(gradients.weights[0].iter().all(|g| g.is_finite()));
    assert!((gradients.weights[0].iter().sum::<f32>() - 3.0).abs() < 1e-5);
    assert_eq!(gradients.biases[0], [3.0]);
}

fn tiny_grid_hidden_gradient(order: usize, constant: bool, threshold: usize, batch: usize) {
    for std in [1.0, 2.0] {
        // Include both endpoints and saturation: the latter must retain parameter gradients.
        for z in [-1e-40, 0.0, 5e-41, 1e-40, 2e-40] {
            let mut cfg = config(order, vec![1], threshold);
            cfg.grid_range = (0.0, 1e-40);
            let mut model = KanNetwork::try_new(cfg).unwrap();
            model.layers[0].weights.fill(0.0);
            model.layers[0].bias[0] = z * std;
            model.layers[1].std[0] = std;
            for (k, weight) in model.layers[1].weights.iter_mut().enumerate() {
                *weight = if constant { 1.0 } else { k as f32 * 1e-41 };
            }
            model.layers[1].bias.fill(0.0);
            let mut workspace = model.create_workspace(batch);
            let mut output = vec![0.0; batch];
            let pass = model
                .try_forward_for_backward(&vec![0.0; batch], &mut output, &mut workspace)
                .unwrap();
            let gradients = pass.backward(&vec![1.0; batch]).unwrap();
            assert!(output.iter().all(|y| y.is_finite()));
            if constant {
                assert!(output.iter().all(|&y| (y - 1.0).abs() < 1e-6));
            }
            // Uniform coefficient increments give slope 1e-41 / 2e-41 for any order.
            let expected = if constant || !(0.0..=1e-40).contains(&z) {
                0.0
            } else {
                0.5 * batch as f32 / std
            };
            let actual = gradients.biases[0][0];
            if expected == 0.0 {
                assert_eq!(actual, 0.0, "order={order}, constant={constant}, z={z:e}");
                assert!(gradients.weights[0].iter().all(|&g| g == 0.0));
            } else {
                assert!(
                    actual.is_finite() && (actual - expected).abs() < expected * 2e-4,
                    "order={order}, std={std}, z={z:e}: got {actual}, expected {expected}"
                );
                assert!(gradients.weights[0].iter().all(|g| g.is_finite()));
            }
            assert_eq!(gradients.biases[1], [batch as f32]);
            assert!(gradients.weights[1].iter().all(|g| g.is_finite()));
            assert!((gradients.weights[1].iter().sum::<f32>() - batch as f32).abs() < 1e-4);
        }
    }
}

#[test]
fn tiny_grid_linear_hidden_slope_is_finite() {
    tiny_grid_hidden_gradient(1, false, usize::MAX, 1);
}

#[test]
fn tiny_grid_cubic_hidden_slope_is_finite() {
    tiny_grid_hidden_gradient(3, false, usize::MAX, 1);
}

#[test]
fn tiny_grid_linear_hidden_constant_has_zero_derivative() {
    tiny_grid_hidden_gradient(1, true, usize::MAX, 1);
}

#[test]
fn tiny_grid_cubic_hidden_constant_has_zero_derivative() {
    tiny_grid_hidden_gradient(3, true, usize::MAX, 1);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_tiny_grid_linear_hidden_slope_is_finite() {
    tiny_grid_hidden_gradient(1, false, 1, 64);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_tiny_grid_cubic_hidden_slope_is_finite() {
    tiny_grid_hidden_gradient(3, false, 1, 64);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_tiny_grid_linear_hidden_constant_has_zero_derivative() {
    tiny_grid_hidden_gradient(1, true, 1, 64);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_tiny_grid_cubic_hidden_constant_has_zero_derivative() {
    tiny_grid_hidden_gradient(3, true, 1, 64);
}

#[test]
fn builder_rejects_nonfinite_std_before_clamping() {
    let results: Vec<_> = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY]
        .into_iter()
        .map(|bad| {
            KanConfig::builder()
                .input_dim(1)
                .output_dim(1)
                .normalization(vec![0.0], vec![bad])
                .build()
        })
        .collect();
    assert!(
        results.iter().all(|result| matches!(
            result,
            Err(ConfigError::NonFiniteNormalization("input_std"))
        )),
        "invalid supplied std accepted: {results:?}"
    );
}

#[test]
fn builder_preserves_finite_std_clamping_and_validation_precedence() {
    for std in [0.0, -1.0, 1e-7] {
        let cfg = KanConfig::builder()
            .input_dim(1)
            .output_dim(1)
            .normalization(vec![0.0], vec![std])
            .build()
            .unwrap();
        assert_eq!(cfg.input_std, [EPSILON]);
    }
    assert!(matches!(
        KanConfig::builder()
            .input_dim(1)
            .output_dim(1)
            .grid_size(0)
            .normalization(vec![0.0], vec![f32::NAN])
            .build(),
        Err(ConfigError::InvalidGridSize(0))
    ));
    assert!(matches!(
        KanConfig::builder()
            .input_dim(1)
            .output_dim(1)
            .normalization(vec![], vec![f32::NAN])
            .build(),
        Err(ConfigError::MismatchedNormalization("input_mean"))
    ));
}
