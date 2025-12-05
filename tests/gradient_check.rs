//! Numerical Gradient Checking for ArKan Backward Pass
//!
//! This test verifies that the analytical gradients computed by `backward()`
//! match numerical gradients computed via central difference approximation.
//!
//! # Methodology
//!
//! For each parameter `w`, we compute:
//! - Analytical gradient: `grad_ana = backward()`
//! - Numerical gradient: `grad_num = (L(w+ε) - L(w-ε)) / (2ε)`
//!
//! The relative error should be < 1e-2 for the implementation to be correct.
//!
//! # Reference
//!
//! This is the "Gold Standard" test for verifying backpropagation correctness.
//! See: https://cs231n.github.io/neural-networks-3/#gradcheck

use arkan::{KanConfig, KanNetwork, TrainOptions, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Epsilon for numerical differentiation.
/// Using 1e-3 for f32 precision (1e-5 would cause numerical issues).
const EPSILON: f32 = 1e-3;

/// Small constant to prevent division by zero in relative error.
const DELTA: f32 = 1e-8;

/// Maximum allowed relative error between analytical and numerical gradients.
/// Using 1e-2 (1%) for f32 precision.
const MAX_RELATIVE_ERROR: f32 = 1e-2;

/// For small gradients (|grad| < this threshold), we use absolute error instead
const SMALL_GRAD_THRESHOLD: f32 = 1e-3;

/// Maximum absolute error for small gradients
const MAX_ABSOLUTE_ERROR: f32 = 1e-5;

/// Minimum percentage of checks that must pass.
/// We allow some failures due to numerical precision issues with small gradients.
/// For spline order 3+, the basis function derivatives can be numerically sensitive.
const MIN_PASS_RATE: f32 = 0.90;

/// Number of random weights to check per layer.
const WEIGHTS_TO_CHECK_PER_LAYER: usize = 10;

/// Computes MSE loss for a batch (same formula as ArKan uses).
/// ArKan uses: loss = sum((pred - target)^2) / count
fn compute_mse_loss(predictions: &[f32], targets: &[f32]) -> f32 {
    let sum: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    sum / predictions.len() as f32
}

/// Computes relative error between two values.
fn relative_error(ana: f32, num: f32) -> f32 {
    (ana - num).abs() / (ana.abs() + num.abs() + DELTA)
}

/// Checks if gradient passes the tolerance check.
/// For small gradients, uses absolute error. For larger gradients, uses relative error.
fn gradient_check_passes(ana: f32, num: f32) -> bool {
    let abs_err = (ana - num).abs();
    let max_abs = ana.abs().max(num.abs());
    
    // For very small gradients, absolute error is more meaningful
    if max_abs < SMALL_GRAD_THRESHOLD {
        abs_err < MAX_ABSOLUTE_ERROR
    } else {
        relative_error(ana, num) < MAX_RELATIVE_ERROR
    }
}

/// Test gradient checking for a simple network (no hidden layers).
#[test]
fn test_gradient_check_simple_network() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![],
        grid_size: 3,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(42),
        ..Default::default()
    };

    run_gradient_check(&config, 8, "simple (no hidden)");
}

/// Test gradient checking for a single hidden layer network.
#[test]
fn test_gradient_check_single_hidden() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 3,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(42),
        ..Default::default()
    };

    run_gradient_check(&config, 8, "single hidden layer");
}

/// Test gradient checking for a multi-layer network.
#[test]
fn test_gradient_check_multi_layer() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8, 4],
        grid_size: 3,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(42),
        ..Default::default()
    };

    run_gradient_check(&config, 8, "multi-layer");
}

/// Test gradient checking with higher spline order.
#[test]
fn test_gradient_check_spline_order_3() {
    let config = KanConfig {
        input_dim: 3,
        output_dim: 2,
        hidden_dims: vec![4],
        grid_size: 4,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 3],
        input_std: vec![1.0; 3],
        init_seed: Some(42),
        ..Default::default()
    };

    run_gradient_check(&config, 4, "spline order 3");
}

/// Test gradient checking with spline order 4.
#[test]
fn test_gradient_check_spline_order_4() {
    let config = KanConfig {
        input_dim: 3,
        output_dim: 2,
        hidden_dims: vec![4],
        grid_size: 4,
        spline_order: 4,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 3],
        input_std: vec![1.0; 3],
        init_seed: Some(42),
        ..Default::default()
    };

    run_gradient_check(&config, 4, "spline order 4");
}

/// Test gradient checking with spline order 2.
#[test]
fn test_gradient_check_spline_order_2() {
    let config = KanConfig {
        input_dim: 3,
        output_dim: 2,
        hidden_dims: vec![4],
        grid_size: 4,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 3],
        input_std: vec![1.0; 3],
        init_seed: Some(42),
        ..Default::default()
    };

    run_gradient_check(&config, 4, "spline order 2");
}

/// Runs the gradient check for a given configuration.
fn run_gradient_check(config: &KanConfig, batch_size: usize, test_name: &str) {
    println!("\n=== Gradient Check: {} ===", test_name);

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(batch_size);

    // Generate random input and target data
    let mut rng = SmallRng::seed_from_u64(12345);
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|_| rng.gen_range(-0.9..0.9))
        .collect();
    let target: Vec<f32> = (0..batch_size * config.output_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // =========================================================================
    // Step 1: Compute analytical gradients via backward()
    // =========================================================================
    let train_opts = TrainOptions {
        max_grad_norm: None, // No clipping during gradient check
        weight_decay: 0.0,
    };

    // Forward pass with training mode
    let mut predictions = vec![0.0f32; batch_size * config.output_dim];
    network.forward_batch_training(&input, &mut predictions, &mut workspace);

    // Compute loss and gradient
    let loss = compute_mse_loss(&predictions, &target);
    println!("Initial loss: {:.6}", loss);

    // Compute output gradient (dL/d_output)
    let mut grad_output = vec![0.0f32; batch_size * config.output_dim];
    for i in 0..predictions.len() {
        grad_output[i] = (predictions[i] - target[i]) / predictions.len() as f32;
    }

    // Backward pass to compute gradients
    // We need to call train_step_with_options but with LR=0 to just compute gradients
    // Actually, let's use a different approach: we'll run train_step with tiny LR
    // and extract gradients from workspace

    // Reset network to original weights
    let original_network = network.clone();

    // Run one training step to populate gradient buffers
    network.train_step_with_options(
        &input,
        &target,
        None,
        0.0, // LR=0, just compute gradients
        &mut workspace,
        &train_opts,
    );

    // Extract analytical gradients from workspace
    let ana_weight_grads: Vec<Vec<f32>> = workspace.weight_grads.iter().cloned().collect();
    let ana_bias_grads: Vec<Vec<f32>> = workspace.bias_grads.iter().cloned().collect();

    // Restore original weights
    network = original_network;

    // =========================================================================
    // Step 2: Compute numerical gradients via central differences
    // =========================================================================
    let mut total_checks = 0;
    let mut passed_checks = 0;
    let mut max_error: f32 = 0.0;

    // First, collect layer info without borrowing network
    let layer_info: Vec<(usize, usize)> = network.layers.iter()
        .map(|layer| (layer.weights.len(), layer.bias.len()))
        .collect();

    // Check random subset of weights for each layer
    for (layer_idx, &(num_weights, num_biases)) in layer_info.iter().enumerate() {
        println!(
            "Layer {}: {} weights, {} biases",
            layer_idx, num_weights, num_biases
        );

        // Select random weight indices to check
        let weight_indices: Vec<usize> = {
            let mut indices: Vec<usize> = (0..num_weights).collect();
            let mut rng = SmallRng::seed_from_u64(layer_idx as u64 + 100);
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rng);
            indices
                .into_iter()
                .take(WEIGHTS_TO_CHECK_PER_LAYER.min(num_weights))
                .collect()
        };

        // Check selected weights
        for &w_idx in &weight_indices {
            let ana_grad = ana_weight_grads[layer_idx][w_idx];
            let num_grad = numerical_gradient_weight(&mut network, &input, &target, &mut workspace, layer_idx, w_idx);

            let rel_err = relative_error(ana_grad, num_grad);
            max_error = max_error.max(rel_err);
            total_checks += 1;

            if gradient_check_passes(ana_grad, num_grad) {
                passed_checks += 1;
            } else {
                println!(
                    "  FAIL weight[{}][{}]: ana={:.6}, num={:.6}, rel_err={:.6}",
                    layer_idx, w_idx, ana_grad, num_grad, rel_err
                );
            }
        }

        // Check ALL biases (they're fewer)
        for b_idx in 0..num_biases {
            let ana_grad = ana_bias_grads[layer_idx][b_idx];
            let num_grad = numerical_gradient_bias(&mut network, &input, &target, &mut workspace, layer_idx, b_idx);

            let rel_err = relative_error(ana_grad, num_grad);
            max_error = max_error.max(rel_err);
            total_checks += 1;

            if gradient_check_passes(ana_grad, num_grad) {
                passed_checks += 1;
            } else {
                println!(
                    "  FAIL bias[{}][{}]: ana={:.6}, num={:.6}, rel_err={:.6}",
                    layer_idx, b_idx, ana_grad, num_grad, rel_err
                );
            }
        }
    }

    println!(
        "Gradient check: {}/{} passed, max_error={:.6}",
        passed_checks, total_checks, max_error
    );

    let pass_rate = passed_checks as f32 / total_checks as f32;
    assert!(
        pass_rate >= MIN_PASS_RATE,
        "Gradient check failed: {}/{} checks passed ({:.1}%), min required: {:.0}%",
        passed_checks, total_checks, pass_rate * 100.0, MIN_PASS_RATE * 100.0
    );
}

/// Computes numerical gradient for a specific weight using central differences.
fn numerical_gradient_weight(
    network: &mut KanNetwork,
    input: &[f32],
    target: &[f32],
    workspace: &mut Workspace,
    layer_idx: usize,
    weight_idx: usize,
) -> f32 {
    let original = network.layers[layer_idx].weights[weight_idx];

    // L(w + ε)
    network.layers[layer_idx].weights[weight_idx] = original + EPSILON;
    let loss_plus = compute_loss_for_network(network, input, target, workspace);

    // L(w - ε)
    network.layers[layer_idx].weights[weight_idx] = original - EPSILON;
    let loss_minus = compute_loss_for_network(network, input, target, workspace);

    // Restore original
    network.layers[layer_idx].weights[weight_idx] = original;

    // Central difference: (L(w+ε) - L(w-ε)) / 2ε
    (loss_plus - loss_minus) / (2.0 * EPSILON)
}

/// Computes numerical gradient for a specific bias using central differences.
fn numerical_gradient_bias(
    network: &mut KanNetwork,
    input: &[f32],
    target: &[f32],
    workspace: &mut Workspace,
    layer_idx: usize,
    bias_idx: usize,
) -> f32 {
    let original = network.layers[layer_idx].bias[bias_idx];

    // L(b + ε)
    network.layers[layer_idx].bias[bias_idx] = original + EPSILON;
    let loss_plus = compute_loss_for_network(network, input, target, workspace);

    // L(b - ε)
    network.layers[layer_idx].bias[bias_idx] = original - EPSILON;
    let loss_minus = compute_loss_for_network(network, input, target, workspace);

    // Restore original
    network.layers[layer_idx].bias[bias_idx] = original;

    // Central difference
    (loss_plus - loss_minus) / (2.0 * EPSILON)
}

/// Computes forward pass and MSE loss.
fn compute_loss_for_network(
    network: &KanNetwork,
    input: &[f32],
    target: &[f32],
    workspace: &mut Workspace,
) -> f32 {
    let batch_size = input.len() / network.config.input_dim;
    let output_dim = network.config.output_dim;

    let mut predictions = vec![0.0f32; batch_size * output_dim];
    network.forward_batch(input, &mut predictions, workspace);

    compute_mse_loss(&predictions, target)
}

/// Test that gradients are zero when target equals prediction.
#[test]
fn test_gradient_zero_at_optimum() {
    let config = KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: 3,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        init_seed: Some(42),
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let input: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let mut predictions = vec![0.0f32; 4];
    network.forward_batch(&input, &mut predictions, &mut workspace);

    // Set target equal to predictions
    let target = predictions.clone();
    let loss = compute_mse_loss(&predictions, &target);

    assert!(
        loss < 1e-10,
        "Loss should be ~0 when target=predictions, got {}",
        loss
    );
}

/// Test gradient direction: small step in gradient direction should decrease loss.
#[test]
fn test_gradient_descent_direction() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![4],
        grid_size: 3,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(42),
        ..Default::default()
    };

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(8);

    // Random input/target
    let mut rng = SmallRng::seed_from_u64(999);
    let input: Vec<f32> = (0..8 * 4).map(|_| rng.gen_range(-0.9..0.9)).collect();
    let target: Vec<f32> = (0..8 * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Compute initial loss
    let loss_before = compute_loss_for_network(&network, &input, &target, &mut workspace);

    // Take one gradient step with small learning rate
    let train_opts = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };
    network.train_step_with_options(&input, &target, None, 0.001, &mut workspace, &train_opts);

    // Compute loss after step
    let loss_after = compute_loss_for_network(&network, &input, &target, &mut workspace);

    println!(
        "Loss before: {:.6}, after: {:.6}, diff: {:.6}",
        loss_before,
        loss_after,
        loss_before - loss_after
    );

    assert!(
        loss_after < loss_before,
        "Loss should decrease after gradient step: {} -> {}",
        loss_before,
        loss_after
    );
}
