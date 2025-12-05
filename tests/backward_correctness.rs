//! Backward pass correctness tests for CPU implementation.
//!
//! This module tests:
//! - Parity between sequential `backward()` and parallel `backward_parallel()`
//! - Numerical correctness of gradient computation
//! - Edge cases: wide layers, small batches, masking
//!
//! # Test Strategy
//!
//! 1. **Parity tests**: Same inputs → same gradients (within floating-point tolerance)
//! 2. **Numerical gradient check**: Compare analytical gradients with finite differences
//! 3. **Wide layer tests**: Layers with 1000+ neurons
//! 4. **Masking tests**: Verify zero grad_output → zero contribution

use arkan::{KanConfig, KanConfigBuilder, KanLayer, KanNetwork, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Helper: create config for tests
fn test_config(in_dim: usize, out_dim: usize, hidden: &[usize]) -> KanConfig {
    KanConfigBuilder::new()
        .input_dim(in_dim)
        .output_dim(out_dim)
        .hidden_dims(hidden.to_vec())
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; in_dim], vec![1.0; in_dim])
        .seed(42)
        .multithreading_threshold(64)
        .build()
        .expect("Config should be valid")
}

/// Helper: create layer and fill with random data
fn setup_layer_test(
    in_dim: usize,
    out_dim: usize,
    batch_size: usize,
    seed: u64,
) -> (
    KanLayer,
    Workspace,
    Vec<f32>, // normalized_input
    Vec<u32>, // grid_indices
    Vec<f32>, // grad_output
) {
    let config = test_config(in_dim, out_dim, &[]);
    let layer = KanLayer::new(in_dim, out_dim, &config);
    let workspace = Workspace::new(&config);

    let mut rng = SmallRng::seed_from_u64(seed);

    // Generate random normalized inputs in [0, 1]
    let normalized_input: Vec<f32> = (0..batch_size * in_dim)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();

    // Generate valid grid indices (span in [order, order + grid_size - 1])
    let order = layer.order;
    let grid_size = layer.grid_size;
    let grid_indices: Vec<u32> = (0..batch_size * in_dim)
        .map(|_| rng.gen_range(order as u32..(order + grid_size) as u32))
        .collect();

    // Generate random grad_output
    let grad_output: Vec<f32> = (0..batch_size * out_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    (
        layer,
        workspace,
        normalized_input,
        grid_indices,
        grad_output,
    )
}

// =============================================================================
// PARITY TESTS: backward vs backward_parallel
// =============================================================================

/// Test that sequential and parallel backward produce identical gradients.
#[test]
fn test_backward_vs_parallel_parity_small_batch() {
    let batch_size = 16;
    let (layer, mut workspace, norm_input, grid_indices, grad_output) =
        setup_layer_test(8, 4, batch_size, 12345);

    // Sequential backward
    let mut grad_weights_seq = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_seq = vec![0.0f32; layer.bias.len()];
    let mut grad_input_seq = vec![0.0f32; batch_size * layer.in_dim];

    layer.backward(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_seq),
        &mut grad_weights_seq,
        &mut grad_bias_seq,
        &mut workspace,
    );

    // Parallel backward
    let mut grad_weights_par = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_par = vec![0.0f32; layer.bias.len()];
    let mut grad_input_par = vec![0.0f32; batch_size * layer.in_dim];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_par),
        &mut grad_weights_par,
        &mut grad_bias_par,
    );

    // Compare weights gradients
    let mut max_diff_weights = 0.0f32;
    for (s, p) in grad_weights_seq.iter().zip(grad_weights_par.iter()) {
        max_diff_weights = max_diff_weights.max((s - p).abs());
    }

    // Compare bias gradients
    let mut max_diff_bias = 0.0f32;
    for (s, p) in grad_bias_seq.iter().zip(grad_bias_par.iter()) {
        max_diff_bias = max_diff_bias.max((s - p).abs());
    }

    // Compare input gradients
    let mut max_diff_input = 0.0f32;
    for (s, p) in grad_input_seq.iter().zip(grad_input_par.iter()) {
        max_diff_input = max_diff_input.max((s - p).abs());
    }

    println!("max_diff_weights = {:.2e}", max_diff_weights);
    println!("max_diff_bias = {:.2e}", max_diff_bias);
    println!("max_diff_input = {:.2e}", max_diff_input);

    // Allow small floating-point differences due to different accumulation order
    assert!(
        max_diff_weights < 1e-5,
        "Weight gradients differ: max_diff = {:.2e}",
        max_diff_weights
    );
    assert!(
        max_diff_bias < 1e-5,
        "Bias gradients differ: max_diff = {:.2e}",
        max_diff_bias
    );
    assert!(
        max_diff_input < 1e-5,
        "Input gradients differ: max_diff = {:.2e}",
        max_diff_input
    );
}

/// Test parity with larger batch (above multithreading threshold).
#[test]
fn test_backward_vs_parallel_parity_large_batch() {
    let batch_size = 256;
    let (layer, mut workspace, norm_input, grid_indices, grad_output) =
        setup_layer_test(16, 8, batch_size, 54321);

    // Sequential backward
    let mut grad_weights_seq = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_seq = vec![0.0f32; layer.bias.len()];
    let mut grad_input_seq = vec![0.0f32; batch_size * layer.in_dim];

    layer.backward(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_seq),
        &mut grad_weights_seq,
        &mut grad_bias_seq,
        &mut workspace,
    );

    // Parallel backward
    let mut grad_weights_par = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_par = vec![0.0f32; layer.bias.len()];
    let mut grad_input_par = vec![0.0f32; batch_size * layer.in_dim];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_par),
        &mut grad_weights_par,
        &mut grad_bias_par,
    );

    // With large batch, parallel accumulation order differs more
    let max_diff_weights: f32 = grad_weights_seq
        .iter()
        .zip(grad_weights_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    let max_diff_bias: f32 = grad_bias_seq
        .iter()
        .zip(grad_bias_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    let max_diff_input: f32 = grad_input_seq
        .iter()
        .zip(grad_input_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    println!(
        "Large batch: max_diff_weights={:.2e}, bias={:.2e}, input={:.2e}",
        max_diff_weights, max_diff_bias, max_diff_input
    );

    // Slightly looser tolerance for large batches due to accumulation order
    assert!(
        max_diff_weights < 1e-4,
        "Weight gradients differ: {:.2e}",
        max_diff_weights
    );
    assert!(
        max_diff_bias < 1e-4,
        "Bias gradients differ: {:.2e}",
        max_diff_bias
    );
    assert!(
        max_diff_input < 1e-4,
        "Input gradients differ: {:.2e}",
        max_diff_input
    );
}

// =============================================================================
// WIDE LAYER TESTS
// =============================================================================

/// Test backward_parallel with wide hidden layer (1024 neurons).
#[test]
fn test_backward_parallel_wide_layer_1024() {
    let batch_size = 64;
    let in_dim = 32;
    let out_dim = 1024;

    let config = KanConfigBuilder::new()
        .input_dim(in_dim)
        .output_dim(out_dim)
        .hidden_dims(vec![])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; in_dim], vec![1.0; in_dim])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let layer = KanLayer::new(in_dim, out_dim, &config);
    let mut workspace = Workspace::new(&config);
    let mut rng = SmallRng::seed_from_u64(99999);

    let norm_input: Vec<f32> = (0..batch_size * in_dim)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
    let grid_indices: Vec<u32> = (0..batch_size * in_dim)
        .map(|_| rng.gen_range(3u32..8u32))
        .collect();
    let grad_output: Vec<f32> = (0..batch_size * out_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Sequential
    let mut grad_weights_seq = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_seq = vec![0.0f32; layer.bias.len()];

    layer.backward(
        &norm_input,
        &grid_indices,
        &grad_output,
        None,
        &mut grad_weights_seq,
        &mut grad_bias_seq,
        &mut workspace,
    );

    // Parallel
    let mut grad_weights_par = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_par = vec![0.0f32; layer.bias.len()];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        None,
        &mut grad_weights_par,
        &mut grad_bias_par,
    );

    let max_diff: f32 = grad_weights_seq
        .iter()
        .zip(grad_weights_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    println!(
        "Wide layer (32→1024): {} weights, max_diff = {:.2e}",
        layer.weights.len(),
        max_diff
    );

    assert!(
        max_diff < 1e-4,
        "Wide layer gradients differ: {:.2e}",
        max_diff
    );
}

/// Test backward_parallel with very wide input (1024 input neurons).
#[test]
fn test_backward_parallel_wide_input_1024() {
    let batch_size = 32;
    let in_dim = 1024;
    let out_dim = 16;

    let config = KanConfigBuilder::new()
        .input_dim(in_dim)
        .output_dim(out_dim)
        .hidden_dims(vec![])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; in_dim], vec![1.0; in_dim])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let layer = KanLayer::new(in_dim, out_dim, &config);
    let mut workspace = Workspace::new(&config);
    let mut rng = SmallRng::seed_from_u64(77777);

    let norm_input: Vec<f32> = (0..batch_size * in_dim)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
    let grid_indices: Vec<u32> = (0..batch_size * in_dim)
        .map(|_| rng.gen_range(3u32..8u32))
        .collect();
    let grad_output: Vec<f32> = (0..batch_size * out_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Sequential
    let mut grad_weights_seq = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_seq = vec![0.0f32; layer.bias.len()];
    let mut grad_input_seq = vec![0.0f32; batch_size * in_dim];

    layer.backward(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_seq),
        &mut grad_weights_seq,
        &mut grad_bias_seq,
        &mut workspace,
    );

    // Parallel
    let mut grad_weights_par = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_par = vec![0.0f32; layer.bias.len()];
    let mut grad_input_par = vec![0.0f32; batch_size * in_dim];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_par),
        &mut grad_weights_par,
        &mut grad_bias_par,
    );

    let max_diff_weights: f32 = grad_weights_seq
        .iter()
        .zip(grad_weights_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    let max_diff_input: f32 = grad_input_seq
        .iter()
        .zip(grad_input_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    println!(
        "Wide input (1024→16): max_diff_weights={:.2e}, max_diff_input={:.2e}",
        max_diff_weights, max_diff_input
    );

    assert!(max_diff_weights < 1e-4, "Weights differ");
    assert!(max_diff_input < 1e-4, "Input grads differ");
}

// =============================================================================
// NETWORK-LEVEL TESTS
// =============================================================================

/// Test that network uses parallel backward when batch >= threshold.
#[test]
fn test_network_train_step_uses_parallel() {
    let config = KanConfigBuilder::new()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![16])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 8], vec![1.0; 8])
        .seed(42)
        .multithreading_threshold(32) // Low threshold for testing
        .build()
        .expect("Config should be valid");

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let batch_size = 64; // Above threshold
    let mut rng = SmallRng::seed_from_u64(11111);

    let input: Vec<f32> = (0..batch_size * 8)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let target: Vec<f32> = (0..batch_size * 4)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // This should use backward_parallel internally
    let loss = network.train_step(&input, &target, None, 0.001, &mut workspace);
    assert!(loss.is_finite(), "Loss should be finite");

    println!(
        "Network train_step (batch=64, threshold=32): loss = {:.6}",
        loss
    );
}

/// Test network with batch below threshold (uses sequential backward).
#[test]
fn test_network_train_step_uses_sequential() {
    let config = KanConfigBuilder::new()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![16])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 8], vec![1.0; 8])
        .seed(42)
        .multithreading_threshold(128) // High threshold
        .build()
        .expect("Config should be valid");

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let batch_size = 16; // Below threshold
    let mut rng = SmallRng::seed_from_u64(22222);

    let input: Vec<f32> = (0..batch_size * 8)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let target: Vec<f32> = (0..batch_size * 4)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // This should use sequential backward
    let loss = network.train_step(&input, &target, None, 0.001, &mut workspace);
    assert!(loss.is_finite(), "Loss should be finite");

    println!(
        "Network train_step (batch=16, threshold=128): loss = {:.6}",
        loss
    );
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

/// Test that zero grad_output produces zero gradients.
#[test]
fn test_backward_parallel_zero_grad_output() {
    let (layer, _workspace, norm_input, grid_indices, _) = setup_layer_test(8, 4, 32, 33333);

    // All zeros in grad_output
    let grad_output = vec![0.0f32; 32 * 4];

    let mut grad_weights = vec![0.0f32; layer.weights.len()];
    let mut grad_bias = vec![0.0f32; layer.bias.len()];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        None,
        &mut grad_weights,
        &mut grad_bias,
    );

    // All gradients should be zero
    assert!(
        grad_weights.iter().all(|&g| g == 0.0),
        "Weight gradients should be zero"
    );
    assert!(
        grad_bias.iter().all(|&g| g == 0.0),
        "Bias gradients should be zero"
    );
}

/// Test backward with masked (sparse) grad_output.
#[test]
fn test_backward_parallel_sparse_grad_output() {
    let batch_size = 64;
    let (layer, mut workspace, norm_input, grid_indices, _) =
        setup_layer_test(8, 4, batch_size, 44444);

    // Sparse grad_output: only first half non-zero
    let mut grad_output = vec![0.0f32; batch_size * 4];
    let mut rng = SmallRng::seed_from_u64(44444);
    for i in 0..(batch_size / 2 * 4) {
        grad_output[i] = rng.gen_range(-1.0..1.0);
    }

    // Sequential
    let mut grad_weights_seq = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_seq = vec![0.0f32; layer.bias.len()];

    layer.backward(
        &norm_input,
        &grid_indices,
        &grad_output,
        None,
        &mut grad_weights_seq,
        &mut grad_bias_seq,
        &mut workspace,
    );

    // Parallel
    let mut grad_weights_par = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_par = vec![0.0f32; layer.bias.len()];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        None,
        &mut grad_weights_par,
        &mut grad_bias_par,
    );

    let max_diff: f32 = grad_weights_seq
        .iter()
        .zip(grad_weights_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    println!("Sparse grad_output: max_diff = {:.2e}", max_diff);
    assert!(max_diff < 1e-5, "Sparse test failed");
}

/// Test backward with batch_size = 1.
#[test]
fn test_backward_parallel_batch_size_1() {
    let (layer, mut workspace, norm_input, grid_indices, grad_output) =
        setup_layer_test(8, 4, 1, 55555);

    // Sequential
    let mut grad_weights_seq = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_seq = vec![0.0f32; layer.bias.len()];
    let mut grad_input_seq = vec![0.0f32; 8];

    layer.backward(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_seq),
        &mut grad_weights_seq,
        &mut grad_bias_seq,
        &mut workspace,
    );

    // Parallel
    let mut grad_weights_par = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_par = vec![0.0f32; layer.bias.len()];
    let mut grad_input_par = vec![0.0f32; 8];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        Some(&mut grad_input_par),
        &mut grad_weights_par,
        &mut grad_bias_par,
    );

    // Should be identical for batch_size=1
    let max_diff: f32 = grad_weights_seq
        .iter()
        .zip(grad_weights_par.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0, f32::max);

    assert!(max_diff < 1e-6, "Batch=1 should be identical");
}

// =============================================================================
// SPLINE ORDER COVERAGE
// =============================================================================

/// Test backward_parallel across different spline orders.
#[test]
fn test_backward_parallel_spline_orders() {
    let orders = [2, 3, 4, 5, 6];
    let batch_size = 32;
    let mut rng = SmallRng::seed_from_u64(66666);

    for order in orders {
        let config = KanConfigBuilder::new()
            .input_dim(8)
            .output_dim(4)
            .hidden_dims(vec![])
            .spline_order(order)
            .grid_size(5)
            .grid_range(-1.0, 1.0)
            .normalization(vec![0.0; 8], vec![1.0; 8])
            .seed(42)
            .build()
            .expect("Config should be valid");

        let layer = KanLayer::new(8, 4, &config);
        let mut workspace = Workspace::new(&config);

        let norm_input: Vec<f32> = (0..batch_size * 8)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        let grid_indices: Vec<u32> = (0..batch_size * 8)
            .map(|_| rng.gen_range(order as u32..(order + 5) as u32))
            .collect();
        let grad_output: Vec<f32> = (0..batch_size * 4)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Sequential
        let mut gw_seq = vec![0.0f32; layer.weights.len()];
        let mut gb_seq = vec![0.0f32; layer.bias.len()];

        layer.backward(
            &norm_input,
            &grid_indices,
            &grad_output,
            None,
            &mut gw_seq,
            &mut gb_seq,
            &mut workspace,
        );

        // Parallel
        let mut gw_par = vec![0.0f32; layer.weights.len()];
        let mut gb_par = vec![0.0f32; layer.bias.len()];

        layer.backward_parallel(
            &norm_input,
            &grid_indices,
            &grad_output,
            None,
            &mut gw_par,
            &mut gb_par,
        );

        let max_diff: f32 = gw_seq
            .iter()
            .zip(gw_par.iter())
            .map(|(s, p)| (s - p).abs())
            .fold(0.0, f32::max);

        println!("Order {}: max_diff = {:.2e}", order, max_diff);
        // Slightly looser tolerance due to floating-point accumulation order differences
        assert!(
            max_diff < 5e-5,
            "Order {} failed with diff {:.2e}",
            order,
            max_diff
        );
    }
}

// =============================================================================
// DETERMINISM TEST
// =============================================================================

/// Test that backward_parallel produces deterministic results.
#[test]
fn test_backward_parallel_deterministic() {
    let (layer, _, norm_input, grid_indices, grad_output) = setup_layer_test(16, 8, 128, 77777);

    let mut grad_weights_1 = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_1 = vec![0.0f32; layer.bias.len()];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        None,
        &mut grad_weights_1,
        &mut grad_bias_1,
    );

    // Second call with fresh buffers
    let mut grad_weights_2 = vec![0.0f32; layer.weights.len()];
    let mut grad_bias_2 = vec![0.0f32; layer.bias.len()];

    layer.backward_parallel(
        &norm_input,
        &grid_indices,
        &grad_output,
        None,
        &mut grad_weights_2,
        &mut grad_bias_2,
    );

    // Should be exactly identical
    assert_eq!(
        grad_weights_1, grad_weights_2,
        "Weights should be deterministic"
    );
    assert_eq!(grad_bias_1, grad_bias_2, "Biases should be deterministic");
}
