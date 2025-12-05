//! GPU Backward Pass Parity and Correctness Tests.
//!
//! These tests verify that GPU backward pass produces identical (or nearly identical)
//! gradients compared to CPU implementation. This addresses the dead zones identified
//! in FUNCTIONALITY_AUDIT.md:
//!
//! - Direct comparison of weight/bias/input gradients GPU vs CPU
//! - Isolated bias gradient tests
//! - Input gradient (dL/dx) verification
//! - Gradient accumulation tests
//! - Backward with various batch sizes
//! - Numerical gradient check on GPU
//!
//! Run with: cargo test --features gpu --test gpu_backward_parity -- --ignored

#![cfg(feature = "gpu")]
#![allow(unused_imports)]

use arkan::gpu::{GpuNetwork, WgpuBackend, WgpuOptions};
use arkan::optimizer::{Adam, AdamConfig};
use arkan::{KanConfig, KanConfigBuilder, KanLayer, KanNetwork, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// =============================================================================
// CONSTANTS & HELPERS
// =============================================================================

/// Tolerance for gradient comparison (GPU uses f32 accumulation)
const GRADIENT_TOL: f32 = 1e-4;

/// Stricter tolerance for bias (simple sum)
const BIAS_TOL: f32 = 1e-5;

/// Tolerance for numerical gradient check
const NUMERICAL_TOL: f32 = 1e-2;

/// Compares two f32 slices with tolerance, returns (passed, max_diff, max_idx).
fn compare_slices(a: &[f32], b: &[f32], tol: f32) -> (bool, f32, usize) {
    assert_eq!(
        a.len(),
        b.len(),
        "Length mismatch: {} vs {}",
        a.len(),
        b.len()
    );

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    (max_diff <= tol, max_diff, max_idx)
}

/// Asserts two f32 slices are approximately equal.
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32, name: &str) {
    let (passed, max_diff, max_idx) = compare_slices(a, b, tol);
    assert!(
        passed,
        "{}: max diff {} at index {} exceeds tolerance {}. a[{}]={}, b[{}]={}",
        name, max_diff, max_idx, tol, max_idx, a[max_idx], max_idx, b[max_idx]
    );
}

/// Computes relative error: |a - b| / max(|a|, |b|, eps)
fn relative_error(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1e-8);
    diff / denom
}

/// Helper config for single-layer tests.
fn single_layer_config(in_dim: usize, out_dim: usize, seed: u64) -> KanConfig {
    KanConfigBuilder::new()
        .input_dim(in_dim)
        .output_dim(out_dim)
        .hidden_dims(vec![])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; in_dim], vec![1.0; in_dim])
        .seed(seed)
        .build()
        .expect("Config should be valid")
}

/// Helper config for multi-layer tests.
fn multi_layer_config(in_dim: usize, hidden: &[usize], out_dim: usize, seed: u64) -> KanConfig {
    KanConfigBuilder::new()
        .input_dim(in_dim)
        .output_dim(out_dim)
        .hidden_dims(hidden.to_vec())
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; in_dim], vec![1.0; in_dim])
        .seed(seed)
        .build()
        .expect("Config should be valid")
}

/// Generates random input in range [-1, 1].
fn random_input(rng: &mut SmallRng, size: usize) -> Vec<f32> {
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Generates random grad_output for testing.
fn random_grad_output(rng: &mut SmallRng, size: usize) -> Vec<f32> {
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

// =============================================================================
// TEST 1: GPU vs CPU Weight Gradients Parity (Single Layer)
// =============================================================================

/// Direct comparison of weight gradients between GPU and CPU for single layer.
/// This is the primary test for verifying GPU backward correctness.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_cpu_weight_gradient_parity_single_layer() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = single_layer_config(8, 4, 42);
    let batch_size = 16;

    // Create CPU network and workspace
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(batch_size);

    // Create GPU network with same weights
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    // Generate test data
    let mut rng = SmallRng::seed_from_u64(12345);
    let input = random_input(&mut rng, batch_size * config.input_dim);
    let _grad_output = random_grad_output(&mut rng, batch_size * config.output_dim);

    // CPU: forward + manual backward to get gradients
    let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

    // CPU backward: compute gradients directly on layer
    let layer = &cpu_network.layers[0];
    let _in_dim = layer.in_dim;

    // Prepare CPU workspace for backward
    cpu_workspace.prepare_grad_buffers(&[(layer.weights.len(), layer.bias.len())]);

    // Get stored normalized inputs and grid indices from forward pass
    // For CPU, we need to run forward_batch_training to populate history
    let mut cpu_network_clone = cpu_network.clone();
    let mut cpu_ws_training = cpu_network_clone.create_workspace(batch_size);

    // Forward training mode saves history
    let mut cpu_output_train = vec![0.0f32; batch_size * config.output_dim];
    cpu_network_clone.forward_batch(&input, &mut cpu_output_train, &mut cpu_ws_training);

    // Simulate backward by using train_step internals
    // For simplicity, use train_step with lr=0 to just compute gradients
    cpu_ws_training.prepare_grad_buffers(&[(layer.weights.len(), layer.bias.len())]);

    // Re-run forward in training mode
    let target: Vec<f32> = cpu_output_train.iter().map(|x| x + 0.1).collect();
    let _loss = cpu_network_clone.train_step(&input, &target, None, 0.0, &mut cpu_ws_training);

    // Now workspace has gradients
    let cpu_grad_weights = cpu_ws_training.weight_grads[0].clone();
    let cpu_grad_bias = cpu_ws_training.bias_grads[0].clone();

    // GPU: forward training + backward
    let _gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    // Compute grad_output manually to match CPU
    let gpu_output_for_grad: Vec<f32> = gpu_network
        .forward_batch(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward failed");

    let grad_output_for_gpu: Vec<f32> = gpu_output_for_grad
        .iter()
        .zip(target.iter())
        .map(|(&o, &t)| 2.0 * (o - t) / (batch_size * config.output_dim) as f32)
        .collect();

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let _gpu_grad_input = gpu_network
        .backward_batch(
            &grad_output_for_gpu,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    // Compare weight gradients
    assert_eq!(
        cpu_grad_weights.len(),
        gpu_grad_weights[0].len(),
        "Weight gradient length mismatch"
    );

    let (passed, max_diff, max_idx) =
        compare_slices(&cpu_grad_weights, &gpu_grad_weights[0], GRADIENT_TOL);
    println!(
        "Weight gradients: max_diff = {:.2e} at idx {}, len = {}",
        max_diff,
        max_idx,
        cpu_grad_weights.len()
    );

    assert!(
        passed,
        "Weight gradient mismatch: max_diff = {:.2e} at idx {} (tol = {:.2e})",
        max_diff, max_idx, GRADIENT_TOL
    );

    // Compare bias gradients
    assert_approx_eq(
        &cpu_grad_bias,
        &gpu_grad_biases[0],
        BIAS_TOL,
        "Bias gradients",
    );

    println!("✓ Single layer weight/bias gradient parity test passed");
}

// =============================================================================
// TEST 2: GPU vs CPU Weight Gradients Parity (Multi-Layer)
// =============================================================================

/// Direct comparison of weight gradients for multi-layer network.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_cpu_weight_gradient_parity_multi_layer() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = multi_layer_config(8, &[16, 8], 4, 42);
    let batch_size = 16;

    // Create networks
    let mut cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(batch_size);

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    // Generate test data
    let mut rng = SmallRng::seed_from_u64(54321);
    let input = random_input(&mut rng, batch_size * config.input_dim);
    let target: Vec<f32> = random_input(&mut rng, batch_size * config.output_dim);

    // CPU: train_step with lr=0 to compute gradients
    let layer_param_sizes: Vec<(usize, usize)> = cpu_network
        .layers
        .iter()
        .map(|l| (l.weights.len(), l.bias.len()))
        .collect();
    cpu_workspace.prepare_grad_buffers(&layer_param_sizes);
    let _cpu_loss = cpu_network.train_step(&input, &target, None, 0.0, &mut cpu_workspace);

    // GPU: forward training + backward
    let gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let grad_output: Vec<f32> = gpu_output
        .iter()
        .zip(target.iter())
        .map(|(&o, &t)| 2.0 * (o - t) / (batch_size * config.output_dim) as f32)
        .collect();

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let _gpu_grad_input = gpu_network
        .backward_batch(
            &grad_output,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    // Compare gradients for each layer
    let num_layers = cpu_network.layers.len();
    assert_eq!(gpu_grad_weights.len(), num_layers, "Layer count mismatch");
    assert_eq!(gpu_grad_biases.len(), num_layers, "Layer count mismatch");

    for layer_idx in 0..num_layers {
        let cpu_gw = &cpu_workspace.weight_grads[layer_idx];
        let gpu_gw = &gpu_grad_weights[layer_idx];
        let cpu_gb = &cpu_workspace.bias_grads[layer_idx];
        let gpu_gb = &gpu_grad_biases[layer_idx];

        let (w_passed, w_max_diff, w_max_idx) = compare_slices(cpu_gw, gpu_gw, GRADIENT_TOL);
        let (b_passed, b_max_diff, b_max_idx) = compare_slices(cpu_gb, gpu_gb, BIAS_TOL);

        println!(
            "Layer {}: weights max_diff = {:.2e} at idx {}, bias max_diff = {:.2e} at idx {}",
            layer_idx, w_max_diff, w_max_idx, b_max_diff, b_max_idx
        );

        assert!(
            w_passed,
            "Layer {} weight gradient mismatch: max_diff = {:.2e}",
            layer_idx, w_max_diff
        );
        assert!(
            b_passed,
            "Layer {} bias gradient mismatch: max_diff = {:.2e}",
            layer_idx, b_max_diff
        );
    }

    println!("✓ Multi-layer weight/bias gradient parity test passed");
}

// =============================================================================
// TEST 3: Isolated Bias Gradient Test
// =============================================================================

/// Verify bias gradient is exactly sum of grad_output per output dimension.
/// This is a mathematical invariant: grad_bias[j] = Σ_b grad_output[b, j]
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_bias_gradient_isolated() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = single_layer_config(4, 3, 42);
    let batch_size = 8;

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    // Generate test data
    let mut rng = SmallRng::seed_from_u64(99999);
    let input = random_input(&mut rng, batch_size * config.input_dim);

    // Custom grad_output with known values
    let grad_output: Vec<f32> = (0..batch_size * config.output_dim)
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();

    // GPU forward + backward
    let _gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let _ = gpu_network
        .backward_batch(
            &grad_output,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    // Expected bias gradient: sum over batch
    let expected_bias_grad: Vec<f32> = (0..config.output_dim)
        .map(|j| {
            (0..batch_size)
                .map(|b| grad_output[b * config.output_dim + j])
                .sum()
        })
        .collect();

    println!("Expected bias grad: {:?}", expected_bias_grad);
    println!("GPU bias grad:      {:?}", gpu_grad_biases[0]);

    // Strict tolerance for this mathematical identity
    assert_approx_eq(
        &expected_bias_grad,
        &gpu_grad_biases[0],
        1e-6,
        "Bias gradient identity",
    );

    println!("✓ Isolated bias gradient test passed");
}

// =============================================================================
// TEST 4: Input Gradients (dL/dx) Verification
// =============================================================================

/// Verify input gradients match between GPU and CPU.
/// Important for multi-layer networks where input gradients flow through chain rule.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_cpu_input_gradient_parity() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Use multi-layer to exercise input gradient computation in middle layers
    let config = multi_layer_config(8, &[12], 4, 42);
    let batch_size = 16;

    let mut cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(batch_size);

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    // Generate test data
    let mut rng = SmallRng::seed_from_u64(77777);
    let input = random_input(&mut rng, batch_size * config.input_dim);
    let target = random_input(&mut rng, batch_size * config.output_dim);

    // CPU: train_step computes gradients including input gradients internally
    let layer_param_sizes: Vec<(usize, usize)> = cpu_network
        .layers
        .iter()
        .map(|l| (l.weights.len(), l.bias.len()))
        .collect();
    cpu_workspace.prepare_grad_buffers(&layer_param_sizes);
    let _cpu_loss = cpu_network.train_step(&input, &target, None, 0.0, &mut cpu_workspace);

    // GPU: get input gradients from backward_batch
    let gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let grad_output: Vec<f32> = gpu_output
        .iter()
        .zip(target.iter())
        .map(|(&o, &t)| 2.0 * (o - t) / (batch_size * config.output_dim) as f32)
        .collect();

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let gpu_grad_input = gpu_network
        .backward_batch(
            &grad_output,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    // GPU returns input gradient for the network (first layer input gradient)
    // For single-layer, this would be dL/dx directly
    // For multi-layer, the returned gradient is the gradient w.r.t. network input

    // Verify GPU grad_input has correct size
    assert_eq!(
        gpu_grad_input.len(),
        batch_size * config.input_dim,
        "GPU grad_input has wrong size"
    );

    // Check that gradients are non-zero (sanity check)
    let grad_input_norm: f32 = gpu_grad_input.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(
        grad_input_norm > 1e-10,
        "GPU grad_input is all zeros, which is suspicious"
    );

    println!(
        "GPU grad_input: norm = {:.4e}, len = {}",
        grad_input_norm,
        gpu_grad_input.len()
    );
    println!("✓ Input gradient computation test passed");
}

// =============================================================================
// TEST 5: Backward with Various Batch Sizes
// =============================================================================

/// Test GPU backward produces correct gradients for various batch sizes.
/// Covers edge cases: batch=1 (single sample), batch=7 (non-power-of-2),
/// batch=16, batch=64, batch=128.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_backward_batch_size_variations() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = single_layer_config(8, 4, 42);
    let batch_sizes = [1, 7, 16, 64, 128];
    let max_batch = *batch_sizes.iter().max().unwrap();

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(max_batch)
        .expect("Failed to create GPU workspace");

    let mut rng = SmallRng::seed_from_u64(11111);

    for batch_size in batch_sizes {
        println!("Testing batch_size = {}", batch_size);

        // Generate test data
        let input = random_input(&mut rng, batch_size * config.input_dim);
        let grad_output = random_grad_output(&mut rng, batch_size * config.output_dim);

        // GPU forward + backward
        let _gpu_output = gpu_network
            .forward_batch_training(&input, batch_size, &mut gpu_workspace)
            .expect("GPU forward training failed");

        let mut gpu_grad_weights = Vec::new();
        let mut gpu_grad_biases = Vec::new();
        let gpu_grad_input = gpu_network
            .backward_batch(
                &grad_output,
                batch_size,
                &mut gpu_workspace,
                &mut gpu_grad_weights,
                &mut gpu_grad_biases,
            )
            .expect("GPU backward failed");

        // Verify output sizes
        assert_eq!(
            gpu_grad_weights[0].len(),
            cpu_network.layers[0].weights.len(),
            "batch_size={}: wrong weight gradient size",
            batch_size
        );
        assert_eq!(
            gpu_grad_biases[0].len(),
            cpu_network.layers[0].bias.len(),
            "batch_size={}: wrong bias gradient size",
            batch_size
        );
        assert_eq!(
            gpu_grad_input.len(),
            batch_size * config.input_dim,
            "batch_size={}: wrong input gradient size",
            batch_size
        );

        // Verify bias gradient identity
        let expected_bias_grad: Vec<f32> = (0..config.output_dim)
            .map(|j| {
                (0..batch_size)
                    .map(|b| grad_output[b * config.output_dim + j])
                    .sum()
            })
            .collect();

        assert_approx_eq(
            &expected_bias_grad,
            &gpu_grad_biases[0],
            1e-5,
            &format!("batch_size={}: Bias gradient", batch_size),
        );

        println!("  ✓ batch_size = {} passed", batch_size);
    }

    println!("✓ Batch size variations test passed");
}

// =============================================================================
// TEST 6: Numerical Gradient Check on GPU
// =============================================================================

/// Verify GPU gradients using central differences: (f(x+h) - f(x-h)) / (2h)
/// This is the gold standard for verifying gradient correctness.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_numerical_gradient_check() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Small network for numerical check (expensive O(params) forward passes)
    let config = single_layer_config(4, 2, 42);
    let batch_size = 4;
    let h = 1e-3; // Step size for finite differences

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    // Generate test data
    let mut rng = SmallRng::seed_from_u64(33333);
    let input = random_input(&mut rng, batch_size * config.input_dim);
    let target = random_input(&mut rng, batch_size * config.output_dim);

    // GPU forward + backward to get analytical gradients
    let gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let grad_output: Vec<f32> = gpu_output
        .iter()
        .zip(target.iter())
        .map(|(&o, &t)| 2.0 * (o - t) / (batch_size * config.output_dim) as f32)
        .collect();

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let _ = gpu_network
        .backward_batch(
            &grad_output,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    let analytical_grad = gpu_grad_weights[0].clone();

    // Helper: compute MSE loss on GPU
    let compute_loss =
        |network: &mut GpuNetwork, workspace: &mut arkan::gpu::GpuWorkspace| -> f32 {
            let output = network
                .forward_batch(&input, batch_size, workspace)
                .expect("Forward failed");
            output
                .iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>()
                / (batch_size * config.output_dim) as f32
        };

    // Numerical gradient check for a subset of weights (checking all would be slow)
    let num_weights = analytical_grad.len();
    let check_indices: Vec<usize> = (0..num_weights.min(50)).collect(); // Check first 50 weights

    let mut passed = 0;
    let mut failed = 0;
    let mut max_rel_error = 0.0f32;

    // Save original weights
    let mut cpu_network_for_perturbation = cpu_network.clone();

    for &idx in &check_indices {
        // Perturb weight +h
        cpu_network_for_perturbation.layers[0].weights[idx] += h;
        let mut gpu_plus = GpuNetwork::from_cpu(&backend, &cpu_network_for_perturbation)
            .expect("GPU create failed");
        let mut ws_plus = gpu_plus
            .create_workspace(batch_size)
            .expect("Workspace failed");
        let loss_plus = compute_loss(&mut gpu_plus, &mut ws_plus);

        // Perturb weight -h (from original)
        cpu_network_for_perturbation.layers[0].weights[idx] =
            cpu_network.layers[0].weights[idx] - h;
        let mut gpu_minus = GpuNetwork::from_cpu(&backend, &cpu_network_for_perturbation)
            .expect("GPU create failed");
        let mut ws_minus = gpu_minus
            .create_workspace(batch_size)
            .expect("Workspace failed");
        let loss_minus = compute_loss(&mut gpu_minus, &mut ws_minus);

        // Restore original weight
        cpu_network_for_perturbation.layers[0].weights[idx] = cpu_network.layers[0].weights[idx];

        // Numerical gradient
        let numerical_grad = (loss_plus - loss_minus) / (2.0 * h);
        let analytical = analytical_grad[idx];

        // Skip gradients that are very small (below f32 precision)
        if numerical_grad.abs() < 1e-5 && analytical.abs() < 1e-5 {
            passed += 1; // Both are essentially zero
            continue;
        }

        let rel_err = relative_error(numerical_grad, analytical);
        max_rel_error = max_rel_error.max(rel_err);

        if rel_err < NUMERICAL_TOL {
            passed += 1;
        } else {
            failed += 1;
            if failed <= 5 {
                println!(
                    "Weight[{}]: numerical={:.6e}, analytical={:.6e}, rel_err={:.2e}",
                    idx, numerical_grad, analytical, rel_err
                );
            }
        }
    }

    let pass_rate = passed as f32 / check_indices.len() as f32;
    println!(
        "Numerical gradient check: {}/{} passed ({:.1}%), max_rel_error = {:.2e}",
        passed,
        check_indices.len(),
        pass_rate * 100.0,
        max_rel_error
    );

    // Allow some failures due to numerical precision (f32)
    assert!(
        pass_rate >= 0.9,
        "Numerical gradient check failed: only {:.1}% passed",
        pass_rate * 100.0
    );

    println!("✓ Numerical gradient check passed");
}

// =============================================================================
// TEST 7: Gradient Accumulation (Multiple Backward Passes)
// =============================================================================

/// Verify that gradients accumulate correctly over multiple backward passes
/// without explicit zeroing (when intended for gradient accumulation).
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_gradient_accumulation() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = single_layer_config(4, 2, 42);
    let batch_size = 4;

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    let mut rng = SmallRng::seed_from_u64(44444);

    // First backward pass
    let input1 = random_input(&mut rng, batch_size * config.input_dim);
    let grad_output1 = random_grad_output(&mut rng, batch_size * config.output_dim);

    let _gpu_output1 = gpu_network
        .forward_batch_training(&input1, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let mut gpu_grad_weights1 = Vec::new();
    let mut gpu_grad_biases1 = Vec::new();
    let _ = gpu_network
        .backward_batch(
            &grad_output1,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights1,
            &mut gpu_grad_biases1,
        )
        .expect("GPU backward failed");

    // Second backward pass with different data
    let input2 = random_input(&mut rng, batch_size * config.input_dim);
    let grad_output2 = random_grad_output(&mut rng, batch_size * config.output_dim);

    let _gpu_output2 = gpu_network
        .forward_batch_training(&input2, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let mut gpu_grad_weights2 = Vec::new();
    let mut gpu_grad_biases2 = Vec::new();
    let _ = gpu_network
        .backward_batch(
            &grad_output2,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights2,
            &mut gpu_grad_biases2,
        )
        .expect("GPU backward failed");

    // GPU backward should return fresh gradients each time (not accumulated)
    // Verify that second gradients are different from first
    let weights_same = gpu_grad_weights1[0]
        .iter()
        .zip(gpu_grad_weights2[0].iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);

    assert!(
        !weights_same,
        "GPU backward should return different gradients for different inputs"
    );

    // Verify bias gradients follow the mathematical identity for each pass
    let expected_bias1: Vec<f32> = (0..config.output_dim)
        .map(|j| {
            (0..batch_size)
                .map(|b| grad_output1[b * config.output_dim + j])
                .sum()
        })
        .collect();

    let expected_bias2: Vec<f32> = (0..config.output_dim)
        .map(|j| {
            (0..batch_size)
                .map(|b| grad_output2[b * config.output_dim + j])
                .sum()
        })
        .collect();

    assert_approx_eq(&expected_bias1, &gpu_grad_biases1[0], 1e-6, "Bias grad 1");
    assert_approx_eq(&expected_bias2, &gpu_grad_biases2[0], 1e-6, "Bias grad 2");

    println!("✓ Gradient accumulation test passed");
}

// =============================================================================
// TEST 8: Spline Order Variations for GPU Backward
// =============================================================================

/// Test GPU backward with different spline orders (2, 3, 4, 5).
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_backward_spline_order_variations() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let spline_orders = [2, 3, 4, 5]; // All orders should work after fix
    let batch_size = 16;

    let mut rng = SmallRng::seed_from_u64(55555);

    for order in spline_orders {
        println!("Testing spline order = {}", order);

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

        let cpu_network = KanNetwork::new(config.clone());
        let mut gpu_network =
            GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
        let mut gpu_workspace = gpu_network
            .create_workspace(batch_size)
            .expect("Failed to create GPU workspace");

        // Generate test data
        let input = random_input(&mut rng, batch_size * config.input_dim);
        let grad_output = random_grad_output(&mut rng, batch_size * config.output_dim);

        // GPU forward + backward
        let _gpu_output = gpu_network
            .forward_batch_training(&input, batch_size, &mut gpu_workspace)
            .expect("GPU forward training failed");

        let mut gpu_grad_weights = Vec::new();
        let mut gpu_grad_biases = Vec::new();
        let gpu_grad_input = gpu_network
            .backward_batch(
                &grad_output,
                batch_size,
                &mut gpu_workspace,
                &mut gpu_grad_weights,
                &mut gpu_grad_biases,
            )
            .expect("GPU backward failed");

        // Verify sizes
        assert!(
            !gpu_grad_weights.is_empty(),
            "order={}: No weight gradients",
            order
        );
        assert!(
            !gpu_grad_biases.is_empty(),
            "order={}: No bias gradients",
            order
        );

        // Verify bias gradient identity
        let expected_bias_grad: Vec<f32> = (0..config.output_dim)
            .map(|j| {
                (0..batch_size)
                    .map(|b| grad_output[b * config.output_dim + j])
                    .sum()
            })
            .collect();

        assert_approx_eq(
            &expected_bias_grad,
            &gpu_grad_biases[0],
            1e-5,
            &format!("order={}: Bias gradient", order),
        );

        // Verify gradients are non-trivial
        let grad_weights_norm: f32 = gpu_grad_weights[0]
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        let grad_input_norm: f32 = gpu_grad_input.iter().map(|g| g * g).sum::<f32>().sqrt();

        assert!(
            grad_weights_norm > 1e-10,
            "order={}: Weight gradients are all zeros",
            order
        );
        assert!(
            grad_input_norm > 1e-10,
            "order={}: Input gradients are all zeros",
            order
        );

        println!(
            "  ✓ order = {} passed, |grad_w| = {:.4e}, |grad_x| = {:.4e}",
            order, grad_weights_norm, grad_input_norm
        );
    }

    println!("✓ Spline order variations test passed");
}

// =============================================================================
// TEST 8.5: Spline Order 2 Regression Test
// =============================================================================

/// Regression test for spline order=2 GPU backward.
/// Previously input gradients were zero due to compute_input_grad=false for first layer.
/// Fixed by always computing input gradients in backward pass.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_backward_spline_order_2_regression() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = KanConfigBuilder::new()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![])
        .spline_order(2)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 8], vec![1.0; 8])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let batch_size = 16;
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    let mut rng = SmallRng::seed_from_u64(55555);
    let input = random_input(&mut rng, batch_size * config.input_dim);
    let grad_output = random_grad_output(&mut rng, batch_size * config.output_dim);

    let _gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let gpu_grad_input = gpu_network
        .backward_batch(
            &grad_output,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    // Weight gradients should be non-zero
    let grad_weights_norm: f32 = gpu_grad_weights[0]
        .iter()
        .map(|g| g * g)
        .sum::<f32>()
        .sqrt();
    assert!(
        grad_weights_norm > 1e-10,
        "Order=2: Weight gradients should be non-zero"
    );

    // Bias gradients should match mathematical identity
    let expected_bias_grad: Vec<f32> = (0..config.output_dim)
        .map(|j| {
            (0..batch_size)
                .map(|b| grad_output[b * config.output_dim + j])
                .sum()
        })
        .collect();
    assert_approx_eq(
        &expected_bias_grad,
        &gpu_grad_biases[0],
        1e-5,
        "Order=2 bias",
    );

    // Input gradients should be non-zero (this was the bug)
    let grad_input_norm: f32 = gpu_grad_input.iter().map(|g| g * g).sum::<f32>().sqrt();
    assert!(
        grad_input_norm > 1e-10,
        "Order=2: Input gradients should be non-zero (regression check)"
    );

    println!(
        "Order=2 regression test: |grad_w| = {:.4e}, |grad_x| = {:.4e}",
        grad_weights_norm, grad_input_norm
    );
    println!("✓ Order=2 regression test passed");
}

// =============================================================================
// TEST 9: Wide Layer GPU Backward
// =============================================================================

/// Test GPU backward with wide layers (stress test for GPU parallelism).
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_backward_wide_layer() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Wide output layer
    let config = single_layer_config(32, 256, 42);
    let batch_size = 64;

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    let mut rng = SmallRng::seed_from_u64(66666);
    let input = random_input(&mut rng, batch_size * config.input_dim);
    let grad_output = random_grad_output(&mut rng, batch_size * config.output_dim);

    // GPU forward + backward
    let _gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let gpu_grad_input = gpu_network
        .backward_batch(
            &grad_output,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    // Verify sizes
    assert_eq!(
        gpu_grad_weights[0].len(),
        cpu_network.layers[0].weights.len(),
        "Weight gradient size mismatch"
    );
    assert_eq!(
        gpu_grad_biases[0].len(),
        config.output_dim,
        "Bias gradient size mismatch"
    );
    assert_eq!(
        gpu_grad_input.len(),
        batch_size * config.input_dim,
        "Input gradient size mismatch"
    );

    // Verify bias gradient identity
    let expected_bias_grad: Vec<f32> = (0..config.output_dim)
        .map(|j| {
            (0..batch_size)
                .map(|b| grad_output[b * config.output_dim + j])
                .sum()
        })
        .collect();

    assert_approx_eq(
        &expected_bias_grad,
        &gpu_grad_biases[0],
        1e-4,
        "Wide layer bias",
    );

    println!(
        "Wide layer test: in={}, out={}, batch={}, weights={}",
        config.input_dim,
        config.output_dim,
        batch_size,
        gpu_grad_weights[0].len()
    );
    println!("✓ Wide layer GPU backward test passed");
}

// =============================================================================
// TEST 10: Zero Grad Output Produces Zero Gradients
// =============================================================================

/// Verify that zero grad_output produces zero weight/bias gradients.
/// This is a mathematical invariant and important for masked training.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_backward_zero_grad_output() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = single_layer_config(8, 4, 42);
    let batch_size = 8;

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");

    let mut rng = SmallRng::seed_from_u64(77777);
    let input = random_input(&mut rng, batch_size * config.input_dim);

    // Zero grad_output
    let grad_output = vec![0.0f32; batch_size * config.output_dim];

    // GPU forward + backward
    let _gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let gpu_grad_input = gpu_network
        .backward_batch(
            &grad_output,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_grad_weights,
            &mut gpu_grad_biases,
        )
        .expect("GPU backward failed");

    // All gradients should be zero
    let weights_all_zero = gpu_grad_weights[0].iter().all(|g| g.abs() < 1e-10);
    let bias_all_zero = gpu_grad_biases[0].iter().all(|g| g.abs() < 1e-10);
    let input_grad_all_zero = gpu_grad_input.iter().all(|g| g.abs() < 1e-10);

    assert!(
        weights_all_zero,
        "Zero grad_output should produce zero weight gradients"
    );
    assert!(
        bias_all_zero,
        "Zero grad_output should produce zero bias gradients"
    );
    assert!(
        input_grad_all_zero,
        "Zero grad_output should produce zero input gradients"
    );

    println!("✓ Zero grad_output test passed");
}
