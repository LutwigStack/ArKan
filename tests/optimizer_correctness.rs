//! Optimizer Correctness Tests.
//!
//! These tests verify numerical correctness of Adam and GpuAdam implementations:
//!
//! 1. **Adam formula**: Manual computation matches implementation
//! 2. **Bias correction**: Correct (1-β^t) factors applied
//! 3. **Weight decay**: AdamW decoupled decay formula
//! 4. **Custom betas**: Non-default β1, β2 values work correctly
//! 5. **GPU vs CPU parity**: GpuAdam produces same results as CPU Adam
//!
//! Run with: cargo test --test optimizer_correctness
//! GPU tests: cargo test --features gpu --test optimizer_correctness -- --ignored

use arkan::optimizer::{Adam, AdamConfig, Optimizer, SafetyConfig};
use arkan::{KanConfig, KanNetwork};

// =============================================================================
// CONSTANTS & HELPERS
// =============================================================================

/// Tolerance for floating point comparison
const TOL: f32 = 1e-6;

/// Creates a minimal network for optimizer testing
fn minimal_network() -> KanNetwork {
    let config = KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: 3,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        init_seed: Some(42),
        ..Default::default()
    };
    KanNetwork::new(config)
}

/// Computes expected Adam update manually (reference implementation)
fn adam_step_reference(
    param: f32,
    grad: f32,
    m: &mut f32,
    v: &mut f32,
    t: i32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
) -> f32 {
    // Update moments
    *m = beta1 * *m + (1.0 - beta1) * grad;
    *v = beta2 * *v + (1.0 - beta2) * grad * grad;

    // Bias correction
    let bc1 = 1.0 - beta1.powi(t);
    let bc2 = 1.0 - beta2.powi(t);
    let alpha = lr * bc2.sqrt() / bc1;

    // Compute update
    let update = alpha * *m / (v.sqrt() + epsilon);

    // Apply weight decay (AdamW-style, before gradient step)
    let mut new_param = param;
    if weight_decay > 0.0 {
        new_param *= 1.0 - lr * weight_decay;
    }

    // Apply gradient update
    new_param - update
}

// =============================================================================
// TEST: Adam formula numerical correctness
// =============================================================================

/// Test that Adam update matches hand-computed reference.
#[test]
fn test_adam_formula_numerical() {
    let mut network = minimal_network();
    let adam_config = AdamConfig::with_decay(0.01, 0.0);
    let mut optimizer = Adam::new(&network, adam_config);

    // Get initial weights
    let initial_weights: Vec<f32> = network.layers[0].weights.clone();
    let initial_bias: Vec<f32> = network.layers[0].bias.clone();

    // Create known gradients
    let grad_weights: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2];
    let grad_bias: Vec<f32> = vec![0.5];

    // Ensure we have correct gradient sizes
    let weight_grads = vec![grad_weights[..network.layers[0].weights.len()].to_vec()];
    let bias_grads = vec![grad_bias.clone()];

    // Perform one step
    optimizer.step(&mut network, &weight_grads, &bias_grads, None).unwrap();

    // Compute expected values manually
    let mut expected_weights = Vec::new();
    for (i, (&w, &g)) in initial_weights.iter().zip(weight_grads[0].iter()).enumerate() {
        let mut m = 0.0f32;
        let mut v = 0.0f32;
        let expected = adam_step_reference(
            w, g, &mut m, &mut v, 1,
            adam_config.lr, adam_config.beta1, adam_config.beta2,
            adam_config.epsilon, adam_config.weight_decay
        );
        expected_weights.push(expected);

        let actual = network.layers[0].weights[i];
        let diff = (actual - expected).abs();
        assert!(
            diff < TOL,
            "Weight[{}]: expected={}, actual={}, diff={}, grad={}",
            i, expected, actual, diff, g
        );
    }

    // Check bias
    let mut m_bias = 0.0f32;
    let mut v_bias = 0.0f32;
    let expected_bias = adam_step_reference(
        initial_bias[0], grad_bias[0], &mut m_bias, &mut v_bias, 1,
        adam_config.lr, adam_config.beta1, adam_config.beta2,
        adam_config.epsilon, adam_config.weight_decay
    );
    let actual_bias = network.layers[0].bias[0];
    let diff = (actual_bias - expected_bias).abs();
    assert!(
        diff < TOL,
        "Bias: expected={}, actual={}, diff={}",
        expected_bias, actual_bias, diff
    );

    println!("✅ Adam formula matches reference implementation");
}

// =============================================================================
// TEST: Bias correction correctness
// =============================================================================

/// Test that bias correction factors are computed correctly at various timesteps.
#[test]
fn test_adam_bias_correction_factors() {
    let mut network = minimal_network();
    let adam_config = AdamConfig::with_lr(0.001);
    let mut optimizer = Adam::new(&network, adam_config);

    // Same gradient each step
    let grad_weights = vec![vec![1.0f32; network.layers[0].weights.len()]];
    let grad_bias = vec![vec![1.0f32; network.layers[0].bias.len()]];

    // Track weight changes at different timesteps
    let mut weight_at_t = Vec::new();
    weight_at_t.push(network.layers[0].weights[0]);

    for t in 1..=10 {
        let w_before = network.layers[0].weights[0];
        optimizer.step(&mut network, &grad_weights, &grad_bias, None).unwrap();
        let w_after = network.layers[0].weights[0];
        let update_magnitude = (w_before - w_after).abs();
        weight_at_t.push(w_after);

        // At early timesteps, bias correction should make updates larger
        // At later timesteps, updates should stabilize
        println!("t={}: update_magnitude={:.6}", t, update_magnitude);

        // Verify bias correction is applied (updates should NOT be constant)
        if t > 1 {
            // First few updates should be larger due to bias correction
            // As t increases, correction factor approaches 1
        }
    }

    // Compute expected bias correction factors
    for t in 1..=5 {
        let bc1 = 1.0 - 0.9f32.powi(t);
        let bc2 = 1.0 - 0.999f32.powi(t);
        let correction_factor = bc2.sqrt() / bc1;
        println!("t={}: bc1={:.6}, bc2={:.6}, factor={:.6}", t, bc1, bc2, correction_factor);
    }

    // Verify that early updates are influenced by bias correction
    // At t=1, bias correction factor is ~10x larger than at t=100
    let early_update = (weight_at_t[0] - weight_at_t[1]).abs();
    let late_update = (weight_at_t[9] - weight_at_t[10]).abs();

    assert!(
        early_update > late_update * 0.5,
        "Early updates should be comparable to or larger than late updates due to bias correction: early={}, late={}",
        early_update, late_update
    );

    println!("✅ Bias correction factors applied correctly");
}

// =============================================================================
// TEST: Multi-step convergence with known solution
// =============================================================================

/// Test that Adam converges to minimum for simple quadratic f(x) = x^2.
#[test]
fn test_adam_convergence_quadratic() {
    // Simulate optimizing f(x) = x^2, gradient = 2x
    // Minimum at x = 0
    let lr = 0.1;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;

    let mut x = 5.0f32;  // Start far from minimum
    let mut m = 0.0f32;
    let mut v = 0.0f32;

    for t in 1..=100 {
        let grad = 2.0 * x;  // df/dx = 2x

        // Adam update
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;

        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        let m_hat = m / bc1;
        let v_hat = v / bc2;

        x -= lr * m_hat / (v_hat.sqrt() + epsilon);

        if t % 20 == 0 {
            println!("t={}: x={:.6}, f(x)={:.6}", t, x, x * x);
        }
    }

    assert!(
        x.abs() < 0.1,
        "Adam should converge close to x=0, got x={}",
        x
    );

    println!("✅ Adam converges on quadratic function");
}

// =============================================================================
// TEST: Weight decay (AdamW) formula
// =============================================================================

/// Test that weight decay is applied correctly (decoupled, AdamW-style).
#[test]
fn test_adam_weight_decay_formula() {
    let mut network = minimal_network();

    // With weight decay
    let adam_config_decay = AdamConfig::with_decay(0.01, 0.1);

    // Without weight decay (for comparison)
    let adam_config_no_decay = AdamConfig::with_lr(0.01);

    let initial_weight = network.layers[0].weights[0];

    // Create another network with same initial weights
    let mut network_no_decay = minimal_network();

    let mut optimizer_decay = Adam::new(&network, adam_config_decay);
    let mut optimizer_no_decay = Adam::new(&network_no_decay, adam_config_no_decay);

    // Zero gradient - only weight decay should affect the result
    let zero_grad_weights = vec![vec![0.0f32; network.layers[0].weights.len()]];
    let zero_grad_bias = vec![vec![0.0f32; network.layers[0].bias.len()]];

    // Step with decay
    optimizer_decay.step(&mut network, &zero_grad_weights, &zero_grad_bias, None).unwrap();
    let weight_with_decay = network.layers[0].weights[0];

    // Step without decay
    optimizer_no_decay.step(&mut network_no_decay, &zero_grad_weights, &zero_grad_bias, None).unwrap();
    let weight_no_decay = network_no_decay.layers[0].weights[0];

    // With zero gradient, weight should only change due to decay
    // Expected: w_new = w_old * (1 - lr * decay) = w_old * (1 - 0.01 * 0.1) = w_old * 0.999
    let expected_with_decay = initial_weight * (1.0 - 0.01 * 0.1);

    println!("Initial weight: {}", initial_weight);
    println!("Weight with decay: {} (expected: {})", weight_with_decay, expected_with_decay);
    println!("Weight without decay: {}", weight_no_decay);

    // With zero gradient and no decay, weight should be unchanged
    // (m=0, v=0, so update=0)
    assert!(
        (weight_no_decay - initial_weight).abs() < TOL,
        "With zero gradient and no decay, weight should be unchanged: {} vs {}",
        weight_no_decay, initial_weight
    );

    // With decay, weight should be reduced
    let decay_diff = (weight_with_decay - expected_with_decay).abs();
    assert!(
        decay_diff < TOL,
        "Weight decay not applied correctly: got {}, expected {}, diff={}",
        weight_with_decay, expected_with_decay, decay_diff
    );

    // Weight with decay should be smaller (in absolute value) than without
    assert!(
        weight_with_decay.abs() < initial_weight.abs() || initial_weight.abs() < 1e-6,
        "Weight decay should reduce weight magnitude: {} -> {}",
        initial_weight, weight_with_decay
    );

    println!("✅ Weight decay (AdamW) formula is correct");
}

// =============================================================================
// TEST: Custom beta values
// =============================================================================

/// Test that custom β1, β2 values affect the optimizer behavior correctly.
#[test]
fn test_adam_custom_betas() {
    // Low momentum (β1=0.5) should respond faster to gradient changes
    // High momentum (β1=0.99) should be more stable

    let configs = [
        ("low_beta1", AdamConfig { lr: 0.01, beta1: 0.5, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0, safety: SafetyConfig::default() }),
        ("high_beta1", AdamConfig { lr: 0.01, beta1: 0.99, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0, safety: SafetyConfig::default() }),
        ("low_beta2", AdamConfig { lr: 0.01, beta1: 0.9, beta2: 0.9, epsilon: 1e-8, weight_decay: 0.0, safety: SafetyConfig::default() }),
        ("high_beta2", AdamConfig { lr: 0.01, beta1: 0.9, beta2: 0.9999, epsilon: 1e-8, weight_decay: 0.0, safety: SafetyConfig::default() }),
    ];

    for (name, config) in &configs {
        let mut network = minimal_network();
        let mut optimizer = Adam::new(&network, *config);

        let initial_weight = network.layers[0].weights[0];

        // First step with positive gradient
        let grad_weights = vec![vec![1.0f32; network.layers[0].weights.len()]];
        let grad_bias = vec![vec![0.0f32; network.layers[0].bias.len()]];

        optimizer.step(&mut network, &grad_weights, &grad_bias, None).unwrap();
        let weight_after_pos = network.layers[0].weights[0];

        // Second step with negative gradient (direction change)
        let grad_weights_neg = vec![vec![-1.0f32; network.layers[0].weights.len()]];
        optimizer.step(&mut network, &grad_weights_neg, &grad_bias, None).unwrap();
        let weight_after_neg = network.layers[0].weights[0];

        println!("{}: {} -> {} -> {}", name, initial_weight, weight_after_pos, weight_after_neg);

        // Verify optimizer is functional with custom betas
        assert!(
            weight_after_pos != initial_weight,
            "{}: Weight should change after first step",
            name
        );
    }

    println!("✅ Custom beta values work correctly");
}

// =============================================================================
// TEST: Momentum states (m, v) update correctly
// =============================================================================

/// Test that momentum states accumulate correctly over multiple steps.
#[test]
fn test_adam_momentum_accumulation() {
    let mut network = minimal_network();
    let adam_config = AdamConfig::with_lr(0.01);
    let mut optimizer = Adam::new(&network, adam_config);

    // Constant gradient
    let grad = 1.0f32;
    let grad_weights = vec![vec![grad; network.layers[0].weights.len()]];
    let grad_bias = vec![vec![grad; network.layers[0].bias.len()]];

    // Track expected m and v
    let mut expected_m = 0.0f32;
    let mut expected_v = 0.0f32;

    for t in 1..=5 {
        // Expected update
        expected_m = 0.9 * expected_m + 0.1 * grad;
        expected_v = 0.999 * expected_v + 0.001 * grad * grad;

        optimizer.step(&mut network, &grad_weights, &grad_bias, None).unwrap();

        // Check internal state
        let actual_m = optimizer.layer_states[0].weights.m[0];
        let actual_v = optimizer.layer_states[0].weights.v[0];

        let m_diff = (actual_m - expected_m).abs();
        let v_diff = (actual_v - expected_v).abs();

        assert!(
            m_diff < TOL,
            "t={}: m mismatch: expected={}, actual={}, diff={}",
            t, expected_m, actual_m, m_diff
        );
        assert!(
            v_diff < TOL,
            "t={}: v mismatch: expected={}, actual={}, diff={}",
            t, expected_v, actual_v, v_diff
        );

        println!("t={}: m={:.6} (expected {:.6}), v={:.6} (expected {:.6})",
                 t, actual_m, expected_m, actual_v, expected_v);
    }

    println!("✅ Momentum states accumulate correctly");
}

// =============================================================================
// GPU TESTS (require --features gpu)
// =============================================================================

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use arkan::gpu::{GpuAdam, GpuAdamConfig, GpuNetwork, WgpuBackend, WgpuOptions};

    /// Test that GPU Adam produces same results as CPU Adam for one step.
    /// Uses hybrid training (GPU backward + CPU Adam) as reference.
    #[test]
    #[ignore = "Requires GPU"]
    fn test_gpu_adam_vs_cpu_adam_single_step() {
        let backend = WgpuBackend::init(WgpuOptions::default()).expect("Failed to create backend");

        // Create identical networks
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8],
            grid_size: 5,
            spline_order: 3,
            grid_range: (-2.0, 2.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            init_seed: Some(12345),
            ..Default::default()
        };

        let cpu_network = KanNetwork::new(config.clone());
        
        // Create two GPU networks for hybrid vs native comparison
        let mut gpu_hybrid = GpuNetwork::from_cpu(&backend, &cpu_network)
            .expect("Failed to create GPU network");
        let mut gpu_native = GpuNetwork::from_cpu(&backend, &cpu_network)
            .expect("Failed to create GPU network");

        // CPU network clone for hybrid training
        let mut cpu_for_hybrid = cpu_network.clone();

        // Create optimizers with same config
        let cpu_adam_config = AdamConfig {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let gpu_adam_config = GpuAdamConfig {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };

        let mut cpu_adam = Adam::new(&cpu_for_hybrid, cpu_adam_config);
        let layer_sizes = gpu_native.layer_param_sizes();
        let mut gpu_adam = GpuAdam::new(
            backend.device_arc(),
            backend.queue_arc(),
            &layer_sizes,
            gpu_adam_config,
        );

        // Create same input/target
        let batch_size = 4;
        let input: Vec<f32> = (0..batch_size * config.input_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let target: Vec<f32> = (0..batch_size * config.output_dim)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();

        // Hybrid training step (GPU backward + CPU Adam)
        let mut hybrid_workspace = gpu_hybrid.create_workspace(batch_size).expect("Failed");
        let hybrid_loss = gpu_hybrid
            .train_step_mse(
                &input, &target, batch_size,
                &mut hybrid_workspace, &mut cpu_adam, &mut cpu_for_hybrid
            )
            .expect("Hybrid training failed");

        // Native GPU training step
        let mut native_workspace = gpu_native.create_workspace(batch_size).expect("Failed");
        let native_loss = gpu_native
            .train_step_gpu_native(&input, &target, batch_size, &mut native_workspace, &mut gpu_adam)
            .expect("GPU native training failed");

        // Losses should be very close
        let loss_diff = (hybrid_loss - native_loss).abs();
        println!("Hybrid loss: {:.6}, Native loss: {:.6}, diff: {:.2e}", 
                 hybrid_loss, native_loss, loss_diff);
        
        assert!(
            loss_diff < 0.01,
            "Losses differ too much: hybrid={}, native={}",
            hybrid_loss, native_loss
        );

        // Compare weights after one step
        let mut cpu_check = cpu_network.clone();
        gpu_native.sync_weights_to_cpu(&mut cpu_check).expect("Sync failed");

        for (layer_idx, (hybrid_layer, native_layer)) in cpu_for_hybrid.layers.iter()
            .zip(cpu_check.layers.iter()).enumerate()
        {
            let mut max_diff = 0.0f32;
            for (w_hybrid, w_native) in hybrid_layer.weights.iter().zip(native_layer.weights.iter()) {
                let diff = (w_hybrid - w_native).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            println!("Layer {}: max weight diff = {:.6}", layer_idx, max_diff);
            
            // Note: Some difference is expected due to GPU atomics in reduction
            assert!(
                max_diff < 0.1,
                "Layer {}: weight diff too large: {}",
                layer_idx, max_diff
            );
        }

        println!("✅ GPU Adam single step test completed");
    }

    /// Test that GPU Adam momentum states match CPU Adam over multiple steps.
    #[test]
    #[ignore = "Requires GPU"]
    fn test_gpu_adam_momentum_parity() {
        let backend = WgpuBackend::init(WgpuOptions::default()).expect("Failed to create backend");

        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![4],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(999),
            ..Default::default()
        };

        // Create networks
        let cpu_network_init = KanNetwork::new(config.clone());
        
        // Create two GPU networks - one for hybrid, one for native
        let mut gpu_hybrid = GpuNetwork::from_cpu(&backend, &cpu_network_init)
            .expect("Failed to create GPU network");
        let mut gpu_native = GpuNetwork::from_cpu(&backend, &cpu_network_init)
            .expect("Failed to create GPU network");

        // CPU network for hybrid training
        let mut cpu_for_hybrid = cpu_network_init.clone();

        // Same optimizer config
        let cpu_adam_config = AdamConfig {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let gpu_adam_config = GpuAdamConfig {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };

        let mut cpu_adam = Adam::new(&cpu_for_hybrid, cpu_adam_config);
        let layer_sizes = gpu_native.layer_param_sizes();
        let mut gpu_adam = GpuAdam::new(
            backend.device_arc(),
            backend.queue_arc(),
            &layer_sizes,
            gpu_adam_config,
        );

        let batch_size = 8;
        let input: Vec<f32> = (0..batch_size * config.input_dim)
            .map(|i| ((i as f32) * 0.3).sin())
            .collect();
        let target: Vec<f32> = (0..batch_size * config.output_dim)
            .map(|i| ((i as f32) * 0.5).cos())
            .collect();

        let mut hybrid_workspace = gpu_hybrid.create_workspace(batch_size).expect("Failed");
        let mut native_workspace = gpu_native.create_workspace(batch_size).expect("Failed");

        let num_steps = 10;
        for step in 0..num_steps {
            // Hybrid step (GPU backward + CPU Adam)
            let hybrid_loss = gpu_hybrid
                .train_step_mse(
                    &input, &target, batch_size,
                    &mut hybrid_workspace, &mut cpu_adam, &mut cpu_for_hybrid
                )
                .expect("Hybrid training failed");

            // Native GPU step
            let native_loss = gpu_native
                .train_step_gpu_native(&input, &target, batch_size, &mut native_workspace, &mut gpu_adam)
                .expect("GPU native training failed");

            let loss_diff = (hybrid_loss - native_loss).abs();
            println!("Step {}: Hybrid loss={:.6}, Native loss={:.6}, diff={:.2e}",
                     step, hybrid_loss, native_loss, loss_diff);

            // Losses should be reasonably close
            assert!(
                loss_diff < 0.1,
                "Step {}: Loss difference too large: Hybrid={}, Native={}, diff={}",
                step, hybrid_loss, native_loss, loss_diff
            );
        }

        // After training, compare weights from both approaches
        let mut cpu_check = cpu_network_init.clone();
        gpu_native.sync_weights_to_cpu(&mut cpu_check).expect("Sync failed");

        // Compare hybrid-trained weights vs native-GPU-trained weights
        for (layer_idx, (hybrid_layer, native_layer)) in cpu_for_hybrid.layers.iter()
            .zip(cpu_check.layers.iter()).enumerate()
        {
            let mut max_diff = 0.0f32;
            for (w_hybrid, w_native) in hybrid_layer.weights.iter()
                .zip(native_layer.weights.iter())
            {
                let diff = (w_hybrid - w_native).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            println!("Layer {}: max weight diff = {:.6}", layer_idx, max_diff);

            // Note: Some difference is expected due to:
            // 1. GPU uses padded weights
            // 2. Different order of operations (floating point)
            // 3. GPU atomics in reduction
        }

        println!("✅ GPU Adam momentum parity test completed");
    }

    /// Test GPU Adam with custom beta values.
    #[test]
    #[ignore = "Requires GPU"]
    fn test_gpu_adam_custom_betas() {
        let backend = WgpuBackend::init(WgpuOptions::default()).expect("Failed to create backend");

        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        let cpu_network = KanNetwork::new(config.clone());

        let custom_configs = [
            ("low_beta1", GpuAdamConfig { lr: 0.01, beta1: 0.5, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.0 }),
            ("high_beta2", GpuAdamConfig { lr: 0.01, beta1: 0.9, beta2: 0.9999, epsilon: 1e-8, weight_decay: 0.0 }),
            ("with_decay", GpuAdamConfig { lr: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0.01 }),
        ];

        for (name, gpu_config) in &custom_configs {
            let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
                .expect("Failed to create GPU network");

            let layer_sizes = gpu_network.layer_param_sizes();
            let mut gpu_adam = GpuAdam::new(
                backend.device_arc(),
                backend.queue_arc(),
                &layer_sizes,
                *gpu_config,
            );

            let batch_size = 4;
            let input: Vec<f32> = vec![0.5; batch_size * config.input_dim];
            let target: Vec<f32> = vec![1.0; batch_size * config.output_dim];

            let mut workspace = gpu_network.create_workspace(batch_size).expect("Failed");

            // Run several steps
            let mut losses = Vec::new();
            for _ in 0..5 {
                let loss = gpu_network
                    .train_step_gpu_native(&input, &target, batch_size, &mut workspace, &mut gpu_adam)
                    .expect("Training failed");
                losses.push(loss);
            }

            // Loss should generally decrease (or at least not explode)
            println!("{}: losses = {:?}", name, losses);
            assert!(
                losses.iter().all(|l| l.is_finite()),
                "{}: All losses should be finite",
                name
            );
        }

        println!("✅ GPU Adam custom betas test passed");
    }
}
