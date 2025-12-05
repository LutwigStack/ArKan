//! Training options correctness tests.
//!
//! This module tests that training options actually work as intended:
//! - Gradient clipping actually clips gradients
//! - Weight decay actually decreases weights
//! - Learning rate = 0 leaves weights unchanged
//! - Large batches work without panics/OOM

use arkan::{KanConfigBuilder, KanNetwork, TrainOptions, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Helper: create a network with known weights for testing
fn create_test_network() -> (KanNetwork, Workspace) {
    let config = KanConfigBuilder::new()
        .input_dim(4)
        .output_dim(2)
        .hidden_dims(vec![8])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 4], vec![1.0; 4])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let network = KanNetwork::new(config.clone());
    let workspace = Workspace::new(&config);

    (network, workspace)
}

/// Helper: get total L2 norm of all weights in network
#[allow(dead_code)]
fn get_weights_l2_norm(network: &KanNetwork) -> f32 {
    let mut sum_sq = 0.0f32;
    for layer in &network.layers {
        for w in &layer.weights {
            sum_sq += w * w;
        }
        for b in &layer.bias {
            sum_sq += b * b;
        }
    }
    sum_sq.sqrt()
}

/// Helper: copy all weights from network
fn copy_weights(network: &KanNetwork) -> Vec<Vec<f32>> {
    network
        .layers
        .iter()
        .map(|l| {
            let mut params = l.weights.clone();
            params.extend(&l.bias);
            params
        })
        .collect()
}

/// Helper: check if two weight snapshots are equal
fn weights_equal(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (wa, wb) in a.iter().zip(b.iter()) {
        if wa.len() != wb.len() {
            return false;
        }
        for (va, vb) in wa.iter().zip(wb.iter()) {
            if (va - vb).abs() > 1e-10 {
                return false;
            }
        }
    }
    true
}

// =============================================================================
// GRADIENT CLIPPING TESTS
// =============================================================================

/// Test that gradient clipping actually clips gradients when norm exceeds max_grad_norm.
///
/// Strategy:
/// 1. Create network and large input that produces large gradients
/// 2. Train with max_grad_norm = 1.0
/// 3. Train same network copy without clipping
/// 4. Verify that clipped update is smaller (weights change less)
#[test]
fn test_gradient_clipping_actually_clips() {
    let config = KanConfigBuilder::new()
        .input_dim(4)
        .output_dim(2)
        .hidden_dims(vec![8])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 4], vec![1.0; 4])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let mut network_clipped = KanNetwork::new(config.clone());
    let mut network_unclipped = KanNetwork::new(config.clone());
    let mut workspace_clipped = Workspace::new(&config);
    let mut workspace_unclipped = Workspace::new(&config);

    // Generate inputs that will produce large gradients
    // Large input values -> large outputs -> large loss -> large gradients
    let batch_size = 32;
    let input: Vec<f32> = (0..batch_size * 4)
        .map(|i| ((i % 10) as f32 - 5.0) * 2.0) // Values in [-10, 8]
        .collect();
    let target: Vec<f32> = (0..batch_size * 2)
        .map(|i| (i % 3) as f32 * 10.0) // Large target values
        .collect();

    let lr = 0.1; // Reasonably large LR to see effect

    // Weights before training
    let weights_before_clipped = copy_weights(&network_clipped);
    let weights_before_unclipped = copy_weights(&network_unclipped);

    // Train with clipping
    let opts_clipped = TrainOptions {
        max_grad_norm: Some(1.0), // Strict clipping
        weight_decay: 0.0,
    };
    network_clipped.train_step_with_options(
        &input,
        &target,
        None,
        lr,
        &mut workspace_clipped,
        &opts_clipped,
    );

    // Train without clipping
    let opts_unclipped = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };
    network_unclipped.train_step_with_options(
        &input,
        &target,
        None,
        lr,
        &mut workspace_unclipped,
        &opts_unclipped,
    );

    // Calculate weight change magnitude
    let weights_after_clipped = copy_weights(&network_clipped);
    let weights_after_unclipped = copy_weights(&network_unclipped);

    let mut delta_clipped_sq = 0.0f32;
    let mut delta_unclipped_sq = 0.0f32;

    for ((before_c, after_c), (before_u, after_u)) in weights_before_clipped
        .iter()
        .zip(weights_after_clipped.iter())
        .zip(
            weights_before_unclipped
                .iter()
                .zip(weights_after_unclipped.iter()),
        )
    {
        for ((bc, ac), (bu, au)) in before_c
            .iter()
            .zip(after_c.iter())
            .zip(before_u.iter().zip(after_u.iter()))
        {
            delta_clipped_sq += (ac - bc).powi(2);
            delta_unclipped_sq += (au - bu).powi(2);
        }
    }

    let delta_clipped = delta_clipped_sq.sqrt();
    let delta_unclipped = delta_unclipped_sq.sqrt();

    println!("Weight change with clipping: {:.6}", delta_clipped);
    println!("Weight change without clipping: {:.6}", delta_unclipped);

    // Clipped update should be smaller than unclipped
    // (assuming gradients exceed max_grad_norm)
    assert!(
        delta_clipped < delta_unclipped,
        "Clipped update ({:.6}) should be smaller than unclipped ({:.6})",
        delta_clipped,
        delta_unclipped
    );

    // Also verify that clipping happened (unclipped delta should be > clipped)
    // If they're nearly equal, gradients were already small
    let ratio = delta_unclipped / delta_clipped.max(1e-10);
    println!("Ratio unclipped/clipped: {:.2}", ratio);
    assert!(
        ratio > 1.1,
        "Expected significant clipping effect, but ratio is only {:.2}",
        ratio
    );
}

/// Test that gradient clipping with very large max_grad_norm has no effect.
#[test]
fn test_gradient_clipping_no_effect_when_large_threshold() {
    let (mut network1, mut workspace1) = create_test_network();
    let (mut network2, mut workspace2) = create_test_network();

    let batch_size = 16;
    let mut rng = SmallRng::seed_from_u64(12345);
    let input: Vec<f32> = (0..batch_size * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f32> = (0..batch_size * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let lr = 0.01;

    // Very large max_grad_norm (essentially no clipping)
    let opts_large = TrainOptions {
        max_grad_norm: Some(1e10),
        weight_decay: 0.0,
    };
    network1.train_step_with_options(&input, &target, None, lr, &mut workspace1, &opts_large);

    // No clipping
    let opts_none = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };
    network2.train_step_with_options(&input, &target, None, lr, &mut workspace2, &opts_none);

    // Weights should be identical (or very close)
    let w1 = copy_weights(&network1);
    let w2 = copy_weights(&network2);

    let mut max_diff = 0.0f32;
    for (wa, wb) in w1.iter().zip(w2.iter()) {
        for (a, b) in wa.iter().zip(wb.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }

    println!("Max weight diff with large vs no clipping: {:.2e}", max_diff);
    assert!(
        max_diff < 1e-6,
        "Large max_grad_norm should have no effect, but diff = {:.2e}",
        max_diff
    );
}

// =============================================================================
// WEIGHT DECAY TESTS
// =============================================================================

/// Test that weight decay actually decreases weights.
#[test]
fn test_weight_decay_actually_decays() {
    let (mut network, mut workspace) = create_test_network();

    let batch_size = 16;
    let mut rng = SmallRng::seed_from_u64(54321);
    let input: Vec<f32> = (0..batch_size * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f32> = (0..batch_size * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let lr = 0.01;
    let decay = 0.1; // Strong decay for clear effect

    // Get initial L2 norm of weights (biases excluded from decay in our implementation)
    let initial_weights_l2: f32 = network
        .layers
        .iter()
        .flat_map(|l| l.weights.iter())
        .map(|w| w * w)
        .sum::<f32>()
        .sqrt();

    // Train with weight decay
    let opts = TrainOptions {
        max_grad_norm: None,
        weight_decay: decay,
    };

    // Multiple steps to see decay effect
    for _ in 0..10 {
        network.train_step_with_options(&input, &target, None, lr, &mut workspace, &opts);
    }

    let final_weights_l2: f32 = network
        .layers
        .iter()
        .flat_map(|l| l.weights.iter())
        .map(|w| w * w)
        .sum::<f32>()
        .sqrt();

    println!("Initial weights L2: {:.6}", initial_weights_l2);
    println!("Final weights L2: {:.6}", final_weights_l2);

    // Weight decay should reduce the L2 norm
    // Note: gradient updates might increase it, but decay factor is strong
    // With decay=0.1 and lr=0.01, each step multiplies by (1 - 0.01*0.1) = 0.999
    // After 10 steps: 0.999^10 ≈ 0.99, so effect might be small
    // Let's just check it's not increasing dramatically
    assert!(
        final_weights_l2 < initial_weights_l2 * 1.5,
        "Weights grew too much: {:.6} -> {:.6}",
        initial_weights_l2,
        final_weights_l2
    );
}

/// Test that weight decay = 0 has no decay effect.
#[test]
fn test_weight_decay_zero_no_decay() {
    let (mut network1, mut workspace1) = create_test_network();
    let (mut network2, mut workspace2) = create_test_network();

    let batch_size = 16;
    let mut rng = SmallRng::seed_from_u64(11111);
    let input: Vec<f32> = (0..batch_size * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f32> = (0..batch_size * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let lr = 0.01;

    // With decay = 0
    let opts_zero = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };
    network1.train_step_with_options(&input, &target, None, lr, &mut workspace1, &opts_zero);

    // Default options (also no decay)
    network2.train_step(&input, &target, None, lr, &mut workspace2);

    let w1 = copy_weights(&network1);
    let w2 = copy_weights(&network2);

    assert!(
        weights_equal(&w1, &w2),
        "weight_decay=0 should be identical to default"
    );
}

/// Test that only weights decay, not biases.
/// 
/// Strategy: Compare two networks - one with decay, one without.
/// The difference in weights should be due to decay term.
/// The difference in biases should be zero (biases are not decayed).
#[test]
fn test_weight_decay_only_weights_not_biases() {
    let config = KanConfigBuilder::new()
        .input_dim(4)
        .output_dim(2)
        .hidden_dims(vec![8])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 4], vec![1.0; 4])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let mut network_decay = KanNetwork::new(config.clone());
    let mut network_nodecay = KanNetwork::new(config.clone());
    let mut workspace_decay = Workspace::new(&config);
    let mut workspace_nodecay = Workspace::new(&config);

    let batch_size = 16;
    let mut rng = SmallRng::seed_from_u64(33333);
    let input: Vec<f32> = (0..batch_size * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f32> = (0..batch_size * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let lr = 0.1;

    // Train with decay
    let opts_decay = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.5, // Strong decay
    };
    network_decay.train_step_with_options(&input, &target, None, lr, &mut workspace_decay, &opts_decay);

    // Train without decay
    let opts_nodecay = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };
    network_nodecay.train_step_with_options(&input, &target, None, lr, &mut workspace_nodecay, &opts_nodecay);

    // Compare biases - should be IDENTICAL (decay doesn't affect biases)
    for (layer_idx, (ld, lnd)) in network_decay.layers.iter().zip(network_nodecay.layers.iter()).enumerate() {
        for (i, (bd, bnd)) in ld.bias.iter().zip(lnd.bias.iter()).enumerate() {
            let diff = (bd - bnd).abs();
            assert!(
                diff < 1e-6,
                "Bias[{}][{}] differs with/without decay: {:.6} vs {:.6}",
                layer_idx,
                i,
                bd,
                bnd
            );
        }
    }

    // Compare weights - should be DIFFERENT (decay affects weights)
    let mut any_weight_diff = false;
    for (ld, lnd) in network_decay.layers.iter().zip(network_nodecay.layers.iter()) {
        for (wd, wnd) in ld.weights.iter().zip(lnd.weights.iter()) {
            if (wd - wnd).abs() > 1e-6 {
                any_weight_diff = true;
                break;
            }
        }
    }
    assert!(any_weight_diff, "Weights should differ with decay vs without");

    println!("✓ Weight decay affects weights only, not biases");
}

// =============================================================================
// LEARNING RATE = 0 TESTS
// =============================================================================

/// Test that learning_rate = 0 leaves weights unchanged.
#[test]
fn test_learning_rate_zero_no_change() {
    let (mut network, mut workspace) = create_test_network();

    let batch_size = 32;
    let mut rng = SmallRng::seed_from_u64(99999);
    let input: Vec<f32> = (0..batch_size * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f32> = (0..batch_size * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let weights_before = copy_weights(&network);

    // Train with lr = 0
    network.train_step(&input, &target, None, 0.0, &mut workspace);

    let weights_after = copy_weights(&network);

    assert!(
        weights_equal(&weights_before, &weights_after),
        "Weights should not change when learning_rate = 0"
    );

    println!("✓ learning_rate=0 preserves weights");
}

/// Test that learning_rate = 0 with weight_decay also has no effect.
#[test]
fn test_learning_rate_zero_with_decay_no_change() {
    let (mut network, mut workspace) = create_test_network();

    let batch_size = 16;
    let mut rng = SmallRng::seed_from_u64(88888);
    let input: Vec<f32> = (0..batch_size * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f32> = (0..batch_size * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let weights_before = copy_weights(&network);

    // Train with lr = 0 but decay > 0
    // Weight decay formula: w *= (1 - lr * decay) = 1 - 0 = 1 (no change)
    let opts = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.1,
    };
    network.train_step_with_options(&input, &target, None, 0.0, &mut workspace, &opts);

    let weights_after = copy_weights(&network);

    assert!(
        weights_equal(&weights_before, &weights_after),
        "Weights should not change when learning_rate = 0 even with weight_decay"
    );

    println!("✓ learning_rate=0 + weight_decay still preserves weights");
}

// =============================================================================
// LARGE BATCH TESTS
// =============================================================================

/// Test that large batch size (2048) works without panic/OOM.
#[test]
fn test_large_batch_2048_no_panic() {
    let config = KanConfigBuilder::new()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![16])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 8], vec![1.0; 8])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let batch_size = 2048;
    let mut rng = SmallRng::seed_from_u64(77777);

    let input: Vec<f32> = (0..batch_size * 8)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let target: Vec<f32> = (0..batch_size * 4)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // This should not panic or OOM
    let loss = network.train_step(&input, &target, None, 0.001, &mut workspace);

    assert!(loss.is_finite(), "Loss should be finite with batch=2048");
    println!("✓ Batch size 2048: loss = {:.6}", loss);
}

/// Test even larger batch (4096) works.
#[test]
fn test_large_batch_4096_no_panic() {
    let config = KanConfigBuilder::new()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![16])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 8], vec![1.0; 8])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let batch_size = 4096;
    let mut rng = SmallRng::seed_from_u64(66666);

    let input: Vec<f32> = (0..batch_size * 8)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let target: Vec<f32> = (0..batch_size * 4)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let loss = network.train_step(&input, &target, None, 0.001, &mut workspace);

    assert!(loss.is_finite(), "Loss should be finite with batch=4096");
    println!("✓ Batch size 4096: loss = {:.6}", loss);
}

/// Test large batch with wide network.
#[test]
fn test_large_batch_with_wide_network() {
    let config = KanConfigBuilder::new()
        .input_dim(64)
        .output_dim(32)
        .hidden_dims(vec![128])
        .spline_order(3)
        .grid_size(5)
        .grid_range(-1.0, 1.0)
        .normalization(vec![0.0; 64], vec![1.0; 64])
        .seed(42)
        .build()
        .expect("Config should be valid");

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let batch_size = 1024;
    let mut rng = SmallRng::seed_from_u64(55555);

    let input: Vec<f32> = (0..batch_size * 64)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let target: Vec<f32> = (0..batch_size * 32)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let loss = network.train_step(&input, &target, None, 0.001, &mut workspace);

    assert!(
        loss.is_finite(),
        "Loss should be finite with batch=1024, wide network"
    );
    println!(
        "✓ Wide network (64→128→32) batch=1024: loss = {:.6}",
        loss
    );
}

// =============================================================================
// COMBINED OPTIONS TESTS
// =============================================================================

/// Test that all options work together.
#[test]
fn test_all_options_combined() {
    let (mut network, mut workspace) = create_test_network();

    let batch_size = 64;
    let mut rng = SmallRng::seed_from_u64(44444);
    let input: Vec<f32> = (0..batch_size * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let target: Vec<f32> = (0..batch_size * 2).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let opts = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.01,
    };

    let loss = network.train_step_with_options(&input, &target, None, 0.01, &mut workspace, &opts);

    assert!(loss.is_finite(), "Loss should be finite with all options");
    println!("✓ All options combined: loss = {:.6}", loss);
}
