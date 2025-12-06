//! Integration tests for examples.
//!
//! These tests verify that the example patterns work correctly.
//! They don't run the full examples (which may be slow) but test
//! the core functionality demonstrated in each example.

use arkan::optimizer::{Adam, AdamConfig};
use arkan::{KanConfig, KanNetwork, TrainOptions};

// =============================================================================
// basic.rs example tests
// =============================================================================

/// Tests the basic inference pattern from basic.rs example.
#[test]
fn test_basic_example_pattern() {
    // Same pattern as basic.rs
    let config = KanConfig::preset();
    
    // Verify config is valid
    config.validate().expect("Preset config should be valid");
    
    // Create network
    let network = KanNetwork::new(config.clone());
    
    // Verify network properties
    assert!(network.num_layers() > 0, "Network should have layers");
    assert!(network.param_count() > 0, "Network should have parameters");
    
    // Create workspace
    let mut workspace = network.create_workspace(8);
    
    // Single forward pass
    let inputs = vec![0.5f32; config.input_dim];
    let mut outputs = vec![0.0f32; config.output_dim];
    
    network.forward_single(&inputs, &mut outputs, &mut workspace);
    
    // Outputs should be finite (not NaN/Inf)
    for (i, &out) in outputs.iter().enumerate() {
        assert!(out.is_finite(), "Output {} should be finite, got {}", i, out);
    }
}

/// Tests batch forward pass pattern.
#[test]
fn test_basic_batch_forward() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(8);
    
    let batch_size = 4;
    let inputs = vec![0.5f32; batch_size * config.input_dim];
    let mut outputs = vec![0.0f32; batch_size * config.output_dim];
    
    network.forward_batch(&inputs, &mut outputs, &mut workspace);
    
    // All outputs should be finite
    for &out in &outputs {
        assert!(out.is_finite(), "All outputs should be finite");
    }
}

// =============================================================================
// training.rs example tests
// =============================================================================

/// Tests the training setup pattern from training.rs.
#[test]
fn test_training_example_setup() {
    // Pattern from training.rs
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![16, 16],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.5; 4],
        input_std: vec![0.3; 4],
        ..Default::default()
    };
    config.validate().expect("Config should be valid");
    
    let mut network = KanNetwork::new(config.clone());
    let batch_size = 32;
    let mut workspace = network.create_workspace(batch_size);
    
    // Adam optimizer
    let optimizer_config = AdamConfig::with_lr(0.001);
    let optimizer = Adam::new(&network, optimizer_config);
    
    assert_eq!(optimizer.learning_rate(), 0.001);
    
    // Train options
    let train_opts = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.01,
    };
    network.set_default_train_options(train_opts);
    
    // Generate simple data
    let inputs = vec![0.5f32; batch_size * config.input_dim];
    let targets = vec![0.5f32; batch_size * config.output_dim];
    
    // One training step
    let loss = network.train_step(
        &inputs,
        &targets,
        None,
        optimizer.learning_rate(),
        &mut workspace,
    );
    
    assert!(loss.is_finite(), "Training loss should be finite");
    assert!(loss >= 0.0, "MSE loss should be non-negative");
}

/// Tests training convergence on simple pattern.
#[test]
fn test_training_convergence_simple() {
    let config = KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![8],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.5; 2],
        input_std: vec![0.3; 2],
        ..Default::default()
    };
    
    let mut network = KanNetwork::new(config.clone());
    let batch_size = 16;
    let mut workspace = network.create_workspace(batch_size);
    
    // Simple target: output = mean(inputs)
    let inputs: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| (i as f32 / 32.0))
        .collect();
    
    let targets: Vec<f32> = (0..batch_size)
        .map(|i| {
            let start = i * config.input_dim;
            let end = start + config.input_dim;
            inputs[start..end].iter().sum::<f32>() / config.input_dim as f32
        })
        .collect();
    
    let lr = 0.01;
    let initial_loss = network.train_step(&inputs, &targets, None, lr, &mut workspace);
    
    // Train for a few steps
    let mut final_loss = initial_loss;
    for _ in 0..50 {
        final_loss = network.train_step(&inputs, &targets, None, lr, &mut workspace);
    }
    
    // Loss should decrease
    assert!(
        final_loss < initial_loss,
        "Loss should decrease: initial={:.6}, final={:.6}",
        initial_loss, final_loss
    );
}

// =============================================================================
// Optimizer integration tests
// =============================================================================

/// Tests Adam optimizer with network training.
#[test]
fn test_adam_optimizer_integration() {
    let config = KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![8],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.5; 2],
        input_std: vec![0.3; 2],
        ..Default::default()
    };
    
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(16);
    
    let optimizer_config = AdamConfig::with_lr(0.01);
    let optimizer = Adam::new(&network, optimizer_config);
    
    // Simple data
    let batch_size = 16;
    let inputs = vec![0.5f32; batch_size * config.input_dim];
    let targets = vec![0.5f32; batch_size * config.output_dim];
    
    // Train with optimizer learning rate
    let loss1 = network.train_step(&inputs, &targets, None, optimizer.learning_rate(), &mut workspace);
    let loss2 = network.train_step(&inputs, &targets, None, optimizer.learning_rate(), &mut workspace);
    
    // Both losses should be finite
    assert!(loss1.is_finite(), "First loss should be finite");
    assert!(loss2.is_finite(), "Second loss should be finite");
    
    // Network should still work after training
    let mut outputs = vec![0.0f32; batch_size * config.output_dim];
    network.forward_batch(&inputs, &mut outputs, &mut workspace);
    for &out in &outputs {
        assert!(out.is_finite(), "Outputs should be finite after training");
    }
}

// =============================================================================
// Config validation tests
// =============================================================================

/// Tests various config patterns.
#[test]
fn test_config_patterns() {
    // Minimal config
    let minimal = KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: 3,
        spline_order: 2,
        input_mean: vec![0.5; 1],
        input_std: vec![0.3; 1],
        ..Default::default()
    };
    minimal.validate().expect("Minimal config should be valid");
    
    // Deep config
    let deep = KanConfig {
        input_dim: 10,
        output_dim: 5,
        hidden_dims: vec![32, 32, 32, 32],
        grid_size: 8,
        spline_order: 4,
        input_mean: vec![0.5; 10],
        input_std: vec![0.3; 10],
        ..Default::default()
    };
    deep.validate().expect("Deep config should be valid");
    
    // Wide config
    let wide = KanConfig {
        input_dim: 100,
        output_dim: 50,
        hidden_dims: vec![512],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.5; 100],
        input_std: vec![0.3; 100],
        ..Default::default()
    };
    wide.validate().expect("Wide config should be valid");
}

/// Tests that invalid configs are rejected.
#[test]
fn test_invalid_configs() {
    // Zero input dim
    let invalid1 = KanConfig {
        input_dim: 0,
        output_dim: 1,
        hidden_dims: vec![8],
        ..Default::default()
    };
    assert!(invalid1.validate().is_err(), "Zero input_dim should be invalid");
    
    // Zero output dim
    let invalid2 = KanConfig {
        input_dim: 1,
        output_dim: 0,
        hidden_dims: vec![8],
        ..Default::default()
    };
    assert!(invalid2.validate().is_err(), "Zero output_dim should be invalid");
}

// =============================================================================
// Workspace tests
// =============================================================================

/// Tests workspace reuse pattern.
#[test]
fn test_workspace_reuse() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(8);
    
    // Multiple forward passes with same workspace
    for _ in 0..10 {
        let inputs = vec![0.5f32; config.input_dim];
        let mut outputs = vec![0.0f32; config.output_dim];
        
        network.forward_single(&inputs, &mut outputs, &mut workspace);
        
        // Should work every time
        assert!(outputs[0].is_finite());
    }
}

/// Tests batch size flexibility.
#[test]
fn test_variable_batch_size() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(8);
    
    // Test different batch sizes up to max
    for batch_size in [1, 2, 4, 8] {
        let inputs = vec![0.5f32; batch_size * config.input_dim];
        let mut outputs = vec![0.0f32; batch_size * config.output_dim];
        
        network.forward_batch(&inputs, &mut outputs, &mut workspace);
        
        for &out in &outputs {
            assert!(out.is_finite(), "batch_size={} should work", batch_size);
        }
    }
}

// =============================================================================
// Loss function tests
// =============================================================================

/// Tests MSE loss computation.
#[test]
fn test_mse_loss_example() {
    let config = KanConfig {
        input_dim: 2,
        output_dim: 2,
        hidden_dims: vec![4],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.5; 2],
        input_std: vec![0.3; 2],
        ..Default::default()
    };
    
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);
    
    let batch_size = 4;
    let inputs = vec![0.5f32; batch_size * config.input_dim];
    
    // Perfect match should give zero loss
    let mut outputs = vec![0.0f32; batch_size * config.output_dim];
    network.forward_batch(&inputs, &mut outputs, &mut workspace);
    
    let loss = network.train_step(&inputs, &outputs, None, 0.0, &mut workspace);
    assert!(loss < 1e-6, "Loss with targets=outputs should be ~0, got {}", loss);
}

// =============================================================================
// Parallel forward tests
// =============================================================================

/// Tests parallel batch forward.
#[test]
fn test_parallel_forward() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(64);
    
    let batch_size = 64;
    let inputs = vec![0.5f32; batch_size * config.input_dim];
    let mut outputs_seq = vec![0.0f32; batch_size * config.output_dim];
    let mut outputs_par = vec![0.0f32; batch_size * config.output_dim];
    
    // Sequential
    network.forward_batch(&inputs, &mut outputs_seq, &mut workspace);
    
    // Parallel (doesn't need workspace)
    network.forward_batch_parallel(&inputs, &mut outputs_par);
    
    // Should match
    for (i, (&seq, &par)) in outputs_seq.iter().zip(outputs_par.iter()).enumerate() {
        assert!(
            (seq - par).abs() < 1e-5,
            "Output {} mismatch: seq={}, par={}",
            i, seq, par
        );
    }
}

// =============================================================================
// Edge case tests
// =============================================================================

/// Tests network with extreme input values.
#[test]
fn test_extreme_inputs() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.5; 4],
        input_std: vec![0.3; 4],
        ..Default::default()
    };
    
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);
    
    // Test various input ranges
    let test_cases = vec![
        vec![0.0; config.input_dim],           // zeros
        vec![1.0; config.input_dim],           // ones
        vec![-1.0; config.input_dim],          // negative
        vec![10.0; config.input_dim],          // large positive
        vec![-10.0; config.input_dim],         // large negative
    ];
    
    for inputs in test_cases {
        let mut outputs = vec![0.0f32; config.output_dim];
        network.forward_single(&inputs, &mut outputs, &mut workspace);
        
        // Outputs should be finite (clamped inputs prevent NaN)
        for &out in &outputs {
            assert!(out.is_finite(), "Output should be finite for inputs {:?}", inputs);
        }
    }
}
