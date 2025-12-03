//! GPU/CPU parity tests.
//!
//! These tests verify that GPU and CPU implementations produce
//! identical (or nearly identical) results.
//!
//! Run with: cargo test --features gpu --test gpu_parity -- --ignored

#![cfg(feature = "gpu")]

use arkan::{KanConfig, KanNetwork};
use arkan::gpu::{WgpuBackend, WgpuOptions, GpuNetwork};
use arkan::optimizer::{Adam, AdamConfig};

/// Tolerance for floating-point comparison.
const EPSILON: f32 = 1e-5;

/// Compares two f32 slices with tolerance.
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "Length mismatch: {} vs {}", a.len(), b.len());
    
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    
    assert!(
        max_diff <= tol,
        "Max difference {} at index {} exceeds tolerance {}. a[{}]={}, b[{}]={}",
        max_diff, max_idx, tol, max_idx, a[max_idx], max_idx, b[max_idx]
    );
}

/// Helper to create a simple KAN config.
fn simple_config() -> KanConfig {
    KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    }
}

/// Helper to create multi-layer config.
fn multi_layer_config() -> KanConfig {
    KanConfig {
        input_dim: 8,
        output_dim: 3,
        hidden_dims: vec![16, 8],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 8],
        input_std: vec![1.0; 8],
        ..Default::default()
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_forward_single_parity() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(1);
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(1)
        .expect("Failed to create workspace");
    
    // Test input
    let input = vec![0.5, -0.3, 0.8, -0.1];
    
    // CPU forward
    let mut cpu_output = vec![0.0f32; config.output_dim];
    cpu_network.forward_single(&input, &mut cpu_output, &mut cpu_workspace);
    
    // GPU forward
    let gpu_output = gpu_network.forward_single(&input, &mut gpu_workspace)
        .expect("GPU forward failed");
    
    // Compare results
    println!("CPU output: {:?}", cpu_output);
    println!("GPU output: {:?}", gpu_output);
    
    assert_approx_eq(&cpu_output, &gpu_output, EPSILON);
}

#[test]
#[ignore = "Requires GPU"]
fn test_forward_batch_parity() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(4);
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(4)
        .expect("Failed to create workspace");
    
    // Test batch
    let batch_size = 4;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    
    // CPU forward batch
    let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
    
    // GPU forward batch
    let gpu_output = gpu_network.forward_batch(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward failed");
    
    // Compare
    println!("CPU output: {:?}", cpu_output);
    println!("GPU output: {:?}", gpu_output);
    
    assert_approx_eq(&cpu_output, &gpu_output, EPSILON);
}

#[test]
#[ignore = "Requires GPU"]
fn test_multi_layer_forward_parity() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network with multiple layers
    let config = multi_layer_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(8);
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(8)
        .expect("Failed to create workspace");
    
    // Test batch
    let batch_size = 8;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    
    // CPU forward
    let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
    
    // GPU forward
    let gpu_output = gpu_network.forward_batch(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward failed");
    
    // Compare (allow slightly higher tolerance for multi-layer)
    println!("CPU output: {:?}", cpu_output);
    println!("GPU output: {:?}", gpu_output);
    
    assert_approx_eq(&cpu_output, &gpu_output, 1e-4);
}

#[test]
#[ignore = "Requires GPU"]
fn test_forward_training_parity() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(4);
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(4)
        .expect("Failed to create workspace");
    
    // Test batch
    let batch_size = 4;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    
    // CPU forward (training mode has same output as inference for forward)
    let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
    
    // GPU forward training
    let gpu_output = gpu_network.forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");
    
    // Compare
    println!("CPU output: {:?}", cpu_output);
    println!("GPU output (training): {:?}", gpu_output);
    
    assert_approx_eq(&cpu_output, &gpu_output, EPSILON);
}

#[test]
#[ignore = "Requires GPU"]
fn test_weight_sync_roundtrip() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    
    // Store original weights
    let original_weights: Vec<f32> = cpu_network.layers[0].weights.to_vec();
    let original_bias: Vec<f32> = cpu_network.layers[0].bias.to_vec();
    
    // Create GPU network (uploads weights)
    let gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    
    // Download weights back to new CPU network
    let mut cpu_network2 = KanNetwork::new(config);
    gpu_network.sync_weights_to_cpu(&mut cpu_network2)
        .expect("Failed to sync weights back");
    
    // Compare weights
    let roundtrip_weights: Vec<f32> = cpu_network2.layers[0].weights.to_vec();
    let roundtrip_bias: Vec<f32> = cpu_network2.layers[0].bias.to_vec();
    
    assert_approx_eq(&original_weights, &roundtrip_weights, 1e-6);
    assert_approx_eq(&original_bias, &roundtrip_bias, 1e-6);
}

#[test]
#[ignore = "Requires GPU"]
fn test_train_step_runs() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network
    let config = simple_config();
    let mut cpu_network = KanNetwork::new(config.clone());
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(4)
        .expect("Failed to create workspace");
    
    // Create optimizer
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));
    
    // Test data
    let batch_size = 4;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let target: Vec<f32> = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.8, 0.2];
    
    // Run a few training steps
    let mut losses = Vec::new();
    for _ in 0..5 {
        let loss = gpu_network.train_step_mse(
            &input,
            &target,
            batch_size,
            &mut gpu_workspace,
            &mut optimizer,
            &mut cpu_network,
        ).expect("Train step failed");
        losses.push(loss);
    }
    
    println!("Training losses: {:?}", losses);
    
    // Loss should generally decrease (not guaranteed but typical)
    assert!(losses[4] <= losses[0] * 1.5, "Loss increased significantly");
}

#[test]
#[ignore = "Requires GPU"]
fn test_batch_size_edge_cases() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(64);
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(64)
        .expect("Failed to create workspace");
    
    // Test various batch sizes
    for batch_size in [1, 2, 3, 7, 16, 64] {
        let input: Vec<f32> = (0..batch_size * config.input_dim)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        
        let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
        cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
        
        let gpu_output = gpu_network.forward_batch(&input, batch_size, &mut gpu_workspace)
            .expect(&format!("GPU forward failed for batch_size={}", batch_size));
        
        assert_approx_eq(&cpu_output, &gpu_output, EPSILON);
        println!("Batch size {} passed", batch_size);
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_backward_parity() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(4);
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(4)
        .expect("Failed to create workspace");
    
    // Test batch
    let batch_size = 4;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let target: Vec<f32> = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.8, 0.2];
    
    // CPU forward
    let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
    
    // Compute MSE gradient
    let grad_output: Vec<f32> = cpu_output.iter()
        .zip(target.iter())
        .map(|(&o, &t)| 2.0 * (o - t) / (batch_size * config.output_dim) as f32)
        .collect();
    
    // GPU forward training + backward
    let _gpu_output = gpu_network.forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");
    
    let mut gpu_grad_weights = Vec::new();
    let mut gpu_grad_biases = Vec::new();
    let _gpu_grad_input = gpu_network.backward_batch(
        &grad_output,
        batch_size,
        &mut gpu_workspace,
        &mut gpu_grad_weights,
        &mut gpu_grad_biases,
    ).expect("GPU backward failed");
    
    // Verify gradients were computed
    assert!(!gpu_grad_weights.is_empty(), "No weight gradients");
    assert!(!gpu_grad_biases.is_empty(), "No bias gradients");
    
    // Bias gradients should match simple sum of grad_output
    let expected_bias_grad: Vec<f32> = (0..config.output_dim)
        .map(|j| {
            (0..batch_size).map(|b| grad_output[b * config.output_dim + j]).sum()
        })
        .collect();
    
    println!("Expected grad_bias: {:?}", expected_bias_grad);
    println!("GPU grad_bias: {:?}", gpu_grad_biases[0]);
    assert_approx_eq(&expected_bias_grad, &gpu_grad_biases[0], 1e-4);
    
    println!("GPU grad_weights len: {}", gpu_grad_weights[0].len());
    println!("Backward test passed!");
}

#[test]
#[ignore = "Requires GPU"]
fn test_train_step_parity() {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())
        .expect("Failed to initialize GPU backend");
    
    // Create two identical networks
    let config = simple_config();
    let mut cpu_network = KanNetwork::new(config.clone());
    let mut gpu_cpu_network = cpu_network.clone(); // Clone for GPU-synced version
    let mut cpu_workspace = cpu_network.create_workspace(4);
    
    // Create GPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &gpu_cpu_network)
        .expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(4)
        .expect("Failed to create workspace");
    
    // Create optimizer for GPU
    let mut gpu_optimizer = Adam::new(&gpu_cpu_network, AdamConfig::with_lr(0.01));
    
    // Test data
    let batch_size = 4;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let target: Vec<f32> = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.8, 0.2];
    
    // CPU train step  
    let cpu_loss = cpu_network.train_step(
        &input, &target, None, 0.01, &mut cpu_workspace
    );
    
    // GPU train step
    let gpu_loss = gpu_network.train_step_mse(
        &input,
        &target,
        batch_size,
        &mut gpu_workspace,
        &mut gpu_optimizer,
        &mut gpu_cpu_network,
    ).expect("GPU train step failed");
    
    println!("CPU loss: {}", cpu_loss);
    println!("GPU loss: {}", gpu_loss);
    
    // Losses should be close
    assert!((cpu_loss - gpu_loss).abs() < 0.1, "Losses differ too much: CPU={}, GPU={}", cpu_loss, gpu_loss);
    
    // Run a few more steps
    for i in 0..5 {
        let cpu_loss = cpu_network.train_step(
            &input, &target, None, 0.01, &mut cpu_workspace
        );
        let gpu_loss = gpu_network.train_step_mse(
            &input,
            &target,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_optimizer,
            &mut gpu_cpu_network,
        ).expect("GPU train step failed");
        
        println!("Step {}: CPU={:.6}, GPU={:.6}", i, cpu_loss, gpu_loss);
    }
}
