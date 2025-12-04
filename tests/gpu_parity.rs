//! GPU/CPU parity tests.
//!
//! These tests verify that GPU and CPU implementations produce
//! identical (or nearly identical) results.
//!
//! Run with: cargo test --features gpu --test gpu_parity -- --ignored

#![cfg(feature = "gpu")]
#![allow(unused_imports)]

use arkan::gpu::{GpuNetwork, WgpuBackend, WgpuOptions};
use arkan::optimizer::{Adam, AdamConfig, SGD};
use arkan::{KanConfig, KanNetwork, TrainOptions};

/// Tolerance for floating-point comparison.
const EPSILON: f32 = 1e-5;

/// Compares two f32 slices with tolerance.
fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
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

    assert!(
        max_diff <= tol,
        "Max difference {} at index {} exceeds tolerance {}. a[{}]={}, b[{}]={}",
        max_diff,
        max_idx,
        tol,
        max_idx,
        a[max_idx],
        max_idx,
        b[max_idx]
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(1);

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(1)
        .expect("Failed to create workspace");

    // Test input
    let input = vec![0.5, -0.3, 0.8, -0.1];

    // CPU forward
    let mut cpu_output = vec![0.0f32; config.output_dim];
    cpu_network.forward_single(&input, &mut cpu_output, &mut cpu_workspace);

    // GPU forward
    let gpu_output = gpu_network
        .forward_single(&input, &mut gpu_workspace)
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(4);

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
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
    let gpu_output = gpu_network
        .forward_batch(&input, batch_size, &mut gpu_workspace)
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network with multiple layers
    let config = multi_layer_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(8);

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(8)
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
    let gpu_output = gpu_network
        .forward_batch(&input, batch_size, &mut gpu_workspace)
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(4);

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
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
    let gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());

    // Store original weights
    let original_weights: Vec<f32> = cpu_network.layers[0].weights.to_vec();
    let original_bias: Vec<f32> = cpu_network.layers[0].bias.to_vec();

    // Create GPU network (uploads weights)
    let gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Download weights back to new CPU network
    let mut cpu_network2 = KanNetwork::new(config);
    gpu_network
        .sync_weights_to_cpu(&mut cpu_network2)
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network
    let config = simple_config();
    let mut cpu_network = KanNetwork::new(config.clone());

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
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
        let loss = gpu_network
            .train_step_mse(
                &input,
                &target,
                batch_size,
                &mut gpu_workspace,
                &mut optimizer,
                &mut cpu_network,
            )
            .expect("Train step failed");
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(64);

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(64)
        .expect("Failed to create workspace");

    // Test various batch sizes
    for batch_size in [1, 2, 3, 7, 16, 64] {
        let input: Vec<f32> = (0..batch_size * config.input_dim)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();

        let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
        cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

        let gpu_output = gpu_network
            .forward_batch(&input, batch_size, &mut gpu_workspace)
            .expect(&format!("GPU forward failed for batch_size={}", batch_size));

        assert_approx_eq(&cpu_output, &gpu_output, EPSILON);
        println!("Batch size {} passed", batch_size);
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_backward_parity() {
    // Initialize GPU backend
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create CPU network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(4);

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
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
    let grad_output: Vec<f32> = cpu_output
        .iter()
        .zip(target.iter())
        .map(|(&o, &t)| 2.0 * (o - t) / (batch_size * config.output_dim) as f32)
        .collect();

    // GPU forward training + backward
    let _gpu_output = gpu_network
        .forward_batch_training(&input, batch_size, &mut gpu_workspace)
        .expect("GPU forward training failed");

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

    // Verify gradients were computed
    assert!(!gpu_grad_weights.is_empty(), "No weight gradients");
    assert!(!gpu_grad_biases.is_empty(), "No bias gradients");

    // Bias gradients should match simple sum of grad_output
    let expected_bias_grad: Vec<f32> = (0..config.output_dim)
        .map(|j| {
            (0..batch_size)
                .map(|b| grad_output[b * config.output_dim + j])
                .sum()
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
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create two identical networks
    let config = simple_config();
    let mut cpu_network = KanNetwork::new(config.clone());
    let mut gpu_cpu_network = cpu_network.clone(); // Clone for GPU-synced version
    let mut cpu_workspace = cpu_network.create_workspace(4);

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &gpu_cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
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
    let cpu_loss = cpu_network.train_step(&input, &target, None, 0.01, &mut cpu_workspace);

    // GPU train step
    let gpu_loss = gpu_network
        .train_step_mse(
            &input,
            &target,
            batch_size,
            &mut gpu_workspace,
            &mut gpu_optimizer,
            &mut gpu_cpu_network,
        )
        .expect("GPU train step failed");

    println!("CPU loss: {}", cpu_loss);
    println!("GPU loss: {}", gpu_loss);

    // Losses should be close
    assert!(
        (cpu_loss - gpu_loss).abs() < 0.1,
        "Losses differ too much: CPU={}, GPU={}",
        cpu_loss,
        gpu_loss
    );

    // Run a few more steps
    for i in 0..5 {
        let cpu_loss = cpu_network.train_step(&input, &target, None, 0.01, &mut cpu_workspace);
        let gpu_loss = gpu_network
            .train_step_mse(
                &input,
                &target,
                batch_size,
                &mut gpu_workspace,
                &mut gpu_optimizer,
                &mut gpu_cpu_network,
            )
            .expect("GPU train step failed");

        println!("Step {}: CPU={:.6}, GPU={:.6}", i, cpu_loss, gpu_loss);
    }
}

// ==================== Error and Limits Tests ====================

#[test]
#[ignore = "Requires GPU"]
fn test_shape_mismatch_input() {
    // Test that shape mismatch errors are informative
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
        .expect("Failed to create workspace");

    // Wrong input size (3 elements instead of 4)
    let wrong_input = vec![0.5f32; 3];

    let result = gpu_network.forward_single(&wrong_input, &mut gpu_workspace);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = err.to_string();
    println!("Shape mismatch error: {}", err_str);
    assert!(
        err_str.contains("Shape mismatch"),
        "Error should indicate shape mismatch"
    );
    assert!(
        err_str.contains("expected"),
        "Error should contain 'expected'"
    );
    assert!(err_str.contains("got"), "Error should contain 'got'");
}

#[test]
#[ignore = "Requires GPU"]
fn test_shape_mismatch_target() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let mut cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
        .expect("Failed to create workspace");
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.01));

    // Correct input
    let input = vec![0.5f32; 4 * 4]; // batch=4, input_dim=4
                                     // Wrong target size
    let wrong_target = vec![0.5f32; 4 * 3]; // batch=4, but output_dim=2, not 3

    let result = gpu_network.train_step_mse(
        &input,
        &wrong_target,
        4,
        &mut gpu_workspace,
        &mut optimizer,
        &mut cpu_network,
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    println!("Target shape mismatch error: {}", err);
    assert!(err.to_string().contains("Shape mismatch"));
}

#[test]
#[ignore = "Requires GPU"]
fn test_validate_layer_weights() {
    // Test that weight validation catches oversized layers
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create a network with large dimensions to test limits
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![],
        spline_order: 3,
        grid_size: 5,
        ..Default::default()
    };
    let cpu_network = KanNetwork::new(config.clone());

    // Validate should pass for normal-sized network
    let result = backend.validate_layer_weights(2, 4, 8);
    assert!(result.is_ok(), "Normal-sized network should validate");

    // Test that validation is called during GpuNetwork creation
    let result = GpuNetwork::from_cpu(&backend, &cpu_network);
    assert!(
        result.is_ok(),
        "Creating GPU network from valid CPU network should succeed"
    );
}

#[test]
#[ignore = "Requires GPU"]
fn test_tensor_upload_download() {
    use arkan::gpu::GpuTensor;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Upload data
    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
    let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![32, 32])
        .expect("Failed to upload tensor");

    assert_eq!(tensor.shape, vec![32, 32]);
    let total_len: usize = tensor.shape.iter().product();
    assert_eq!(total_len, 1024);

    // Download data
    let downloaded = tensor
        .download(&backend.device, &backend.queue)
        .expect("Failed to download tensor");

    assert_eq!(downloaded.len(), 1024);
    assert_approx_eq(&data, &downloaded, 1e-6);

    println!("Tensor upload/download test passed");
}

#[test]
#[ignore = "Requires GPU"]
fn test_workspace_resize() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Start with small workspace
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
        .expect("Failed to create workspace");

    // Small batch should work
    let input_small = vec![0.5f32; 4 * config.input_dim];
    let result = gpu_network.forward_batch(&input_small, 4, &mut gpu_workspace);
    assert!(result.is_ok(), "Small batch should work");

    // Larger batch should trigger resize
    let input_large = vec![0.5f32; 16 * config.input_dim];
    let result = gpu_network.forward_batch(&input_large, 16, &mut gpu_workspace);
    assert!(
        result.is_ok(),
        "Larger batch should trigger resize and work"
    );

    // Verify workspace was resized
    assert!(gpu_workspace.max_batch >= 16, "Workspace should be resized");

    // Even larger batch
    let input_xl = vec![0.5f32; 64 * config.input_dim];
    let result = gpu_network.forward_batch(&input_xl, 64, &mut gpu_workspace);
    assert!(result.is_ok(), "XL batch should work after resize");

    println!("Workspace resize test passed");
}

#[test]
#[ignore = "Requires GPU"]
fn test_train_step_with_options() {
    use arkan::TrainOptions;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let mut cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
        .expect("Failed to create workspace");
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.01));

    let batch_size = 4;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let target: Vec<f32> = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.8, 0.2];

    // Test with gradient clipping
    let opts_clip = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.0,
    };

    let loss1 = gpu_network
        .train_step_with_options(
            &input,
            &target,
            None,
            batch_size,
            &mut gpu_workspace,
            &mut optimizer,
            &mut cpu_network,
            &opts_clip,
        )
        .expect("Train step with options failed");

    println!("Loss with grad clipping: {}", loss1);

    // Test with weight decay
    let opts_decay = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.01,
    };

    let loss2 = gpu_network
        .train_step_with_options(
            &input,
            &target,
            None,
            batch_size,
            &mut gpu_workspace,
            &mut optimizer,
            &mut cpu_network,
            &opts_decay,
        )
        .expect("Train step with weight decay failed");

    println!("Loss with weight decay: {}", loss2);

    // Test with both
    let opts_both = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.01,
    };

    let loss3 = gpu_network
        .train_step_with_options(
            &input,
            &target,
            None,
            batch_size,
            &mut gpu_workspace,
            &mut optimizer,
            &mut cpu_network,
            &opts_both,
        )
        .expect("Train step with both options failed");

    println!("Loss with both options: {}", loss3);
}

#[test]
#[ignore = "Requires GPU"]
fn test_train_step_sgd() {
    use arkan::optimizer::SGD;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let mut cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(4)
        .expect("Failed to create workspace");
    let mut optimizer = SGD::new(&cpu_network, 0.01, 0.9, 0.0);

    let batch_size = 4;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let target: Vec<f32> = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 0.8, 0.2];

    // Run a few training steps with SGD
    let mut losses = Vec::new();
    for _ in 0..5 {
        let loss = gpu_network
            .train_step_sgd(
                &input,
                &target,
                batch_size,
                &mut gpu_workspace,
                &mut optimizer,
                &mut cpu_network,
            )
            .expect("SGD train step failed");
        losses.push(loss);
    }

    println!("SGD training losses: {:?}", losses);
    assert!(
        losses[4] <= losses[0] * 1.5,
        "Loss should not increase significantly"
    );
}

#[test]
#[ignore = "Requires GPU"]
fn test_weight_sync_methods() {
    // Test the various weight sync method aliases
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let original_weights: Vec<f32> = cpu_network.layers[0].weights.to_vec();

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Test sync_weights_cpu_to_gpu
    let mut modified_cpu = cpu_network.clone();
    for w in modified_cpu.layers[0].weights.as_mut_slice() {
        *w *= 1.1; // Modify weights
    }

    gpu_network
        .sync_weights_cpu_to_gpu(&modified_cpu)
        .expect("sync_weights_cpu_to_gpu failed");

    // Test sync_weights_gpu_to_cpu
    let mut recovered_cpu = KanNetwork::new(config.clone());
    gpu_network
        .sync_weights_gpu_to_cpu(&mut recovered_cpu)
        .expect("sync_weights_gpu_to_cpu failed");

    // Verify the modified weights came back
    let recovered_weights: Vec<f32> = recovered_cpu.layers[0].weights.to_vec();
    let modified_weights: Vec<f32> = modified_cpu.layers[0].weights.to_vec();

    assert_approx_eq(&recovered_weights, &modified_weights, 1e-5);

    // Make sure they're different from original
    let diff: f32 = original_weights
        .iter()
        .zip(recovered_weights.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.01,
        "Weights should be different from original after modification"
    );

    println!("Weight sync methods test passed");
}

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_softmax() {
    // Test that GPU softmax produces valid probability distribution
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());

    // Create GPU network
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network
        .create_workspace(4)
        .expect("Failed to create workspace");

    // Test input (batch of 4)
    let input: Vec<f32> = (0..config.input_dim * 4)
        .map(|i| (i as f32 * 0.1) - 0.5)
        .collect();

    // Forward with softmax
    let probs = gpu_network
        .forward_batch_softmax(&input, 4, &mut workspace)
        .expect("GPU forward_batch_softmax failed");

    println!(
        "Softmax output (batch 4, dim {}): {:?}",
        config.output_dim, probs
    );

    // Verify each sample sums to ~1.0
    for batch_idx in 0..4 {
        let start = batch_idx * config.output_dim;
        let end = start + config.output_dim;
        let sample_probs = &probs[start..end];

        let sum: f32 = sample_probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Sample {} probabilities sum to {} (expected ~1.0)",
            batch_idx,
            sum
        );

        // Verify all values are in [0, 1]
        for (i, &p) in sample_probs.iter().enumerate() {
            assert!(
                p >= 0.0 && p <= 1.0,
                "Sample {} probability[{}] = {} not in [0,1]",
                batch_idx,
                i,
                p
            );
        }
    }

    println!("GPU softmax test passed");
}

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_softmax_vs_cpu() {
    // Compare GPU softmax with manual CPU softmax
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(2);

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(2)
        .expect("Failed to create workspace");

    // Test input
    let input: Vec<f32> = vec![0.5, -0.3, 0.8, -0.1, 0.2, 0.4, -0.5, 0.6];

    // GPU forward with softmax
    let gpu_probs = gpu_network
        .forward_batch_softmax(&input, 2, &mut gpu_workspace)
        .expect("GPU forward_batch_softmax failed");

    // CPU forward without softmax
    let mut cpu_logits = vec![0.0f32; config.output_dim * 2];
    cpu_network.forward_batch(&input, &mut cpu_logits, &mut cpu_workspace);

    // Manual CPU softmax
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|x| x / sum).collect()
    }

    let mut cpu_probs = Vec::new();
    for batch in 0..2 {
        let start = batch * config.output_dim;
        let end = start + config.output_dim;
        cpu_probs.extend(softmax(&cpu_logits[start..end]));
    }

    println!("GPU softmax: {:?}", gpu_probs);
    println!("CPU softmax: {:?}", cpu_probs);

    // Compare with tolerance
    assert_approx_eq(&gpu_probs, &cpu_probs, 1e-4);

    println!("GPU vs CPU softmax test passed");
}

// ==================== Native GPU Training Tests ====================

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_native_training_adam() {
    use arkan::gpu::{GpuAdam, GpuAdamConfig};

    // Initialize GPU backend
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(16)
        .expect("Failed to create workspace");

    // Create GPU Adam optimizer
    let layer_sizes = gpu_network.layer_param_sizes();
    let mut optimizer = GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.01),
    );

    // Generate training data
    let batch_size = 16;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32 * 0.1).sin()) * 0.5)
        .collect();
    let target: Vec<f32> = (0..batch_size * config.output_dim)
        .map(|i| ((i as f32 * 0.2).cos()) * 0.5)
        .collect();

    // Run several training steps
    let mut losses = Vec::new();
    for _step in 0..10 {
        let loss = gpu_network
            .train_step_gpu_native(&input, &target, batch_size, &mut gpu_workspace, &mut optimizer)
            .expect("Native GPU training step failed");
        losses.push(loss);
    }

    // Verify loss is decreasing (at least not increasing dramatically)
    println!("Native GPU training losses: {:?}", losses);
    assert!(
        losses.len() == 10,
        "Expected 10 training steps, got {}",
        losses.len()
    );

    // Loss should generally decrease or stay stable
    let first_loss = losses[0];
    let last_loss = losses[losses.len() - 1];
    println!(
        "First loss: {:.6}, Last loss: {:.6}, Change: {:.2}%",
        first_loss,
        last_loss,
        (last_loss - first_loss) / first_loss * 100.0
    );

    // Not strictly required to decrease (depends on data), but shouldn't explode
    assert!(
        last_loss < first_loss * 10.0,
        "Loss exploded: {} -> {}",
        first_loss,
        last_loss
    );

    println!("Native GPU Adam training test passed");
}

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_native_training_sgd() {
    use arkan::gpu::{GpuSgd, GpuSgdConfig};

    // Initialize GPU backend
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create network
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(8)
        .expect("Failed to create workspace");

    // Create GPU SGD optimizer
    let layer_sizes = gpu_network.layer_param_sizes();
    let mut optimizer = GpuSgd::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuSgdConfig::with_lr(0.1),
    );

    // Generate training data
    let batch_size = 8;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32 * 0.1).sin()) * 0.5)
        .collect();
    let target: Vec<f32> = (0..batch_size * config.output_dim)
        .map(|i| ((i as f32 * 0.2).cos()) * 0.5)
        .collect();

    // Run training steps
    for step in 0..5 {
        let loss = gpu_network
            .train_step_gpu_native_sgd(
                &input,
                &target,
                batch_size,
                &mut gpu_workspace,
                &mut optimizer,
            )
            .expect("Native GPU SGD training step failed");
        println!("SGD Step {}: loss = {:.6}", step, loss);
    }

    println!("Native GPU SGD training test passed");
}

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_native_training_multi_layer() {
    use arkan::gpu::{GpuAdam, GpuAdamConfig};

    // Initialize GPU backend
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create multi-layer network
    let config = multi_layer_config();
    let cpu_network = KanNetwork::new(config.clone());

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(32)
        .expect("Failed to create workspace");

    // Create optimizer
    let layer_sizes = gpu_network.layer_param_sizes();
    println!("Layer sizes for optimizer: {:?}", layer_sizes);

    let mut optimizer = GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.001),
    );

    // Training data
    let batch_size = 32;
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| ((i as f32 * 0.05).sin()) * 0.5)
        .collect();
    let target: Vec<f32> = (0..batch_size * config.output_dim)
        .map(|i| ((i as f32 * 0.1).cos()) * 0.3 + 0.5)
        .collect();

    // Train for several epochs
    let mut prev_loss = f32::MAX;
    for epoch in 0..20 {
        let loss = gpu_network
            .train_step_gpu_native(&input, &target, batch_size, &mut gpu_workspace, &mut optimizer)
            .expect("Multi-layer GPU training failed");

        if epoch % 5 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
        prev_loss = loss;
    }

    println!("Final loss: {:.6}", prev_loss);
    println!("Multi-layer native GPU training test passed");
}

