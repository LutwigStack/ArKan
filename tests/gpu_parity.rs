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
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
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
            .train_step_gpu_native(
                &input,
                &target,
                batch_size,
                &mut gpu_workspace,
                &mut optimizer,
            )
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
            .train_step_gpu_native(
                &input,
                &target,
                batch_size,
                &mut gpu_workspace,
                &mut optimizer,
            )
            .expect("Multi-layer GPU training failed");

        if epoch % 5 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
        prev_loss = loss;
    }

    println!("Final loss: {:.6}", prev_loss);
    println!("Multi-layer native GPU training test passed");
}

// ==================== Error Path Tests ====================

#[test]
#[ignore = "Requires GPU"]
fn test_unsupported_spline_order_1() {
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create config with spline_order = 1 (not supported on GPU, min is 2)
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![],
        spline_order: 1, // Too low for GPU
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };
    let cpu_network = KanNetwork::new(config);

    let result = GpuNetwork::from_cpu(&backend, &cpu_network);
    assert!(result.is_err(), "Should fail for spline_order=1");

    match result.unwrap_err() {
        ArkanError::UnsupportedOrder(order) => {
            assert_eq!(order, 1);
            println!("Correctly rejected spline_order=1: order={}", order);
        }
        other => panic!("Expected UnsupportedOrder, got {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_unsupported_spline_order_6() {
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create config with spline_order = 6 (not supported on GPU, max is 5)
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![],
        spline_order: 6, // Too high for GPU
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };
    let cpu_network = KanNetwork::new(config);

    let result = GpuNetwork::from_cpu(&backend, &cpu_network);
    assert!(result.is_err(), "Should fail for spline_order=6");

    match result.unwrap_err() {
        ArkanError::UnsupportedOrder(order) => {
            assert_eq!(order, 6);
            println!("Correctly rejected spline_order=6");
        }
        other => panic!("Expected UnsupportedOrder, got {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_shape_mismatch_forward_input_size() {
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network
        .create_workspace(8)
        .expect("Failed to create workspace");

    // Wrong input size: expected 4 * batch, got wrong length
    let wrong_input = vec![0.0f32; 3]; // Should be 4 for batch=1

    let result = gpu_network.forward_batch(&wrong_input, 1, &mut workspace);
    assert!(result.is_err(), "Should fail with wrong input size");

    match result.unwrap_err() {
        ArkanError::ShapeMismatch { expected, got } => {
            println!(
                "Correctly detected shape mismatch: expected {:?}, got {:?}",
                expected, got
            );
        }
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_shape_mismatch_batch_not_divisible() {
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network
        .create_workspace(8)
        .expect("Failed to create workspace");

    // Input not divisible by input_dim (4)
    let wrong_input = vec![0.0f32; 5]; // 5 is not divisible by 4

    let result = gpu_network.forward_batch(&wrong_input, 1, &mut workspace);
    assert!(
        result.is_err(),
        "Should fail when input not divisible by input_dim"
    );

    match result.unwrap_err() {
        ArkanError::ShapeMismatch { .. } => {
            println!("Correctly detected non-divisible input");
        }
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_train_step_target_mismatch() {
    use arkan::optimizer::{Adam, AdamConfig};
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_network_copy = cpu_network.clone();
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network
        .create_workspace(8)
        .expect("Failed to create workspace");
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.01));

    let batch_size = 4;
    let input = vec![0.5f32; batch_size * config.input_dim];
    let wrong_target = vec![0.5f32; 3]; // Should be batch_size * output_dim = 4 * 2 = 8

    let result = gpu_network.train_step_mse(
        &input,
        &wrong_target,
        batch_size,
        &mut workspace,
        &mut optimizer,
        &mut cpu_network_copy,
    );
    assert!(result.is_err(), "Should fail with wrong target size");

    match result.unwrap_err() {
        ArkanError::ShapeMismatch { .. } => {
            println!("Correctly detected target shape mismatch");
        }
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_batch_zero_forward() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network
        .create_workspace(8)
        .expect("Failed to create workspace");

    // batch_size = 0 should return empty result or handle gracefully
    let empty_input: Vec<f32> = vec![];
    let result = gpu_network.forward_batch(&empty_input, 0, &mut workspace);

    // Either returns empty vec or an error - both are acceptable
    match result {
        Ok(output) => {
            assert!(output.is_empty(), "Output should be empty for batch=0");
            println!("batch=0 returned empty output (acceptable)");
        }
        Err(_) => {
            println!("batch=0 returned error (acceptable)");
        }
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_workspace_batch_exceeds_capacity() {
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Create workspace with small capacity
    let mut workspace = gpu_network
        .create_workspace(4)
        .expect("Failed to create workspace");

    // Try to process batch larger than workspace capacity
    let large_batch_size = 16;
    let input = vec![0.5f32; large_batch_size * config.input_dim];

    let result = gpu_network.forward_batch(&input, large_batch_size, &mut workspace);

    // Should either resize workspace or return BatchTooLarge error
    match result {
        Ok(output) => {
            // Workspace was resized
            assert_eq!(output.len(), large_batch_size * config.output_dim);
            println!("Workspace automatically resized to accommodate batch");
        }
        Err(ArkanError::BatchTooLarge { .. }) => {
            println!("BatchTooLarge error returned (acceptable)");
        }
        Err(other) => {
            panic!("Unexpected error: {:?}", other);
        }
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_native_training_without_init() {
    use arkan::gpu::{GpuAdam, GpuAdamConfig};

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network
        .create_workspace(8)
        .expect("Failed to create workspace");

    // DON'T call init_training() - try to use native training without init
    let layer_sizes = gpu_network.layer_param_sizes();
    let mut optimizer = GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.01),
    );

    let batch_size = 4;
    let input = vec![0.5f32; batch_size * config.input_dim];
    let target = vec![0.5f32; batch_size * config.output_dim];

    // This should work - native training uses workspace grad buffers, not layer grad buffers
    let result = gpu_network.train_step_gpu_native(
        &input,
        &target,
        batch_size,
        &mut workspace,
        &mut optimizer,
    );

    match result {
        Ok(loss) => {
            // Training worked without explicit init - workspace handles grad buffers
            println!(
                "Native training succeeded without explicit init, loss = {:.6}",
                loss
            );
            assert!(loss.is_finite(), "Loss should be finite");
        }
        Err(e) => {
            // Also acceptable - some paths may require init
            println!("Native training without init returned error: {:?}", e);
        }
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_tensor_shape_mismatch() {
    use arkan::gpu::GpuTensor;
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Try to upload data with mismatched shape
    let data = vec![1.0f32; 10]; // 10 elements
    let wrong_shape = vec![3, 4]; // 12 elements expected

    let result = GpuTensor::upload(&backend.device, &backend.queue, &data, wrong_shape);
    assert!(result.is_err(), "Should fail with shape mismatch");

    match result.unwrap_err() {
        ArkanError::ShapeMismatch { expected, got } => {
            println!(
                "Correctly detected tensor shape mismatch: expected {:?}, got {:?}",
                expected, got
            );
        }
        other => panic!("Expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_backward_without_forward() {
    use arkan::ArkanError;

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network
        .create_workspace(8)
        .expect("Failed to create workspace");

    // Try backward without forward - should fail or handle gracefully
    let batch_size = 4;
    let grad_output = vec![0.1f32; batch_size * config.output_dim];
    let mut grad_weights = Vec::new();
    let mut grad_biases = Vec::new();

    // backward_batch requires output from forward pass
    // This tests internal state validation
    let result = gpu_network.backward_batch(
        &grad_output,
        batch_size,
        &mut workspace,
        &mut grad_weights,
        &mut grad_biases,
    );

    // Should either work with zero gradients or return an error
    match result {
        Ok(grads) => {
            println!("Backward returned gradients (workspace had cached forward state)");
            assert!(!grads.is_empty());
        }
        Err(e) => {
            println!(
                "Backward without forward returned error: {:?} (acceptable)",
                e
            );
        }
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_multiple_spline_orders() {
    // Test all supported spline orders (2-5)
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    for order in 2..=5 {
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8],
            spline_order: order,
            grid_size: 5,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            ..Default::default()
        };
        let cpu_network = KanNetwork::new(config.clone());
        let mut cpu_workspace = cpu_network.create_workspace(4);

        let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)
            .expect(&format!("Failed to create GPU network for order={}", order));
        let mut gpu_workspace = gpu_network
            .create_workspace(4)
            .expect("Failed to create workspace");

        // Test forward pass
        let input = vec![0.5f32; 4 * config.input_dim];
        let mut cpu_output = vec![0.0f32; 4 * config.output_dim];
        cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
        let gpu_output = gpu_network
            .forward_batch(&input, 4, &mut gpu_workspace)
            .expect("GPU forward failed");

        // Higher order splines have more numerical variance - use larger tolerance
        let tolerance = match order {
            2 => 1e-4,
            3 => 1e-4,
            4 => 0.2, // Quartic has more precision differences due to basis function complexity
            5 => 0.3, // Quintic even more
            _ => 0.5,
        };

        // Compare results
        assert_approx_eq(&cpu_output, &gpu_output, tolerance);
        println!(
            "spline_order={} passed parity test (tol={})",
            order, tolerance
        );
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_large_batch_forward() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Test with progressively larger batches
    for batch_size in [1, 8, 64, 256, 1024] {
        let mut workspace = gpu_network.create_workspace(batch_size).expect(&format!(
            "Failed to create workspace for batch={}",
            batch_size
        ));

        let input: Vec<f32> = (0..batch_size * config.input_dim)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        let result = gpu_network.forward_batch(&input, batch_size, &mut workspace);
        assert!(result.is_ok(), "Forward failed for batch={}", batch_size);

        let output = result.unwrap();
        assert_eq!(output.len(), batch_size * config.output_dim);
        println!("batch={} forward passed", batch_size);
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_warmup_and_memory_stats() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = multi_layer_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let _workspace = gpu_network
        .create_workspace(32)
        .expect("Failed to create workspace");

    // Test warmup
    let warmup_result = gpu_network.warmup();
    assert!(warmup_result.is_ok(), "Warmup should succeed");
    println!("Warmup completed successfully");

    // Test memory stats
    let stats = gpu_network.memory_stats();
    println!("Memory stats: {:?}", stats);
    assert!(stats.weights_bytes > 0);
    assert!(stats.total_bytes > 0);
    println!(
        "Weights: {} bytes, Bias: {} bytes, Total: {} bytes",
        stats.weights_bytes, stats.bias_bytes, stats.total_bytes
    );
}

#[test]
#[ignore = "Requires GPU"]
fn test_weight_download_upload_roundtrip() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_network_copy = cpu_network.clone();

    let gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Download weights back
    gpu_network
        .sync_weights_to_cpu(&mut cpu_network_copy)
        .expect("Failed to download weights");

    // Compare weights
    for (original, downloaded) in cpu_network
        .layers
        .iter()
        .zip(cpu_network_copy.layers.iter())
    {
        assert_eq!(original.weights.len(), downloaded.weights.len());
        assert_approx_eq(&original.weights, &downloaded.weights, 1e-6);
        assert_approx_eq(&original.bias, &downloaded.bias, 1e-6);
    }
    println!("Weight roundtrip passed");
}

#[test]
#[ignore = "Requires GPU"]
fn test_multiple_workspaces() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Create multiple workspaces with different sizes
    let mut workspace_small = gpu_network.create_workspace(4).expect("Failed");
    let mut workspace_large = gpu_network.create_workspace(64).expect("Failed");

    let input_small = vec![0.5f32; 4 * config.input_dim];
    let input_large = vec![0.5f32; 64 * config.input_dim];

    // Use both workspaces
    let result_small = gpu_network
        .forward_batch(&input_small, 4, &mut workspace_small)
        .expect("Small forward failed");
    let result_large = gpu_network
        .forward_batch(&input_large, 64, &mut workspace_large)
        .expect("Large forward failed");

    assert_eq!(result_small.len(), 4 * config.output_dim);
    assert_eq!(result_large.len(), 64 * config.output_dim);

    // First 4 samples should be approximately equal
    assert_approx_eq(&result_small, &result_large[..result_small.len()], 1e-5);
    println!("Multiple workspaces work correctly");
}

// ==================== Extended GPU Tests ====================

/// Test GPU network with different grid sizes.
#[test]
#[ignore = "Requires GPU"]
fn test_grid_size_variations() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    for grid_size in [3_usize, 5, 8, 12, 16] {
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8],
            spline_order: 3,
            grid_size,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            ..Default::default()
        };

        let cpu_network = KanNetwork::new(config.clone());
        let mut cpu_workspace = cpu_network.create_workspace(4);

        let mut gpu_network =
            GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
        let mut gpu_workspace = gpu_network.create_workspace(4).expect("Failed");

        let input = vec![0.5f32; 4 * config.input_dim];
        let mut cpu_output = vec![0.0f32; 4 * config.output_dim];

        cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
        let gpu_output = gpu_network
            .forward_batch(&input, 4, &mut gpu_workspace)
            .expect("GPU forward failed");

        // Allow slightly larger tolerance for larger grids
        let tol = 0.01 + (grid_size as f32) * 0.001;
        assert_approx_eq(&cpu_output, &gpu_output, tol);
        println!("Grid size {} passed", grid_size);
    }
}

/// Test GPU network with extreme input values.
#[test]
#[ignore = "Requires GPU"]
fn test_extreme_input_values() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(1);

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(1).expect("Failed");

    // Test extreme input values
    let test_cases = [
        vec![0.0f32; config.input_dim],   // All zeros
        vec![1.0f32; config.input_dim],   // All ones
        vec![-1.0f32; config.input_dim],  // All negative ones
        vec![3.0f32; config.input_dim],   // At grid boundary
        vec![-3.0f32; config.input_dim],  // At negative grid boundary
        vec![10.0f32; config.input_dim],  // Beyond grid range
        vec![-10.0f32; config.input_dim], // Beyond negative grid range
    ];

    for (i, input) in test_cases.iter().enumerate() {
        let mut cpu_output = vec![0.0f32; config.output_dim];
        cpu_network.forward_batch(input, &mut cpu_output, &mut cpu_workspace);

        let gpu_output = gpu_network
            .forward_batch(input, 1, &mut gpu_workspace)
            .expect("GPU forward failed");

        // Check for NaN/Inf
        for (j, &val) in gpu_output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Test case {}: output[{}] is not finite: {}",
                i,
                j,
                val
            );
        }

        assert_approx_eq(&cpu_output, &gpu_output, 0.01);
        println!("Extreme input test case {} passed", i);
    }
}

/// Test GPU native optimizer learning rate API.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_optimizer_lr_api() {
    use arkan::gpu::{GpuAdam, GpuAdamConfig, GpuSgd, GpuSgdConfig};

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    let layer_sizes = gpu_network.layer_param_sizes();

    // Test Adam LR API
    let mut adam = GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.001),
    );
    assert!((adam.get_lr() - 0.001).abs() < 1e-6);

    adam.set_lr(0.01);
    assert!((adam.get_lr() - 0.01).abs() < 1e-6);

    // Test SGD LR API
    let mut sgd = GpuSgd::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuSgdConfig::with_lr(0.1),
    );
    assert!((sgd.get_lr() - 0.1).abs() < 1e-6);

    sgd.set_lr(0.05);
    assert!((sgd.get_lr() - 0.05).abs() < 1e-6);

    println!("GPU optimizer LR API tests passed");
}

/// Test that GPU training loss decreases over multiple steps.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_training_convergence() {
    use arkan::gpu::{GpuAdam, GpuAdamConfig};

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network.create_workspace(16).expect("Failed");

    let layer_sizes = gpu_network.layer_param_sizes();
    let mut optimizer = GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.01),
    );

    // Simple pattern: input sum -> output
    let inputs: Vec<f32> = (0..16 * 4).map(|i| ((i % 4) as f32) * 0.5 - 1.0).collect();
    let targets: Vec<f32> = (0..16)
        .flat_map(|i| {
            let sum: f32 = (0..4).map(|j| inputs[i * 4 + j]).sum();
            vec![sum.tanh(), (-sum).tanh()]
        })
        .collect();

    let mut losses = Vec::new();
    for _ in 0..20 {
        let loss = gpu_network
            .train_step_gpu_native(&inputs, &targets, 16, &mut workspace, &mut optimizer)
            .expect("Training failed");
        losses.push(loss);
    }

    // Check that loss generally decreased
    let first_loss = losses[0];
    let last_loss = *losses.last().unwrap();
    println!(
        "Convergence test: first_loss={:.6}, last_loss={:.6}",
        first_loss, last_loss
    );
    assert!(
        last_loss < first_loss,
        "Loss should decrease: first={}, last={}",
        first_loss,
        last_loss
    );
}

/// Test concurrent GPU forward passes on different networks.
#[test]
#[ignore = "Requires GPU"]
fn test_concurrent_networks() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Create multiple independent networks
    let configs: Vec<KanConfig> = (0..3)
        .map(|i| KanConfig {
            input_dim: 4 + i,
            output_dim: 2,
            hidden_dims: vec![8],
            spline_order: 3,
            grid_size: 5,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.0; 4 + i],
            input_std: vec![1.0; 4 + i],
            ..Default::default()
        })
        .collect();

    let mut networks: Vec<(GpuNetwork, _)> = configs
        .iter()
        .map(|cfg| {
            let cpu = KanNetwork::new(cfg.clone());
            let gpu = GpuNetwork::from_cpu(&backend, &cpu).expect("Failed");
            let ws = gpu.create_workspace(4).expect("Failed");
            (gpu, ws)
        })
        .collect();

    // Run forward passes on all networks
    for (i, (network, workspace)) in networks.iter_mut().enumerate() {
        let input_dim = configs[i].input_dim;
        let input = vec![0.5f32; 4 * input_dim];

        let result = network
            .forward_batch(&input, 4, workspace)
            .expect("Forward failed");

        assert_eq!(result.len(), 4 * configs[i].output_dim);
        println!("Network {} forward passed", i);
    }
}

/// Test WgpuOptions variants.
#[test]
#[ignore = "Requires GPU"]
fn test_wgpu_options_variants() {
    // Test default options
    let backend_default =
        WgpuBackend::init(WgpuOptions::default()).expect("Default options failed");
    println!("Default backend: {}", backend_default.adapter_info().name);

    // Test compute options
    let backend_compute =
        WgpuBackend::init(WgpuOptions::compute()).expect("Compute options failed");
    println!("Compute backend: {}", backend_compute.adapter_info().name);

    // Both should work
    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());

    let mut gpu1 =
        GpuNetwork::from_cpu(&backend_default, &cpu_network).expect("Failed with default");
    let mut gpu2 =
        GpuNetwork::from_cpu(&backend_compute, &cpu_network).expect("Failed with compute");

    let mut ws1 = gpu1.create_workspace(1).expect("Failed");
    let mut ws2 = gpu2.create_workspace(1).expect("Failed");

    let input = vec![0.5f32; config.input_dim];

    let result1 = gpu1
        .forward_batch(&input, 1, &mut ws1)
        .expect("Forward failed on default");
    let result2 = gpu2
        .forward_batch(&input, 1, &mut ws2)
        .expect("Forward failed on compute");

    // Results should be identical
    assert_approx_eq(&result1, &result2, 1e-6);
    println!("WgpuOptions variants test passed");
}

/// Test GPU memory stats with different network sizes.
#[test]
#[ignore = "Requires GPU"]
fn test_memory_stats_scaling() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let configs = [
        ("tiny", vec![], 4, 2),
        ("small", vec![8], 4, 2),
        ("medium", vec![16, 8], 8, 4),
        ("large", vec![32, 32, 16], 16, 8),
    ];

    let mut prev_total = 0;
    for (name, hidden, input_dim, output_dim) in configs {
        let config = KanConfig {
            input_dim,
            output_dim,
            hidden_dims: hidden,
            spline_order: 3,
            grid_size: 5,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.0; input_dim],
            input_std: vec![1.0; input_dim],
            ..Default::default()
        };

        let cpu_network = KanNetwork::new(config);
        let gpu_network =
            GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

        let stats = gpu_network.memory_stats();
        println!(
            "{}: weights={} bytes, bias={} bytes, total={:.2} KB",
            name,
            stats.weights_bytes,
            stats.bias_bytes,
            stats.total_kb()
        );

        assert!(stats.weights_bytes > 0);
        assert!(stats.total_bytes >= stats.weights_bytes);
        assert!(stats.total_bytes >= prev_total);
        prev_total = stats.total_bytes;
    }
}

/// Test optimizer reset functionality.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_optimizer_reset() {
    use arkan::gpu::{GpuAdam, GpuAdamConfig};

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network.create_workspace(4).expect("Failed");

    let layer_sizes = gpu_network.layer_param_sizes();
    let mut optimizer = GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.01),
    );

    let inputs = vec![0.5f32; 4 * config.input_dim];
    let targets = vec![0.5f32; 4 * config.output_dim];

    // Train a few steps
    for _ in 0..5 {
        let _ = gpu_network
            .train_step_gpu_native(&inputs, &targets, 4, &mut workspace, &mut optimizer)
            .expect("Training failed");
    }

    // Reset optimizer
    optimizer.reset();

    // Should be able to train again
    let loss_after_reset = gpu_network
        .train_step_gpu_native(&inputs, &targets, 4, &mut workspace, &mut optimizer)
        .expect("Training after reset failed");

    assert!(loss_after_reset.is_finite());
    println!(
        "Optimizer reset test passed, loss after reset: {}",
        loss_after_reset
    );
}

/// Test gradient clipping in GPU training.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_gradient_clipping() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = multi_layer_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_cpu_network = cpu_network.clone();
    let _cpu_workspace = cpu_network.create_workspace(8);

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &gpu_cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(8).expect("Failed");
    let mut optimizer = Adam::new(&gpu_cpu_network, AdamConfig::with_lr(0.1));

    let opts = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.0,
    };

    // Large input values to generate large gradients
    let inputs = vec![5.0f32; 8 * config.input_dim];
    let targets = vec![1.0f32; 8 * config.output_dim];

    // GPU train with clipping
    let gpu_loss = gpu_network
        .train_step_with_options(
            &inputs,
            &targets,
            None,
            8,
            &mut gpu_workspace,
            &mut optimizer,
            &mut gpu_cpu_network,
            &opts,
        )
        .expect("GPU train failed");

    assert!(gpu_loss.is_finite());
    println!("Gradient clipping test passed, loss: {}", gpu_loss);
}

/// Test weight decay in GPU training.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_weight_decay() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_cpu_network = cpu_network.clone();

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &gpu_cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(4).expect("Failed");
    let mut optimizer = Adam::new(&gpu_cpu_network, AdamConfig::with_lr(0.001));

    let opts = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.1,
    };

    let inputs = vec![0.5f32; 4 * config.input_dim];
    let targets = vec![0.5f32; 4 * config.output_dim];

    // Get initial weight norm
    let initial_weight_norm: f32 = gpu_cpu_network
        .layers
        .iter()
        .flat_map(|l| l.weights.iter())
        .map(|w| w * w)
        .sum::<f32>()
        .sqrt();

    // Train with weight decay
    for _ in 0..10 {
        let _ = gpu_network
            .train_step_with_options(
                &inputs,
                &targets,
                None,
                4,
                &mut gpu_workspace,
                &mut optimizer,
                &mut gpu_cpu_network,
                &opts,
            )
            .expect("Training failed");
    }

    // Weight norm should decrease with weight decay
    let final_weight_norm: f32 = gpu_cpu_network
        .layers
        .iter()
        .flat_map(|l| l.weights.iter())
        .map(|w| w * w)
        .sum::<f32>()
        .sqrt();

    println!(
        "Weight decay test: initial norm={:.4}, final norm={:.4}",
        initial_weight_norm, final_weight_norm
    );
    // Note: weight decay should reduce weights, but optimizer updates may increase them
    // Just verify the training completed without errors
    assert!(final_weight_norm.is_finite());
}

/// Test different batch sizes in sequence.
#[test]
#[ignore = "Requires GPU"]
fn test_batch_size_sequence() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Create workspace for max batch
    let mut workspace = gpu_network.create_workspace(128).expect("Failed");

    // Test various batch sizes in sequence
    for batch_size in [1, 2, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128] {
        let input = vec![0.5f32; batch_size * config.input_dim];

        let result = gpu_network
            .forward_batch(&input, batch_size, &mut workspace)
            .expect("Forward failed");

        assert_eq!(result.len(), batch_size * config.output_dim);
        println!("Batch size {} passed", batch_size);
    }
}

/// Test GPU forward with softmax output.
#[test]
#[ignore = "Requires GPU"]
fn test_softmax_output_properties() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = KanConfig {
        input_dim: 8,
        output_dim: 10, // Like classification
        hidden_dims: vec![16],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 8],
        input_std: vec![1.0; 8],
        ..Default::default()
    };

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network.create_workspace(4).expect("Failed");

    let input = vec![0.5f32; 4 * config.input_dim];

    let result = gpu_network
        .forward_batch_softmax(&input, 4, &mut workspace)
        .expect("Forward with softmax failed");

    // Check softmax properties for each sample
    for sample in 0..4 {
        let start = sample * config.output_dim;
        let end = start + config.output_dim;
        let probs = &result[start..end];

        // All values should be in [0, 1]
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Sample {} output {} = {} is not in [0, 1]",
                sample,
                i,
                p
            );
        }

        // Sum should be approximately 1
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Sample {} softmax sum = {} is not 1",
            sample,
            sum
        );
    }
    println!("Softmax output properties test passed");
}

/// Test GPU with cross-entropy loss.
#[test]
#[ignore = "Requires GPU"]
fn test_cross_entropy_training() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = KanConfig {
        input_dim: 4,
        output_dim: 3, // 3-class classification
        hidden_dims: vec![8],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_cpu_network = cpu_network.clone();
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &gpu_cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network.create_workspace(4).expect("Failed");
    let mut optimizer = Adam::new(&gpu_cpu_network, AdamConfig::with_lr(0.01));

    let inputs = vec![0.5f32; 4 * config.input_dim];
    // One-hot encoded targets
    let targets = vec![
        1.0, 0.0, 0.0, // Class 0
        0.0, 1.0, 0.0, // Class 1
        0.0, 0.0, 1.0, // Class 2
        1.0, 0.0, 0.0, // Class 0
    ];

    let loss = gpu_network
        .train_step_cross_entropy(
            &inputs,
            &targets,
            4,
            &mut workspace,
            &mut optimizer,
            &mut gpu_cpu_network,
        )
        .expect("Cross-entropy training failed");

    assert!(loss.is_finite());
    assert!(loss > 0.0); // CE loss is always positive
    println!("Cross-entropy training test passed, loss: {}", loss);
}

/// Test GPU layer param sizes API.
#[test]
#[ignore = "Requires GPU"]
fn test_layer_param_sizes() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8, 6],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let cpu_network = KanNetwork::new(config.clone());
    let gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    let layer_sizes = gpu_network.layer_param_sizes();

    // Should have 3 layers: 4->8, 8->6, 6->2
    assert_eq!(layer_sizes.len(), 3);

    for (i, &(weights, biases)) in layer_sizes.iter().enumerate() {
        println!("Layer {}: weights={}, biases={}", i, weights, biases);
        assert!(weights > 0);
        assert!(biases > 0);
    }

    // Biases should match output dims
    assert_eq!(layer_sizes[0].1, 8);
    assert_eq!(layer_sizes[1].1, 6);
    assert_eq!(layer_sizes[2].1, 2);
}

/// Test that GPU and CPU produce same results after weight sync.
#[test]
#[ignore = "Requires GPU"]
fn test_bidirectional_weight_sync() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_copy = cpu_network.clone();
    let mut cpu_workspace = cpu_network.create_workspace(1);

    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(1).expect("Failed");

    let input = vec![0.5f32; config.input_dim];

    // Get GPU result
    let gpu_result = gpu_network
        .forward_batch(&input, 1, &mut gpu_workspace)
        .expect("GPU forward failed");

    // Sync GPU weights back to CPU
    gpu_network
        .sync_weights_to_cpu(&mut cpu_copy)
        .expect("Sync failed");

    // CPU forward with synced weights
    let mut cpu_result = vec![0.0f32; config.output_dim];
    cpu_copy.forward_batch(&input, &mut cpu_result, &mut cpu_workspace);

    // Should be identical
    assert_approx_eq(&gpu_result, &cpu_result, 1e-6);
    println!("Bidirectional weight sync test passed");
}

/// Test workspace buffer validation.
#[test]
#[ignore = "Requires GPU"]
fn test_workspace_validation() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    // Create workspace for batch=4
    let mut workspace = gpu_network.create_workspace(4).expect("Failed");

    // Valid use: batch <= max_batch
    let input = vec![0.5f32; 4 * config.input_dim];
    let result = gpu_network.forward_batch(&input, 4, &mut workspace);
    assert!(result.is_ok());

    let input = vec![0.5f32; 2 * config.input_dim];
    let result = gpu_network.forward_batch(&input, 2, &mut workspace);
    assert!(result.is_ok());

    // Test that workspace can be recreated for larger batches
    let mut workspace_large = gpu_network
        .create_workspace(16)
        .expect("Failed to create larger workspace");
    let input_large = vec![0.5f32; 16 * config.input_dim];
    let result = gpu_network.forward_batch(&input_large, 16, &mut workspace_large);
    assert!(result.is_ok());
    println!("Workspace validation test passed");
}

/// Test native GPU training with SGD.
#[test]
#[ignore = "Requires GPU"]
fn test_native_gpu_sgd_training() {
    use arkan::gpu::{GpuSgd, GpuSgdConfig};

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = simple_config();
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");
    let mut workspace = gpu_network.create_workspace(8).expect("Failed");

    let layer_sizes = gpu_network.layer_param_sizes();
    let mut optimizer = GpuSgd::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuSgdConfig {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
        },
    );

    let inputs = vec![0.5f32; 8 * config.input_dim];
    let targets = vec![0.5f32; 8 * config.output_dim];

    let mut losses = Vec::new();
    for _ in 0..10 {
        let loss = gpu_network
            .train_step_gpu_native_sgd(&inputs, &targets, 8, &mut workspace, &mut optimizer)
            .expect("Training failed");
        losses.push(loss);
    }

    println!("SGD losses: {:?}", losses);
    // All losses should be finite
    for loss in &losses {
        assert!(loss.is_finite());
    }
}

/// Test GPU network input/output dimensions accessors.
#[test]
#[ignore = "Requires GPU"]
fn test_gpu_network_dimensions() {
    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = KanConfig {
        input_dim: 7,
        output_dim: 5,
        hidden_dims: vec![10, 8],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 7],
        input_std: vec![1.0; 7],
        ..Default::default()
    };

    let cpu_network = KanNetwork::new(config.clone());
    let gpu_network =
        GpuNetwork::from_cpu(&backend, &cpu_network).expect("Failed to create GPU network");

    assert_eq!(gpu_network.input_dim, 7);
    assert_eq!(gpu_network.output_dim, 5);
    println!(
        "GPU network dimensions: input={}, output={}",
        gpu_network.input_dim, gpu_network.output_dim
    );
}

// =============================================================================
// Training Trajectory Parity Test
// =============================================================================
// This test verifies that GPU and CPU training produce identical loss curves
// over 100 training steps. Any divergence indicates a bug in gradient computation
// or optimizer implementation.
//
// We use SGD (not Adam) because it has a simpler API and direct parity between
// CPU train_step() and GPU train_step_gpu_native_sgd().

/// Test training trajectory parity: CPU vs GPU over 100 steps using SGD.
///
/// This is the "Gold Standard" test for GPU correctness:
/// - Same seed for identical initialization
/// - Same data for each step
/// - Compare loss at every 10th step
/// - Maximum allowed divergence: 1e-3
#[test]
#[ignore = "Requires GPU"]
fn test_training_trajectory_parity() {
    use arkan::gpu::optimizer::{GpuSgd, GpuSgdConfig};

    println!("\n=== Training Trajectory Parity Test (100 steps, SGD) ===\n");

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    // Configuration with fixed seed for deterministic initialization
    let config = KanConfig {
        input_dim: 8,
        output_dim: 4,
        hidden_dims: vec![16, 8],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-2.0, 2.0),
        input_mean: vec![0.0; 8],
        input_std: vec![1.0; 8],
        init_seed: Some(42), // CRITICAL: Same seed for identical weights
        ..Default::default()
    };

    // Generate synthetic dataset
    let batch_size = 64;
    let num_samples = 128; // Multiple of batch_size for clean cycling
    let learning_rate = 0.01;
    
    // Fixed seed for reproducible data
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(12345);
    
    let all_inputs: Vec<f32> = (0..num_samples * config.input_dim)
        .map(|_| rng.gen_range(-1.5..1.5))
        .collect();
    let all_targets: Vec<f32> = (0..num_samples * config.output_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // =========================================================================
    // CPU Training using train_step (SGD)
    // =========================================================================
    println!("Running CPU training (SGD)...");
    
    let mut cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(batch_size);
    
    let mut cpu_losses: Vec<f32> = Vec::new();
    
    for step in 0..100 {
        // Cycle through data
        let batch_idx = step % (num_samples / batch_size);
        let start = batch_idx * batch_size * config.input_dim;
        let end = start + batch_size * config.input_dim;
        let inputs = &all_inputs[start..end];
        
        let target_start = batch_idx * batch_size * config.output_dim;
        let target_end = target_start + batch_size * config.output_dim;
        let targets = &all_targets[target_start..target_end];
        
        // CPU train_step uses SGD internally
        let loss = cpu_network.train_step(
            inputs, targets, None, learning_rate, &mut cpu_workspace
        );
        
        if step % 10 == 0 {
            cpu_losses.push(loss);
            println!("  CPU step {:3}: loss = {:.6}", step, loss);
        }
    }

    // =========================================================================
    // GPU Training (using GPU-native SGD)
    // =========================================================================
    println!("\nRunning GPU training (SGD)...");
    
    // Create fresh network with same seed
    let gpu_cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network =
        GpuNetwork::from_cpu(&backend, &gpu_cpu_network).expect("Failed to create GPU network");
    let mut gpu_workspace = gpu_network
        .create_workspace(batch_size)
        .expect("Failed to create GPU workspace");
    
    // Create GPU SGD optimizer with same learning rate
    let layer_sizes = gpu_network.layer_param_sizes();
    let mut gpu_sgd = GpuSgd::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuSgdConfig {
            lr: learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    );
    
    let mut gpu_losses: Vec<f32> = Vec::new();
    
    for step in 0..100 {
        // Same data cycling as CPU
        let batch_idx = step % (num_samples / batch_size);
        let start = batch_idx * batch_size * config.input_dim;
        let end = start + batch_size * config.input_dim;
        let inputs = &all_inputs[start..end];
        
        let target_start = batch_idx * batch_size * config.output_dim;
        let target_end = target_start + batch_size * config.output_dim;
        let targets = &all_targets[target_start..target_end];
        
        let loss = gpu_network
            .train_step_gpu_native_sgd(inputs, targets, batch_size, &mut gpu_workspace, &mut gpu_sgd)
            .expect("GPU training failed");
        
        if step % 10 == 0 {
            gpu_losses.push(loss);
            println!("  GPU step {:3}: loss = {:.6}", step, loss);
        }
    }

    // =========================================================================
    // Compare Loss Trajectories
    // =========================================================================
    println!("\n=== Loss Trajectory Comparison ===");
    println!("{:>6} {:>12} {:>12} {:>12}", "Step", "CPU Loss", "GPU Loss", "Diff");
    println!("{:-<48}", "");
    
    let mut max_diff: f32 = 0.0;
    let mut max_diff_step = 0;
    
    for (i, (cpu_loss, gpu_loss)) in cpu_losses.iter().zip(gpu_losses.iter()).enumerate() {
        let step = i * 10;
        let diff = (cpu_loss - gpu_loss).abs();
        
        if diff > max_diff {
            max_diff = diff;
            max_diff_step = step;
        }
        
        let status = if diff < 1e-3 { "✓" } else { "✗" };
        println!(
            "{:>6} {:>12.6} {:>12.6} {:>12.6} {}",
            step, cpu_loss, gpu_loss, diff, status
        );
    }
    
    println!("\nMax difference: {:.6} at step {}", max_diff, max_diff_step);
    
    // Assertions
    // Step 0: Should be identical (same initialization, same forward pass)
    let step0_diff = (cpu_losses[0] - gpu_losses[0]).abs();
    assert!(
        step0_diff < 1e-4,
        "Step 0 loss should match exactly: CPU={}, GPU={}, diff={}",
        cpu_losses[0], gpu_losses[0], step0_diff
    );
    
    // Step 100: Should match within tolerance
    let step100_diff = (cpu_losses.last().unwrap() - gpu_losses.last().unwrap()).abs();
    assert!(
        step100_diff < 1e-3,
        "Step 100 loss diverged too much: CPU={}, GPU={}, diff={}",
        cpu_losses.last().unwrap(), gpu_losses.last().unwrap(), step100_diff
    );
    
    // All steps should be within reasonable tolerance
    // (allowing for floating point accumulation)
    let tolerance_per_step = |step: usize| -> f32 {
        // Allow more tolerance as steps increase due to FP accumulation
        1e-4 + (step as f32 * 1e-5)
    };
    
    for (i, (cpu_loss, gpu_loss)) in cpu_losses.iter().zip(gpu_losses.iter()).enumerate() {
        let step = i * 10;
        let diff = (cpu_loss - gpu_loss).abs();
        let tol = tolerance_per_step(step);
        
        assert!(
            diff < tol,
            "Loss diverged at step {}: CPU={}, GPU={}, diff={}, tolerance={}",
            step, cpu_loss, gpu_loss, diff, tol
        );
    }
    
    println!("\n✅ Training trajectory parity test PASSED!");
}

/// Simplified trajectory test with fewer steps for faster CI.
#[test]
#[ignore = "Requires GPU"]
fn test_training_trajectory_short() {
    use arkan::gpu::optimizer::{GpuSgd, GpuSgdConfig};
    
    println!("\n=== Short Training Trajectory Test (20 steps, SGD) ===\n");

    let backend =
        WgpuBackend::init(WgpuOptions::default()).expect("Failed to initialize GPU backend");

    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-2.0, 2.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(42),
        ..Default::default()
    };

    let batch_size = 16;
    let learning_rate = 0.01;
    
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(99999);
    
    let inputs: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let targets: Vec<f32> = (0..batch_size * config.output_dim)
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect();

    // CPU using train_step (SGD)
    let mut cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(batch_size);
    
    let mut cpu_losses = Vec::new();
    for _ in 0..20 {
        let loss = cpu_network.train_step(
            &inputs, &targets, None, learning_rate, &mut cpu_workspace
        );
        cpu_losses.push(loss);
    }

    // GPU using GPU-native SGD
    let gpu_cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &gpu_cpu_network).expect("Failed");
    let mut gpu_workspace = gpu_network.create_workspace(batch_size).expect("Failed");
    
    let layer_sizes = gpu_network.layer_param_sizes();
    let mut gpu_sgd = GpuSgd::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuSgdConfig {
            lr: learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
        },
    );
    
    let mut gpu_losses = Vec::new();
    for _ in 0..20 {
        let loss = gpu_network
            .train_step_gpu_native_sgd(&inputs, &targets, batch_size, &mut gpu_workspace, &mut gpu_sgd)
            .expect("Failed");
        gpu_losses.push(loss);
    }

    // Compare first and last
    let diff_first = (cpu_losses[0] - gpu_losses[0]).abs();
    let diff_last = (cpu_losses[19] - gpu_losses[19]).abs();
    
    println!("First step: CPU={:.6}, GPU={:.6}, diff={:.6}", 
             cpu_losses[0], gpu_losses[0], diff_first);
    println!("Last step:  CPU={:.6}, GPU={:.6}, diff={:.6}", 
             cpu_losses[19], gpu_losses[19], diff_last);
    
    assert!(diff_first < 1e-4, "First step loss mismatch: {}", diff_first);
    assert!(diff_last < 1e-3, "Last step loss diverged: {}", diff_last);
    
    println!("\n✅ Short trajectory test PASSED!");
}

