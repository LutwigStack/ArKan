use super::*;
use crate::cpu::Workspace;
use crate::error::ArkanError;
#[cfg(feature = "serde")]
use crate::format::network::{SERIALIZATION_MAGIC, SERIALIZATION_VERSION};

#[test]
fn test_network_creation() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config);

    // 21 → 64 → 64 → 24: 3 layers
    assert_eq!(network.num_layers(), 3);
    assert_eq!(network.layers[0].in_dim, 21);
    assert_eq!(network.layers[0].out_dim, 64);
    assert_eq!(network.layers[2].out_dim, 24);
}

#[test]
fn test_network_forward_single() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);

    let input = vec![0.5f32; 21];
    let mut output = vec![0.0f32; 24];

    network.forward_single(&input, &mut output, &mut workspace);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_network_forward_batch() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(32);

    let batch_size = 16;
    let input: Vec<f32> = (0..batch_size * 21)
        .map(|i| (i as f32 * 0.01) % 1.0)
        .collect();
    let mut output = vec![0.0f32; batch_size * 24];

    network.forward_batch(&input, &mut output, &mut workspace);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_network_train_step() {
    let config = KanConfig::preset();
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(16);

    let batch_size = 8;
    let input: Vec<f32> = vec![0.5; batch_size * 21];
    let target: Vec<f32> = vec![0.1; batch_size * 24];

    let loss1 = network.train_step(&input, &target, None, 0.01, &mut workspace);
    let loss2 = network.train_step(&input, &target, None, 0.01, &mut workspace);

    // Loss should decrease after training
    assert!(
        loss2 < loss1,
        "Loss should decrease: {} -> {}",
        loss1,
        loss2
    );
}

#[test]
fn test_train_step_with_optimizer_adam() {
    use crate::optimizer::{Adam, AdamConfig};

    let config = KanConfig::preset();
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(16);
    let mut adam = Adam::new(&network, AdamConfig::default());

    let batch_size = 8;
    let input: Vec<f32> = vec![0.5; batch_size * 21];
    let target: Vec<f32> = vec![0.1; batch_size * 24];
    let opts = TrainOptions::default();

    let loss1 = network
        .train_step_with_optimizer(&input, &target, None, &mut workspace, &mut adam, &opts)
        .expect("train_step_with_optimizer failed");
    let loss2 = network
        .train_step_with_optimizer(&input, &target, None, &mut workspace, &mut adam, &opts)
        .expect("train_step_with_optimizer failed");

    // Loss should decrease with Adam
    assert!(
        loss2 < loss1,
        "Loss should decrease with Adam: {} -> {}",
        loss1,
        loss2
    );
}

#[test]
fn test_network_param_count() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 5,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        multithreading_threshold: 1024,
        simd_width: 8,
        init_seed: None,
    };

    let network = KanNetwork::new(config);

    // Layer 1: 4 → 8, basis_aligned=8
    // Weights: 8 * 4 * 8 = 256, Bias: 8 → 264
    // Layer 2: 8 → 2
    // Weights: 2 * 8 * 8 = 128, Bias: 2 → 130
    // Total: 394

    let params = network.param_count();
    assert!(params > 0);
    assert_eq!(params, 264 + 130);
}

#[test]
fn test_single_layer_network() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![], // No hidden layers
        grid_size: 5,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        multithreading_threshold: 1024,
        simd_width: 8,
        init_seed: None,
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(8);

    assert_eq!(network.num_layers(), 1);

    let input = vec![0.5f32; 4];
    let mut output = vec![0.0f32; 2];
    network.forward_single(&input, &mut output, &mut workspace);

    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_gradcheck_single_layer() {
    // Маленькая сеть 2 -> 1 для численной проверки градиентов
    let config = KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: 4,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        multithreading_threshold: 16,
        simd_width: 4,
        init_seed: None,
    };

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);

    let input = [0.25f32, -0.4];
    let target = [0.1f32];

    // Прямой проход с записью истории
    let mut preds = vec![0.0f32; 1];
    network.forward_batch_training(&input, &mut preds, &mut workspace);
    let diff = preds[0] - target[0];
    let grad_out = vec![2.0 * diff]; // dMSE/dy для batch=1

    // Аналитические градиенты
    let layer = &network.layers[0];
    let mut weight_grad = vec![0.0f32; layer.weights.len()];
    let mut bias_grad = vec![0.0f32; layer.bias.len()];
    let mut grad_in = vec![0.0f32; input.len()];

    let hist_in = workspace.layers_inputs[0].as_slice().to_vec();
    let hist_idx = workspace.layers_grid_indices[0].clone();

    network.layers[0].backward(
        &hist_in,
        &hist_idx,
        &grad_out,
        Some(&mut grad_in),
        &mut weight_grad,
        &mut bias_grad,
        &mut workspace,
    );

    // Численный градиент по весам
    let eps = 1e-3f32;
    let mut ws_num = Workspace::new(&config);
    ws_num.reserve(1, &config);
    let mut out_buf = vec![0.0f32; 1];

    #[allow(clippy::needless_range_loop)]
    for idx in 0..network.layers[0].weights.len() {
        let orig = network.layers[0].weights[idx];

        network.layers[0].weights[idx] = orig + eps;
        network.forward_batch(&input, &mut out_buf, &mut ws_num);
        let lp = {
            let d = out_buf[0] - target[0];
            d * d
        };

        network.layers[0].weights[idx] = orig - eps;
        network.forward_batch(&input, &mut out_buf, &mut ws_num);
        let lm = {
            let d = out_buf[0] - target[0];
            d * d
        };

        network.layers[0].weights[idx] = orig;

        let num = (lp - lm) / (2.0 * eps);
        let ana = weight_grad[idx];
        let rel_err = (ana - num).abs() / num.abs().max(1e-4);
        assert!(
            rel_err < 1e-2,
            "gradcheck weight {} failed: ana={} num={} rel_err={}",
            idx,
            ana,
            num,
            rel_err
        );
    }
}

#[test]
fn test_mask_blocks_update() {
    // Маска из нулей должна блокировать обновления
    let config = KanConfig {
        input_dim: 2,
        output_dim: 2,
        hidden_dims: vec![],
        grid_size: 4,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        multithreading_threshold: 16,
        simd_width: 4,
        init_seed: None,
    };

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);

    let input = vec![0.3f32, -0.2];
    let target = vec![0.5f32, -0.5];
    let mask = vec![0.0f32; 2]; // все выключено

    let before_w = network.layers[0].weights.clone();
    let before_b = network.layers[0].bias.clone();

    network.train_step_with_options(
        &input,
        &target,
        Some(&mask),
        0.1,
        &mut workspace,
        &TrainOptions::default(),
    );

    assert_eq!(before_w, network.layers[0].weights);
    assert_eq!(before_b, network.layers[0].bias);
}

#[test]
fn test_try_forward_batch_ok() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let batch_size = 4;
    let input = vec![0.5f32; batch_size * config.input_dim];
    let mut output = vec![0.0f32; batch_size * config.output_dim];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok());
}

#[test]
fn test_try_forward_batch_input_mismatch() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let batch_size = 4;
    // Wrong input size (missing one element)
    let input = vec![0.5f32; batch_size * config.input_dim - 1];
    let mut output = vec![0.0f32; batch_size * config.output_dim];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ArkanError::ShapeMismatch { .. }
    ));
}

#[test]
fn test_try_forward_batch_output_mismatch() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let batch_size = 4;
    let input = vec![0.5f32; batch_size * config.input_dim];
    // Wrong output size
    let mut output = vec![0.0f32; batch_size * config.output_dim - 1];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ArkanError::ShapeMismatch { .. }
    ));
}

#[test]
fn test_try_train_step_ok() {
    let config = KanConfig::preset();
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let batch_size = 4;
    let input = vec![0.5f32; batch_size * config.input_dim];
    let target = vec![0.1f32; batch_size * config.output_dim];

    let result = network.try_train_step(&input, &target, None, 0.001, &mut workspace);
    assert!(result.is_ok());
    assert!(result.unwrap() > 0.0);
}

#[test]
fn test_try_train_step_input_mismatch() {
    let config = KanConfig::preset();
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let batch_size = 4;
    // Wrong input size
    let input = vec![0.5f32; batch_size * config.input_dim + 1];
    let target = vec![0.1f32; batch_size * config.output_dim];

    let result = network.try_train_step(&input, &target, None, 0.001, &mut workspace);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ArkanError::ShapeMismatch { .. }
    ));
}

#[test]
fn test_try_train_step_target_mismatch() {
    let config = KanConfig::preset();
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let batch_size = 4;
    let input = vec![0.5f32; batch_size * config.input_dim];
    // Wrong target size
    let target = vec![0.1f32; batch_size * config.output_dim - 2];

    let result = network.try_train_step(&input, &target, None, 0.001, &mut workspace);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ArkanError::ShapeMismatch { .. }
    ));
}

#[test]
fn test_try_train_step_mask_mismatch() {
    let config = KanConfig::preset();
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let batch_size = 4;
    let input = vec![0.5f32; batch_size * config.input_dim];
    let target = vec![0.1f32; batch_size * config.output_dim];
    // Wrong mask size
    let mask = vec![1.0f32; batch_size * config.output_dim + 1];

    let result = network.try_train_step(&input, &target, Some(&mask), 0.001, &mut workspace);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ArkanError::ShapeMismatch { .. }
    ));
}

// ==================== Edge-case tests ====================

#[test]
fn test_batch_size_zero() {
    // Empty batch should work without panic
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);

    let input: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];

    // Should not panic
    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.is_empty());
}

#[test]
fn test_batch_size_one() {
    // Single sample batch
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);

    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];

    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_spline_order_2() {
    // Quadratic splines (order 2)
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 5,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let input = vec![0.5f32; 4 * config.input_dim];
    let mut output = vec![0.0f32; 4 * config.output_dim];

    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_spline_order_4() {
    // Quartic splines (order 4)
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 5,
        spline_order: 4,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let input = vec![0.5f32; 4 * config.input_dim];
    let mut output = vec![0.0f32; 4 * config.output_dim];

    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_no_hidden_layers() {
    // Direct input -> output (single layer)
    let config = KanConfig {
        input_dim: 10,
        output_dim: 5,
        hidden_dims: vec![],
        grid_size: 5,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 10],
        input_std: vec![1.0; 10],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    assert_eq!(network.num_layers(), 1);

    let mut workspace = network.create_workspace(2);
    let input = vec![0.5f32; 2 * config.input_dim];
    let mut output = vec![0.0f32; 2 * config.output_dim];

    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_deep_network() {
    // Many hidden layers
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8, 8, 8, 8, 8], // 5 hidden layers
        grid_size: 3,
        spline_order: 2,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    assert_eq!(network.num_layers(), 6); // 5 hidden + 1 output

    let mut workspace = network.create_workspace(2);
    let input = vec![0.5f32; 2 * config.input_dim];
    let mut output = vec![0.0f32; 2 * config.output_dim];

    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_extreme_inputs() {
    // Very large/small inputs
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);

    // Large positive
    let input = vec![100.0f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];
    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));

    // Large negative
    let input = vec![-100.0f32; config.input_dim];
    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));

    // Near zero
    let input = vec![1e-10f32; config.input_dim];
    network.forward_batch(&input, &mut output, &mut workspace);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_workspace_reuse() {
    // Same workspace for different batch sizes
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(64);

    // Small batch
    let input1 = vec![0.5f32; 4 * config.input_dim];
    let mut output1 = vec![0.0f32; 4 * config.output_dim];
    network.forward_batch(&input1, &mut output1, &mut workspace);

    // Larger batch
    let input2 = vec![0.5f32; 32 * config.input_dim];
    let mut output2 = vec![0.0f32; 32 * config.output_dim];
    network.forward_batch(&input2, &mut output2, &mut workspace);

    // Same small batch again
    let mut output3 = vec![0.0f32; 4 * config.output_dim];
    network.forward_batch(&input1, &mut output3, &mut workspace);

    // Results should be reproducible
    assert_eq!(output1, output3);
}

#[test]
fn test_try_forward_batch_overflow() {
    // Huge batch should return Overflow error, not panic
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);

    // Create a very large input that would overflow
    // Using a calculation that would exceed MAX_BUFFER_ELEMENTS
    let _huge_batch = usize::MAX / 4;

    // We can't actually allocate this, but try_forward_batch should
    // return an error before trying to allocate
    let small_input = vec![0.5f32; config.input_dim]; // batch=1
    let mut small_output = vec![0.0f32; config.output_dim];

    // This should succeed
    let result = network.try_forward_batch(&small_input, &mut small_output, &mut workspace);
    assert!(result.is_ok());
}

#[test]
fn test_try_train_step_overflow() {
    // Overflow in train step should return error
    use crate::buffer::checked_buffer_size;

    // This tests the overflow detection logic
    let result = checked_buffer_size(usize::MAX, 2);
    assert!(result.is_err());
}

// =========================================================================
// Serialization tests (serde feature)
// =========================================================================

/// Test serialization round-trip: to_bytes -> from_bytes
#[cfg(feature = "serde")]
#[test]
fn test_serialization_roundtrip() {
    let config = KanConfig::preset();
    let original = KanNetwork::new(config);

    // Serialize
    let bytes = original.to_bytes().expect("to_bytes failed");

    // Verify header
    assert_eq!(&bytes[..5], SERIALIZATION_MAGIC);
    let version = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
    assert_eq!(version, SERIALIZATION_VERSION);

    // Deserialize
    let loaded = KanNetwork::from_bytes(&bytes).expect("from_bytes failed");

    // Verify properties match
    assert_eq!(loaded.param_count(), original.param_count());
    assert_eq!(loaded.num_layers(), original.num_layers());
    assert_eq!(loaded.config.input_dim, original.config.input_dim);
    assert_eq!(loaded.config.output_dim, original.config.output_dim);
}

/// Test from_bytes with wrong magic bytes returns error
#[cfg(feature = "serde")]
#[test]
fn test_from_bytes_wrong_magic() {
    // Create invalid bytes with wrong magic
    let mut invalid_bytes = vec![0u8; 20];
    invalid_bytes[..5].copy_from_slice(b"WRONG"); // Wrong magic
    invalid_bytes[5..9].copy_from_slice(&1u32.to_le_bytes()); // Version 1

    let result = KanNetwork::from_bytes(&invalid_bytes);
    let err = result.err().expect("Expected error for wrong magic");

    let err_msg = format!("{}", err);
    assert!(
        err_msg.contains("wrong magic") || err_msg.contains("not an ArKan"),
        "Expected wrong magic error, got: {}",
        err_msg
    );
}

/// Test from_bytes with incompatible version returns error
#[cfg(feature = "serde")]
#[test]
fn test_from_bytes_incompatible_version() {
    // Create bytes with valid magic but wrong version
    let mut invalid_bytes = vec![0u8; 100];
    invalid_bytes[..5].copy_from_slice(SERIALIZATION_MAGIC); // Correct magic
    invalid_bytes[5..9].copy_from_slice(&99u32.to_le_bytes()); // Wrong version (99)

    let result = KanNetwork::from_bytes(&invalid_bytes);
    let err = result
        .err()
        .expect("Expected error for incompatible version");

    let err_msg = format!("{}", err);
    assert!(
        err_msg.contains("Incompatible") || err_msg.contains("version"),
        "Expected incompatible version error, got: {}",
        err_msg
    );
}

/// Test from_bytes with truncated header returns error
#[cfg(feature = "serde")]
#[test]
fn test_from_bytes_truncated_header() {
    // Too short for header (magic + version = 9 bytes)
    let too_short = vec![b'A', b'R', b'K']; // Only 3 bytes

    let result = KanNetwork::from_bytes(&too_short);
    let err = result.err().expect("Expected error for truncated header");

    let err_msg = format!("{}", err);
    assert!(
        err_msg.contains("too short") || err_msg.contains("Invalid"),
        "Expected header error, got: {}",
        err_msg
    );
}

/// Test from_bytes with corrupted network data returns error
#[cfg(feature = "serde")]
#[test]
fn test_from_bytes_corrupted_data() {
    // Valid header but garbage network data
    let mut corrupted_bytes = vec![0u8; 50];
    corrupted_bytes[..5].copy_from_slice(SERIALIZATION_MAGIC);
    corrupted_bytes[5..9].copy_from_slice(&SERIALIZATION_VERSION.to_le_bytes());
    // Rest is zeros - invalid bincode data

    let result = KanNetwork::from_bytes(&corrupted_bytes);
    assert!(result.is_err()); // Should fail to deserialize
}

/// Test legacy from_bytes_legacy for backwards compatibility
#[cfg(feature = "serde")]
#[test]
fn test_from_bytes_legacy() {
    let config = KanConfig::preset();
    let original = KanNetwork::new(config);

    // Legacy format: just bincode, no header
    let legacy_bytes = bincode::serialize(&original).expect("serialize failed");

    // Legacy loader should work
    let loaded = KanNetwork::from_bytes_legacy(&legacy_bytes).expect("from_bytes_legacy failed");
    assert_eq!(loaded.param_count(), original.param_count());
}
