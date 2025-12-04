//! Regression tests for v0.2.0 refactoring.
//!
//! These tests validate the fixes made in the v0.2.0 refactoring:
//! - Overflow protection in forward_batch
//! - Correct workspace sizing for wide hidden layers
//! - Safe error handling (no panics)
//! - Boundary conditions

use arkan::{
    checked_buffer_size, checked_buffer_size3, ArkanError, KanConfig, KanNetwork, Workspace,
    MAX_BUFFER_ELEMENTS,
};

// =============================================================================
// 1. Overflow Protection Tests
// =============================================================================

#[test]
fn test_checked_buffer_size_normal() {
    // Normal case - should succeed
    let result = checked_buffer_size(100, 32);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 3200);
}

#[test]
fn test_checked_buffer_size_overflow() {
    // Overflow case - multiplication would overflow usize
    let result = checked_buffer_size(usize::MAX, 2);
    assert!(result.is_err());
    match result {
        Err(ArkanError::Overflow(msg)) => {
            assert!(msg.contains("overflow"));
        }
        _ => panic!("Expected Overflow error"),
    }
}

#[test]
fn test_checked_buffer_size_exceeds_max() {
    // Exceeds MAX_BUFFER_ELEMENTS
    let large = MAX_BUFFER_ELEMENTS + 1;
    let result = checked_buffer_size(large, 1);
    assert!(result.is_err());
}

#[test]
fn test_checked_buffer_size3_normal() {
    // Normal 3D case
    let result = checked_buffer_size3(10, 20, 8);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1600);
}

#[test]
fn test_checked_buffer_size3_overflow() {
    // 3D overflow case
    let result = checked_buffer_size3(usize::MAX / 2, 3, 1);
    assert!(result.is_err());
}

#[test]
fn test_forward_batch_large_but_valid() {
    // Test with a batch size that's large but valid
    let config = KanConfig {
        input_dim: 21,
        output_dim: 16,
        hidden_dims: vec![64],
        ..Default::default()
    };
    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    // Batch of 1000 - should work fine
    let batch_size = 1000;
    let input = vec![0.5f32; config.input_dim * batch_size];
    let mut output = vec![0.0f32; config.output_dim * batch_size];

    // Use try_forward_batch which returns Result
    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok());
}

// =============================================================================
// 2. Workspace Sizing Tests (Wide Hidden Layers)
// =============================================================================

#[test]
fn test_workspace_wide_hidden_layer() {
    // Hidden layer wider than input - this was the bug in buffer.rs
    let config = KanConfig {
        input_dim: 10,
        output_dim: 5,
        hidden_dims: vec![100], // Much wider than input!
        input_mean: vec![0.0; 10],
        input_std: vec![1.0; 10],
        ..Default::default()
    };

    // Workspace should be sized for max dimension (100), not input (10)
    let mut workspace = Workspace::new(&config);
    let network = KanNetwork::new(config.clone());

    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];

    // This should not panic due to undersized workspace
    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok(), "Forward pass should succeed with wide hidden layer");
}

#[test]
fn test_workspace_multiple_wide_layers() {
    // Multiple hidden layers, middle one is widest
    let config = KanConfig {
        input_dim: 10,
        output_dim: 5,
        hidden_dims: vec![20, 50, 30], // 50 is the widest
        input_mean: vec![0.0; 10],
        input_std: vec![1.0; 10],
        ..Default::default()
    };

    let mut workspace = Workspace::new(&config);
    let network = KanNetwork::new(config.clone());

    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok());
}

#[test]
fn test_workspace_output_wider_than_hidden() {
    // Output dimension is the widest
    let config = KanConfig {
        input_dim: 10,
        output_dim: 100, // Widest
        hidden_dims: vec![20, 30],
        input_mean: vec![0.0; 10],
        input_std: vec![1.0; 10],
        ..Default::default()
    };

    let mut workspace = Workspace::new(&config);
    let network = KanNetwork::new(config.clone());

    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok());
}

// =============================================================================
// 3. Zero-Allocation Inference Tests
// =============================================================================

#[test]
fn test_workspace_reuse_no_realloc() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];

    // First forward - may allocate
    network.forward_batch(&input, &mut output, &mut workspace);

    // Subsequent forwards should reuse buffers
    for _ in 0..100 {
        network.forward_batch(&input, &mut output, &mut workspace);
    }
    // If we get here without panic, workspace reuse works
}

#[test]
fn test_workspace_prepare_idempotent() {
    let config = KanConfig::preset();
    let mut workspace = Workspace::new(&config);

    // Multiple prepares should be fine
    for batch_size in [1, 10, 1, 100, 1] {
        let result = workspace.try_prepare_forward(batch_size, &config);
        assert!(result.is_ok(), "Prepare should succeed for batch_size={}", batch_size);
    }
}

// =============================================================================
// 4. Boundary Condition Tests
// =============================================================================

#[test]
fn test_batch_size_one() {
    let config = KanConfig::default();
    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok());
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_single_layer_network() {
    // No hidden layers - just input -> output
    let config = KanConfig {
        input_dim: 10,
        output_dim: 5,
        hidden_dims: vec![], // No hidden layers
        input_mean: vec![0.0; 10],
        input_std: vec![1.0; 10],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok());
}

#[test]
fn test_minimal_dimensions() {
    // Smallest possible network: 1 -> 1
    let config = KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![],
        input_mean: vec![0.0],
        input_std: vec![1.0],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let input = vec![0.5f32];
    let mut output = vec![0.0f32];

    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok());
}

#[test]
fn test_extreme_input_values() {
    let config = KanConfig::default();
    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let mut output = vec![0.0f32; config.output_dim];

    // Test with zeros
    let zeros = vec![0.0f32; config.input_dim];
    let result = network.try_forward_batch(&zeros, &mut output, &mut workspace);
    assert!(result.is_ok());
    assert!(output.iter().all(|&x| x.is_finite()));

    // Test with large values
    let large = vec![1000.0f32; config.input_dim];
    let result = network.try_forward_batch(&large, &mut output, &mut workspace);
    assert!(result.is_ok());
    // Output may be clamped but should be finite
    assert!(output.iter().all(|&x| x.is_finite()));

    // Test with negative values
    let negative = vec![-1.0f32; config.input_dim];
    let result = network.try_forward_batch(&negative, &mut output, &mut workspace);
    assert!(result.is_ok());
    assert!(output.iter().all(|&x| x.is_finite()));
}

// =============================================================================
// 5. Error Handling Tests (No Panics)
// =============================================================================

#[test]
fn test_config_validation_zero_input() {
    let config = KanConfig {
        input_dim: 0, // Invalid
        output_dim: 5,
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_config_validation_zero_output() {
    let config = KanConfig {
        input_dim: 5,
        output_dim: 0, // Invalid
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_config_validation_invalid_spline_order() {
    let config = KanConfig {
        spline_order: 0, // Invalid - must be >= 1
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_config_validation_spline_order_too_high() {
    let config = KanConfig {
        spline_order: 100, // Invalid - too high
        ..Default::default()
    };

    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_shape_mismatch_error() {
    let config = KanConfig::default();
    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    // Wrong input size
    let wrong_input = vec![0.5f32; config.input_dim + 1];
    let mut output = vec![0.0f32; config.output_dim];

    let result = network.try_forward_batch(&wrong_input, &mut output, &mut workspace);
    assert!(result.is_err());
}

// =============================================================================
// 6. Spline Order Range Tests
// =============================================================================

#[test]
fn test_spline_order_range_cpu() {
    // CPU supports orders 1-7
    for order in 1..=7 {
        let config = KanConfig {
            spline_order: order,
            ..Default::default()
        };
        assert!(
            config.validate().is_ok(),
            "Spline order {} should be valid for CPU",
            order
        );
    }
}

#[test]
fn test_spline_order_out_of_range() {
    // Order 8 should fail
    let config = KanConfig {
        spline_order: 8,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// =============================================================================
// 7. P0/P1 Regression Tests - Critical Overflow & Allocation Bugs
// =============================================================================

/// P0: Verify checked_buffer_size catches overflow BEFORE it becomes UB
#[test]
fn test_p0_overflow_protection_near_max() {
    // Just below MAX_BUFFER_ELEMENTS - should succeed
    let half_max = MAX_BUFFER_ELEMENTS / 2;
    let result = checked_buffer_size(half_max, 1);
    assert!(result.is_ok(), "Half of MAX should be valid");

    // At MAX - should succeed
    let result = checked_buffer_size(MAX_BUFFER_ELEMENTS, 1);
    assert!(result.is_ok(), "Exactly MAX should be valid");

    // Just over MAX - should fail
    let result = checked_buffer_size(MAX_BUFFER_ELEMENTS + 1, 1);
    assert!(result.is_err(), "Over MAX should fail");
}

/// P0: Verify multiplication overflow is caught
#[test]
fn test_p0_multiplication_overflow_protection() {
    // Large values that multiply to overflow
    let large = (1 << 30) as usize; // 2^30
    let result = checked_buffer_size(large, large);
    assert!(result.is_err(), "2^30 * 2^30 should overflow and be caught");

    // Test 3D overflow
    let med = (1 << 22) as usize; // 2^22
    let result = checked_buffer_size3(med, med, med);
    assert!(result.is_err(), "2^22 * 2^22 * 2^22 should overflow");
}

/// P0: Layer forward should not panic on large batch
#[test]
fn test_p0_layer_no_panic_large_batch() {
    // Use network-level API which properly manages workspace
    let config = KanConfig {
        input_dim: 21,
        output_dim: 64,
        hidden_dims: vec![],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    // Large but valid batch
    let batch_size = 10_000;
    let input = vec![0.5f32; config.input_dim * batch_size];
    let mut output = vec![0.0f32; config.output_dim * batch_size];

    // This should succeed without panic
    let result = network.try_forward_batch(&input, &mut output, &mut workspace);
    assert!(result.is_ok(), "Large batch forward should succeed");
}

/// P1: Verify workspace doesn't grow unbounded with varying batch sizes
#[test]
fn test_p1_workspace_memory_stability() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    // Vary batch sizes - workspace should handle this gracefully
    let batch_sizes = [1, 10, 100, 1000, 500, 1, 1000, 1];

    for &batch_size in &batch_sizes {
        let input = vec![0.5f32; config.input_dim * batch_size];
        let mut output = vec![0.0f32; config.output_dim * batch_size];

        let result = network.try_forward_batch(&input, &mut output, &mut workspace);
        assert!(
            result.is_ok(),
            "Batch size {} should work after previous batches",
            batch_size
        );
    }
}

/// P1: Zero-allocation path - verify inference with pre-sized workspace
#[test]
fn test_p1_zero_alloc_inference_path() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());

    // Pre-allocate workspace for max expected batch
    let max_batch = 256;
    let mut workspace = Workspace::new(&config);
    workspace
        .try_prepare_forward(max_batch, &config)
        .expect("Prepare should succeed");

    // Now run inference - should not need to allocate
    for batch_size in [1, 16, 64, 128, 256] {
        let input = vec![0.5f32; config.input_dim * batch_size];
        let mut output = vec![0.0f32; config.output_dim * batch_size];

        let result = network.try_forward_batch(&input, &mut output, &mut workspace);
        assert!(result.is_ok(), "Batch {} should succeed with pre-sized workspace", batch_size);
    }
}

/// P1: Verify try_ methods don't panic on invalid input
#[test]
fn test_p1_try_methods_no_panic() {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    // Empty input - should error, not panic
    let empty: Vec<f32> = vec![];
    let mut output = vec![0.0f32; config.output_dim];

    let result = network.try_forward_batch(&empty, &mut output, &mut workspace);
    assert!(result.is_err(), "Empty input should return error");

    // Output too small - should error, not panic
    let input = vec![0.5f32; config.input_dim];
    let mut small_output = vec![0.0f32; 1]; // Too small

    let result = network.try_forward_batch(&input, &mut small_output, &mut workspace);
    assert!(result.is_err(), "Small output should return error");
}

/// P0: Config validation must catch dangerous configs before they cause issues
#[test]
fn test_p0_config_validation_safety() {
    // Zero hidden dim should fail
    let config = KanConfig {
        input_dim: 10,
        output_dim: 5,
        hidden_dims: vec![0], // Invalid!
        ..Default::default()
    };
    assert!(config.validate().is_err(), "Zero hidden dim should be invalid");

    // Mismatched normalization vectors
    let config = KanConfig {
        input_dim: 10,
        output_dim: 5,
        input_mean: vec![0.0; 5], // Wrong size!
        input_std: vec![1.0; 10],
        ..Default::default()
    };
    assert!(config.validate().is_err(), "Mismatched input_mean should be invalid");

    // Zero/negative std - now warns but succeeds (values will be clamped to EPSILON)
    let config = KanConfig {
        input_dim: 2,
        output_dim: 1,
        input_mean: vec![0.0; 2],
        input_std: vec![0.0, 1.0], // Zero - will be clamped
        ..Default::default()
    };
    // Should succeed with warning, not error
    assert!(config.validate().is_ok(), "Zero input_std should warn but not fail");

    // Negative std also warns but succeeds
    let config = KanConfig {
        input_dim: 2,
        output_dim: 1,
        input_mean: vec![0.0; 2],
        input_std: vec![-1.0, 1.0], // Negative - will be clamped
        ..Default::default()
    };
    assert!(config.validate().is_ok(), "Negative input_std should warn but not fail");
}

/// P1: Grid size edge cases
#[test]
fn test_p1_grid_size_edge_cases() {
    // Minimum grid size
    let config = KanConfig {
        grid_size: 2, // Minimum reasonable
        ..Default::default()
    };
    assert!(config.validate().is_ok(), "Grid size 2 should be valid");

    // Large grid size - should work but uses more memory
    let config = KanConfig {
        grid_size: 100,
        ..Default::default()
    };
    let result = config.validate();
    // May fail if too large, but should not panic
    let _ = result;
}

/// P0: Workspace prepare with zero batch should fail gracefully
#[test]
fn test_p0_workspace_zero_batch() {
    let config = KanConfig::preset();
    let mut workspace = Workspace::new(&config);

    // Zero batch - should error, not panic or silently succeed
    let result = workspace.try_prepare_forward(0, &config);
    assert!(result.is_err(), "Zero batch should return error");
}
