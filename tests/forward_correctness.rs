//! Tests for forward pass numerical correctness and SIMD path coverage.
//!
//! These tests verify:
//! - Numerical correctness (not just NaN checks)
//! - SIMD paths (simd4, simd8) vs scalar fallback parity
//! - Wide layer support (>1000 dimensions)
//! - Batch vs parallel parity

use arkan::{KanConfig, KanNetwork};

// =============================================================================
// SIMD Path Isolation Tests
// =============================================================================

/// Test that SIMD8 path gives same result as SIMD4 path with same seed
#[test]
fn test_simd8_vs_simd4_parity() {
    // Create network with simd_width=8
    let config_simd8 = KanConfig::builder()
        .input_dim(16) // Multiple of 8 for SIMD8
        .output_dim(4)
        .hidden_dims(vec![8])
        .grid_size(5)
        .spline_order(3) // basis_size = 4, fits in SIMD8
        .simd_width(8)
        .seed(42) // Same seed for reproducibility
        .build()
        .unwrap();

    // Create network with simd_width=4
    let config_simd4 = KanConfig::builder()
        .input_dim(16)
        .output_dim(4)
        .hidden_dims(vec![8])
        .grid_size(5)
        .spline_order(3)
        .simd_width(4)
        .seed(42) // Same seed
        .build()
        .unwrap();

    let network_simd8 = KanNetwork::new(config_simd8.clone());
    let network_simd4 = KanNetwork::new(config_simd4.clone());

    // Test input
    let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();

    let mut output_simd8 = vec![0.0f32; config_simd8.output_dim];
    let mut output_simd4 = vec![0.0f32; config_simd4.output_dim];

    let mut ws_simd8 = network_simd8.create_workspace(1);
    let mut ws_simd4 = network_simd4.create_workspace(1);

    network_simd8.forward_single(&input, &mut output_simd8, &mut ws_simd8);
    network_simd4.forward_single(&input, &mut output_simd4, &mut ws_simd4);

    // Compare outputs
    for (i, (a, b)) in output_simd8.iter().zip(output_simd4.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-5,
            "SIMD8 vs SIMD4 mismatch at output[{}]: {} vs {}, diff={}",
            i,
            a,
            b,
            diff
        );
    }

    println!("✓ SIMD8 vs SIMD4 parity: max_diff < 1e-5");
}

/// Test scalar fallback when in_dim is not divisible by SIMD width
#[test]
fn test_scalar_fallback_odd_dimensions() {
    // in_dim=7 is not divisible by 4 or 8 - forces scalar tail processing
    let config = KanConfig::builder()
        .input_dim(7) // Odd dimension
        .output_dim(3)
        .hidden_dims(vec![5])
        .grid_size(4)
        .spline_order(3)
        .simd_width(8)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());

    // Test multiple inputs
    let test_inputs = [
        vec![0.1f32; 7],
        vec![0.5f32; 7],
        (0..7).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        (0..7).map(|i| ((i as f32) * 0.5).sin()).collect::<Vec<_>>(),
    ];

    let mut ws = network.create_workspace(1);

    for (idx, input) in test_inputs.iter().enumerate() {
        let mut output = vec![0.0f32; config.output_dim];
        network.forward_single(input, &mut output, &mut ws);

        // Verify no NaN or Inf
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Test {}, output[{}] is not finite: {}",
                idx,
                i,
                val
            );
        }
    }

    println!("✓ Scalar fallback (odd dimensions): all outputs finite");
}

/// Test scalar fallback when basis_size > simd_width
#[test]
fn test_scalar_fallback_large_basis() {
    // spline_order=6 gives basis_size=7, which is > simd_width=4
    let config = KanConfig::builder()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![6])
        .grid_size(8)
        .spline_order(6) // basis_size = 7 > 4
        .simd_width(4)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let mut output = vec![0.0f32; config.output_dim];

    network.forward_single(&input, &mut output, &mut ws);

    // Verify outputs
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Output[{}] is not finite: {}", i, val);
    }

    println!("✓ Scalar fallback (large basis_size=7): OK");
}

// =============================================================================
// Numerical Correctness Tests
// =============================================================================

/// Test that forward pass produces consistent results
#[test]
fn test_forward_deterministic() {
    let config = KanConfig::builder()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![6])
        .grid_size(5)
        .spline_order(3)
        .seed(123)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let mut output1 = vec![0.0f32; config.output_dim];
    let mut output2 = vec![0.0f32; config.output_dim];

    // Run twice
    network.forward_single(&input, &mut output1, &mut ws);
    network.forward_single(&input, &mut output2, &mut ws);

    // Results should be identical
    for (i, (a, b)) in output1.iter().zip(output2.iter()).enumerate() {
        assert_eq!(a, b, "Non-deterministic output at [{}]: {} vs {}", i, a, b);
    }

    println!("✓ Forward deterministic: outputs identical");
}

/// Test that forward_single and forward_batch produce identical results
#[test]
fn test_forward_single_vs_batch_parity() {
    let config = KanConfig::builder()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![6])
        .grid_size(5)
        .spline_order(3)
        .seed(456)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());

    let batch_size = 16;
    let mut ws_single = network.create_workspace(1);
    let mut ws_batch = network.create_workspace(batch_size);

    // Create batch input
    let mut batch_input = vec![0.0f32; batch_size * config.input_dim];
    for b in 0..batch_size {
        for i in 0..config.input_dim {
            batch_input[b * config.input_dim + i] = ((b * i) as f32 * 0.1).sin();
        }
    }

    // Forward batch
    let mut batch_output = vec![0.0f32; batch_size * config.output_dim];
    network.forward_batch(&batch_input, &mut batch_output, &mut ws_batch);

    // Forward single for each sample and compare
    let mut max_diff = 0.0f32;
    for b in 0..batch_size {
        let in_start = b * config.input_dim;
        let out_start = b * config.output_dim;

        let single_input = &batch_input[in_start..in_start + config.input_dim];
        let mut single_output = vec![0.0f32; config.output_dim];

        network.forward_single(single_input, &mut single_output, &mut ws_single);

        for i in 0..config.output_dim {
            let diff = (single_output[i] - batch_output[out_start + i]).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < 1e-5,
                "Batch[{}] output[{}] mismatch: single={}, batch={}, diff={}",
                b,
                i,
                single_output[i],
                batch_output[out_start + i],
                diff
            );
        }
    }

    println!(
        "✓ forward_single vs forward_batch parity: max_diff = {:.2e}",
        max_diff
    );
}

/// Test that forward_batch and forward_batch_parallel produce identical results
#[test]
fn test_forward_batch_vs_parallel_parity() {
    let config = KanConfig::builder()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![6, 6])
        .grid_size(5)
        .spline_order(3)
        .seed(789)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());

    let batch_size = 64;
    let mut ws = network.create_workspace(batch_size);

    // Create batch input
    let mut batch_input = vec![0.0f32; batch_size * config.input_dim];
    for b in 0..batch_size {
        for i in 0..config.input_dim {
            batch_input[b * config.input_dim + i] = ((b + i) as f32 * 0.05).sin();
        }
    }

    let mut output_seq = vec![0.0f32; batch_size * config.output_dim];
    let mut output_par = vec![0.0f32; batch_size * config.output_dim];

    network.forward_batch(&batch_input, &mut output_seq, &mut ws);
    network.forward_batch_parallel(&batch_input, &mut output_par);

    let mut max_diff = 0.0f32;
    for (i, (&seq, &par)) in output_seq.iter().zip(output_par.iter()).enumerate() {
        let diff = (seq - par).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-5,
            "Sequential vs Parallel mismatch at [{}]: {} vs {}, diff={}",
            i,
            seq,
            par,
            diff
        );
    }

    println!(
        "✓ forward_batch vs forward_batch_parallel parity: max_diff = {:.2e}",
        max_diff
    );
}

// =============================================================================
// Wide Layer Tests
// =============================================================================

/// Test network with very wide hidden layer (>1000 neurons)
#[test]
fn test_wide_hidden_layer_1024() {
    let config = KanConfig::builder()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![1024]) // Wide hidden layer
        .grid_size(5)
        .spline_order(3)
        .seed(111)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let mut output = vec![0.0f32; config.output_dim];

    network.forward_single(&input, &mut output, &mut ws);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Wide hidden 1024: output[{}] is not finite: {}",
            i,
            val
        );
    }

    println!("✓ Wide hidden layer (1024): forward OK, outputs finite");
}

/// Test network with very wide input (>1000 dimensions)
#[test]
fn test_wide_input_1024() {
    let in_dim = 1024;
    let config = KanConfig::builder()
        .input_dim(in_dim)
        .output_dim(4)
        .hidden_dims(vec![16])
        .grid_size(5)
        .spline_order(3)
        .normalization(vec![0.0; in_dim], vec![1.0; in_dim])
        .seed(222)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 100) as f32) * 0.01).collect();
    let mut output = vec![0.0f32; config.output_dim];

    network.forward_single(&input, &mut output, &mut ws);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Wide input 1024: output[{}] is not finite: {}",
            i,
            val
        );
    }

    println!("✓ Wide input (1024): forward OK, outputs finite");
}

/// Test network with very wide output (>1000 dimensions)
#[test]
fn test_wide_output_1024() {
    let out_dim = 1024;
    let config = KanConfig::builder()
        .input_dim(8)
        .output_dim(out_dim)
        .hidden_dims(vec![16])
        .grid_size(5)
        .spline_order(3)
        .seed(333)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let mut output = vec![0.0f32; out_dim];

    network.forward_single(&input, &mut output, &mut ws);

    let mut nan_count = 0;
    let mut inf_count = 0;
    for &val in &output {
        if val.is_nan() {
            nan_count += 1;
        }
        if val.is_infinite() {
            inf_count += 1;
        }
    }

    assert_eq!(nan_count, 0, "Wide output 1024: {} NaN values", nan_count);
    assert_eq!(inf_count, 0, "Wide output 1024: {} Inf values", inf_count);

    println!("✓ Wide output (1024): forward OK, all outputs finite");
}

/// Test very wide network (all dimensions >1000)
#[test]
fn test_very_wide_network() {
    let config = KanConfig::builder()
        .input_dim(1024)
        .output_dim(256)
        .hidden_dims(vec![1024])
        .grid_size(5)
        .spline_order(3)
        .normalization(vec![0.0; 1024], vec![1.0; 1024])
        .seed(444)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..1024).map(|i| ((i % 50) as f32) * 0.02).collect();
    let mut output = vec![0.0f32; 256];

    network.forward_single(&input, &mut output, &mut ws);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Very wide network: output[{}] is not finite: {}",
            i,
            val
        );
    }

    println!("✓ Very wide network (1024->1024->256): forward OK");
}

/// Test wide network with batch processing
#[test]
fn test_wide_network_batch() {
    let config = KanConfig::builder()
        .input_dim(512)
        .output_dim(128)
        .hidden_dims(vec![512])
        .grid_size(5)
        .spline_order(3)
        .normalization(vec![0.0; 512], vec![1.0; 512])
        .seed(555)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());

    let batch_size = 32;
    let mut ws = network.create_workspace(batch_size);

    // Create batch input
    let mut batch_input = vec![0.0f32; batch_size * 512];
    for (i, val) in batch_input.iter_mut().enumerate() {
        *val = ((i % 100) as f32) * 0.01;
    }

    let mut output = vec![0.0f32; batch_size * 128];

    network.forward_batch(&batch_input, &mut output, &mut ws);

    let mut nan_count = 0;
    for &val in &output {
        if !val.is_finite() {
            nan_count += 1;
        }
    }

    assert_eq!(nan_count, 0, "Wide batch: {} non-finite values", nan_count);

    println!("✓ Wide network batch (512->512->128, batch=32): OK");
}

// =============================================================================
// SIMD Edge Cases
// =============================================================================

/// Test SIMD with exact multiple of 8 (no scalar tail)
#[test]
fn test_simd8_exact_multiple() {
    let config = KanConfig::builder()
        .input_dim(24) // 24 = 3 * 8
        .output_dim(8)
        .hidden_dims(vec![16])
        .grid_size(5)
        .spline_order(3) // basis_size=4 fits in SIMD8
        .simd_width(8)
        .normalization(vec![0.0; 24], vec![1.0; 24])
        .seed(666)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..24).map(|i| (i as f32) * 0.04).collect();
    let mut output = vec![0.0f32; 8];

    network.forward_single(&input, &mut output, &mut ws);

    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "SIMD8 exact: output[{}] = {}", i, val);
    }

    println!("✓ SIMD8 exact multiple (24): OK");
}

/// Test SIMD4 with exact multiple of 4 (no scalar tail)
#[test]
fn test_simd4_exact_multiple() {
    let config = KanConfig::builder()
        .input_dim(20) // 20 = 5 * 4
        .output_dim(8)
        .hidden_dims(vec![12])
        .grid_size(5)
        .spline_order(3)
        .simd_width(4)
        .normalization(vec![0.0; 20], vec![1.0; 20])
        .seed(777)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.05).collect();
    let mut output = vec![0.0f32; 8];

    network.forward_single(&input, &mut output, &mut ws);

    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "SIMD4 exact: output[{}] = {}", i, val);
    }

    println!("✓ SIMD4 exact multiple (20): OK");
}

/// Test SIMD8 with scalar tail (not multiple of 8)
#[test]
fn test_simd8_with_tail() {
    let config = KanConfig::builder()
        .input_dim(19) // 19 = 2*8 + 3 (has tail)
        .output_dim(4)
        .hidden_dims(vec![8])
        .grid_size(5)
        .spline_order(3)
        .simd_width(8)
        .normalization(vec![0.0; 19], vec![1.0; 19])
        .seed(888)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..19).map(|i| (i as f32) * 0.05).collect();
    let mut output = vec![0.0f32; 4];

    network.forward_single(&input, &mut output, &mut ws);

    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "SIMD8+tail: output[{}] = {}", i, val);
    }

    println!("✓ SIMD8 with tail (19): OK");
}

/// Test SIMD4 with scalar tail (not multiple of 4)
#[test]
fn test_simd4_with_tail() {
    let config = KanConfig::builder()
        .input_dim(11) // 11 = 2*4 + 3 (has tail)
        .output_dim(4)
        .hidden_dims(vec![6])
        .grid_size(5)
        .spline_order(3)
        .simd_width(4)
        .normalization(vec![0.0; 11], vec![1.0; 11])
        .seed(999)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input: Vec<f32> = (0..11).map(|i| (i as f32) * 0.08).collect();
    let mut output = vec![0.0f32; 4];

    network.forward_single(&input, &mut output, &mut ws);

    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "SIMD4+tail: output[{}] = {}", i, val);
    }

    println!("✓ SIMD4 with tail (11): OK");
}

/// Comprehensive SIMD coverage test - various dimension combinations
#[test]
fn test_simd_coverage_matrix() {
    let in_dims = [
        1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128,
    ];
    let simd_widths = [4, 8];
    let spline_orders = [2, 3, 4, 5, 6]; // Different basis sizes

    let mut passed = 0;
    let mut total = 0;

    for &in_dim in &in_dims {
        for &simd_width in &simd_widths {
            for &order in &spline_orders {
                total += 1;

                let config = KanConfig::builder()
                    .input_dim(in_dim)
                    .output_dim(4)
                    .hidden_dims(vec![8])
                    .grid_size(6)
                    .spline_order(order)
                    .simd_width(simd_width)
                    .normalization(vec![0.0; in_dim], vec![1.0; in_dim])
                    .seed(total as u64)
                    .build()
                    .unwrap();

                let network = KanNetwork::new(config.clone());
                let mut ws = network.create_workspace(1);

                let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.1).sin()).collect();
                let mut output = vec![0.0f32; 4];

                network.forward_single(&input, &mut output, &mut ws);

                let all_finite = output.iter().all(|x| x.is_finite());
                if all_finite {
                    passed += 1;
                } else {
                    println!(
                        "✗ Failed: in_dim={}, simd_width={}, order={}",
                        in_dim, simd_width, order
                    );
                }
            }
        }
    }

    assert_eq!(passed, total, "SIMD matrix: {}/{} passed", passed, total);
    println!(
        "✓ SIMD coverage matrix: {}/{} combinations passed",
        passed, total
    );
}

// =============================================================================
// Numerical Value Correctness Tests
// =============================================================================

/// Test that output values are bounded (not exploding)
#[test]
fn test_output_bounded() {
    let config = KanConfig::builder()
        .input_dim(8)
        .output_dim(4)
        .hidden_dims(vec![16, 16])
        .grid_size(5)
        .spline_order(3)
        .seed(1234)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    // Test with various input ranges
    let test_cases = [
        (0..8).map(|i| (i as f32) * 0.1).collect::<Vec<_>>(),
        (0..8).map(|i| ((i as f32) * 0.5).sin()).collect::<Vec<_>>(),
        vec![0.0f32; 8],
        vec![1.0f32; 8],
        vec![-1.0f32; 8],
    ];

    for (idx, input) in test_cases.iter().enumerate() {
        let mut output = vec![0.0f32; config.output_dim];
        network.forward_single(input, &mut output, &mut ws);

        // With random weights, outputs should be roughly bounded
        // (not a strict test, but sanity check)
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1000.0,
                "Test {}: output[{}] = {} is too large (possible explosion)",
                idx,
                i,
                val
            );
        }
    }

    println!("✓ Output bounded: all values < 1000.0");
}

/// Test that changing input changes output (sensitivity)
#[test]
fn test_input_sensitivity() {
    let config = KanConfig::builder()
        .input_dim(4)
        .output_dim(2)
        .hidden_dims(vec![8])
        .grid_size(5)
        .spline_order(3)
        .seed(5678)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());
    let mut ws = network.create_workspace(1);

    let input1 = vec![0.1, 0.2, 0.3, 0.4];
    let input2 = vec![0.1, 0.2, 0.3, 0.5]; // Changed last element

    let mut output1 = vec![0.0f32; config.output_dim];
    let mut output2 = vec![0.0f32; config.output_dim];

    network.forward_single(&input1, &mut output1, &mut ws);
    network.forward_single(&input2, &mut output2, &mut ws);

    // Outputs should be different
    let diff: f32 = output1
        .iter()
        .zip(output2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff > 1e-6,
        "Input change produced no output change (diff={})",
        diff
    );

    println!("✓ Input sensitivity: output changed by {:.2e}", diff);
}

/// Test batch consistency - same sample in different positions gives same result
#[test]
fn test_batch_position_invariance() {
    let config = KanConfig::builder()
        .input_dim(4)
        .output_dim(2)
        .hidden_dims(vec![8])
        .grid_size(5)
        .spline_order(3)
        .seed(9999)
        .build()
        .unwrap();

    let network = KanNetwork::new(config.clone());

    let batch_size = 8;
    let mut ws = network.create_workspace(batch_size);

    // Create batch where sample at position 0 and position 5 are identical
    let test_sample = vec![0.1, 0.2, 0.3, 0.4];
    let mut batch_input = vec![0.0f32; batch_size * config.input_dim];

    // Fill with random data
    for (i, val) in batch_input.iter_mut().enumerate() {
        *val = ((i as f32) * 0.17).sin();
    }

    // Place identical samples at positions 0 and 5
    for (i, &val) in test_sample.iter().enumerate() {
        batch_input[0 * config.input_dim + i] = val;
        batch_input[5 * config.input_dim + i] = val;
    }

    let mut output = vec![0.0f32; batch_size * config.output_dim];
    network.forward_batch(&batch_input, &mut output, &mut ws);

    // Compare outputs at position 0 and 5
    for i in 0..config.output_dim {
        let out0 = output[0 * config.output_dim + i];
        let out5 = output[5 * config.output_dim + i];
        let diff = (out0 - out5).abs();
        assert!(
            diff < 1e-6,
            "Position invariance failed: output[0][{}]={}, output[5][{}]={}, diff={}",
            i,
            out0,
            i,
            out5,
            diff
        );
    }

    println!("✓ Batch position invariance: identical samples produce identical outputs");
}
