//! B-Spline Edge Case Tests: Coverage of dead zones
//!
//! Tests for edge cases that were previously untested:
//! - Extreme x values (very small, very large)
//! - grid_size = 2 (minimum)
//! - High spline orders (5, 6)
//! - Large grid sizes (32, 64)
//! - Denormalized floats

use arkan::spline::{compute_basis, compute_basis_and_deriv, compute_knots, find_span};
use arkan::{KanConfig, KanNetwork, MAX_GRID_SIZE};

// =============================================================================
// Grid Size Tests
// =============================================================================

/// Test minimum grid_size = 2
#[test]
fn test_grid_size_2_minimum() {
    let grid_size = 2;
    let order = 3;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Expected: n_knots = grid_size + 2*order + 1 = 2 + 6 + 1 = 9
    assert_eq!(
        knots.len(),
        grid_size + 2 * order + 1,
        "Wrong number of knots for grid_size=2"
    );

    // Test partition of unity at various points
    let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0];
    for &x in &test_points {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed for grid_size=2 at x={}: sum={}",
            x,
            sum
        );

        // All basis values should be non-negative
        for (i, &b) in basis.iter().enumerate() {
            assert!(
                b >= -1e-7,
                "Negative basis value at x={}: basis[{}]={}",
                x,
                i,
                b
            );
        }
    }

    println!("✓ grid_size=2 minimum: partition of unity OK");
}

/// Test grid_size = 32 (large)
#[test]
fn test_grid_size_32() {
    let grid_size = 32;
    let order = 3;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);
    assert_eq!(knots.len(), grid_size + 2 * order + 1);

    // Test multiple points
    let num_points = 100;
    for i in 0..num_points {
        let x = range.0 + (range.1 - range.0) * (i as f32 / (num_points - 1) as f32);
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed for grid_size=32 at x={}: sum={}",
            x,
            sum
        );
    }

    println!("✓ grid_size=32: partition of unity OK for 100 points");
}

/// Test grid_size = 64 (maximum)
#[test]
fn test_grid_size_64_maximum() {
    let grid_size = 64;
    let order = 3;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);
    assert_eq!(knots.len(), grid_size + 2 * order + 1);

    // Test multiple points
    let num_points = 100;
    for i in 0..num_points {
        let x = range.0 + (range.1 - range.0) * (i as f32 / (num_points - 1) as f32);
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed for grid_size=64 at x={}: sum={}",
            x,
            sum
        );
    }

    println!("✓ grid_size=64 maximum: partition of unity OK for 100 points");
}

/// Test that MAX_GRID_SIZE = 64 in validation
#[test]
fn test_max_grid_size_constant() {
    assert_eq!(MAX_GRID_SIZE, 64, "MAX_GRID_SIZE should be 64");

    // grid_size = 64 should be valid
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 64,
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };
    assert!(config.validate().is_ok(), "grid_size=64 should be valid");

    // grid_size = 65 should be invalid
    let config_invalid = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 65,
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };
    assert!(
        config_invalid.validate().is_err(),
        "grid_size=65 should be invalid"
    );

    println!("✓ MAX_GRID_SIZE validation works correctly");
}

// =============================================================================
// Spline Order Tests
// =============================================================================

/// Test spline order = 5
#[test]
fn test_spline_order_5() {
    let grid_size = 5;
    let order = 5;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Test partition of unity
    let num_points = 50;
    for i in 0..num_points {
        let x = range.0 + (range.1 - range.0) * (i as f32 / (num_points - 1) as f32);
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed for order=5 at x={}: sum={}",
            x,
            sum
        );
    }

    println!("✓ spline_order=5: partition of unity OK");
}

/// Test spline order = 6
#[test]
fn test_spline_order_6() {
    let grid_size = 5;
    let order = 6;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Test partition of unity
    let num_points = 50;
    for i in 0..num_points {
        let x = range.0 + (range.1 - range.0) * (i as f32 / (num_points - 1) as f32);
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed for order=6 at x={}: sum={}",
            x,
            sum
        );
    }

    println!("✓ spline_order=6: partition of unity OK");
}

/// Test order=5 derivatives
#[test]
fn test_derivative_order_5() {
    let grid_size = 5;
    let order = 5;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Test that derivative sum = 0 (partition of unity derivative)
    let test_points = [-0.8, -0.3, 0.0, 0.4, 0.9];
    for &x in &test_points {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        let mut deriv = vec![0.0f32; order + 1];
        compute_basis_and_deriv(x, span, &knots, order, &mut basis, &mut deriv);

        let deriv_sum: f32 = deriv.iter().sum();
        assert!(
            deriv_sum.abs() < 1e-4,
            "Derivative sum should be ~0 for order=5 at x={}: sum={}",
            x,
            deriv_sum
        );
    }

    println!("✓ spline_order=5 derivatives: sum ≈ 0 OK");
}

/// Test order=6 derivatives
#[test]
fn test_derivative_order_6() {
    let grid_size = 5;
    let order = 6;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Test that derivative sum = 0
    let test_points = [-0.8, -0.3, 0.0, 0.4, 0.9];
    for &x in &test_points {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        let mut deriv = vec![0.0f32; order + 1];
        compute_basis_and_deriv(x, span, &knots, order, &mut basis, &mut deriv);

        let deriv_sum: f32 = deriv.iter().sum();
        assert!(
            deriv_sum.abs() < 1e-4,
            "Derivative sum should be ~0 for order=6 at x={}: sum={}",
            x,
            deriv_sum
        );
    }

    println!("✓ spline_order=6 derivatives: sum ≈ 0 OK");
}

// =============================================================================
// Extreme X Value Tests
// =============================================================================

/// Test extremely small x values (near zero, not denormalized)
#[test]
fn test_extreme_x_small() {
    let grid_size = 5;
    let order = 3;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Very small x values still in range
    let small_values = [1e-10, 1e-20, 1e-30, -1e-10, -1e-20, -1e-30];

    for &x in &small_values {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();

        // Should still satisfy partition of unity
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed for extreme small x={:.2e}: sum={}",
            x,
            sum
        );

        // All basis values should be non-negative
        for &b in &basis {
            assert!(b >= -1e-7, "Negative basis for x={:.2e}: b={}", x, b);
        }
    }

    println!("✓ Extreme small x values: OK");
}

/// Test x values at exact boundaries with float precision
#[test]
fn test_x_boundary_precision() {
    let grid_size = 5;
    let order = 3;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Test at exact boundaries and just inside
    let boundary_values = [
        -1.0,
        -1.0 + f32::EPSILON,
        -1.0 + 1e-6,
        1.0 - 1e-6,
        1.0 - f32::EPSILON,
        1.0,
    ];

    for &x in &boundary_values {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed at boundary x={:.15}: sum={}",
            x,
            sum
        );
    }

    println!("✓ Boundary precision test: OK");
}

/// Test x values outside range (should clamp)
#[test]
fn test_x_outside_range() {
    let grid_size = 5;
    let order = 3;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Values outside the range (should be handled gracefully)
    let outside_values = [-10.0, -2.0, -1.001, 1.001, 2.0, 10.0];

    for &x in &outside_values {
        let span = find_span(x, &knots, order, grid_size);

        // Span should be clamped to valid range
        assert!(span >= order, "Span too small for x={}: span={}", x, span);
        assert!(
            span <= grid_size + order - 1,
            "Span too large for x={}: span={}",
            x,
            span
        );

        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        // Even outside range, should not panic
        let sum: f32 = basis.iter().sum();
        // Partition of unity may not hold exactly outside range, but should be finite
        assert!(sum.is_finite(), "Non-finite basis sum for x={}: sum={}", x, sum);
    }

    println!("✓ Outside range handling: OK (no panics)");
}

/// Test very large x values
#[test]
fn test_extreme_x_large() {
    let grid_size = 5;
    let order = 3;
    // Use a range that includes large values
    let range = (-100.0f32, 100.0f32);

    let knots = compute_knots(grid_size, order, range);

    let test_values = [-99.9, -50.0, 0.0, 50.0, 99.9];

    for &x in &test_values {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Partition of unity failed for large-range x={}: sum={}",
            x,
            sum
        );
    }

    println!("✓ Large range values: OK");
}

/// Test with very wide range
#[test]
fn test_very_wide_range() {
    let grid_size = 5;
    let order = 3;
    let range = (-1000.0f32, 1000.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Test at various points in the wide range
    let test_values = [-999.0, -500.0, -0.001, 0.0, 0.001, 500.0, 999.0];

    for &x in &test_values {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Partition of unity failed for wide-range x={}: sum={}",
            x,
            sum
        );
    }

    println!("✓ Very wide range (-1000, 1000): OK");
}

// =============================================================================
// Network Tests with Extended Grid Sizes
// =============================================================================

/// Test network forward with grid_size = 32
#[test]
fn test_network_forward_grid_32() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 32,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let input = vec![0.5f32; 4 * 4]; // batch of 4
    let mut output = vec![0.0f32; 4 * 2];

    network.forward_batch(&input, &mut output, &mut workspace);

    // Check all outputs are finite
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite output at index {}: {}", i, val);
    }

    println!("✓ Network forward with grid_size=32: OK");
}

/// Test network forward with grid_size = 64
#[test]
fn test_network_forward_grid_64() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 64,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        ..Default::default()
    };

    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let input = vec![0.5f32; 4 * 4]; // batch of 4
    let mut output = vec![0.0f32; 4 * 2];

    network.forward_batch(&input, &mut output, &mut workspace);

    // Check all outputs are finite
    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite output at index {}: {}", i, val);
    }

    println!("✓ Network forward with grid_size=64: OK");
}

/// Test network training with grid_size = 32
#[test]
fn test_network_train_grid_32() {
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 32,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(42),
        ..Default::default()
    };

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(4);

    let input = vec![0.5f32; 4 * 4];
    let target = vec![0.1f32; 4 * 2];

    // Get initial loss
    let mut output = vec![0.0f32; 4 * 2];
    network.forward_batch(&input, &mut output, &mut workspace);
    let initial_loss: f32 = output
        .iter()
        .zip(target.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f32>()
        / output.len() as f32;

    // Train for a few steps
    for _ in 0..10 {
        network.train_step(&input, &target, None, 0.01, &mut workspace);
    }

    // Get final loss
    network.forward_batch(&input, &mut output, &mut workspace);
    let final_loss: f32 = output
        .iter()
        .zip(target.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f32>()
        / output.len() as f32;

    assert!(
        final_loss < initial_loss,
        "Training should reduce loss: initial={}, final={}",
        initial_loss,
        final_loss
    );

    println!(
        "✓ Network training with grid_size=32: loss {} → {}",
        initial_loss, final_loss
    );
}

// =============================================================================
// Denormalized Float Tests
// =============================================================================

/// Test behavior with denormalized floats (subnormal numbers)
#[test]
fn test_denormalized_floats() {
    let grid_size = 5;
    let order = 3;
    let range = (-1.0f32, 1.0f32);

    let knots = compute_knots(grid_size, order, range);

    // Subnormal numbers (smallest positive f32 values)
    let subnormal = f32::MIN_POSITIVE / 2.0;
    assert!(subnormal > 0.0 && subnormal < f32::MIN_POSITIVE);

    // Test with subnormal x values
    let test_values = [subnormal, -subnormal, subnormal * 100.0, -subnormal * 100.0];

    for &x in &test_values {
        let span = find_span(x, &knots, order, grid_size);
        let mut basis = vec![0.0f32; order + 1];
        compute_basis(x, span, &knots, order, &mut basis);

        let sum: f32 = basis.iter().sum();

        // Should still work (all these values are essentially 0)
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Partition of unity failed for subnormal x={:.2e}: sum={}",
            x,
            sum
        );
    }

    println!("✓ Denormalized float handling: OK");
}

// =============================================================================
// Combined Stress Test
// =============================================================================

/// Stress test combining large grid_size and high order
#[test]
fn test_stress_large_grid_high_order() {
    // Maximum supported configuration
    let grid_size = 64;
    let order = 6; // Near maximum order

    let range = (-1.0f32, 1.0f32);
    let knots = compute_knots(grid_size, order, range);

    // Test multiple points
    let num_points = 50;
    let mut max_error: f32 = 0.0;

    for i in 0..num_points {
        let x = range.0 + (range.1 - range.0) * (i as f32 / (num_points - 1) as f32);
        let span = find_span(x, &knots, order, grid_size);

        let mut basis = vec![0.0f32; order + 1];
        let mut deriv = vec![0.0f32; order + 1];
        compute_basis_and_deriv(x, span, &knots, order, &mut basis, &mut deriv);

        let basis_sum: f32 = basis.iter().sum();
        let deriv_sum: f32 = deriv.iter().sum();

        let basis_error = (basis_sum - 1.0).abs();
        max_error = max_error.max(basis_error);

        assert!(
            basis_error < 1e-4,
            "Partition of unity failed at x={}: sum={}, error={}",
            x,
            basis_sum,
            basis_error
        );

        assert!(
            deriv_sum.abs() < 1e-3,
            "Derivative sum should be ~0 at x={}: sum={}",
            x,
            deriv_sum
        );
    }

    println!(
        "✓ Stress test (grid=64, order=6): max basis error = {:.2e}",
        max_error
    );
}
