use arkan::loss::{
    masked_categorical_cross_entropy_with_logits, masked_mse_into, pde_residual_loss, r_squared,
};

#[test]
fn mse_normalizes_tiny_masks_before_narrowing_and_clears_inactive_slots() {
    for weight in [f32::from_bits(1), 1e-40, f32::MAX] {
        let mut gradient = [99.0; 2];
        let loss = masked_mse_into(
            &[1.0, f32::NAN],
            &[0.0, f32::INFINITY],
            Some(&[weight, 0.0]),
            &mut gradient,
        )
        .unwrap();
        assert_eq!(loss, 1.0, "mask {weight}");
        assert_eq!(gradient[0], 2.0, "mask {weight}");
        assert_eq!(gradient[1].to_bits(), 0);
        let (pde_loss, pde_gradient) = pde_residual_loss(&[1.0, f32::NAN], Some(&[weight, 0.0]));
        assert_eq!(pde_loss, loss);
        assert_eq!(pde_gradient, gradient);
    }
}

#[test]
fn mse_normalizes_large_mask_sums_and_residuals_in_widened_arithmetic() {
    let mut gradient = [99.0; 2];
    let loss =
        masked_mse_into(&[1.0, 2.0], &[0.0; 2], Some(&[f32::MAX; 2]), &mut gradient).unwrap();
    assert_eq!(loss, 2.5);
    assert_eq!(gradient, [1.0, 2.0]);
    let loss = masked_mse_into(
        &[f32::MAX, 0.0],
        &[-f32::MAX, 0.0],
        Some(&[f32::from_bits(1), 1.0]),
        &mut gradient,
    )
    .unwrap();
    // The residual overflows f32, but its weighted square and derivative fit.
    assert!(loss.is_finite() && loss > 0.0);
    assert!(gradient[0].is_finite() && gradient[0] > 0.0);
    assert_eq!(gradient[1], 0.0);
}

#[test]
fn logits_gradients_cancel_tiny_and_large_single_sample_masks() {
    for weight in [f32::from_bits(1), 1e-40, f32::MAX] {
        let (loss, gradient) = masked_categorical_cross_entropy_with_logits(
            &[0.0, 0.0],
            &[1.0, 0.0],
            2,
            Some(&[weight]),
        );
        assert!((loss - std::f32::consts::LN_2).abs() < 1e-6);
        assert_eq!(gradient, [-0.5, 0.5], "mask {weight}");
    }
}

#[test]
fn logits_normalize_large_mask_sums_and_preserve_inactive_zero() {
    let (loss, gradient) = masked_categorical_cross_entropy_with_logits(
        &[0.0, 0.0, 0.0, 0.0, f32::NAN, f32::INFINITY],
        &[1.0, 0.0, 0.0, 1.0, f32::NAN, f32::INFINITY],
        2,
        Some(&[f32::MAX, f32::MAX, 0.0]),
    );
    assert!((loss - std::f32::consts::LN_2).abs() < 1e-6);
    assert_eq!(gradient, [-0.25, 0.25, 0.25, -0.25, 0.0, 0.0]);
    assert_eq!(gradient[4].to_bits(), 0);
    assert_eq!(gradient[5].to_bits(), 0);
}

#[test]
fn zero_active_weight_clears_gradients_without_reading_invalid_values() {
    let mut gradient = [99.0; 2];
    assert_eq!(
        masked_mse_into(
            &[f32::NAN; 2],
            &[f32::NAN; 2],
            Some(&[0.0, -1.0]),
            &mut gradient
        )
        .unwrap()
        .to_bits(),
        0
    );
    assert!(gradient.iter().all(|value| value.to_bits() == 0));
    let (loss, gradient) = masked_categorical_cross_entropy_with_logits(
        &[f32::NAN; 2],
        &[f32::NAN; 2],
        2,
        Some(&[0.0]),
    );
    assert_eq!(loss.to_bits(), 0);
    assert!(gradient.iter().all(|value| value.to_bits() == 0));
}

#[test]
fn r_squared_is_scale_independent_for_nonconstant_targets() {
    for scale in [f32::from_bits(2), 1e-20, 1e-4, 1.0, 1e20, 1e38] {
        assert_eq!(
            r_squared(&[scale * 0.5; 2], &[0.0, scale]),
            0.0,
            "scale {scale}"
        );
        assert_eq!(
            r_squared(&[0.0, scale], &[0.0, scale]),
            1.0,
            "scale {scale}"
        );
        assert_eq!(
            r_squared(&[scale, 0.0], &[0.0, scale]),
            -3.0,
            "scale {scale}"
        );
    }
}

#[test]
fn r_squared_constant_targets_require_actual_zero_error() {
    assert_eq!(r_squared(&[1e-20; 2], &[1e-20; 2]), 1.0);
    assert_eq!(r_squared(&[0.0; 2], &[1e-20; 2]), 0.0);
    assert_eq!(r_squared(&[], &[]), 0.0);
    assert_eq!(r_squared(&[f32::MAX; 2], &[f32::MAX; 2]), 1.0);
}

#[test]
fn logits_masks_preserve_nan_propagation_and_inactive_zero() {
    for inactive in [f32::NEG_INFINITY, -1.0, -0.0, 0.0] {
        let (loss, gradient) = masked_categorical_cross_entropy_with_logits(
            &[0.0, 0.0, 0.0, 0.0, f32::NAN, f32::INFINITY],
            &[1.0, 0.0, 1.0, 0.0, f32::NAN, f32::INFINITY],
            2,
            Some(&[f32::NAN, 2.0, inactive]),
        );
        assert!(loss.is_nan());
        assert!(gradient[0].is_nan() && gradient[1].is_nan());
        // A NaN total leaves the other active sample's derivative unnormalized.
        assert_eq!(&gradient[2..4], &[-1.0, 1.0]);
        assert_eq!(gradient[4].to_bits(), 0);
        assert_eq!(gradient[5].to_bits(), 0);
    }
    let (loss, gradient) = masked_categorical_cross_entropy_with_logits(
        &[0.0; 4],
        &[1.0, 0.0, 1.0, 0.0],
        2,
        Some(&[f32::INFINITY, 2.0]),
    );
    assert!(loss.is_nan());
    assert!(gradient[0].is_nan() && gradient[1].is_nan());
    assert_eq!(gradient[2].to_bits(), (-0.0f32).to_bits());
    assert_eq!(gradient[3].to_bits(), 0);
}
