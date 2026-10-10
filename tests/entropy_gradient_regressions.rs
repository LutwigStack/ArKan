use arkan::loss::entropy_regularization_gradient;

#[test]
fn inactive_entropy_coefficients_preserve_signed_zero_and_ignored_tail() {
    let coefficients = [1.0, 0.0, -0.0, f32::from_bits(0x7fc0_1234)];
    let before = coefficients.map(f32::to_bits);
    let gradient = entropy_regularization_gradient(&coefficients, 3);
    assert_eq!(gradient.len(), coefficients.len());
    assert!(gradient[0].is_finite() && gradient[0] < 0.0);
    assert_eq!(gradient[1].to_bits(), 0.0_f32.to_bits());
    assert_eq!(gradient[2].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(gradient[3].to_bits(), 0.0_f32.to_bits());
    for group_size in [0, coefficients.len() + 1, usize::MAX] {
        let inactive = entropy_regularization_gradient(&coefficients, group_size);
        assert_eq!(inactive.len(), coefficients.len());
        assert!(inactive.iter().all(|value| value.to_bits() == 0));
    }
    assert_eq!(coefficients.map(f32::to_bits), before);
}
