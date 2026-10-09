use arkan::loss::*;

// Frozen allocating objective from accepted C 3c129df; keep its storage/reduction.
fn frozen_c_entropy(coefficients: &[f32], group_size: usize) -> f32 {
    const EPSILON: f32 = 1e-6;
    if coefficients.is_empty() || group_size == 0 {
        return 0.0;
    }
    let num_groups = coefficients.len() / group_size;
    if num_groups == 0 {
        return 0.0;
    }
    let mut total_entropy = 0.0f32;
    for g in 0..num_groups {
        let group = &coefficients[g * group_size..(g + 1) * group_size];
        let squared: Vec<f32> = group.iter().map(|c| c * c).collect();
        let sum: f32 = squared.iter().sum::<f32>() + EPSILON;
        let mut entropy = 0.0f32;
        for &sq in &squared {
            let p = sq / sum;
            if p > EPSILON {
                entropy -= p * p.ln();
            }
        }
        total_entropy += entropy;
    }
    total_entropy / num_groups as f32
}

fn assert_same_float(actual: f32, expected: f32, context: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{context}: {actual} should be NaN");
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}");
    }
}

#[test]
fn entropy_and_combined_loss_preserve_frozen_objective_bits() {
    use std::hint::black_box;

    // For [1,c], the real cutoff has c² = epsilon*(1+epsilon)/(1-epsilon).
    // Find its neighboring f32 inputs and verify the actual rounded straddle.
    let probability = |c: f32| c * c / ([1.0, c * c].iter().sum::<f32>() + 1e-6);
    let center = 0.001f32.to_bits();
    let above_bits = (center - 32..=center + 32)
        .find(|&bits| probability(f32::from_bits(bits)) > 1e-6)
        .expect("cutoff lies within the analytically derived bracket");
    let below = f32::from_bits(above_bits - 1);
    let above = f32::from_bits(above_bits);
    assert_eq!(probability(below).to_bits(), 1e-6f32.to_bits());
    assert!(probability(above) > 1e-6);
    assert_eq!(above.to_bits() - below.to_bits(), 1);
    println!(
        "adjacent cutoff coefficients bits={} / {}, probabilities={} / {}",
        below.to_bits(),
        above.to_bits(),
        probability(below),
        probability(above)
    );

    let coefficients: Vec<f32> = (0..18).map(|i| (i as f32 - 7.0) * 0.13).collect();
    let cases = [
        ("nonconstant groups", coefficients.clone(), 4),
        ("complete groups", coefficients[..16].to_vec(), 4),
        ("whole group", coefficients.clone(), 18),
        ("singleton groups", coefficients.clone(), 1),
        ("empty", vec![], 4),
        ("zero group", coefficients.clone(), 0),
        ("oversized group", coefficients, 19),
        ("signed zeros", vec![0.0, -0.0, -0.0, 0.0], 2),
        ("uneven magnitudes", vec![1e-10, -1e5, 0.3, 8.0], 4),
        ("inactive cutoff neighbor", vec![1.0, below], 2),
        ("active cutoff neighbor", vec![1.0, above], 2),
        ("subnormal square", vec![1e-22, -1e-22, 0.0, -0.0], 4),
        ("underflow square", vec![f32::from_bits(1), -1e-30], 2),
        ("overflow square", vec![f32::MAX, 0.5, -1e20, 1.0], 4),
        ("NaN group", vec![f32::NAN, 0.25, 0.5, 1.0], 4),
        ("positive infinity group", vec![f32::INFINITY, 0.25], 2),
        ("negative infinity group", vec![f32::NEG_INFINITY, 0.25], 2),
        ("ignored NaN tail", vec![0.2, -0.7, 0.4, 0.9, f32::NAN], 4),
        (
            "ignored infinite tail",
            vec![0.2, -0.7, 0.4, 0.9, f32::INFINITY],
            4,
        ),
    ];
    let configs = [
        KanLossConfig {
            lambda_l1: 0.3,
            lambda_entropy: 0.7,
            lambda_smooth: 0.2,
        },
        KanLossConfig {
            lambda_l1: 0.3,
            lambda_entropy: 0.0,
            lambda_smooth: 0.2,
        },
        KanLossConfig {
            lambda_l1: 0.0,
            lambda_entropy: 0.7,
            lambda_smooth: 0.0,
        },
    ];
    for (name, coefficients, group_size) in cases {
        let coefficients = black_box(coefficients.as_slice());
        let group_size = black_box(group_size);
        let entropy = frozen_c_entropy(coefficients, group_size);
        assert_same_float(
            entropy_regularization(coefficients, group_size),
            entropy,
            name,
        );

        for config in &configs {
            // These MSE/gradient literals are hand-derived from the active residuals.
            let predictions = [0.5, f32::NAN, -1.25, f32::INFINITY];
            let targets = [-0.25, 2.0, 0.5, -1.0];
            let weighted_mask = [0.5, 0.0, 1.5, -1.0];
            let inactive_mask = [0.0; 4];
            let mse_cases = [
                (
                    &[0.5, -1.25][..],
                    &[-0.25, 0.5][..],
                    None,
                    1.8125,
                    &[0.75, -1.75][..],
                ),
                (
                    &predictions[..],
                    &targets[..],
                    Some(&weighted_mask[..]),
                    2.4375,
                    &[0.375, 0.0, -2.625, 0.0][..],
                ),
                (
                    &predictions[..],
                    &targets[..],
                    Some(&inactive_mask[..]),
                    0.0,
                    &[0.0; 4][..],
                ),
            ];
            for (predictions, targets, mask, pred, gradient) in mse_cases {
                let l1 = l1_sparsity_loss(coefficients);
                let smooth = smoothness_penalty(coefficients, group_size);
                let reg = config.lambda_l1 * l1
                    + config.lambda_entropy * entropy
                    + config.lambda_smooth * smooth;
                let (actual_total, actual_pred, actual_reg, actual_gradient) =
                    kan_combined_loss(predictions, targets, coefficients, group_size, config, mask);
                assert_same_float(actual_total, pred + reg, name);
                assert_same_float(actual_pred, pred, name);
                assert_same_float(actual_reg, reg, name);
                assert_eq!(actual_gradient.len(), gradient.len());
                for (&actual, &expected) in actual_gradient.iter().zip(gradient) {
                    assert_same_float(actual, expected, name);
                }
            }
        }
    }
}

#[test]
fn tiny_nonzero_rmse_keeps_chain_rule_gradient() {
    let (loss, grad) = masked_rmse(&[1e-7], &[0.0], None);
    assert!((loss - 1e-7).abs() < 1e-12);
    assert!((grad[0] - 1.0).abs() < 1e-6);
    assert_eq!(masked_rmse(&[0.0], &[0.0], None).1, vec![0.0]);
}
#[test]
fn regularization_gradient_matches_entire_combined_objective() {
    let config = KanLossConfig {
        lambda_l1: 0.3,
        lambda_entropy: 0.7,
        lambda_smooth: 0.2,
    };
    let coefficients = vec![0.3, -0.9, 0.4, 0.6, -0.7, 0.2];
    let gradient = kan_regularization_gradient(&coefficients, 3, &config);
    for i in 0..coefficients.len() {
        let mut plus = coefficients.clone();
        plus[i] += 1e-3;
        let mut minus = coefficients.clone();
        minus[i] -= 1e-3;
        let numeric = (kan_combined_loss(&[0.0], &[0.0], &plus, 3, &config, None).0
            - kan_combined_loss(&[0.0], &[0.0], &minus, 3, &config, None).0)
            / 2e-3;
        assert!(
            (gradient[i] - numeric).abs() < 3e-4,
            "{i}: {} vs {numeric}",
            gradient[i]
        );
    }
}

#[test]
fn categorical_probability_gradient_matches_independent_probability_differences() {
    let probabilities = [0.2, 0.8];
    let targets = [1.0, 0.0];
    let (_, grad) =
        masked_categorical_cross_entropy_probabilities(&probabilities, &targets, 2, None);
    for i in 0..2 {
        let mut plus = probabilities;
        plus[i] += 1e-3;
        let mut minus = probabilities;
        minus[i] -= 1e-3;
        let numeric = (masked_categorical_cross_entropy_probabilities(&plus, &targets, 2, None).0
            - masked_categorical_cross_entropy_probabilities(&minus, &targets, 2, None).0)
            / 2e-3;
        assert!((grad[i] - numeric).abs() < 1e-3);
    }
}
#[test]
fn poker_probability_gradient_matches_probability_differences() {
    let mut predictions = [0.0; 24];
    predictions[0] = 0.2;
    let mut targets = [0.0; 24];
    targets[0] = 1.0;
    targets[16] = 1.0;
    let (_, _, _, grad) = poker_combined_loss_probabilities(&predictions, &targets, 0.5);
    let mut plus = predictions;
    plus[0] += 1e-3;
    let mut minus = predictions;
    minus[0] -= 1e-3;
    let numeric = (poker_combined_loss_probabilities(&plus, &targets, 0.5).0
        - poker_combined_loss_probabilities(&minus, &targets, 0.5).0)
        / 2e-3;
    assert!((grad[0] - numeric).abs() < 1e-3);
}
#[test]
fn categorical_logits_objective_is_stable_and_has_logit_derivative() {
    fn objective(logits: &[f32], targets: &[f32]) -> (f32, Vec<f32>) {
        masked_categorical_cross_entropy_with_logits(logits, targets, 2, None)
    }
    let (loss, gradient) = objective(&[1000.0, -1000.0], &[0.0, 1.0]);
    assert!((loss - 2000.0).abs() < 1e-4);
    assert_eq!(gradient, vec![1.0, -1.0]);
    let logits = [0.2, -0.8];
    let targets = [0.25, 0.75];
    let (_, grad) = objective(&logits, &targets);
    for i in 0..2 {
        let mut plus = logits;
        plus[i] += 1e-3;
        let mut minus = logits;
        minus[i] -= 1e-3;
        let numeric = (objective(&plus, &targets).0 - objective(&minus, &targets).0) / 2e-3;
        assert!((grad[i] - numeric).abs() < 1e-4);
    }
}
#[test]
fn rmse_avoids_squaring_underflow_and_overflow_for_finite_residuals() {
    for residual in [1e-30, 1e20] {
        let (loss, gradient) = masked_rmse(&[residual], &[0.0], None);
        assert!((loss / residual - 1.0).abs() < 1e-6);
        assert!((gradient[0] - 1.0).abs() < 1e-6);
    }
}
