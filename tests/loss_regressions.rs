use arkan::loss::*;
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
