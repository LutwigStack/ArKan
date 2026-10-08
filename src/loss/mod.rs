//! Loss functions with masking support.
//!
//! This module provides loss functions for training KAN networks:
//!
//! # Standard Task-Specific Losses
//!
//! **Regression:**
//! - [`masked_mse`] - Mean Squared Error (standard for regression)
//! - [`masked_rmse`] - Root MSE (error in original units)
//! - [`masked_mae`] - Mean Absolute Error (robust to outliers)
//! - [`masked_huber`] - Huber loss (smooth L1, combines MSE and MAE)
//!
//! **Classification:**
//! - [`masked_cross_entropy`] - Cross-Entropy for probability outputs
//! - [`masked_bce_with_logits`] - Binary Cross-Entropy with logits (numerically stable)
//!
//! **Game/RL:**
//! - [`poker_combined_loss`] - Specialized loss for poker Q-learning
//!
//! # KAN-Specific Regularization
//!
//! These are critical for KAN to find interpretable formulas:
//!
//! - [`l1_sparsity_loss`] - L1 norm of spline coefficients (promotes sparsity)
//! - [`entropy_regularization`] - Encourages selecting one activation function
//! - [`smoothness_penalty`] - Second derivative penalty (prevents overfitting)
//! - [`kan_combined_loss`] - All-in-one: task loss + sparsity + entropy + smoothness
//!
//! # Masking
//!
//! All loss functions support optional masks to:
//! - Handle variable-length sequences
//! - Ignore invalid outputs
//! - Implement multi-task learning
//!
//! # Example
//!
//! ```rust
//! use arkan::loss::masked_mse;
//!
//! let predictions = vec![0.5, 1.0, 1.5];
//! let targets = vec![0.0, 1.0, 2.0];
//! let mask = vec![1.0, 1.0, 0.0]; // Ignore last element
//!
//! let (loss, grad) = masked_mse(&predictions, &targets, Some(&mask));
//! ```
//!
//! # KAN Regularization Example
//!
//! ```rust
//! use arkan::loss::{masked_mse, l1_sparsity_loss, smoothness_penalty, kan_combined_loss, KanLossConfig};
//!
//! // Spline coefficients from a KAN layer
//! let coefficients = vec![0.1, 0.0, -0.2, 0.5, 0.0, 0.0, 0.3, -0.1];
//!
//! // L1 regularization encourages sparsity
//! let l1_loss = l1_sparsity_loss(&coefficients);
//!
//! // Smoothness penalty (using coefficient differences as approximation)
//! let smooth_loss = smoothness_penalty(&coefficients, 4); // 4 basis functions per input
//!
//! // Or use the combined loss helper
//! let predictions = vec![0.5, 1.0];
//! let targets = vec![0.6, 0.9];
//!
//! let config = KanLossConfig {
//!     lambda_l1: 0.001,        // Sparsity weight
//!     lambda_entropy: 0.0001,  // Entropy weight
//!     lambda_smooth: 0.001,    // Smoothness weight
//! };
//!
//! let (total, pred_loss, reg_loss, grad) = kan_combined_loss(
//!     &predictions, &targets, &coefficients, 4, &config, None
//! );
//! ```

use crate::config::EPSILON;

mod classification;
mod regression;
mod regularization;

pub use classification::*;
pub use regression::*;
pub use regularization::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_mse() {
        let pred = vec![1.0, 2.0, 3.0, 4.0];
        let target = vec![1.0, 2.0, 3.0, 4.0];

        let (loss, grad) = masked_mse(&pred, &target, None);
        assert!(loss < EPSILON);
        assert!(grad.iter().all(|&g| g.abs() < EPSILON));
    }

    #[test]
    fn test_masked_mse_with_mask() {
        let pred = vec![0.0, 1.0, 0.0, 1.0];
        let target = vec![1.0, 1.0, 1.0, 1.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0]; // Only positions 0 and 2 active

        let (loss, grad) = masked_mse(&pred, &target, Some(&mask));

        // Only (0-1)^2 and (0-1)^2 = 1 + 1 = 2, mean = 1.0
        assert!((loss - 1.0).abs() < EPSILON);

        // Masked positions should have zero gradient
        assert!(grad[1].abs() < EPSILON);
        assert!(grad[3].abs() < EPSILON);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x, 3);

        // Sum should be 1.0
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < EPSILON);

        // Should be monotonically increasing
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_masked_softmax() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];

        masked_softmax(&mut x, &mask, 4);

        // Masked positions should be zero
        assert!(x[1] < EPSILON);
        assert!(x[3] < EPSILON);

        // Sum of active positions should be ~1.0
        let sum = x[0] + x[2];
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_huber_loss() {
        let pred = vec![0.0, 0.0, 0.0];
        let target = vec![0.5, 2.0, 10.0]; // Small, medium, large errors

        let (loss1, _) = masked_huber(&pred, &target, 1.0, None);
        let (loss2, _) = masked_mse(&pred, &target, None);

        // Huber should be smaller than MSE for large errors
        assert!(loss1 < loss2);
    }

    #[test]
    fn test_poker_combined_loss() {
        // Create dummy predictions and targets
        let batch_size = 2;
        let mut predictions = vec![0.0f32; batch_size * 24];
        let mut targets = vec![0.0f32; batch_size * 24];

        // Set some probabilities and Q-values
        for b in 0..batch_size {
            let base = b * 24;
            // Probabilities (softmax-like)
            predictions[base] = 0.5;
            predictions[base + 1] = 0.5;
            targets[base] = 0.6;
            targets[base + 1] = 0.4;

            // Q-values
            predictions[base + 8] = 0.1;
            predictions[base + 9] = -0.1;
            targets[base + 8] = 0.2;
            targets[base + 9] = -0.2;

            // Mask (only first two actions active)
            targets[base + 16] = 1.0;
            targets[base + 17] = 1.0;
        }

        let (total, prob_loss, q_loss, grad) = poker_combined_loss(&predictions, &targets, 0.5);

        assert!(total.is_finite());
        assert!(prob_loss.is_finite());
        assert!(q_loss.is_finite());
        assert!(grad.iter().all(|&g| g.is_finite()));
    }

    // =========================================================================
    // RMSE Tests
    // =========================================================================

    #[test]
    fn test_rmse_perfect() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];

        let (loss, grad) = masked_rmse(&pred, &target, None);
        assert!(loss < EPSILON);
        assert!(grad.iter().all(|&g| g.abs() < EPSILON));
    }

    #[test]
    fn test_rmse_value() {
        let pred = vec![0.0, 0.0];
        let target = vec![1.0, 1.0];

        let (loss, _) = masked_rmse(&pred, &target, None);
        // MSE = 1.0, RMSE = 1.0
        assert!((loss - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rmse_vs_mse() {
        let pred = vec![0.0, 0.0, 0.0];
        let target = vec![2.0, 2.0, 2.0];

        let (mse, _) = masked_mse(&pred, &target, None);
        let (rmse, _) = masked_rmse(&pred, &target, None);

        // RMSE should be sqrt(MSE)
        assert!((rmse - mse.sqrt()).abs() < EPSILON);
    }

    // =========================================================================
    // MAE Tests
    // =========================================================================

    #[test]
    fn test_mae_perfect() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];

        let (loss, grad) = masked_mae(&pred, &target, None);
        assert!(loss < EPSILON);
        assert!(grad.iter().all(|&g| g.abs() < EPSILON));
    }

    #[test]
    fn test_mae_value() {
        let pred = vec![0.0, 0.0, 0.0];
        let target = vec![1.0, 2.0, 3.0];

        let (loss, _) = masked_mae(&pred, &target, None);
        // MAE = (1 + 2 + 3) / 3 = 2.0
        assert!((loss - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_mae_robust_to_outliers() {
        let pred = vec![0.0, 0.0, 0.0];
        let target_normal = vec![1.0, 1.0, 1.0];
        let target_outlier = vec![1.0, 1.0, 100.0]; // One outlier

        let (mae_normal, _) = masked_mae(&pred, &target_normal, None);
        let (mae_outlier, _) = masked_mae(&pred, &target_outlier, None);
        let (mse_outlier, _) = masked_mse(&pred, &target_outlier, None);

        // MAE should be less affected by outlier than MSE
        let mae_ratio = mae_outlier / mae_normal;
        let mse_ratio = mse_outlier / masked_mse(&pred, &target_normal, None).0;

        assert!(mae_ratio < mse_ratio);
    }

    // =========================================================================
    // BCE with Logits Tests
    // =========================================================================

    #[test]
    fn test_bce_logits_confident_correct() {
        // High logit for class 1, target is 1
        let logits = vec![5.0];
        let targets = vec![1.0];

        let (loss, _) = masked_bce_with_logits(&logits, &targets, None);
        // Should be low loss
        assert!(loss < 0.01);
    }

    #[test]
    fn test_bce_logits_confident_wrong() {
        // High logit for class 1, target is 0
        let logits = vec![5.0];
        let targets = vec![0.0];

        let (loss, _) = masked_bce_with_logits(&logits, &targets, None);
        // Should be high loss
        assert!(loss > 4.0);
    }

    #[test]
    fn test_bce_logits_gradient() {
        let logits = vec![0.0]; // sigmoid(0) = 0.5
        let targets = vec![1.0];

        let (_, grad) = masked_bce_with_logits(&logits, &targets, None);
        // Gradient = sigmoid(0) - 1 = 0.5 - 1 = -0.5
        assert!((grad[0] - (-0.5)).abs() < 0.01);
    }

    // =========================================================================
    // L1 Sparsity Tests
    // =========================================================================

    #[test]
    fn test_l1_all_zeros() {
        let coeffs = vec![0.0, 0.0, 0.0, 0.0];
        let loss = l1_sparsity_loss(&coeffs);
        assert!(loss < EPSILON);
    }

    #[test]
    fn test_l1_value() {
        let coeffs = vec![0.5, 0.0, -0.3, 0.0, 0.1];
        let loss = l1_sparsity_loss(&coeffs);
        // L1 = (0.5 + 0 + 0.3 + 0 + 0.1) / 5 = 0.18
        assert!((loss - 0.18).abs() < 0.001);
    }

    #[test]
    fn test_l1_gradient() {
        let coeffs = vec![0.5, -0.3, 0.0];
        let grad = l1_sparsity_gradient(&coeffs);

        // sign(0.5) / 3 = 1/3
        assert!((grad[0] - 1.0 / 3.0).abs() < 0.01);
        // sign(-0.3) / 3 = -1/3
        assert!((grad[1] - (-1.0 / 3.0)).abs() < 0.01);
        // sign(0) / 3 = 0
        assert!(grad[2].abs() < EPSILON);
    }

    // =========================================================================
    // Entropy Regularization Tests
    // =========================================================================

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution should have high entropy
        let coeffs = vec![0.5, 0.5, 0.5, 0.5];
        let entropy = entropy_regularization(&coeffs, 4);
        assert!(entropy > 1.0);
    }

    #[test]
    fn test_entropy_concentrated() {
        // Concentrated should have low entropy
        let coeffs = vec![1.0, 0.0, 0.0, 0.0];
        let entropy = entropy_regularization(&coeffs, 4);
        assert!(entropy < 0.01);
    }

    #[test]
    fn test_entropy_comparison() {
        let uniform = vec![0.5, 0.5, 0.5, 0.5];
        let concentrated = vec![1.0, 0.01, 0.01, 0.01];

        let h_uniform = entropy_regularization(&uniform, 4);
        let h_concentrated = entropy_regularization(&concentrated, 4);

        assert!(h_concentrated < h_uniform);
    }

    // =========================================================================
    // Smoothness Penalty Tests
    // =========================================================================

    #[test]
    fn test_smoothness_linear() {
        // Linear coefficients: [0, 1, 2, 3, 4] should have zero second derivative
        let coeffs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let penalty = smoothness_penalty(&coeffs, 5);
        assert!(penalty < EPSILON);
    }

    #[test]
    fn test_smoothness_oscillating() {
        // Oscillating should have high penalty
        let coeffs = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let penalty = smoothness_penalty(&coeffs, 5);
        assert!(penalty > 0.5);
    }

    #[test]
    fn test_smoothness_comparison() {
        let smooth = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let rough = vec![0.1, 0.5, 0.1, 0.5, 0.1];

        let s_smooth = smoothness_penalty(&smooth, 5);
        let s_rough = smoothness_penalty(&rough, 5);

        assert!(s_smooth < s_rough);
    }

    // =========================================================================
    // KAN Combined Loss Tests
    // =========================================================================

    #[test]
    fn test_kan_combined_basic() {
        let predictions = vec![0.5, 1.0];
        let targets = vec![0.6, 1.1];
        let coefficients = vec![0.1, 0.0, -0.2, 0.5, 0.0, 0.0, 0.3, -0.1];

        let config = KanLossConfig::default();

        let (total, pred, reg, grad) =
            kan_combined_loss(&predictions, &targets, &coefficients, 4, &config, None);

        assert!(total.is_finite());
        assert!(pred.is_finite());
        assert!(reg.is_finite());
        assert!(grad.len() == predictions.len());
        assert!(grad.iter().all(|&g| g.is_finite()));

        // Total should be >= pred (regularization adds)
        assert!(total >= pred - EPSILON);
    }

    #[test]
    fn test_kan_combined_zero_reg() {
        let predictions = vec![0.5, 1.0];
        let targets = vec![0.6, 1.1];
        let coefficients = vec![0.0; 8];

        let config = KanLossConfig {
            lambda_l1: 0.0,
            lambda_entropy: 0.0,
            lambda_smooth: 0.0,
        };

        let (total, pred, reg, _) =
            kan_combined_loss(&predictions, &targets, &coefficients, 4, &config, None);

        // With zero lambdas and zero coeffs, reg should be minimal
        assert!(reg < EPSILON);
        assert!((total - pred).abs() < EPSILON);
    }

    // =========================================================================
    // R² Tests
    // =========================================================================

    #[test]
    fn test_r_squared_perfect() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];

        let r2 = r_squared(&predictions, &targets);
        assert!((r2 - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_r_squared_mean_predictor() {
        // If we predict the mean, R² should be 0
        let targets = vec![1.0, 2.0, 3.0];
        let mean = 2.0;
        let predictions = vec![mean, mean, mean];

        let r2 = r_squared(&predictions, &targets);
        assert!(r2.abs() < EPSILON);
    }

    #[test]
    fn test_r_squared_good_fit() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 1.9, 3.1];

        let r2 = r_squared(&predictions, &targets);
        assert!(r2 > 0.95);
    }

    // =========================================================================
    // PDE Residual Tests
    // =========================================================================

    #[test]
    fn test_pde_residual_zero() {
        let residuals = vec![0.0, 0.0, 0.0];
        let (loss, _) = pde_residual_loss(&residuals, None);
        assert!(loss < EPSILON);
    }

    #[test]
    fn test_pde_residual_nonzero() {
        let residuals = vec![0.1, -0.1, 0.05];
        let (loss, grad) = pde_residual_loss(&residuals, None);

        assert!(loss > 0.0);
        // Gradient should push residuals toward zero
        assert!(grad[0] > 0.0); // residual is positive, grad positive
        assert!(grad[1] < 0.0); // residual is negative, grad negative
    }

    // =========================================================================
    // Categorical Cross-Entropy Tests
    // =========================================================================

    #[test]
    fn test_categorical_ce_perfect() {
        // Use probabilities slightly below 1.0 to account for clamping
        let predictions = vec![0.99, 0.005, 0.005];
        let targets = vec![1.0, 0.0, 0.0];

        let (loss, _) = masked_categorical_cross_entropy(&predictions, &targets, 3, None);
        // Loss should be very small for confident correct prediction
        assert!(loss < 0.02);
    }

    #[test]
    fn test_categorical_ce_wrong() {
        let predictions = vec![0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0];

        let (loss, _) = masked_categorical_cross_entropy(&predictions, &targets, 3, None);
        // Should be high loss
        assert!(loss > 1.0);
    }

    #[test]
    fn test_categorical_ce_batch() {
        // Two samples
        let predictions = vec![0.9, 0.1, 0.1, 0.9]; // 2 samples x 2 classes
        let targets = vec![1.0, 0.0, 0.0, 1.0];

        let (loss, grad) = masked_categorical_cross_entropy(&predictions, &targets, 2, None);

        assert!(loss.is_finite());
        assert!(loss < 0.5); // Good predictions
        assert_eq!(grad.len(), 4);
    }

    // =========================================================================
    // Cross-Entropy PyTorch Parity Tests
    // =========================================================================
    //
    // These tests compare masked_cross_entropy (binary CE) with PyTorch's
    // F.binary_cross_entropy. Reference values generated via:
    //   python -c "import torch; import torch.nn.functional as F; ..."
    //
    // Formula: BCE = -Σ[t*log(p) + (1-t)*log(1-p)] / n

    #[test]
    fn test_cross_entropy_pytorch_perfect_prediction() {
        // PyTorch: pred=[0.9,0.1], target=[1.0,0.0], loss=0.10536053776741028
        let predictions = vec![0.9f32, 0.1];
        let targets = vec![1.0f32, 0.0];

        let (loss, _) = masked_cross_entropy(&predictions, &targets, None);

        // PyTorch reference: 0.10536053776741028
        let pytorch_loss = 0.10536054f32;
        let tolerance = 1e-5;

        assert!(
            (loss - pytorch_loss).abs() < tolerance,
            "Cross-entropy perfect prediction: ArKan={}, PyTorch={}, diff={}",
            loss,
            pytorch_loss,
            (loss - pytorch_loss).abs()
        );
    }

    #[test]
    fn test_cross_entropy_pytorch_confident_wrong() {
        // PyTorch: pred=[0.1,0.9], target=[1.0,0.0], loss=2.3025851249694824
        let predictions = vec![0.1f32, 0.9];
        let targets = vec![1.0f32, 0.0];

        let (loss, _) = masked_cross_entropy(&predictions, &targets, None);

        // PyTorch reference: 2.3025851249694824, which is f32 ln(10) exactly.
        let pytorch_loss = std::f32::consts::LN_10;
        let tolerance = 1e-4; // Slightly higher tolerance for large loss

        assert!(
            (loss - pytorch_loss).abs() < tolerance,
            "Cross-entropy confident wrong: ArKan={}, PyTorch={}, diff={}",
            loss,
            pytorch_loss,
            (loss - pytorch_loss).abs()
        );
    }

    #[test]
    fn test_cross_entropy_pytorch_uncertain() {
        // PyTorch: pred=[0.5,0.5], target=[1.0,0.0], loss=0.6931471824645996
        let predictions = vec![0.5f32, 0.5];
        let targets = vec![1.0f32, 0.0];

        let (loss, _) = masked_cross_entropy(&predictions, &targets, None);

        // PyTorch reference: 0.6931471824645996, which is f32 ln(2) exactly.
        let pytorch_loss = std::f32::consts::LN_2;
        let tolerance = 1e-5;

        assert!(
            (loss - pytorch_loss).abs() < tolerance,
            "Cross-entropy uncertain: ArKan={}, PyTorch={}, diff={}",
            loss,
            pytorch_loss,
            (loss - pytorch_loss).abs()
        );
    }

    #[test]
    fn test_cross_entropy_pytorch_multiclass() {
        // PyTorch: pred=[0.7,0.1,0.1,0.1], target=[1.0,0.0,0.0,0.0], loss=0.1681891232728958
        let predictions = vec![0.7f32, 0.1, 0.1, 0.1];
        let targets = vec![1.0f32, 0.0, 0.0, 0.0];

        let (loss, _) = masked_cross_entropy(&predictions, &targets, None);

        // PyTorch reference: 0.1681891232728958
        let pytorch_loss = 0.16818912f32;
        let tolerance = 1e-5;

        assert!(
            (loss - pytorch_loss).abs() < tolerance,
            "Cross-entropy multiclass: ArKan={}, PyTorch={}, diff={}",
            loss,
            pytorch_loss,
            (loss - pytorch_loss).abs()
        );
    }

    #[test]
    fn test_cross_entropy_pytorch_soft_targets() {
        // PyTorch: pred=[0.6,0.4], target=[0.7,0.3], loss=0.632465124130249
        let predictions = vec![0.6f32, 0.4];
        let targets = vec![0.7f32, 0.3]; // Soft labels (not one-hot)

        let (loss, _) = masked_cross_entropy(&predictions, &targets, None);

        // PyTorch reference: 0.632465124130249
        let pytorch_loss = 0.6324651f32;
        let tolerance = 1e-5;

        assert!(
            (loss - pytorch_loss).abs() < tolerance,
            "Cross-entropy soft targets: ArKan={}, PyTorch={}, diff={}",
            loss,
            pytorch_loss,
            (loss - pytorch_loss).abs()
        );
    }

    #[test]
    fn test_cross_entropy_gradient_direction() {
        // Gradient should point toward correct answer
        // If prediction < target, gradient should be negative (increase prediction)
        // If prediction > target, gradient should be positive (decrease prediction)

        let predictions = vec![0.3f32, 0.7]; // pred[0] < target[0], pred[1] > target[1]
        let targets = vec![0.6f32, 0.4];

        let (_, grad) = masked_cross_entropy(&predictions, &targets, None);

        // grad = (p - t) / (p * (1-p))
        // For p=0.3, t=0.6: gradient should be negative
        assert!(
            grad[0] < 0.0,
            "Gradient should be negative when pred < target"
        );
        // For p=0.7, t=0.4: gradient should be positive (0.7 - 0.4 = 0.3)
        assert!(
            grad[1] > 0.0,
            "Gradient should be positive when pred > target"
        );
    }

    #[test]
    fn test_cross_entropy_with_mask() {
        // Test that masking works correctly
        let predictions = vec![0.9f32, 0.1, 0.5, 0.5];
        let targets = vec![1.0f32, 0.0, 1.0, 0.0];
        let mask = vec![1.0f32, 1.0, 0.0, 0.0]; // Only first two elements

        let (loss_masked, grad_masked) = masked_cross_entropy(&predictions, &targets, Some(&mask));
        let (loss_first_two, _) = masked_cross_entropy(&predictions[..2], &targets[..2], None);

        // Masked loss should equal loss of first two elements only
        let tolerance = 1e-6;
        assert!(
            (loss_masked - loss_first_two).abs() < tolerance,
            "Masked loss should match: masked={}, first_two={}",
            loss_masked,
            loss_first_two
        );

        // Masked elements should have zero gradient
        assert!(
            grad_masked[2].abs() < EPSILON,
            "Masked element should have zero gradient"
        );
        assert!(
            grad_masked[3].abs() < EPSILON,
            "Masked element should have zero gradient"
        );
    }

    #[test]
    fn test_cross_entropy_numerical_stability() {
        // Test edge cases near 0 and 1 (should not produce NaN/Inf)
        let predictions = vec![0.0001f32, 0.9999, 0.5]; // Very close to boundaries
        let targets = vec![0.0f32, 1.0, 0.5];

        let (loss, grad) = masked_cross_entropy(&predictions, &targets, None);

        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "All gradients should be finite"
        );

        // Loss should be reasonable (not exploding)
        // Note: p=0.5, t=0.5 gives -0.5*ln(0.5) - 0.5*ln(0.5) = ln(2) ≈ 0.693
        assert!(loss < 1.0, "Loss should not explode, got {}", loss);
    }

    // =========================================================================
    // Finite-Difference Gradient Check for masked_cross_entropy
    // =========================================================================

    /// Verifies that the analytic BCE gradient (p-t)/(p*(1-p)) agrees with a
    /// numerical central-difference estimate at several (p, t) test points.
    #[test]
    fn test_cross_entropy_finite_difference_gradient() {
        // Points chosen to be safely away from 0 and 1
        let test_points: &[(f32, f32)] = &[
            (0.3, 0.0),
            (0.3, 1.0),
            (0.7, 0.0),
            (0.7, 1.0),
            (0.5, 0.5),
            (0.2, 0.8),
            (0.8, 0.2),
        ];

        let fd_eps = 1e-4f32; // central-difference step
        let tolerance = 1e-3f32;

        for &(p, t) in test_points {
            // Analytic gradient from masked_cross_entropy
            let (_, grad) = masked_cross_entropy(&[p], &[t], None);
            // grad is already divided by count (=1), so it equals the per-element gradient
            let analytic = grad[0];

            // Numerical estimate via central differences on the per-element BCE loss
            // L(p) = -t*ln(p) - (1-t)*ln(1-p)  (no averaging since n=1)
            let bce = |prob: f32| -> f32 {
                let pc = prob.clamp(EPSILON, 1.0 - EPSILON);
                -t * pc.ln() - (1.0 - t) * (1.0 - pc).ln()
            };
            let numerical = (bce(p + fd_eps) - bce(p - fd_eps)) / (2.0 * fd_eps);

            let abs_err = (analytic - numerical).abs();
            assert!(
                abs_err < tolerance,
                "Finite-diff gradient check failed at (p={}, t={}): analytic={:.6}, numerical={:.6}, err={:.6}",
                p, t, analytic, numerical, abs_err
            );
        }
    }
}
