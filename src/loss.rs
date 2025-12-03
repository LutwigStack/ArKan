//! Loss functions with masking support.
//!
//! This module provides loss functions for training KAN networks:
//!
//! - [`masked_mse`] - Mean Squared Error with optional masking
//! - [`masked_cross_entropy`] - Cross-Entropy for probability outputs
//! - [`masked_huber`] - Huber (smooth L1) loss for robustness
//! - [`poker_combined_loss`] - Specialized loss for poker Q-learning
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

use crate::config::EPSILON;

/// Masked Mean Squared Error loss.
///
/// Computes MSE only for positions where `mask > 0`.
///
/// # Arguments
///
/// * `predictions` - Model output: `[batch_size * output_dim]`
/// * `targets` - Ground truth: `[batch_size * output_dim]`
/// * `mask` - Optional mask: `[batch_size * output_dim]`, 1.0 for active, 0.0 for ignore
///
/// # Returns
///
/// Tuple of (loss, gradient):
/// - `loss`: Scalar MSE value averaged over masked elements
/// - `gradient`: Vector of gradients for each prediction
///
/// # Example
///
/// ```rust
/// use arkan::loss::masked_mse;
///
/// let pred = vec![1.0, 2.0, 3.0];
/// let target = vec![1.0, 1.0, 1.0];
/// let (loss, grad) = masked_mse(&pred, &target, None);
///
/// assert!(loss > 0.0);
/// ```
pub fn masked_mse(predictions: &[f32], targets: &[f32], mask: Option<&[f32]>) -> (f32, Vec<f32>) {
    debug_assert_eq!(predictions.len(), targets.len());

    let n = predictions.len();
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; n];
    let mut count = 0.0f32;

    for i in 0..n {
        let m = mask.map(|m| m[i]).unwrap_or(1.0);

        if m > 0.0 {
            let diff = predictions[i] - targets[i];
            loss += m * diff * diff;
            grad[i] = 2.0 * m * diff;
            count += m;
        }
    }

    if count > 0.0 {
        loss /= count;
        // Gradient is already weighted by mask, but we normalize by count
        for g in &mut grad {
            *g /= count;
        }
    }

    (loss, grad)
}

/// Masked Cross-Entropy loss for probability outputs.
///
/// Computes binary cross-entropy for each element, with optional masking.
/// Expects predictions to be in (0, 1) range (after sigmoid).
///
/// # Arguments
///
/// * `predictions` - Probability predictions: `[batch_size * output_dim]`
/// * `targets` - Target probabilities: `[batch_size * output_dim]`
/// * `mask` - Optional mask
///
/// # Returns
///
/// Tuple of (loss, gradient)
///
/// # Note
///
/// Predictions are clamped to `[EPSILON, 1-EPSILON]` to avoid log(0).
pub fn masked_cross_entropy(
    predictions: &[f32],
    targets: &[f32],
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    debug_assert_eq!(predictions.len(), targets.len());

    let n = predictions.len();
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; n];
    let mut count = 0.0f32;

    for i in 0..n {
        let m = mask.map(|m| m[i]).unwrap_or(1.0);

        if m > 0.0 {
            // Clamp predictions to avoid log(0)
            let p = predictions[i].clamp(EPSILON, 1.0 - EPSILON);
            let t = targets[i];

            // Binary cross-entropy: -t*log(p) - (1-t)*log(1-p)
            loss += m * (-t * p.ln() - (1.0 - t) * (1.0 - p).ln());

            // Gradient: (p - t) / (p * (1 - p))
            // Simplified for stability: just (p - t) works with softmax
            grad[i] = m * (p - t);
            count += m;
        }
    }

    if count > 0.0 {
        loss /= count;
        for g in &mut grad {
            *g /= count;
        }
    }

    (loss, grad)
}

/// Combined loss for poker KAN: MSE for Q-values + Cross-Entropy for probabilities.
///
/// Specialized loss function for poker Q-learning with action masking.
///
/// # Output Layout
///
/// The output has 24 dimensions per sample:
/// - `[0..8]`: Action probabilities
/// - `[8..16]`: Q-values
/// - `[16..24]`: Action mask (1.0 for valid actions)
///
/// # Arguments
///
/// * `predictions` - Model output: `[batch_size * 24]`
/// * `targets` - Ground truth: `[batch_size * 24]`
/// * `alpha` - Weight for probability loss (0.0 to 1.0)
///
/// # Returns
///
/// Tuple of (total_loss, prob_loss, q_loss, gradient)
pub fn poker_combined_loss(
    predictions: &[f32],
    targets: &[f32],
    alpha: f32,
) -> (f32, f32, f32, Vec<f32>) {
    let n = predictions.len();
    let batch_size = n / 24;
    debug_assert_eq!(n, batch_size * 24);

    let mut grad = vec![0.0f32; n];
    let mut prob_loss = 0.0f32;
    let mut q_loss = 0.0f32;
    let mut prob_count = 0.0f32;
    let mut q_count = 0.0f32;

    for b in 0..batch_size {
        let base = b * 24;

        // Get action mask from targets (indices 16..24)
        for action in 0..8 {
            let m = targets[base + 16 + action];

            if m > 0.0 {
                // Probability loss (indices 0..8)
                let p = predictions[base + action].clamp(EPSILON, 1.0 - EPSILON);
                let t = targets[base + action];

                // KL-divergence style: t * log(t/p), but simplified to cross-entropy
                prob_loss += m * (-t * p.ln() - (1.0 - t) * (1.0 - p).ln()).max(0.0);
                grad[base + action] = m * (p - t);
                prob_count += m;

                // Q-value loss (indices 8..16)
                let q_pred = predictions[base + 8 + action];
                let q_true = targets[base + 8 + action];
                let diff = q_pred - q_true;

                q_loss += m * diff * diff;
                grad[base + 8 + action] = 2.0 * m * diff;
                q_count += m;
            }
        }
    }

    // Normalize losses
    if prob_count > 0.0 {
        prob_loss /= prob_count;
    }
    if q_count > 0.0 {
        q_loss /= q_count;
    }

    // Scale gradients
    let prob_scale = if prob_count > 0.0 {
        alpha / prob_count
    } else {
        0.0
    };
    let q_scale = if q_count > 0.0 {
        (1.0 - alpha) / q_count
    } else {
        0.0
    };

    for b in 0..batch_size {
        let base = b * 24;
        for action in 0..8 {
            grad[base + action] *= prob_scale;
            grad[base + 8 + action] *= q_scale;
            // Mask gradients stay zero
        }
    }

    let total_loss = alpha * prob_loss + (1.0 - alpha) * q_loss;

    (total_loss, prob_loss, q_loss, grad)
}

/// Huber loss (smooth L1) for robust training.
///
/// Combines L1 and L2 loss to be robust to outliers:
/// - Quadratic for small errors (|error| ≤ delta)
/// - Linear for large errors (|error| > delta)
///
/// # Formula
///
/// $$L_\delta(a) = \begin{cases}
///   \frac{1}{2}a^2 & |a| \leq \delta \\
///   \delta(|a| - \frac{1}{2}\delta) & |a| > \delta
/// \end{cases}$$
///
/// # Arguments
///
/// * `predictions` - Model output
/// * `targets` - Ground truth
/// * `delta` - Threshold for switching between L1 and L2 (typically 1.0)
/// * `mask` - Optional mask
pub fn masked_huber(
    predictions: &[f32],
    targets: &[f32],
    delta: f32,
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    debug_assert_eq!(predictions.len(), targets.len());

    let n = predictions.len();
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; n];
    let mut count = 0.0f32;

    for i in 0..n {
        let m = mask.map(|m| m[i]).unwrap_or(1.0);

        if m > 0.0 {
            let diff = predictions[i] - targets[i];
            let abs_diff = diff.abs();

            if abs_diff <= delta {
                // Quadratic region
                loss += m * 0.5 * diff * diff;
                grad[i] = m * diff;
            } else {
                // Linear region
                loss += m * delta * (abs_diff - 0.5 * delta);
                grad[i] = m * delta * diff.signum();
            }

            count += m;
        }
    }

    if count > 0.0 {
        loss /= count;
        for g in &mut grad {
            *g /= count;
        }
    }

    (loss, grad)
}

/// Computes softmax in-place.
///
/// Applies softmax over batches of size `dim_size`:
///
/// $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
///
/// # Arguments
///
/// * `x` - Input logits (modified in-place)
/// * `dim_size` - Size of each softmax group
pub fn softmax(x: &mut [f32], dim_size: usize) {
    let batch_size = x.len() / dim_size;

    for b in 0..batch_size {
        let slice = &mut x[b * dim_size..(b + 1) * dim_size];

        // Find max for numerical stability
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp and sum
        let mut sum = 0.0f32;
        for v in slice.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }

        // Normalize
        for v in slice.iter_mut() {
            *v /= sum + EPSILON;
        }
    }
}

/// Applies masked softmax where inactive positions get zero probability.
///
/// Sets masked positions to `-inf` before softmax, effectively
/// giving them zero probability in the output.
///
/// # Arguments
///
/// * `x` - Input logits (modified in-place)
/// * `mask` - Mask: 1.0 for active, 0.0 for inactive
/// * `dim_size` - Size of each softmax group
pub fn masked_softmax(x: &mut [f32], mask: &[f32], dim_size: usize) {
    let batch_size = x.len() / dim_size;

    for b in 0..batch_size {
        let x_slice = &mut x[b * dim_size..(b + 1) * dim_size];
        let m_slice = &mask[b * dim_size..(b + 1) * dim_size];

        // Set masked positions to -inf
        for i in 0..dim_size {
            if m_slice[i] <= 0.0 {
                x_slice[i] = f32::NEG_INFINITY;
            }
        }

        // Find max for numerical stability
        let max = x_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if max.is_finite() {
            // Compute exp and sum
            let mut sum = 0.0f32;
            for v in x_slice.iter_mut() {
                if v.is_finite() {
                    *v = (*v - max).exp();
                    sum += *v;
                } else {
                    *v = 0.0;
                }
            }

            // Normalize
            for v in x_slice.iter_mut() {
                *v /= sum + EPSILON;
            }
        } else {
            // All masked: set to zero
            x_slice.fill(0.0);
        }
    }
}

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
}
