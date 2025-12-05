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

// =============================================================================
// REGRESSION LOSSES
// =============================================================================

/// Masked Root Mean Squared Error loss.
///
/// RMSE provides error in the same units as the target variable,
/// making it easier to interpret than MSE.
///
/// # Formula
///
/// $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}$$
///
/// # Arguments
///
/// * `predictions` - Model output: `[batch_size * output_dim]`
/// * `targets` - Ground truth: `[batch_size * output_dim]`
/// * `mask` - Optional mask: `[batch_size * output_dim]`
///
/// # Returns
///
/// Tuple of (loss, gradient)
///
/// # Note
///
/// Gradient is `grad_MSE / (2 * RMSE)` to account for the square root.
pub fn masked_rmse(predictions: &[f32], targets: &[f32], mask: Option<&[f32]>) -> (f32, Vec<f32>) {
    let (mse, mut grad) = masked_mse(predictions, targets, mask);
    let rmse = mse.sqrt();

    // Gradient of sqrt(MSE) = grad_MSE / (2 * sqrt(MSE))
    if rmse > EPSILON {
        let scale = 0.5 / rmse;
        for g in &mut grad {
            *g *= scale;
        }
    }

    (rmse, grad)
}

/// Masked Mean Absolute Error (L1) loss.
///
/// MAE is more robust to outliers than MSE because it doesn't
/// square the errors. Good choice when data has noise/outliers.
///
/// # Formula
///
/// $$\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$$
///
/// # Arguments
///
/// * `predictions` - Model output
/// * `targets` - Ground truth
/// * `mask` - Optional mask
///
/// # Returns
///
/// Tuple of (loss, gradient)
///
/// # Note
///
/// Gradient is `sign(prediction - target)`, discontinuous at zero.
pub fn masked_mae(predictions: &[f32], targets: &[f32], mask: Option<&[f32]>) -> (f32, Vec<f32>) {
    debug_assert_eq!(predictions.len(), targets.len());

    let n = predictions.len();
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; n];
    let mut count = 0.0f32;

    for i in 0..n {
        let m = mask.map(|m| m[i]).unwrap_or(1.0);

        if m > 0.0 {
            let diff = predictions[i] - targets[i];
            loss += m * diff.abs();
            // Subgradient: sign(diff), use 0 for diff=0
            grad[i] = m * if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };
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

// =============================================================================
// CLASSIFICATION LOSSES
// =============================================================================

/// Binary Cross-Entropy with Logits (numerically stable).
///
/// Combines sigmoid and BCE in one step for numerical stability.
/// Use this instead of `masked_cross_entropy` when predictions are logits
/// (before sigmoid).
///
/// # Formula
///
/// $$\text{BCE}(x, y) = -y \cdot \log(\sigma(x)) - (1-y) \cdot \log(1-\sigma(x))$$
///
/// Computed as:
/// $$\text{BCE}(x, y) = \max(x, 0) - x \cdot y + \log(1 + e^{-|x|})$$
///
/// # Arguments
///
/// * `logits` - Raw model output (before sigmoid)
/// * `targets` - Binary targets (0 or 1)
/// * `mask` - Optional mask
///
/// # Returns
///
/// Tuple of (loss, gradient)
///
/// # Example
///
/// ```rust
/// use arkan::loss::masked_bce_with_logits;
///
/// let logits = vec![2.0, -1.0, 0.5];
/// let targets = vec![1.0, 0.0, 1.0];
///
/// let (loss, grad) = masked_bce_with_logits(&logits, &targets, None);
/// ```
pub fn masked_bce_with_logits(
    logits: &[f32],
    targets: &[f32],
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    debug_assert_eq!(logits.len(), targets.len());

    let n = logits.len();
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; n];
    let mut count = 0.0f32;

    for i in 0..n {
        let m = mask.map(|m| m[i]).unwrap_or(1.0);

        if m > 0.0 {
            let x = logits[i];
            let t = targets[i];

            // Numerically stable BCE:
            // max(x, 0) - x * t + log(1 + exp(-|x|))
            let loss_i = x.max(0.0) - x * t + (1.0 + (-x.abs()).exp()).ln();
            loss += m * loss_i;

            // Gradient: sigmoid(x) - t
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            grad[i] = m * (sigmoid - t);
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

/// Categorical Cross-Entropy loss for multi-class classification.
///
/// Expects `predictions` to be softmax probabilities and `targets` to be
/// one-hot encoded. Use with softmax output.
///
/// # Formula
///
/// $$\text{CE} = -\sum_c y_c \log(\hat{y}_c)$$
///
/// # Arguments
///
/// * `predictions` - Softmax probabilities: `[batch_size * num_classes]`
/// * `targets` - One-hot targets: `[batch_size * num_classes]`
/// * `num_classes` - Number of classes
/// * `mask` - Optional mask per sample: `[batch_size]`
///
/// # Returns
///
/// Tuple of (loss, gradient)
pub fn masked_categorical_cross_entropy(
    predictions: &[f32],
    targets: &[f32],
    num_classes: usize,
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    debug_assert_eq!(predictions.len(), targets.len());
    debug_assert!(num_classes > 0);

    let n = predictions.len();
    let batch_size = n / num_classes;
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; n];
    let mut count = 0.0f32;

    for b in 0..batch_size {
        let m = mask.map(|m| m[b]).unwrap_or(1.0);

        if m > 0.0 {
            let base = b * num_classes;

            for c in 0..num_classes {
                let p = predictions[base + c].clamp(EPSILON, 1.0 - EPSILON);
                let t = targets[base + c];

                if t > 0.0 {
                    loss -= m * t * p.ln();
                }

                // Gradient: -t/p (for softmax + CE, simplifies to p - t)
                grad[base + c] = m * (p - t);
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

// =============================================================================
// KAN-SPECIFIC REGULARIZATION
// =============================================================================

/// Configuration for KAN combined loss.
///
/// These hyperparameters control the balance between fitting the data
/// and regularizing the spline functions for interpretability.
///
/// # Recommended Values
///
/// | Parameter | Range | Effect |
/// |-----------|-------|--------|
/// | `lambda_l1` | 0.0001 - 0.01 | Higher = sparser, simpler functions |
/// | `lambda_entropy` | 0.0001 - 0.001 | Higher = more decisive function choice |
/// | `lambda_smooth` | 0.0001 - 0.01 | Higher = smoother splines |
#[derive(Debug, Clone, Copy)]
pub struct KanLossConfig {
    /// Weight for L1 sparsity regularization.
    ///
    /// Encourages spline coefficients to be exactly zero,
    /// effectively "turning off" unused connections.
    pub lambda_l1: f32,

    /// Weight for entropy regularization.
    ///
    /// Encourages the network to commit to specific activation
    /// functions rather than blending many.
    pub lambda_entropy: f32,

    /// Weight for smoothness penalty.
    ///
    /// Penalizes high-frequency oscillations in the spline,
    /// encouraging simple, interpretable shapes.
    pub lambda_smooth: f32,
}

impl Default for KanLossConfig {
    fn default() -> Self {
        Self {
            lambda_l1: 0.001,
            lambda_entropy: 0.0001,
            lambda_smooth: 0.001,
        }
    }
}

/// L1 sparsity loss for spline coefficients.
///
/// Computes the mean absolute value of coefficients. When added to the
/// main loss, this encourages coefficients to become exactly zero,
/// effectively pruning unused connections.
///
/// # Formula
///
/// $$L_{L1} = \frac{1}{n}\sum_i |c_i|$$
///
/// # Arguments
///
/// * `coefficients` - Spline coefficients from all layers
///
/// # Returns
///
/// Scalar L1 loss value
///
/// # Example
///
/// ```rust
/// use arkan::loss::l1_sparsity_loss;
///
/// let coefficients = vec![0.5, 0.0, -0.3, 0.0, 0.1];
/// let l1 = l1_sparsity_loss(&coefficients);
///
/// // L1 = (0.5 + 0 + 0.3 + 0 + 0.1) / 5 = 0.18
/// assert!((l1 - 0.18).abs() < 0.001);
/// ```
pub fn l1_sparsity_loss(coefficients: &[f32]) -> f32 {
    if coefficients.is_empty() {
        return 0.0;
    }

    let sum: f32 = coefficients.iter().map(|c| c.abs()).sum();
    sum / coefficients.len() as f32
}

/// Compute L1 sparsity gradient for coefficients.
///
/// Returns the subgradient of L1 norm: `sign(c)` for each coefficient.
///
/// # Arguments
///
/// * `coefficients` - Spline coefficients
///
/// # Returns
///
/// Gradient vector (same size as coefficients)
pub fn l1_sparsity_gradient(coefficients: &[f32]) -> Vec<f32> {
    let n = coefficients.len();
    if n == 0 {
        return vec![];
    }

    let scale = 1.0 / n as f32;
    coefficients
        .iter()
        .map(|&c| {
            scale * if c > 0.0 { 1.0 } else if c < 0.0 { -1.0 } else { 0.0 }
        })
        .collect()
}

/// Entropy regularization loss.
///
/// Computes the entropy of coefficient magnitudes (as a soft distribution).
/// Low entropy means the network has "committed" to specific functions.
///
/// # Formula
///
/// First, normalize coefficients to a probability-like distribution:
/// $$p_i = \frac{|c_i|^2}{\sum_j |c_j|^2 + \epsilon}$$
///
/// Then compute entropy:
/// $$H = -\sum_i p_i \log(p_i + \epsilon)$$
///
/// # Arguments
///
/// * `coefficients` - Spline coefficients (usually per input-output pair)
/// * `group_size` - Size of each coefficient group (e.g., `global_basis_size`)
///
/// # Returns
///
/// Entropy loss value (lower = more decisive)
///
/// # Example
///
/// ```rust
/// use arkan::loss::entropy_regularization;
///
/// // Spread out coefficients (high entropy)
/// let spread = vec![0.25, 0.25, 0.25, 0.25];
/// let h_spread = entropy_regularization(&spread, 4);
///
/// // Concentrated coefficients (low entropy)
/// let focused = vec![1.0, 0.0, 0.0, 0.0];
/// let h_focused = entropy_regularization(&focused, 4);
///
/// assert!(h_focused < h_spread);
/// ```
pub fn entropy_regularization(coefficients: &[f32], group_size: usize) -> f32 {
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

        // Compute squared magnitudes
        let squared: Vec<f32> = group.iter().map(|c| c * c).collect();
        let sum: f32 = squared.iter().sum::<f32>() + EPSILON;

        // Compute entropy
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

/// Smoothness penalty (second derivative approximation).
///
/// Penalizes high-frequency oscillations in the spline by computing
/// the squared second differences of adjacent coefficients.
///
/// # Formula
///
/// $$L_{smooth} = \frac{1}{n-2}\sum_i (c_{i+1} - 2c_i + c_{i-1})^2$$
///
/// This approximates $\int (f''(x))^2 dx$ for the spline.
///
/// # Arguments
///
/// * `coefficients` - Spline coefficients
/// * `basis_size` - Number of basis functions per input-output pair
///
/// # Returns
///
/// Smoothness penalty value (lower = smoother)
///
/// # Example
///
/// ```rust
/// use arkan::loss::smoothness_penalty;
///
/// // Smooth coefficients (linear-ish)
/// let smooth = vec![0.1, 0.2, 0.3, 0.4, 0.5];
/// let s_smooth = smoothness_penalty(&smooth, 5);
///
/// // Oscillating coefficients
/// let rough = vec![0.1, 0.5, 0.1, 0.5, 0.1];
/// let s_rough = smoothness_penalty(&rough, 5);
///
/// assert!(s_smooth < s_rough);
/// ```
pub fn smoothness_penalty(coefficients: &[f32], basis_size: usize) -> f32 {
    if coefficients.is_empty() || basis_size < 3 {
        return 0.0;
    }

    let num_groups = coefficients.len() / basis_size;
    if num_groups == 0 {
        return 0.0;
    }

    let mut total_penalty = 0.0f32;
    let mut count = 0;

    for g in 0..num_groups {
        let group = &coefficients[g * basis_size..(g + 1) * basis_size];

        // Second differences
        for i in 1..group.len() - 1 {
            let second_diff = group[i + 1] - 2.0 * group[i] + group[i - 1];
            total_penalty += second_diff * second_diff;
            count += 1;
        }
    }

    if count > 0 {
        total_penalty / count as f32
    } else {
        0.0
    }
}

/// Compute smoothness gradient for coefficients.
///
/// Returns the gradient of the smoothness penalty with respect to coefficients.
///
/// # Arguments
///
/// * `coefficients` - Spline coefficients
/// * `basis_size` - Number of basis functions per input-output pair
///
/// # Returns
///
/// Gradient vector (same size as coefficients)
pub fn smoothness_gradient(coefficients: &[f32], basis_size: usize) -> Vec<f32> {
    let n = coefficients.len();
    let mut grad = vec![0.0f32; n];

    if n == 0 || basis_size < 3 {
        return grad;
    }

    let num_groups = n / basis_size;
    if num_groups == 0 {
        return grad;
    }

    let mut count = 0;
    for g in 0..num_groups {
        let base = g * basis_size;

        for i in 1..basis_size - 1 {
            let c_prev = coefficients[base + i - 1];
            let c_curr = coefficients[base + i];
            let c_next = coefficients[base + i + 1];
            let second_diff = c_next - 2.0 * c_curr + c_prev;

            // d/d(c_{i-1}): 2 * second_diff
            // d/d(c_i): -4 * second_diff
            // d/d(c_{i+1}): 2 * second_diff
            grad[base + i - 1] += 2.0 * second_diff;
            grad[base + i] += -4.0 * second_diff;
            grad[base + i + 1] += 2.0 * second_diff;
            count += 1;
        }
    }

    if count > 0 {
        let scale = 1.0 / count as f32;
        for g in &mut grad {
            *g *= scale;
        }
    }

    grad
}

/// Combined KAN loss with task loss and regularization.
///
/// This function combines:
/// 1. Task loss (MSE) for fitting the data
/// 2. L1 sparsity for interpretable, sparse connections
/// 3. Entropy regularization for decisive function selection
/// 4. Smoothness penalty for preventing overfitting
///
/// # Formula
///
/// $$L_{total} = L_{pred} + \lambda_1 L_{L1} + \lambda_2 H + \lambda_3 L_{smooth}$$
///
/// # Arguments
///
/// * `predictions` - Model output
/// * `targets` - Ground truth
/// * `coefficients` - All spline coefficients from the network
/// * `basis_size` - Number of basis functions per input-output pair
/// * `config` - Regularization weights
/// * `mask` - Optional mask for predictions
///
/// # Returns
///
/// Tuple of:
/// - `total_loss` - Combined loss value
/// - `pred_loss` - Task (MSE) loss only
/// - `reg_loss` - Total regularization loss
/// - `pred_gradient` - Gradient for predictions (backprop through network)
///
/// # Example
///
/// ```rust
/// use arkan::loss::{kan_combined_loss, KanLossConfig};
///
/// let predictions = vec![0.5, 1.0];
/// let targets = vec![0.6, 1.1];
/// let coefficients = vec![0.1, 0.0, -0.2, 0.5, 0.0, 0.0, 0.3, -0.1];
///
/// let config = KanLossConfig::default();
///
/// let (total, pred, reg, grad) = kan_combined_loss(
///     &predictions, &targets, &coefficients, 4, &config, None
/// );
/// ```
pub fn kan_combined_loss(
    predictions: &[f32],
    targets: &[f32],
    coefficients: &[f32],
    basis_size: usize,
    config: &KanLossConfig,
    mask: Option<&[f32]>,
) -> (f32, f32, f32, Vec<f32>) {
    // Task loss
    let (pred_loss, pred_grad) = masked_mse(predictions, targets, mask);

    // Regularization losses
    let l1_loss = l1_sparsity_loss(coefficients);
    let entropy_loss = entropy_regularization(coefficients, basis_size);
    let smooth_loss = smoothness_penalty(coefficients, basis_size);

    let reg_loss = config.lambda_l1 * l1_loss
        + config.lambda_entropy * entropy_loss
        + config.lambda_smooth * smooth_loss;

    let total_loss = pred_loss + reg_loss;

    (total_loss, pred_loss, reg_loss, pred_grad)
}

/// Get regularization gradients for coefficients.
///
/// Returns combined gradient from L1 and smoothness regularization.
/// Call this separately and add to coefficient gradients during training.
///
/// # Arguments
///
/// * `coefficients` - All spline coefficients
/// * `basis_size` - Number of basis functions per input-output pair
/// * `config` - Regularization weights
///
/// # Returns
///
/// Gradient vector for coefficients
pub fn kan_regularization_gradient(
    coefficients: &[f32],
    basis_size: usize,
    config: &KanLossConfig,
) -> Vec<f32> {
    let l1_grad = l1_sparsity_gradient(coefficients);
    let smooth_grad = smoothness_gradient(coefficients, basis_size);

    l1_grad
        .iter()
        .zip(smooth_grad.iter())
        .map(|(&l1, &sm)| config.lambda_l1 * l1 + config.lambda_smooth * sm)
        .collect()
}

// =============================================================================
// PHYSICS-INFORMED LOSS (for PDE solving)
// =============================================================================

/// PDE residual loss for physics-informed KAN.
///
/// Computes the squared residual of a differential equation.
/// The network predicts the solution u(x), and this loss measures
/// how well the equation is satisfied.
///
/// # Arguments
///
/// * `residuals` - Pre-computed PDE residuals: `L[u] - f` where L is the operator
/// * `mask` - Optional mask for boundary conditions
///
/// # Returns
///
/// Tuple of (loss, gradient)
///
/// # Example
///
/// For heat equation: $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$
///
/// Compute residual: $r = \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2}$
///
/// ```rust
/// use arkan::loss::pde_residual_loss;
///
/// // Residuals computed by automatic differentiation
/// let residuals = vec![0.01, -0.02, 0.005];
/// let (loss, grad) = pde_residual_loss(&residuals, None);
/// ```
pub fn pde_residual_loss(residuals: &[f32], mask: Option<&[f32]>) -> (f32, Vec<f32>) {
    // PDE residual loss is essentially MSE with target = 0
    let zeros = vec![0.0f32; residuals.len()];
    masked_mse(residuals, &zeros, mask)
}

// =============================================================================
// SYMBOLIC REGRESSION SUPPORT
// =============================================================================

/// Compute R² (coefficient of determination) for symbolic regression.
///
/// Used when "locking" a spline to a symbolic function to measure
/// how well the symbolic approximation fits.
///
/// # Formula
///
/// $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
///
/// # Arguments
///
/// * `predictions` - Model/symbolic function predictions
/// * `targets` - True values
///
/// # Returns
///
/// R² value (1.0 = perfect fit, 0.0 = no better than mean, negative = worse than mean)
///
/// # Example
///
/// ```rust
/// use arkan::loss::r_squared;
///
/// let predictions = vec![1.0, 2.0, 3.0];
/// let targets = vec![1.1, 1.9, 3.1];
///
/// let r2 = r_squared(&predictions, &targets);
/// assert!(r2 > 0.95); // Very good fit
/// ```
pub fn r_squared(predictions: &[f32], targets: &[f32]) -> f32 {
    debug_assert_eq!(predictions.len(), targets.len());

    if predictions.is_empty() {
        return 0.0;
    }

    let n = predictions.len() as f32;

    // Mean of targets
    let mean: f32 = targets.iter().sum::<f32>() / n;

    // SS_tot = sum((y - mean)^2)
    let ss_tot: f32 = targets.iter().map(|&y| (y - mean).powi(2)).sum();

    // SS_res = sum((y - pred)^2)
    let ss_res: f32 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (t - p).powi(2))
        .sum();

    if ss_tot < EPSILON {
        // All targets are the same
        if ss_res < EPSILON {
            1.0 // Perfect prediction of constant
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
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
}
