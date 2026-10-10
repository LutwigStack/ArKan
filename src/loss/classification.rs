//! Classification losses, softmax, and the fused poker objective.

use super::*;

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
/// The gradient is with respect to the clamped probability, treating the clamp
/// as identity during backpropagation; it can be nonzero outside that interval.
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

            // Gradient of BCE w.r.t. (already-sigmoided, clamped) probability p
            grad[i] = m * (p - t) / (p * (1.0 - p));
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
/// Tuple of (total_loss, prob_loss, q_loss, gradient).
/// The probability-head gradient is the legacy fused derivative with respect
/// to sigmoid logits; pass it directly to the logits head without sigmoid backward.
/// Use [`poker_combined_loss_probabilities`] for derivatives of these inputs.
pub fn poker_combined_loss(
    predictions: &[f32],
    targets: &[f32],
    alpha: f32,
) -> (f32, f32, f32, Vec<f32>) {
    poker_loss(predictions, targets, alpha, true)
}

/// Poker objective with derivatives with respect to the supplied probabilities
/// and Q-values. Clamped probability regions have zero derivative.
pub fn poker_combined_loss_probabilities(
    predictions: &[f32],
    targets: &[f32],
    alpha: f32,
) -> (f32, f32, f32, Vec<f32>) {
    poker_loss(predictions, targets, alpha, false)
}

fn poker_loss(
    predictions: &[f32],
    targets: &[f32],
    alpha: f32,
    fused_logits: bool,
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

                // BCE: inputs must be probabilities in (EPSILON, 1-EPSILON)
                prob_loss += m * (-t * p.ln() - (1.0 - t) * (1.0 - p).ln());
                grad[base + action] = if fused_logits {
                    m * (p - t)
                } else if predictions[base + action] > EPSILON
                    && predictions[base + action] < 1.0 - EPSILON
                {
                    m * (p - t) / (p * (1.0 - p))
                } else {
                    0.0
                };
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

        // Normalize — sum >= 1.0 (max-subtraction guarantees exp(0)=1), so division is safe
        for v in slice.iter_mut() {
            *v /= sum;
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

            // Normalize — sum >= 1.0 (max-subtraction guarantees exp(0)=1), so division is safe
            for v in x_slice.iter_mut() {
                *v /= sum;
            }
        } else {
            // All masked: set to zero
            x_slice.fill(0.0);
        }
    }
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
/// Tuple of (loss, gradient). This legacy API returns the fused derivative
/// with respect to softmax logits for normalized targets (`p - t`). Supply this
/// gradient directly to the logits head without another softmax backward.
/// Use [`masked_categorical_cross_entropy_probabilities`] for input derivatives
/// or [`masked_categorical_cross_entropy_with_logits`] for a stable logits objective.
pub fn masked_categorical_cross_entropy(
    predictions: &[f32],
    targets: &[f32],
    num_classes: usize,
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    categorical_probability_loss(predictions, targets, num_classes, mask, true)
}

/// Categorical cross-entropy with derivatives of the supplied probabilities.
/// Inputs are independent probabilities; no softmax derivative is fused.
/// Clamped regions have zero derivative. Mask weights normalize per sample.
pub fn masked_categorical_cross_entropy_probabilities(
    predictions: &[f32],
    targets: &[f32],
    num_classes: usize,
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    categorical_probability_loss(predictions, targets, num_classes, mask, false)
}

fn categorical_probability_loss(
    predictions: &[f32],
    targets: &[f32],
    num_classes: usize,
    mask: Option<&[f32]>,
    fused_logits: bool,
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

                grad[base + c] = if fused_logits {
                    m * (p - t)
                } else if predictions[base + c] > EPSILON
                    && predictions[base + c] < 1.0 - EPSILON
                    && t > 0.0
                {
                    -m * t / p
                } else {
                    0.0
                };
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

/// Stable categorical cross-entropy of logits, with derivatives of those logits.
/// Computes log-sum-exp without probability clipping. Targets may be one-hot
/// or nonnegative soft labels; sample masks normalize loss and gradient.
pub fn masked_categorical_cross_entropy_with_logits(
    logits: &[f32],
    targets: &[f32],
    num_classes: usize,
    mask: Option<&[f32]>,
) -> (f32, Vec<f32>) {
    debug_assert_eq!(logits.len(), targets.len());
    debug_assert!(num_classes > 0);
    let mut gradient = vec![0.0; logits.len()];
    let mut loss = 0.0f64;
    let samples = logits.len().min(targets.len()) / num_classes;
    let count = (0..samples)
        .map(|sample| mask.map_or(1.0, |mask| f64::from(mask[sample])))
        .filter(|&weight| weight > 0.0 || weight.is_nan())
        .sum::<f64>();
    for (sample, (scores, labels)) in logits
        .chunks_exact(num_classes)
        .zip(targets.chunks_exact(num_classes))
        .enumerate()
    {
        let m = mask.map(|m| m[sample] as f64).unwrap_or(1.0);
        if m <= 0.0 {
            continue;
        }
        let maximum = scores
            .iter()
            .map(|&z| z as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = scores
            .iter()
            .map(|&z| (z as f64 - maximum).exp())
            .sum::<f64>();
        let log_sum = sum.ln();
        let target_sum = labels.iter().map(|&t| t as f64).sum::<f64>();
        for (class, (&z, &t)) in scores.iter().zip(labels).enumerate() {
            loss += m * t as f64 * (log_sum - (z as f64 - maximum));
            let probability = (z as f64 - maximum).exp() / sum;
            let normalized_weight = if count > 0.0 { m / count } else { m };
            gradient[sample * num_classes + class] =
                (normalized_weight * (probability * target_sum - t as f64)) as f32;
        }
    }
    if count > 0.0 {
        loss /= count;
    }
    (loss as f32, gradient)
}
