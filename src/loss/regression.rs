//! Regression losses and fit metrics.

use super::*;

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
/// Gradient is `grad_MSE / (2 * RMSE)` for every nonzero RMSE.
/// At exact zero the zero subgradient is used. Accumulation in f64 prevents
/// finite residuals from losing their gradients when f32 squares overflow or underflow.
pub fn masked_rmse(predictions: &[f32], targets: &[f32], mask: Option<&[f32]>) -> (f32, Vec<f32>) {
    debug_assert_eq!(predictions.len(), targets.len());
    let mut gradient = vec![0.0; predictions.len()];
    let mut squared = 0.0f64;
    let mut count = 0.0f64;
    for (i, (&prediction, &target)) in predictions.iter().zip(targets).enumerate() {
        let m = mask.map(|mask| mask[i] as f64).unwrap_or(1.0);
        if m > 0.0 {
            let residual = prediction as f64 - target as f64;
            squared += m * residual * residual;
            count += m;
        }
    }
    let rmse = if count > 0.0 {
        (squared / count).sqrt()
    } else {
        0.0
    };
    if rmse > 0.0 {
        for (i, (&prediction, &target)) in predictions.iter().zip(targets).enumerate() {
            let m = mask.map(|mask| mask[i] as f64).unwrap_or(1.0);
            if m > 0.0 {
                gradient[i] = (m * (prediction as f64 - target as f64) / (count * rmse)) as f32;
            }
        }
    }
    (rmse as f32, gradient)
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
            // Subgradient: sign(diff), 0 when diff=0
            grad[i] = m * if diff != 0.0 { diff.signum() } else { 0.0 };
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
