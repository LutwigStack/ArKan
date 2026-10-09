//! Spline regularizers and differential-equation residual loss.

use super::*;

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
            scale
                * if c > 0.0 {
                    1.0
                } else if c < 0.0 {
                    -1.0
                } else {
                    0.0
                }
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
/// $$H = -\sum_{i: p_i > \epsilon} p_i \log(p_i)$$
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

        let sum: f32 = group.iter().map(|c| c * c).sum::<f32>() + EPSILON;

        // Compute entropy
        let mut entropy = 0.0f32;
        for &c in group {
            let sq = c * c;
            let p = sq / sum;
            if p > EPSILON {
                entropy -= p * p.ln();
            }
        }

        total_entropy += entropy;
    }

    total_entropy / num_groups as f32
}

/// Derivative of [`entropy_regularization`], including its epsilon normalization.
/// At the cutoff `p == EPSILON`, this uses the inactive branch derivative.
/// Incomplete trailing coefficient groups contribute zero, matching the objective.
pub fn entropy_regularization_gradient(coefficients: &[f32], group_size: usize) -> Vec<f32> {
    let mut gradient = vec![0.0; coefficients.len()];
    if group_size == 0 {
        return gradient;
    }
    let groups = coefficients.len() / group_size;
    if groups == 0 {
        return gradient;
    }
    for (group, output) in coefficients
        .chunks_exact(group_size)
        .zip(gradient.chunks_exact_mut(group_size))
    {
        let sum = group.iter().map(|c| c * c).sum::<f32>() + EPSILON;
        let mean_derivative = group
            .iter()
            .map(|c| {
                let p = c * c / sum;
                if p > EPSILON {
                    -p * (p.ln() + 1.0)
                } else {
                    0.0
                }
            })
            .sum::<f32>();
        for (&c, g) in group.iter().zip(output) {
            let p = c * c / sum;
            let derivative = if p > EPSILON { -(p.ln() + 1.0) } else { 0.0 };
            *g = 2.0 * c / sum * (derivative - mean_derivative) / groups as f32;
        }
    }
    gradient
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
/// Returns combined gradient from L1, entropy, and smoothness regularization.
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
    let mut l1_grad = l1_sparsity_gradient(coefficients);
    let smooth_grad = smoothness_gradient(coefficients, basis_size);
    let entropy_grad = entropy_regularization_gradient(coefficients, basis_size);

    let combine = |l1: f32, sm: f32, entropy: f32| {
        config.lambda_l1 * l1 + config.lambda_smooth * sm + config.lambda_entropy * entropy
    };
    if l1_grad.capacity() == l1_grad.len() {
        for ((l1, &sm), &entropy) in l1_grad
            .iter_mut()
            .zip(smooth_grad.iter())
            .zip(entropy_grad.iter())
        {
            let previous_l1 = *l1;
            *l1 = combine(previous_l1, sm, entropy);
        }
        l1_grad
    } else {
        l1_grad
            .iter()
            .zip(smooth_grad.iter())
            .zip(entropy_grad.iter())
            .map(|((&l1, &sm), &entropy)| combine(l1, sm, entropy))
            .collect()
    }
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
