//! Optimizers for KAN network training.
//!
//! This module provides gradient-based optimizers for training KAN networks:
//!
//! - [`Adam`] - Adaptive moment estimation (recommended for most cases)
//! - [`SGD`] - Stochastic gradient descent with momentum
//! - [`LBFGS`] - Limited-memory BFGS for second-order optimization
//! - Learning rate schedulers: [`StepLR`], [`CosineAnnealingLR`]
//!
//! ## v2.1 Features
//!
//! - **Thread Safety**: All optimizers implement `Send + Sync`
//! - **Versioning**: Support for dynamic topology (Grid Extension) via `bump_version()`
//! - **NaN/Inf Handling**: Configurable behavior for numerical instability (NaN and ±inf)
//! - **AMP Support**: Gradient scaling placeholders for mixed precision training
//!
//! # Example
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork};
//! use arkan::optimizer::{Adam, AdamConfig, Optimizer};
//!
//! let config = KanConfig::preset();
//! let mut network = KanNetwork::new(config);
//! let mut optimizer = Adam::new(&network, AdamConfig::with_lr(0.001));
//!
//! // Training loop
//! // optimizer.step(&mut network, &weight_grads, &bias_grads, Some(1.0));
//! ```
//!
//! # Gradient Clipping
//!
//! First-order optimizers support one global gradient norm threshold via `max_grad_norm`,
//! across all weight and bias tensors after AMP unscaling.
//! This helps prevent exploding gradients during training.
//!
//! # Weight Decay
//!
//! Weight decay is implemented as decoupled weight decay (AdamW style),
//! not L2 regularization. This provides better generalization.

use crate::training::gradients::{global_clip_scale, global_grad_norm};
use std::borrow::Cow;

use crate::buffer::AlignedBuffer;
use crate::error::{ArkanError, ArkanResult};
use crate::layer::KanLayer;
use crate::network::KanNetwork;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// =============================================================================
// TRAIT DEFINITION (v2.1)
// =============================================================================

/// Unified optimizer trait for KAN networks.
///
/// All optimizers must implement this trait, which provides:
/// - Parameter updates via `step()` or `step_with_closure()`
/// - Gradient zeroing via `zero_grad()`
/// - Version management for dynamic topology support
/// - Learning rate access per parameter group
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` for use in multi-threaded training.
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::{Optimizer, Adam, AdamConfig};
/// use arkan::{KanConfig, KanNetwork};
///
/// let config = KanConfig::preset();
/// let mut network = KanNetwork::new(config);
/// let mut optimizer = Adam::new(&network, AdamConfig::default());
///
/// // Get/set learning rate
/// let lr = optimizer.get_lr(0).unwrap();
/// optimizer.set_lr(0, 0.0001).unwrap();
///
/// // Version management for Grid Extension
/// optimizer.bump_version();
/// ```
pub trait Optimizer: Send + Sync {
    /// Performs a single optimization step.
    ///
    /// For first-order optimizers (SGD, Adam), this applies gradient updates.
    /// For second-order optimizers (L-BFGS), the closure may be called multiple times.
    ///
    /// # Arguments
    ///
    /// * `network` - Mutable reference to the network being optimized
    /// * `weight_grads` - Weight gradients per layer
    /// * `bias_grads` - Bias gradients per layer
    /// * `max_grad_norm` - Optional global norm threshold over all unscaled tensors
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after an applied update or a numerical skip under
    /// [`SafetyConfig::skip_step_on_nan`], or an error if:
    /// - Gradient/parameter shape mismatch
    /// - NaN detected (if `fail_on_nan` is enabled)
    /// - Numerical issues in the optimizer
    fn step(
        &mut self,
        network: &mut KanNetwork,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) -> ArkanResult<()>;

    /// Performs optimization step with loss closure (for L-BFGS).
    ///
    /// The closure computes the loss and may be called multiple times
    /// during line search.
    ///
    /// # Arguments
    ///
    /// * `closure` - Closure that computes and returns the loss
    ///
    /// # Returns
    ///
    /// Returns the final loss value, or an error.
    fn step_with_closure<F>(&mut self, closure: F) -> ArkanResult<f64>
    where
        F: FnMut() -> ArkanResult<f64>,
    {
        // Default: not supported for first-order optimizers
        let _ = closure;
        Err(ArkanError::optimizer(
            "step_with_closure not supported for this optimizer",
        ))
    }

    /// Zeros all gradients in the network.
    ///
    /// This should be called at the beginning of each training step
    /// to clear gradients from the previous iteration.
    ///
    /// # Note
    ///
    /// This performs in-place zeroing, not re-allocation.
    fn zero_grad(&mut self, network: &mut KanNetwork) -> ArkanResult<()>;

    /// Gets the current state version.
    ///
    /// Used to detect topology changes (Grid Extension).
    fn get_state_version(&self) -> u64;

    /// Bumps the state version and resets optimizer state.
    ///
    /// Call this after Grid Extension or any topology change.
    /// This clears all momentum/history buffers but does not resize tensor state.
    /// After a topology shape change, Adam and SGD also require `reinitialize(network)`.
    fn bump_version(&mut self);

    /// Gets the learning rate for a parameter group.
    ///
    /// # Arguments
    ///
    /// * `group_index` - Index of the parameter group (usually 0)
    ///
    /// # Returns
    ///
    /// Returns the learning rate, or `GroupIndexOutOfBounds` error.
    fn get_lr(&self, group_index: usize) -> ArkanResult<f64>;

    /// Sets the learning rate for a parameter group.
    ///
    /// # Arguments
    ///
    /// * `group_index` - Index of the parameter group (usually 0)
    /// * `new_lr` - New learning rate value
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `GroupIndexOutOfBounds` error.
    fn set_lr(&mut self, group_index: usize, new_lr: f64) -> ArkanResult<()>;

    /// Gets the total number of parameter groups.
    fn num_groups(&self) -> usize {
        1 // Default: single group
    }
}

// =============================================================================
// COMMON CONFIGURATION
// =============================================================================

/// Parameter-group metadata retained for configuration and checkpoint compatibility.
///
/// Current optimizers do not consume this type: overrides, layer selections,
/// gradient scaling and freezing have no runtime effect. They support one group;
/// configure it through [`AdamConfig`], [`SGDConfig`] or [`LBFGSConfig`] and
/// [`Optimizer::set_lr`].
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::ParamGroup;
///
/// // Store metadata; this does not configure an optimizer
/// let group = ParamGroup {
///     lr_override: Some(0.0001),
///     weight_decay_override: Some(0.01),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParamGroup {
    /// Stored learning-rate override; unused by current optimizers.
    pub lr_override: Option<f64>,

    /// Stored weight-decay override; unused by current optimizers.
    pub weight_decay_override: Option<f64>,

    /// Stored Adam betas override; unused by current optimizers.
    pub betas_override: Option<(f64, f64)>,

    /// Stored gradient-enable flag; setting false does not freeze parameters at runtime.
    pub requires_grad: bool,

    /// Stored per-group AMP scaling factor; unused by current optimizers.
    pub grad_scaling: Option<f64>,

    /// Stored layer selection; unused by current optimizers.
    pub layer_indices: Vec<usize>,
}

impl ParamGroup {
    /// Creates a new parameter group for all layers with default settings.
    pub fn all_layers(num_layers: usize) -> Self {
        Self {
            requires_grad: true,
            layer_indices: (0..num_layers).collect(),
            ..Default::default()
        }
    }

    /// Creates metadata with `requires_grad = false`; this does not freeze runtime updates.
    pub fn frozen(layer_indices: Vec<usize>) -> Self {
        Self {
            requires_grad: false,
            layer_indices,
            ..Default::default()
        }
    }

    /// Creates metadata with a learning-rate override; this does not change runtime updates.
    pub fn with_lr(layer_indices: Vec<usize>, lr: f64) -> Self {
        Self {
            requires_grad: true,
            lr_override: Some(lr),
            layer_indices,
            ..Default::default()
        }
    }
}

/// Safety configuration for optimizers.
///
/// Controls NaN handling and AMP (Automatic Mixed Precision) behavior.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SafetyConfig {
    /// If true, returns error when NaN is detected in gradients.
    pub fail_on_nan: bool,

    /// If true, skips the step when NaN is detected (logs warning).
    /// Takes precedence over `fail_on_nan` if both are true.
    pub skip_step_on_nan: bool,

    /// Gradient scaling factor for AMP.
    /// If Some, gradients are divided by this factor before updates.
    pub grad_scaling_factor: Option<f64>,

    /// Compatibility field retained for checkpoints. Gradients are always unscaled
    /// before clipping and updates; decoupled weight decay acts on parameters.
    pub unscale_before_step: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            fail_on_nan: false,
            skip_step_on_nan: true,
            grad_scaling_factor: None,
            unscale_before_step: true,
        }
    }
}

impl SafetyConfig {
    /// Creates a strict safety config that fails on any NaN.
    pub fn strict() -> Self {
        Self {
            fail_on_nan: true,
            skip_step_on_nan: false,
            ..Default::default()
        }
    }

    /// Creates a config with AMP gradient scaling.
    pub fn with_amp(scaling_factor: f64) -> Self {
        Self {
            grad_scaling_factor: Some(scaling_factor),
            unscale_before_step: true,
            ..Default::default()
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Checks for non-finite values in gradients (NaN or ±inf).
///
/// Returns the index of the first non-finite value found, or None if all values are finite.
fn find_nan_in_grads(grads: &[f32]) -> Option<usize> {
    grads.iter().position(|&g| !g.is_finite())
}

fn validate_nonnegative(value: f64, name: &str) -> ArkanResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(ArkanError::optimizer(format!(
            "{name} must be finite and nonnegative"
        )));
    }
    Ok(())
}

fn validate_safety(safety: &SafetyConfig, max_norm: Option<f32>) -> ArkanResult<()> {
    if let Some(factor) = safety.grad_scaling_factor {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(ArkanError::optimizer(
                "gradient scaling factor must be finite and positive",
            ));
        }
    }
    if let Some(max) = max_norm {
        validate_nonnegative(max as f64, "max_grad_norm")?;
    }
    Ok(())
}

fn validate_shape(expected: usize, actual: usize) -> ArkanResult<()> {
    if expected != actual {
        return Err(ArkanError::tensor_shape_mismatch(&[expected], &[actual]));
    }
    Ok(())
}

fn validate_grad_shapes(
    parameters: &crate::model::ParametersMut<'_>,
    weights: &[Vec<f32>],
    biases: &[Vec<f32>],
) -> ArkanResult<()> {
    validate_shape(parameters.len(), weights.len())?;
    validate_shape(parameters.len(), biases.len())?;
    for (((pw, pb), wg), bg) in parameters.iter().zip(weights).zip(biases) {
        validate_shape(pw.len(), wg.len())?;
        validate_shape(pb.len(), bg.len())?;
    }
    Ok(())
}

/// Ok(true) preserves the legacy skip policy; strict errors precede every mutation.
fn check_finite(values: &[f32], safety: &SafetyConfig, context: &str) -> ArkanResult<bool> {
    if safety.fail_on_nan || safety.skip_step_on_nan {
        if let Some(index) = find_nan_in_grads(values) {
            if safety.skip_step_on_nan {
                return Ok(true);
            }
            return Err(ArkanError::nan_encountered(index, context));
        }
    }
    Ok(false)
}

/// Internal transaction result; public `Optimizer::step` retains its unit return type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepOutcome {
    Applied,
    Skipped,
}

type PreparedGradients<'a> = (Cow<'a, [Vec<f32>]>, Cow<'a, [Vec<f32>]>);

fn prepare_gradients<'a>(
    weights: &'a [Vec<f32>],
    biases: &'a [Vec<f32>],
    safety: &SafetyConfig,
    max_norm: Option<f32>,
) -> ArkanResult<Option<PreparedGradients<'a>>> {
    for tensor in weights.iter().chain(biases) {
        if check_finite(tensor, safety, "gradient")? {
            return Ok(None);
        }
    }
    let unscale = |source: &'a [Vec<f32>]| -> Cow<'a, [Vec<f32>]> {
        match safety.grad_scaling_factor {
            Some(factor) => Cow::Owned(
                source
                    .iter()
                    .map(|tensor| tensor.iter().map(|&g| (g as f64 / factor) as f32).collect())
                    .collect(),
            ),
            None => Cow::Borrowed(source),
        }
    };
    let mut wg = unscale(weights);
    let mut bg = unscale(biases);
    if safety.grad_scaling_factor.is_some() {
        for tensor in wg.iter().chain(bg.iter()) {
            if check_finite(tensor, safety, "unscaled gradient")? {
                return Ok(None);
            }
        }
    }
    if max_norm.is_some() {
        let scale = global_clip_scale(global_grad_norm(&wg, &bg), max_norm);
        if scale != 1.0 {
            for tensor in wg.to_mut().iter_mut().chain(bg.to_mut()) {
                for g in tensor {
                    *g = (*g as f64 * scale) as f32;
                }
            }
        }
    }
    Ok(Some((wg, bg)))
}

mod first_order;
mod lbfgs;
mod scheduler;

pub use first_order::{Adam, AdamConfig, AdamState, LayerAdamState, SGDConfig, SGD};
pub use lbfgs::{LBFGSConfig, LineSearchMethod, LBFGS};
pub use scheduler::{CosineAnnealingLR, LrScheduler, StepLR};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KanConfig;

    #[test]
    fn test_adam_state_creation() {
        let state = AdamState::new(100);
        assert_eq!(state.m.len(), 100);
        assert_eq!(state.v.len(), 100);
        assert_eq!(state.t, 0);
    }

    #[test]
    fn test_adam_optimizer() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config);
        let mut optimizer = Adam::new(&network, AdamConfig::with_lr(0.001));

        assert_eq!(optimizer.layer_states.len(), network.layers.len());
        assert_eq!(optimizer.learning_rate(), 0.001);

        optimizer.set_learning_rate(0.0001);
        assert_eq!(optimizer.learning_rate(), 0.0001);
    }

    #[test]
    fn test_adam_update() {
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![],
            grid_size: 5,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            multithreading_threshold: 1024,
            simd_width: 8,
            init_seed: None,
        };

        let mut network = KanNetwork::new(config);
        let mut optimizer = Adam::new(&network, AdamConfig::with_lr(0.1));

        // Get initial weight
        let initial_weight = network.layers[0].weights[0];

        // Create gradients (all positive)
        let weight_grads = vec![vec![1.0f32; network.layers[0].weights.len()]];
        let bias_grads = vec![vec![0.5f32; network.layers[0].bias.len()]];

        // Step using new API
        optimizer
            .step(&mut network, &weight_grads, &bias_grads, None)
            .unwrap();

        // Weight should have decreased
        let new_weight = network.layers[0].weights[0];
        assert!(
            new_weight < initial_weight,
            "Weight should decrease with positive gradient: {} -> {}",
            initial_weight,
            new_weight
        );
    }

    #[test]
    fn test_step_lr() {
        let scheduler = StepLR::new(0.1, 10, 0.5);

        assert!((scheduler.get_lr(0, 0.1) - 0.1).abs() < 1e-6);
        assert!((scheduler.get_lr(10, 0.1) - 0.05).abs() < 1e-6);
        assert!((scheduler.get_lr(20, 0.1) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_lr() {
        let scheduler = CosineAnnealingLR::new(0.1, 100, 0.001);

        // At start
        let lr_start = scheduler.get_lr(0, 0.1);
        assert!((lr_start - 0.1).abs() < 0.01);

        // At end
        let lr_end = scheduler.get_lr(100, 0.1);
        assert!((lr_end - 0.001).abs() < 0.01);

        // At middle (should be around midpoint)
        let lr_mid = scheduler.get_lr(50, 0.1);
        assert!(lr_mid > 0.001 && lr_mid < 0.1);
    }

    // =========================================================================
    // NEW TESTS FOR v2.1 FEATURES
    // =========================================================================

    #[test]
    fn test_optimizer_trait_get_set_lr() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config);
        let mut optimizer = Adam::new(&network, AdamConfig::with_lr(0.001));

        // Test get_lr via trait
        assert!((optimizer.get_lr(0).unwrap() - 0.001).abs() < 1e-6);

        // Test set_lr via trait
        optimizer.set_lr(0, 0.0001).unwrap();
        assert!((optimizer.get_lr(0).unwrap() - 0.0001).abs() < 1e-6);

        // Test out of bounds
        assert!(optimizer.get_lr(1).is_err());
        assert!(optimizer.set_lr(999, 0.01).is_err());
    }

    #[test]
    fn test_optimizer_versioning() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config);
        let mut optimizer = Adam::new(&network, AdamConfig::with_lr(0.001));

        assert_eq!(optimizer.get_state_version(), 0);

        optimizer.bump_version();
        assert_eq!(optimizer.get_state_version(), 1);

        // All states should be reset
        for state in &optimizer.layer_states {
            assert_eq!(state.weights.t, 0);
            assert_eq!(state.bias.t, 0);
        }
    }

    #[test]
    fn test_sgd_new_api() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config);
        let optimizer = SGD::new(&network, SGDConfig::with_momentum(0.01, 0.9));

        assert!((optimizer.lr() - 0.01).abs() < 1e-6);
        assert!((optimizer.momentum() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_safety_config() {
        let safety = SafetyConfig::strict();
        assert!(safety.fail_on_nan);
        assert!(!safety.skip_step_on_nan);

        let amp = SafetyConfig::with_amp(1024.0);
        assert_eq!(amp.grad_scaling_factor, Some(1024.0));
        assert!(amp.unscale_before_step);
    }

    #[test]
    fn test_nan_detection_skip() {
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        let mut network = KanNetwork::new(config);
        let initial_weights = network.layers[0].weights.clone();

        let mut optimizer = Adam::new(
            &network,
            AdamConfig::with_lr(0.1).with_safety(SafetyConfig::default()),
        );

        // Create gradients with NaN
        let weight_grads = vec![vec![f32::NAN; network.layers[0].weights.len()]];
        let bias_grads = vec![vec![0.5f32; network.layers[0].bias.len()]];

        // Step should succeed but skip (default: skip_step_on_nan = true)
        let result = optimizer.step(&mut network, &weight_grads, &bias_grads, None);
        assert!(result.is_ok());

        // Weights should be unchanged
        assert_eq!(
            network.layers[0].weights.as_slice(),
            initial_weights.as_slice()
        );
    }

    #[test]
    fn test_nan_detection_fail() {
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        let mut network = KanNetwork::new(config);

        let strict_safety = SafetyConfig {
            fail_on_nan: true,
            skip_step_on_nan: false,
            ..Default::default()
        };

        let mut optimizer = Adam::new(
            &network,
            AdamConfig::with_lr(0.1).with_safety(strict_safety),
        );

        // Create gradients with NaN
        let weight_grads = vec![vec![f32::NAN; network.layers[0].weights.len()]];
        let bias_grads = vec![vec![0.5f32; network.layers[0].bias.len()]];

        // Step should fail
        let result = optimizer.step(&mut network, &weight_grads, &bias_grads, None);
        assert!(result.is_err());
    }

    /// Regression test: inf gradients must be caught by SafetyConfig just like NaN.
    ///
    /// Before the fix, `find_nan_in_grads` used `g.is_nan()` which let ±inf through,
    /// causing Adam to silently corrupt parameters to ±inf even under strict().
    #[test]
    fn test_inf_gradient_strict_fails() {
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        let mut network = KanNetwork::new(config.clone());

        // SafetyConfig::strict() sets fail_on_nan=true, skip_step_on_nan=false.
        // It must also reject inf gradients.
        let mut optimizer_strict = Adam::new(
            &network,
            AdamConfig::with_lr(0.1).with_safety(SafetyConfig::strict()),
        );

        let inf_weight_grads = vec![vec![f32::INFINITY; network.layers[0].weights.len()]];
        let normal_bias_grads = vec![vec![0.5f32; network.layers[0].bias.len()]];

        let result =
            optimizer_strict.step(&mut network, &inf_weight_grads, &normal_bias_grads, None);
        assert!(
            result.is_err(),
            "SafetyConfig::strict() must return Err for inf gradient"
        );

        // Also test negative infinity
        let neg_inf_grads = vec![vec![f32::NEG_INFINITY; network.layers[0].weights.len()]];
        let mut network2 = KanNetwork::new(config.clone());
        let mut optimizer_strict2 = Adam::new(
            &network2,
            AdamConfig::with_lr(0.1).with_safety(SafetyConfig::strict()),
        );
        let result2 =
            optimizer_strict2.step(&mut network2, &neg_inf_grads, &normal_bias_grads, None);
        assert!(
            result2.is_err(),
            "SafetyConfig::strict() must return Err for -inf gradient"
        );
    }

    /// Regression test: skip_step_on_nan must skip the step for inf gradients,
    /// leaving parameters unchanged.
    #[test]
    fn test_inf_gradient_skip_leaves_params_unchanged() {
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        let mut network = KanNetwork::new(config);
        let initial_weights = network.layers[0].weights.clone();
        let initial_bias = network.layers[0].bias.clone();

        // Default config has skip_step_on_nan=true, fail_on_nan=false
        let mut optimizer = Adam::new(
            &network,
            AdamConfig::with_lr(0.1).with_safety(SafetyConfig::default()),
        );

        let inf_weight_grads = vec![vec![f32::INFINITY; network.layers[0].weights.len()]];
        let normal_bias_grads = vec![vec![0.5f32; network.layers[0].bias.len()]];

        // Step should succeed (Ok) but skip the update
        let result = optimizer.step(&mut network, &inf_weight_grads, &normal_bias_grads, None);
        assert!(result.is_ok(), "skip_step_on_nan should return Ok, not Err");

        // Parameters must be completely unchanged
        assert_eq!(
            network.layers[0].weights.as_slice(),
            initial_weights.as_slice(),
            "Weights must be unchanged when step is skipped due to inf gradient"
        );
        assert_eq!(
            network.layers[0].bias.as_slice(),
            initial_bias.as_slice(),
            "Bias must be unchanged when step is skipped due to inf gradient"
        );
    }

    #[test]
    fn test_lbfgs_creation() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config);
        let optimizer = LBFGS::new(&network, LBFGSConfig::default());

        assert_eq!(optimizer.get_state_version(), 0);
        assert!((optimizer.get_lr(0).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}

        // These should compile if Adam, SGD, LBFGS implement Send + Sync
        assert_send_sync::<Adam>();
        assert_send_sync::<SGD>();
        assert_send_sync::<LBFGS>();
    }

    // =========================================================================
    // NEW TESTS FOR v2.0 FEATURES
    // =========================================================================

    #[test]
    fn test_sgd_nesterov() {
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        let mut network = KanNetwork::new(config);
        let initial_weights = network.layers[0].weights.clone();

        // Create Nesterov SGD
        let mut optimizer = SGD::new(&network, SGDConfig::with_nesterov(0.1, 0.9));

        // Create constant gradients
        let weight_grads = vec![vec![1.0f32; network.layers[0].weights.len()]];
        let bias_grads = vec![vec![0.5f32; network.layers[0].bias.len()]];

        // Step 1
        optimizer
            .step(&mut network, &weight_grads, &bias_grads, None)
            .unwrap();

        // Weights should have decreased more than standard momentum due to look-ahead
        // For Nesterov: update = μ*(μ*v + g) + g = μ²v + μg + g
        // First step v=0: update = 0 + μ*1 + 1 = μ + 1 = 1.9
        // param -= lr * update = param - 0.1 * 1.9 = param - 0.19
        let expected_change = 0.1 * (0.9 * 1.0 + 1.0); // 0.19
        let actual_change = initial_weights[0] - network.layers[0].weights[0];

        assert!(
            (actual_change - expected_change).abs() < 0.01,
            "Nesterov update: expected ~{:.4}, got {:.4}",
            expected_change,
            actual_change
        );
    }

    #[test]
    fn test_sgd_nesterov_vs_standard() {
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        // Two networks with same init
        let mut net_standard = KanNetwork::new(config.clone());
        let mut net_nesterov = KanNetwork::new(config);

        let mut opt_standard = SGD::new(&net_standard, SGDConfig::with_momentum(0.1, 0.9));
        let mut opt_nesterov = SGD::new(&net_nesterov, SGDConfig::with_nesterov(0.1, 0.9));

        let weight_grads = vec![vec![1.0f32; net_standard.layers[0].weights.len()]];
        let bias_grads = vec![vec![0.5f32; net_standard.layers[0].bias.len()]];

        // After first step, Nesterov should move more aggressively
        opt_standard
            .step(&mut net_standard, &weight_grads, &bias_grads, None)
            .unwrap();
        opt_nesterov
            .step(&mut net_nesterov, &weight_grads, &bias_grads, None)
            .unwrap();

        let w_standard = net_standard.layers[0].weights[0];
        let w_nesterov = net_nesterov.layers[0].weights[0];

        // Nesterov should have moved further (smaller weight = more decrease)
        assert!(
            w_nesterov < w_standard,
            "Nesterov should be more aggressive: standard={:.6}, nesterov={:.6}",
            w_standard,
            w_nesterov
        );
    }

    #[test]
    fn test_param_group_creation() {
        // Test all layers group
        let group = ParamGroup::all_layers(4);
        assert!(group.requires_grad);
        assert_eq!(group.layer_indices, vec![0, 1, 2, 3]);
        assert!(group.lr_override.is_none());

        // Test frozen group
        let frozen = ParamGroup::frozen(vec![0, 1]);
        assert!(!frozen.requires_grad);
        assert_eq!(frozen.layer_indices, vec![0, 1]);

        // Test custom LR group
        let custom = ParamGroup::with_lr(vec![2, 3], 0.0001);
        assert!(custom.requires_grad);
        assert_eq!(custom.lr_override, Some(0.0001));
    }

    #[test]
    fn test_lbfgs_two_loop_recursion() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config);
        let optimizer = LBFGS::new(&network, LBFGSConfig::default());

        // With no history, should return negative gradient (steepest descent)
        let grad = vec![1.0f32, 2.0, 3.0];
        let direction = optimizer.two_loop_recursion(&grad);

        assert_eq!(direction.len(), 3);
        assert!((direction[0] - (-1.0)).abs() < 1e-5);
        assert!((direction[1] - (-2.0)).abs() < 1e-5);
        assert!((direction[2] - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_lbfgs_pack_unpack() {
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![4],
            grid_size: 3,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            init_seed: Some(42),
            ..Default::default()
        };

        let network = KanNetwork::new(config);

        // Flatten params
        let params = LBFGS::flatten_params(&network);

        // Check total size
        let expected_size: usize = network
            .layers
            .iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum();
        assert_eq!(params.len(), expected_size);

        // Restore to new network and verify
        let mut network2 = network.clone();
        // Modify weights
        for layer in &mut network2.layers {
            for w in layer.weights.as_mut_slice() {
                *w = 0.0;
            }
        }

        // Restore original params
        LBFGS::restore_params(&mut network2, &params);

        // Verify restoration
        for (l1, l2) in network.layers.iter().zip(network2.layers.iter()) {
            assert_eq!(l1.weights.as_slice(), l2.weights.as_slice());
            assert_eq!(l1.bias.as_slice(), l2.bias.as_slice());
        }
    }

    #[test]
    fn test_line_search_method_default() {
        let method = LineSearchMethod::default();
        assert_eq!(method, LineSearchMethod::StrongWolfe);
    }

    #[test]
    fn test_lbfgs_config_variants() {
        let config = LBFGSConfig {
            lr: 0.5,
            line_search_fn: LineSearchMethod::Backtracking,
            ..Default::default()
        };
        assert_eq!(config.line_search_fn, LineSearchMethod::Backtracking);
        assert!((config.lr - 0.5).abs() < 1e-6);

        let config2 = LBFGSConfig {
            line_search_fn: LineSearchMethod::NoLineSearch,
            ..Default::default()
        };
        assert_eq!(config2.line_search_fn, LineSearchMethod::NoLineSearch);
    }

    #[test]
    fn test_workspace_zero_grads() {
        use crate::buffer::Workspace;

        let config = KanConfig::preset();
        let mut workspace = Workspace::new(&config);

        // Manually add some gradient data
        workspace.weight_grads.push(vec![1.0, 2.0, 3.0]);
        workspace.weight_grads.push(vec![4.0, 5.0]);
        workspace.bias_grads.push(vec![0.5, 0.6]);
        workspace.bias_grads.push(vec![0.7]);

        // Zero them
        workspace.zero_grads();

        // Verify all zeros
        for wg in &workspace.weight_grads {
            for &v in wg {
                assert_eq!(v, 0.0);
            }
        }
        for bg in &workspace.bias_grads {
            for &v in bg {
                assert_eq!(v, 0.0);
            }
        }
    }
}
