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
//! - **NaN Handling**: Configurable behavior for numerical instability
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
//! All optimizers support gradient clipping via `max_grad_norm` parameter.
//! This helps prevent exploding gradients during training.
//!
//! # Weight Decay
//!
//! Weight decay is implemented as decoupled weight decay (AdamW style),
//! not L2 regularization. This provides better generalization.

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
    /// * `max_grad_norm` - Optional gradient clipping threshold
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if:
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
    /// This clears all momentum/history buffers.
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

/// Parameter group for per-layer or per-parameter-set optimization.
///
/// Allows different learning rates, weight decay, and other hyperparameters
/// for different parts of the network.
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::ParamGroup;
///
/// // Custom group with overrides
/// let group = ParamGroup {
///     lr_override: Some(0.0001),
///     weight_decay_override: Some(0.01),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParamGroup {
    /// Override learning rate for this group. If None, uses optimizer default.
    pub lr_override: Option<f64>,

    /// Override weight decay for this group. If None, uses optimizer default.
    pub weight_decay_override: Option<f64>,

    /// Override betas (for Adam) for this group. If None, uses optimizer default.
    pub betas_override: Option<(f64, f64)>,

    /// If false, parameters in this group are frozen (no gradient updates).
    pub requires_grad: bool,

    /// Per-group gradient scaling factor for AMP.
    pub grad_scaling: Option<f64>,

    /// Layer indices that belong to this group.
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

    /// Creates a frozen group (no gradient updates).
    pub fn frozen(layer_indices: Vec<usize>) -> Self {
        Self {
            requires_grad: false,
            layer_indices,
            ..Default::default()
        }
    }

    /// Creates a group with custom learning rate.
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

    /// If true, unscales gradients before applying weight decay.
    /// Only relevant when `grad_scaling_factor` is set.
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

/// Checks for NaN values in gradients.
///
/// Returns the index of the first NaN found, or None if all values are finite.
fn find_nan_in_grads(grads: &[f32]) -> Option<usize> {
    grads.iter().position(|&g| g.is_nan())
}

/// Applies gradient scaling for AMP.
///
/// Returns scaled gradients if scaling factor is set, otherwise clones input.
fn apply_grad_scaling(grads: &[f32], scaling_factor: Option<f64>) -> Vec<f32> {
    match scaling_factor {
        Some(factor) => {
            let inv_factor = 1.0 / factor as f32;
            grads.iter().map(|&g| g * inv_factor).collect()
        }
        None => grads.to_vec(),
    }
}

/// Applies gradient clipping and returns clipped gradients.
fn clip_gradients(
    weight_grads: &[f32],
    bias_grads: &[f32],
    max_norm: Option<f32>,
) -> (Vec<f32>, Vec<f32>) {
    if let Some(max_norm) = max_norm {
        let mut sq: f32 = weight_grads.iter().map(|g| g * g).sum();
        sq += bias_grads.iter().map(|g| g * g).sum::<f32>();
        let norm = sq.sqrt();

        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            let wg: Vec<f32> = weight_grads.iter().map(|g| g * scale).collect();
            let bg: Vec<f32> = bias_grads.iter().map(|g| g * scale).collect();
            return (wg, bg);
        }
    }
    (weight_grads.to_vec(), bias_grads.to_vec())
}

/// Adam optimizer state for a single parameter tensor.
///
/// Stores the first moment (mean) and second moment (variance)
/// estimates used by the Adam algorithm.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdamState {
    /// First moment estimate (exponential moving average of gradients).
    pub m: AlignedBuffer,

    /// Second moment estimate (exponential moving average of squared gradients).
    pub v: AlignedBuffer,

    /// Timestep counter for bias correction.
    pub t: usize,
}

impl AdamState {
    /// Creates a new Adam state for a parameter tensor of given size.
    ///
    /// Initializes both moments to zero.
    pub fn new(size: usize) -> Self {
        let mut m = AlignedBuffer::with_capacity(size);
        m.resize(size); // Zeros

        let mut v = AlignedBuffer::with_capacity(size);
        v.resize(size); // Zeros

        Self { m, v, t: 0 }
    }

    /// Resets the state.
    pub fn reset(&mut self) {
        self.m.zero();
        self.v.zero();
        self.t = 0;
    }
}

impl Clone for AdamState {
    fn clone(&self) -> Self {
        Self {
            m: self.m.clone(),
            v: self.v.clone(),
            t: self.t,
        }
    }
}

/// Adam optimizer configuration.
///
/// # Default Values
///
/// | Parameter | Default | Description |
/// |-----------|---------|-------------|
/// | `lr` | 0.001 | Learning rate |
/// | `beta1` | 0.9 | First moment decay |
/// | `beta2` | 0.999 | Second moment decay |
/// | `epsilon` | 1e-8 | Numerical stability |
/// | `weight_decay` | 0.0 | L2 regularization |
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::{AdamConfig, SafetyConfig};
///
/// // Default config
/// let config = AdamConfig::default();
///
/// // Custom learning rate
/// let config = AdamConfig::with_lr(0.0001);
///
/// // With weight decay (AdamW)
/// let config = AdamConfig::with_decay(0.001, 0.01);
///
/// // With safety settings
/// let config = AdamConfig::default().with_safety(SafetyConfig::strict());
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdamConfig {
    /// Learning rate (alpha).
    pub lr: f32,

    /// First moment decay (beta1).
    pub beta1: f32,

    /// Second moment decay (beta2).
    pub beta2: f32,

    /// Epsilon for numerical stability.
    pub epsilon: f32,

    /// Weight decay (L2 regularization).
    pub weight_decay: f32,

    /// Safety configuration for NaN handling and AMP.
    #[cfg_attr(feature = "serde", serde(default))]
    pub safety: SafetyConfig,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            safety: SafetyConfig::default(),
        }
    }
}

impl AdamConfig {
    /// Creates config with learning rate.
    pub fn with_lr(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Creates config with learning rate and weight decay.
    pub fn with_decay(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            ..Default::default()
        }
    }

    /// Sets safety configuration.
    pub fn with_safety(mut self, safety: SafetyConfig) -> Self {
        self.safety = safety;
        self
    }
}

/// Per-layer optimizer state for Adam.
///
/// Holds separate states for weights and biases.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LayerAdamState {
    /// State for weights.
    pub weights: AdamState,

    /// State for bias.
    pub bias: AdamState,
}

impl LayerAdamState {
    /// Creates state for a layer.
    pub fn new(layer: &KanLayer) -> Self {
        Self {
            weights: AdamState::new(layer.weights.len()),
            bias: AdamState::new(layer.bias.len()),
        }
    }

    /// Resets all states.
    pub fn reset(&mut self) {
        self.weights.reset();
        self.bias.reset();
    }
}

impl Clone for LayerAdamState {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}

/// Adam optimizer for KAN networks.
///
/// Implements the Adam algorithm with bias correction and optional
/// decoupled weight decay (AdamW).
///
/// # Algorithm
///
/// $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
/// $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
/// $$\hat{m}_t = m_t / (1 - \beta_1^t)$$
/// $$\hat{v}_t = v_t / (1 - \beta_2^t)$$
/// $$\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$
///
/// # Thread Safety
///
/// `Adam` implements `Send + Sync` for use in multi-threaded training.
///
/// # Versioning
///
/// Supports dynamic topology via `bump_version()`. Call this after Grid Extension.
///
/// # Example
///
/// ```rust
/// use arkan::{KanConfig, KanNetwork};
/// use arkan::optimizer::{Adam, AdamConfig, Optimizer};
///
/// let config = KanConfig::preset();
/// let mut network = KanNetwork::new(config);
/// let mut optimizer = Adam::new(&network, AdamConfig::with_lr(0.001));
///
/// // Check optimizer version
/// assert_eq!(optimizer.get_state_version(), 0);
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Adam {
    /// Configuration.
    pub config: AdamConfig,

    /// Per-layer states.
    pub layer_states: Vec<LayerAdamState>,

    /// State version for topology tracking.
    state_version: u64,
}

// SAFETY: Adam uses only thread-safe types (AlignedBuffer is Send+Sync)
unsafe impl Send for Adam {}
unsafe impl Sync for Adam {}

impl Adam {
    /// Creates a new Adam optimizer for the given network.
    pub fn new(network: &KanNetwork, config: AdamConfig) -> Self {
        let layer_states = network.layers.iter().map(LayerAdamState::new).collect();

        Self {
            config,
            layer_states,
            state_version: 0,
        }
    }

    /// Creates optimizer with default config.
    pub fn default_for(network: &KanNetwork) -> Self {
        Self::new(network, AdamConfig::default())
    }

    /// Resets all optimizer state (but preserves version).
    pub fn reset(&mut self) {
        for state in &mut self.layer_states {
            state.reset();
        }
    }

    /// Reinitializes state for a new network topology.
    ///
    /// Call this after Grid Extension when the network structure changes.
    pub fn reinitialize(&mut self, network: &KanNetwork) {
        self.layer_states = network.layers.iter().map(LayerAdamState::new).collect();
    }

    /// Gets current learning rate.
    #[inline]
    pub fn learning_rate(&self) -> f32 {
        self.config.lr
    }

    /// Sets learning rate.
    #[inline]
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.lr = lr;
    }

    /// Checks gradients for NaN values.
    ///
    /// Returns Ok(()) if all gradients are finite, or an error describing
    /// where NaN was found.
    fn check_nan_in_layer(
        weight_grads: &[f32],
        bias_grads: &[f32],
        layer_idx: usize,
        safety: &SafetyConfig,
    ) -> ArkanResult<bool> {
        if !safety.fail_on_nan && !safety.skip_step_on_nan {
            return Ok(false); // No NaN checking needed
        }

        if let Some(idx) = find_nan_in_grads(weight_grads) {
            if safety.fail_on_nan && !safety.skip_step_on_nan {
                return Err(ArkanError::nan_encountered(
                    idx,
                    format!("weight gradient at layer {}", layer_idx),
                ));
            }
            return Ok(true); // Skip this step
        }

        if let Some(idx) = find_nan_in_grads(bias_grads) {
            if safety.fail_on_nan && !safety.skip_step_on_nan {
                return Err(ArkanError::nan_encountered(
                    idx,
                    format!("bias gradient at layer {}", layer_idx),
                ));
            }
            return Ok(true); // Skip this step
        }

        Ok(false)
    }

    /// Updates a single parameter tensor using Adam.
    ///
    /// # Update Order (AdamW-style)
    ///
    /// 1. **Moment update**: $m_t$, $v_t$ computed from gradients
    /// 2. **Weight decay**: `param *= 1 - lr * decay` (applied BEFORE gradient step)
    /// 3. **Gradient step**: `param -= alpha * m / (sqrt(v) + eps)`
    ///
    /// This is "decoupled" weight decay (AdamW), not L2 regularization.
    /// The decay is proportional to `lr`, so it scales with learning rate.
    fn update_params(
        params: &mut [f32],
        grads: &[f32],
        state: &mut AdamState,
        config: &AdamConfig,
    ) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), state.m.len());

        state.t += 1;
        let _t = state.t as f32;

        let beta1 = config.beta1;
        let beta2 = config.beta2;
        let lr = config.lr;
        let eps = config.epsilon;
        let decay = config.weight_decay;

        // Bias correction factors
        let bc1 = 1.0 - beta1.powi(state.t as i32);
        let bc2 = 1.0 - beta2.powi(state.t as i32);
        let alpha = lr * (bc2.sqrt()) / bc1;

        let m = state.m.as_mut_slice();
        let v = state.v.as_mut_slice();

        for i in 0..params.len() {
            let g = grads[i];

            // Update moments
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            // Compute update
            let update = alpha * m[i] / (v[i].sqrt() + eps);

            // Apply weight decay (decoupled, AdamW-style)
            // Applied BEFORE gradient update for proper decoupling
            if decay > 0.0 {
                params[i] *= 1.0 - lr * decay;
            }

            // Update parameter
            params[i] -= update;
        }
    }

    /// Performs one optimization step on a single layer.
    pub fn step_layer(
        &mut self,
        layer_idx: usize,
        layer: &mut KanLayer,
        weight_grads: &[f32],
        bias_grads: &[f32],
        max_grad_norm: Option<f32>,
    ) {
        let state = &mut self.layer_states[layer_idx];

        let (wg_scaled, bg_scaled) = if let Some(max_norm) = max_grad_norm {
            let mut sq: f32 = weight_grads.iter().map(|g| g * g).sum();
            sq += bias_grads.iter().map(|g| g * g).sum::<f32>();
            let norm = sq.sqrt();
            if norm > max_norm && norm > 0.0 {
                let scale = max_norm / norm;
                let mut wg = weight_grads.to_vec();
                let mut bg = bias_grads.to_vec();
                for g in &mut wg {
                    *g *= scale;
                }
                for g in &mut bg {
                    *g *= scale;
                }
                (wg, bg)
            } else {
                (weight_grads.to_vec(), bias_grads.to_vec())
            }
        } else {
            (weight_grads.to_vec(), bias_grads.to_vec())
        };

        Self::update_params(
            layer.weights.as_mut_slice(),
            &wg_scaled,
            &mut state.weights,
            &self.config,
        );

        Self::update_params(
            layer.bias.as_mut_slice(),
            &bg_scaled,
            &mut state.bias,
            &self.config,
        );
    }

    /// Performs one optimization step on the entire network (legacy API).
    ///
    /// # Arguments
    ///
    /// * `network` - Network to update
    /// * `all_weight_grads` - Weight gradients per layer
    /// * `all_bias_grads` - Bias gradients per layer
    /// * `max_grad_norm` - Optional gradient clipping threshold
    ///
    /// # Note
    ///
    /// This is the legacy API. Use the `Optimizer` trait method for new code.
    pub fn step_legacy(
        &mut self,
        network: &mut KanNetwork,
        all_weight_grads: &[Vec<f32>],
        all_bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) {
        debug_assert_eq!(all_weight_grads.len(), network.layers.len());
        debug_assert_eq!(all_bias_grads.len(), network.layers.len());

        for (i, layer) in network.layers.iter_mut().enumerate() {
            self.step_layer(
                i,
                layer,
                &all_weight_grads[i],
                &all_bias_grads[i],
                max_grad_norm,
            );
        }
    }

    /// Backward-compatible step without gradient clipping.
    #[deprecated(since = "0.2.0", note = "use Optimizer::step() instead")]
    pub fn step_unclipped(
        &mut self,
        network: &mut KanNetwork,
        all_weight_grads: &[Vec<f32>],
        all_bias_grads: &[Vec<f32>],
    ) {
        self.step_legacy(network, all_weight_grads, all_bias_grads, None);
    }
}

impl Clone for Adam {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            layer_states: self.layer_states.clone(),
            state_version: self.state_version,
        }
    }
}

// =============================================================================
// TRAIT IMPLEMENTATION FOR ADAM
// =============================================================================

impl Optimizer for Adam {
    fn step(
        &mut self,
        network: &mut KanNetwork,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) -> ArkanResult<()> {
        // Validate layer count
        if weight_grads.len() != network.layers.len() || bias_grads.len() != network.layers.len() {
            return Err(ArkanError::tensor_shape_mismatch(
                &[network.layers.len()],
                &[weight_grads.len()],
            ));
        }

        // Check for NaN and handle according to safety config
        for (i, (wg, bg)) in weight_grads.iter().zip(bias_grads.iter()).enumerate() {
            let should_skip = Self::check_nan_in_layer(wg, bg, i, &self.config.safety)?;
            if should_skip {
                // Skip step due to NaN (warning logged elsewhere if needed)
                return Ok(());
            }
        }

        // Apply AMP scaling if configured
        let scaling_factor = self.config.safety.grad_scaling_factor;

        for (i, layer) in network.layers.iter_mut().enumerate() {
            let state = &mut self.layer_states[i];

            // Apply gradient scaling for AMP
            let wg_scaled = apply_grad_scaling(&weight_grads[i], scaling_factor);
            let bg_scaled = apply_grad_scaling(&bias_grads[i], scaling_factor);

            // Apply gradient clipping
            let (wg_clipped, bg_clipped) = clip_gradients(&wg_scaled, &bg_scaled, max_grad_norm);

            // Update parameters
            Self::update_params(
                layer.weights.as_mut_slice(),
                &wg_clipped,
                &mut state.weights,
                &self.config,
            );

            Self::update_params(
                layer.bias.as_mut_slice(),
                &bg_clipped,
                &mut state.bias,
                &self.config,
            );
        }

        Ok(())
    }

    fn zero_grad(&mut self, network: &mut KanNetwork) -> ArkanResult<()> {
        // In ArKan, gradients are computed fresh each step and passed to optimizer,
        // so zero_grad is a no-op for the network. However, we can reset state if needed.
        let _ = network;
        Ok(())
    }

    fn get_state_version(&self) -> u64 {
        self.state_version
    }

    fn bump_version(&mut self) {
        self.state_version += 1;
        // Clear all momentum buffers as they're no longer relevant
        self.reset();
    }

    fn get_lr(&self, group_index: usize) -> ArkanResult<f64> {
        if group_index == 0 {
            Ok(self.config.lr as f64)
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }

    fn set_lr(&mut self, group_index: usize, new_lr: f64) -> ArkanResult<()> {
        if group_index == 0 {
            self.config.lr = new_lr as f32;
            Ok(())
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }
}

/// SGD optimizer with momentum.
///
/// Implements classic stochastic gradient descent with optional
/// momentum and decoupled weight decay.
///
/// # Algorithm
///
/// $$v_t = \mu \cdot v_{t-1} + g_t$$
/// $$\theta_t = \theta_{t-1} - \alpha \cdot v_t$$
///
/// # Update Order
///
/// 1. **Velocity update**: $v = \mu \cdot v + g$ (momentum accumulation)
/// 2. **Weight decay**: `param *= 1 - lr * decay` (applied BEFORE gradient step)
/// 3. **Gradient step**: `param -= lr * v`
///
/// Weight decay is decoupled (not L2 regularization), matching AdamW behavior.
/// Note: decay is only applied to weights, not biases.
///
/// # Thread Safety
///
/// `SGD` implements `Send + Sync` for use in multi-threaded training.
///
/// # Example
///
/// ```rust
/// use arkan::{KanConfig, KanNetwork};
/// use arkan::optimizer::{SGD, SGDConfig, Optimizer};
///
/// let config = KanConfig::preset();
/// let network = KanNetwork::new(config);
/// let mut optimizer = SGD::new(&network, SGDConfig::with_lr(0.01));
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SGD {
    /// Configuration.
    pub config: SGDConfig,

    /// Velocity for each parameter.
    pub velocities: Vec<(AlignedBuffer, AlignedBuffer)>,

    /// State version for topology tracking.
    state_version: u64,
}

/// SGD optimizer configuration.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SGDConfig {
    /// Learning rate.
    pub lr: f32,

    /// Momentum coefficient (0 = no momentum).
    pub momentum: f32,

    /// Weight decay coefficient.
    pub weight_decay: f32,

    /// Use Nesterov momentum (look-ahead gradients).
    pub nesterov: bool,

    /// Safety configuration for NaN handling and AMP.
    #[cfg_attr(feature = "serde", serde(default))]
    pub safety: SafetyConfig,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            safety: SafetyConfig::default(),
        }
    }
}

impl SGDConfig {
    /// Creates config with learning rate.
    pub fn with_lr(lr: f32) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    /// Creates config with learning rate and momentum.
    pub fn with_momentum(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            ..Default::default()
        }
    }

    /// Creates config with Nesterov momentum.
    ///
    /// Nesterov momentum computes gradients at the "look-ahead" position,
    /// often leading to faster convergence than standard momentum.
    pub fn with_nesterov(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            nesterov: true,
            ..Default::default()
        }
    }

    /// Creates full config.
    pub fn full(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            ..Default::default()
        }
    }

    /// Creates full config with Nesterov option.
    pub fn full_nesterov(lr: f32, momentum: f32, weight_decay: f32, nesterov: bool) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            nesterov,
            ..Default::default()
        }
    }

    /// Sets safety configuration.
    pub fn with_safety(mut self, safety: SafetyConfig) -> Self {
        self.safety = safety;
        self
    }
}

// SAFETY: SGD uses only thread-safe types (AlignedBuffer is Send+Sync)
unsafe impl Send for SGD {}
unsafe impl Sync for SGD {}

impl SGD {
    /// Creates a new SGD optimizer with config.
    pub fn new(network: &KanNetwork, config: SGDConfig) -> Self {
        let velocities = network
            .layers
            .iter()
            .map(|layer| {
                let mut vw = AlignedBuffer::with_capacity(layer.weights.len());
                vw.resize(layer.weights.len());
                let mut vb = AlignedBuffer::with_capacity(layer.bias.len());
                vb.resize(layer.bias.len());
                (vw, vb)
            })
            .collect();

        Self {
            config,
            velocities,
            state_version: 0,
        }
    }

    /// Creates a new SGD optimizer (legacy API).
    #[deprecated(since = "0.3.0", note = "use SGD::new() with SGDConfig instead")]
    pub fn new_legacy(network: &KanNetwork, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self::new(network, SGDConfig::full(lr, momentum, weight_decay))
    }

    /// Creates SGD without momentum.
    pub fn vanilla(network: &KanNetwork, lr: f32) -> Self {
        Self::new(network, SGDConfig::with_lr(lr))
    }

    /// Reinitializes velocities for a new network topology.
    pub fn reinitialize(&mut self, network: &KanNetwork) {
        self.velocities = network
            .layers
            .iter()
            .map(|layer| {
                let mut vw = AlignedBuffer::with_capacity(layer.weights.len());
                vw.resize(layer.weights.len());
                let mut vb = AlignedBuffer::with_capacity(layer.bias.len());
                vb.resize(layer.bias.len());
                (vw, vb)
            })
            .collect();
    }

    /// Legacy learning rate getter.
    #[inline]
    pub fn lr(&self) -> f32 {
        self.config.lr
    }

    /// Legacy momentum getter.
    #[inline]
    pub fn momentum(&self) -> f32 {
        self.config.momentum
    }

    /// Legacy weight decay getter.
    #[inline]
    pub fn weight_decay(&self) -> f32 {
        self.config.weight_decay
    }

    /// Performs one optimization step (legacy API).
    pub fn step_legacy(
        &mut self,
        network: &mut KanNetwork,
        all_weight_grads: &[Vec<f32>],
        all_bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) {
        for (i, layer) in network.layers.iter_mut().enumerate() {
            let (ref mut vw, ref mut vb) = self.velocities[i];

            let weights = layer.weights.as_mut_slice();
            let weight_grads = &all_weight_grads[i];
            let vw_slice = vw.as_mut_slice();
            let bias_grads = &all_bias_grads[i];

            let (wg_view, bg_view) = clip_gradients(weight_grads, bias_grads, max_grad_norm);

            for j in 0..weights.len() {
                vw_slice[j] = self.config.momentum * vw_slice[j] + wg_view[j];
                if self.config.weight_decay > 0.0 {
                    weights[j] *= 1.0 - self.config.lr * self.config.weight_decay;
                }
                weights[j] -= self.config.lr * vw_slice[j];
            }

            let bias = layer.bias.as_mut_slice();
            let vb_slice = vb.as_mut_slice();

            for j in 0..bias.len() {
                vb_slice[j] = self.config.momentum * vb_slice[j] + bg_view[j];
                bias[j] -= self.config.lr * vb_slice[j];
            }
        }
    }

    /// Backward-compatible step without gradient clipping.
    #[deprecated(since = "0.2.0", note = "use Optimizer::step() instead")]
    pub fn step_unclipped(
        &mut self,
        network: &mut KanNetwork,
        all_weight_grads: &[Vec<f32>],
        all_bias_grads: &[Vec<f32>],
    ) {
        self.step_legacy(network, all_weight_grads, all_bias_grads, None);
    }

    /// Resets all velocity buffers to zero.
    pub fn reset(&mut self) {
        for (vw, vb) in &mut self.velocities {
            vw.zero();
            vb.zero();
        }
    }
}

impl Clone for SGD {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            velocities: self.velocities.clone(),
            state_version: self.state_version,
        }
    }
}

// =============================================================================
// TRAIT IMPLEMENTATION FOR SGD
// =============================================================================

impl Optimizer for SGD {
    fn step(
        &mut self,
        network: &mut KanNetwork,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) -> ArkanResult<()> {
        // Validate layer count
        if weight_grads.len() != network.layers.len() || bias_grads.len() != network.layers.len() {
            return Err(ArkanError::tensor_shape_mismatch(
                &[network.layers.len()],
                &[weight_grads.len()],
            ));
        }

        // Check for NaN
        let safety = &self.config.safety;
        for (i, (wg, bg)) in weight_grads.iter().zip(bias_grads.iter()).enumerate() {
            if safety.fail_on_nan || safety.skip_step_on_nan {
                if let Some(idx) = find_nan_in_grads(wg) {
                    if safety.fail_on_nan && !safety.skip_step_on_nan {
                        return Err(ArkanError::nan_encountered(
                            idx,
                            format!("weight gradient at layer {}", i),
                        ));
                    }
                    return Ok(()); // Skip step
                }
                if let Some(idx) = find_nan_in_grads(bg) {
                    if safety.fail_on_nan && !safety.skip_step_on_nan {
                        return Err(ArkanError::nan_encountered(
                            idx,
                            format!("bias gradient at layer {}", i),
                        ));
                    }
                    return Ok(()); // Skip step
                }
            }
        }

        // Apply AMP scaling
        let scaling_factor = self.config.safety.grad_scaling_factor;
        let nesterov = self.config.nesterov;
        let momentum = self.config.momentum;
        let lr = self.config.lr;
        let decay = self.config.weight_decay;

        for (i, layer) in network.layers.iter_mut().enumerate() {
            let (ref mut vw, ref mut vb) = self.velocities[i];

            let wg_scaled = apply_grad_scaling(&weight_grads[i], scaling_factor);
            let bg_scaled = apply_grad_scaling(&bias_grads[i], scaling_factor);

            let (wg_clipped, bg_clipped) = clip_gradients(&wg_scaled, &bg_scaled, max_grad_norm);

            let weights = layer.weights.as_mut_slice();
            let vw_slice = vw.as_mut_slice();

            for j in 0..weights.len() {
                // Update velocity: v = μ*v + g
                vw_slice[j] = momentum * vw_slice[j] + wg_clipped[j];

                // Apply decoupled weight decay BEFORE gradient step
                if decay > 0.0 {
                    weights[j] *= 1.0 - lr * decay;
                }

                // Nesterov vs standard momentum
                // Nesterov: θ -= lr * (μ*v + g)
                // Standard: θ -= lr * v
                let update = if nesterov {
                    momentum * vw_slice[j] + wg_clipped[j]
                } else {
                    vw_slice[j]
                };
                weights[j] -= lr * update;
            }

            let bias = layer.bias.as_mut_slice();
            let vb_slice = vb.as_mut_slice();

            for j in 0..bias.len() {
                vb_slice[j] = momentum * vb_slice[j] + bg_clipped[j];
                let update = if nesterov {
                    momentum * vb_slice[j] + bg_clipped[j]
                } else {
                    vb_slice[j]
                };
                bias[j] -= lr * update;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self, network: &mut KanNetwork) -> ArkanResult<()> {
        let _ = network;
        Ok(())
    }

    fn get_state_version(&self) -> u64 {
        self.state_version
    }

    fn bump_version(&mut self) {
        self.state_version += 1;
        self.reset();
    }

    fn get_lr(&self, group_index: usize) -> ArkanResult<f64> {
        if group_index == 0 {
            Ok(self.config.lr as f64)
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }

    fn set_lr(&mut self, group_index: usize, new_lr: f64) -> ArkanResult<()> {
        if group_index == 0 {
            self.config.lr = new_lr as f32;
            Ok(())
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }
}

// =============================================================================
// L-BFGS OPTIMIZER (Second-Order)
// =============================================================================

/// L-BFGS optimizer configuration.
///
/// Limited-memory BFGS is a quasi-Newton method that approximates the
/// inverse Hessian matrix using a limited number of past gradients.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LBFGSConfig {
    /// Learning rate (step size).
    pub lr: f32,

    /// Maximum number of iterations per step.
    pub max_iter: usize,

    /// Maximum number of function evaluations per step.
    pub max_eval: Option<usize>,

    /// Termination tolerance on function value change.
    pub tolerance_grad: f64,

    /// Termination tolerance on parameter change.
    pub tolerance_change: f64,

    /// Number of corrections to approximate inverse Hessian.
    /// Higher = more memory, better approximation.
    pub history_size: usize,

    /// Line search method.
    pub line_search_fn: LineSearchMethod,

    /// Safety configuration.
    #[cfg_attr(feature = "serde", serde(default))]
    pub safety: SafetyConfig,
}

impl Default for LBFGSConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            max_iter: 20,
            max_eval: Some(25),
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
            line_search_fn: LineSearchMethod::StrongWolfe,
            safety: SafetyConfig::default(),
        }
    }
}

/// Line search methods for L-BFGS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LineSearchMethod {
    /// Strong Wolfe conditions (default, most robust).
    #[default]
    StrongWolfe,
    /// Backtracking with Armijo condition.
    Backtracking,
    /// No line search (fixed step size). Use with caution.
    NoLineSearch,
}

/// L-BFGS optimizer for KAN networks.
///
/// Implements the Limited-memory BFGS algorithm for second-order optimization.
/// Best suited for small to medium networks where function evaluations are cheap.
///
/// # Atomicity and Rollback
///
/// If line search fails, the optimizer:
/// 1. Restores parameters to their pre-step values
/// 2. Returns `Err(LineSearchFailed)`
///
/// This ensures no partial updates occur.
///
/// # Thread Safety
///
/// `LBFGS` implements `Send + Sync`.
///
/// # Example
///
/// ```rust,ignore
/// use arkan::optimizer::{LBFGS, LBFGSConfig, Optimizer};
///
/// let mut optimizer = LBFGS::new(&network, LBFGSConfig::default());
///
/// // L-BFGS uses closures for loss evaluation
/// let loss = optimizer.step_with_closure(|| {
///     // Compute forward pass and loss
///     Ok(compute_loss(&network, &inputs, &targets))
/// })?;
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LBFGS {
    /// Configuration.
    pub config: LBFGSConfig,

    /// History of s vectors (parameter differences).
    s_history: Vec<Vec<f32>>,

    /// History of y vectors (gradient differences).
    y_history: Vec<Vec<f32>>,

    /// History of rho values (1 / (y^T s)).
    rho_history: Vec<f64>,

    /// Previous parameters (for computing s).
    prev_params: Option<Vec<f32>>,

    /// Previous gradients (for computing y).
    prev_grads: Option<Vec<f32>>,

    /// State version for topology tracking.
    state_version: u64,

    /// Number of function evaluations.
    n_eval: usize,
}

// SAFETY: LBFGS uses only thread-safe types
unsafe impl Send for LBFGS {}
unsafe impl Sync for LBFGS {}

impl LBFGS {
    /// Creates a new L-BFGS optimizer.
    pub fn new(_network: &KanNetwork, config: LBFGSConfig) -> Self {
        Self {
            config,
            s_history: Vec::with_capacity(config.history_size),
            y_history: Vec::with_capacity(config.history_size),
            rho_history: Vec::with_capacity(config.history_size),
            prev_params: None,
            prev_grads: None,
            state_version: 0,
            n_eval: 0,
        }
    }

    /// Resets all history buffers.
    pub fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
        self.prev_params = None;
        self.prev_grads = None;
        self.n_eval = 0;
    }

    /// Returns the number of function evaluations.
    pub fn num_evals(&self) -> usize {
        self.n_eval
    }

    /// Flattens network parameters into a single vector.
    pub fn flatten_params(network: &KanNetwork) -> Vec<f32> {
        let total: usize = network
            .layers
            .iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum();
        let mut params = Vec::with_capacity(total);
        for layer in &network.layers {
            params.extend_from_slice(&layer.weights);
            params.extend_from_slice(&layer.bias);
        }
        params
    }

    /// Restores parameters from a flat vector.
    pub fn restore_params(network: &mut KanNetwork, params: &[f32]) {
        let mut offset = 0;
        for layer in &mut network.layers {
            let w_len = layer.weights.len();
            layer
                .weights
                .as_mut_slice()
                .copy_from_slice(&params[offset..offset + w_len]);
            offset += w_len;

            let b_len = layer.bias.len();
            layer
                .bias
                .as_mut_slice()
                .copy_from_slice(&params[offset..offset + b_len]);
            offset += b_len;
        }
    }

    /// Flattens gradients into a single vector.
    pub fn flatten_grads(weight_grads: &[Vec<f32>], bias_grads: &[Vec<f32>]) -> Vec<f32> {
        let total: usize = weight_grads.iter().map(|v| v.len()).sum::<usize>()
            + bias_grads.iter().map(|v| v.len()).sum::<usize>();
        let mut grads = Vec::with_capacity(total);
        for (wg, bg) in weight_grads.iter().zip(bias_grads.iter()) {
            grads.extend_from_slice(wg);
            grads.extend_from_slice(bg);
        }
        grads
    }

    /// Updates history with new s and y vectors.
    fn update_history(&mut self, s: Vec<f32>, y: Vec<f32>) {
        let sy: f64 = s
            .iter()
            .zip(y.iter())
            .map(|(&si, &yi)| si as f64 * yi as f64)
            .sum();

        // Skip if curvature condition not satisfied
        if sy <= 1e-10 {
            return;
        }

        let rho = 1.0 / sy;

        // Maintain history_size limit
        if self.s_history.len() >= self.config.history_size {
            self.s_history.remove(0);
            self.y_history.remove(0);
            self.rho_history.remove(0);
        }

        self.s_history.push(s);
        self.y_history.push(y);
        self.rho_history.push(rho);
    }

    /// Two-loop recursion for computing search direction.
    ///
    /// Returns -H⁻¹g where H⁻¹ is approximated using L-BFGS history.
    pub fn two_loop_recursion(&self, grad: &[f32]) -> Vec<f32> {
        let m = self.s_history.len();

        if m == 0 {
            // No history: steepest descent
            return grad.iter().map(|&g| -g).collect();
        }

        // q = grad
        let mut q: Vec<f64> = grad.iter().map(|&g| g as f64).collect();
        let mut alpha = vec![0.0f64; m];

        // First loop (backward)
        // Note: We need to access alpha[i] and history[i] in tandem, clippy allow is correct here
        #[allow(clippy::needless_range_loop)]
        for i in (0..m).rev() {
            alpha[i] = self.rho_history[i]
                * self.s_history[i]
                    .iter()
                    .zip(q.iter())
                    .map(|(&s, &q)| s as f64 * q)
                    .sum::<f64>();
            for (q_j, &y_ij) in q.iter_mut().zip(self.y_history[i].iter()) {
                *q_j -= alpha[i] * y_ij as f64;
            }
        }

        // Scale initial Hessian approximation: H0 = γI where γ = sᵀy / yᵀy
        let s_last = &self.s_history[m - 1];
        let y_last = &self.y_history[m - 1];
        let yy: f64 = y_last.iter().map(|&y| (y as f64).powi(2)).sum();
        let sy: f64 = s_last
            .iter()
            .zip(y_last.iter())
            .map(|(&s, &y)| s as f64 * y as f64)
            .sum();
        let gamma = if yy > 1e-10 { sy / yy } else { 1.0 };

        // r = γ * q
        let mut r: Vec<f64> = q.iter().map(|&q| gamma * q).collect();

        // Second loop (forward)
        // Note: We need to access alpha[i] and history[i] in tandem, clippy allow is correct here
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            let beta = self.rho_history[i]
                * self.y_history[i]
                    .iter()
                    .zip(r.iter())
                    .map(|(&y, &r)| y as f64 * r)
                    .sum::<f64>();
            for (r_j, &s_ij) in r.iter_mut().zip(self.s_history[i].iter()) {
                *r_j += s_ij as f64 * (alpha[i] - beta);
            }
        }

        // Return -r (descent direction)
        r.iter().map(|&r| -r as f32).collect()
    }

    /// Computes directional derivative: g ⋅ d
    fn directional_derivative(grad: &[f32], direction: &[f32]) -> f64 {
        grad.iter()
            .zip(direction.iter())
            .map(|(&g, &d)| g as f64 * d as f64)
            .sum()
    }

    /// Computes gradient norm: ||g||
    fn grad_norm(grad: &[f32]) -> f64 {
        grad.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt()
    }

    /// Strong Wolfe line search.
    ///
    /// Finds step size α satisfying Strong Wolfe conditions:
    /// 1. Sufficient decrease: f(x + αd) ≤ f(x) + c₁·α·(∇f·d)
    /// 2. Curvature: |∇f(x + αd)·d| ≤ c₂·|∇f(x)·d|
    ///
    /// Parameters from PyTorch LBFGS defaults:
    /// - c1 = 1e-4 (Armijo constant)
    /// - c2 = 0.9 (curvature constant for L-BFGS, 0.1 for CG)
    /// - max_iter = 25
    fn strong_wolfe_line_search<F>(
        &mut self,
        network: &mut KanNetwork,
        closure: &mut F,
        x0: &[f32],
        f0: f64,
        g0: &[f32],
        direction: &[f32],
    ) -> ArkanResult<(f64, Vec<f32>, f64)>
    where
        F: FnMut(&mut KanNetwork) -> ArkanResult<(f64, Vec<f32>)>,
    {
        const C1: f64 = 1e-4;
        const C2: f64 = 0.9;
        const MAX_LS_ITER: usize = 25;
        const ALPHA_MAX: f64 = 50.0;

        let dg0 = Self::directional_derivative(g0, direction);

        // dg0 should be negative (descent direction)
        if dg0 >= 0.0 {
            return Err(ArkanError::line_search_failed(
                "Search direction is not a descent direction",
            ));
        }

        let mut alpha = self.config.lr as f64;
        let mut alpha_lo: f64 = 0.0;
        let mut alpha_hi: f64 = ALPHA_MAX;
        let mut f_lo = f0;
        let mut g_lo = g0.to_vec();

        for iter in 0..MAX_LS_ITER {
            // x = x0 + alpha * d
            let x_new: Vec<f32> = x0
                .iter()
                .zip(direction.iter())
                .map(|(&x, &d)| x + alpha as f32 * d)
                .collect();

            Self::restore_params(network, &x_new);
            self.n_eval += 1;
            let (f_new, g_new) = closure(network)?;

            // Check for NaN
            if f_new.is_nan() || !f_new.is_finite() {
                // Reduce step size
                alpha_hi = alpha;
                alpha = (alpha_lo + alpha_hi) / 2.0;
                continue;
            }

            let dg_new = Self::directional_derivative(&g_new, direction);

            // Check Armijo condition (sufficient decrease)
            if f_new > f0 + C1 * alpha * dg0 || (iter > 0 && f_new >= f_lo) {
                // Zoom into [alpha_lo, alpha]
                alpha_hi = alpha;
            } else {
                // Check Strong Wolfe curvature condition
                if dg_new.abs() <= -C2 * dg0 {
                    // Both conditions satisfied
                    return Ok((alpha, g_new, f_new));
                }

                if dg_new >= 0.0 {
                    // Zoom into [alpha, alpha_lo]
                    alpha_hi = alpha_lo;
                    alpha_lo = alpha;
                    f_lo = f_new;
                    g_lo = g_new;
                } else {
                    // Move to higher alpha
                    alpha_lo = alpha;
                    f_lo = f_new;
                    g_lo = g_new.clone();
                    alpha = (alpha + alpha_hi) / 2.0;
                    continue;
                }
            }

            // Zoom phase: binary search in [alpha_lo, alpha_hi]
            if (alpha_hi - alpha_lo).abs() < 1e-10 {
                // Interval too small
                return Ok((alpha_lo, g_lo, f_lo));
            }

            alpha = (alpha_lo + alpha_hi) / 2.0;
        }

        // Max iterations reached - return best found
        Ok((alpha_lo, g_lo, f_lo))
    }

    /// Backtracking line search with Armijo condition.
    fn backtracking_line_search<F>(
        &mut self,
        network: &mut KanNetwork,
        closure: &mut F,
        x0: &[f32],
        f0: f64,
        g0: &[f32],
        direction: &[f32],
    ) -> ArkanResult<(f64, Vec<f32>, f64)>
    where
        F: FnMut(&mut KanNetwork) -> ArkanResult<(f64, Vec<f32>)>,
    {
        const C1: f64 = 1e-4;
        const RHO: f64 = 0.5; // Backtrack factor
        const MAX_LS_ITER: usize = 20;

        let dg0 = Self::directional_derivative(g0, direction);
        let mut alpha = self.config.lr as f64;

        for _ in 0..MAX_LS_ITER {
            let x_new: Vec<f32> = x0
                .iter()
                .zip(direction.iter())
                .map(|(&x, &d)| x + alpha as f32 * d)
                .collect();

            Self::restore_params(network, &x_new);
            self.n_eval += 1;
            let (f_new, g_new) = closure(network)?;

            // Check Armijo condition
            if f_new <= f0 + C1 * alpha * dg0 {
                return Ok((alpha, g_new, f_new));
            }

            alpha *= RHO;
        }

        Err(ArkanError::line_search_failed(
            "Backtracking line search did not converge",
        ))
    }
}

impl Clone for LBFGS {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            s_history: self.s_history.clone(),
            y_history: self.y_history.clone(),
            rho_history: self.rho_history.clone(),
            prev_params: self.prev_params.clone(),
            prev_grads: self.prev_grads.clone(),
            state_version: self.state_version,
            n_eval: self.n_eval,
        }
    }
}

impl Optimizer for LBFGS {
    fn step(
        &mut self,
        _network: &mut KanNetwork,
        _weight_grads: &[Vec<f32>],
        _bias_grads: &[Vec<f32>],
        _max_grad_norm: Option<f32>,
    ) -> ArkanResult<()> {
        // L-BFGS requires closure-based API
        Err(ArkanError::optimizer(
            "LBFGS requires step_lbfgs() with closure. Use step() only for first-order optimizers.",
        ))
    }

    fn step_with_closure<F>(&mut self, _closure: F) -> ArkanResult<f64>
    where
        F: FnMut() -> ArkanResult<f64>,
    {
        // This default impl doesn't work for LBFGS because we need mutable network access
        // Use step_lbfgs() instead
        Err(ArkanError::optimizer(
            "Use step_lbfgs() for L-BFGS optimization with network access.",
        ))
    }

    fn zero_grad(&mut self, network: &mut KanNetwork) -> ArkanResult<()> {
        let _ = network;
        Ok(())
    }

    fn get_state_version(&self) -> u64 {
        self.state_version
    }

    fn bump_version(&mut self) {
        self.state_version += 1;
        self.reset();
    }

    fn get_lr(&self, group_index: usize) -> ArkanResult<f64> {
        if group_index == 0 {
            Ok(self.config.lr as f64)
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }

    fn set_lr(&mut self, group_index: usize, new_lr: f64) -> ArkanResult<()> {
        if group_index == 0 {
            self.config.lr = new_lr as f32;
            Ok(())
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }
}

impl LBFGS {
    /// Performs L-BFGS optimization step.
    ///
    /// The closure should compute loss and gradients given the current network state.
    /// It will be called multiple times during line search.
    ///
    /// # Arguments
    ///
    /// * `network` - Network to optimize
    /// * `closure` - Closure that computes `(loss, gradient_vector)` for current params
    ///
    /// # Returns
    ///
    /// Final loss value after the optimization step.
    ///
    /// # Rollback Guarantee
    ///
    /// If line search fails, parameters are restored to their initial values.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loss = optimizer.step_lbfgs(&mut network, |net| {
    ///     let mut ws = net.create_workspace(batch_size);
    ///     let output = net.forward_batch(&input, &mut output_buf, &mut ws);
    ///     let loss = compute_mse(&output_buf, &target);
    ///     
    ///     // Compute gradients
    ///     net.train_step_with_options(&input, &target, None, 0.0, &mut ws, &opts);
    ///     let grads = LBFGS::flatten_grads(&ws.weight_grads, &ws.bias_grads);
    ///     
    ///     Ok((loss as f64, grads))
    /// })?;
    /// ```
    pub fn step_lbfgs<F>(&mut self, network: &mut KanNetwork, mut closure: F) -> ArkanResult<f64>
    where
        F: FnMut(&mut KanNetwork) -> ArkanResult<(f64, Vec<f32>)>,
    {
        // Save initial parameters for potential rollback
        let x0 = Self::flatten_params(network);

        // Evaluate closure to get initial loss and gradient
        self.n_eval += 1;
        let (f0, g0) = closure(network)?;

        // Check for NaN in loss
        if f0.is_nan() || !f0.is_finite() {
            if self.config.safety.fail_on_nan && !self.config.safety.skip_step_on_nan {
                return Err(ArkanError::nan_encountered(0, "loss"));
            }
            if self.config.safety.skip_step_on_nan {
                return Ok(f0);
            }
        }

        // Check convergence on gradient norm
        let grad_norm = Self::grad_norm(&g0);
        if grad_norm < self.config.tolerance_grad {
            return Ok(f0);
        }

        // Compute search direction using two-loop recursion
        let direction = self.two_loop_recursion(&g0);

        // Perform line search based on configured method
        let (alpha, g_new, f_new) = match self.config.line_search_fn {
            LineSearchMethod::StrongWolfe => {
                match self.strong_wolfe_line_search(network, &mut closure, &x0, f0, &g0, &direction)
                {
                    Ok(result) => result,
                    Err(_) => {
                        // Rollback on failure
                        Self::restore_params(network, &x0);
                        return Err(ArkanError::line_search_failed(
                            "Strong Wolfe line search failed, parameters restored",
                        ));
                    }
                }
            }
            LineSearchMethod::Backtracking => {
                match self.backtracking_line_search(network, &mut closure, &x0, f0, &g0, &direction)
                {
                    Ok(result) => result,
                    Err(_) => {
                        // Rollback on failure
                        Self::restore_params(network, &x0);
                        return Err(ArkanError::line_search_failed(
                            "Backtracking line search failed, parameters restored",
                        ));
                    }
                }
            }
            LineSearchMethod::NoLineSearch => {
                // Fixed step size (use lr directly)
                let alpha = self.config.lr as f64;
                let x_new: Vec<f32> = x0
                    .iter()
                    .zip(direction.iter())
                    .map(|(&x, &d)| x + alpha as f32 * d)
                    .collect();

                Self::restore_params(network, &x_new);
                self.n_eval += 1;
                let (f_new, g_new) = closure(network)?;

                (alpha, g_new, f_new)
            }
        };

        // Compute s and y for history update
        let x_new = Self::flatten_params(network);
        let s: Vec<f32> = x_new
            .iter()
            .zip(x0.iter())
            .map(|(&xn, &x0)| xn - x0)
            .collect();
        let y: Vec<f32> = g_new
            .iter()
            .zip(g0.iter())
            .map(|(&gn, &g0)| gn - g0)
            .collect();

        // Update history
        self.update_history(s, y);

        // Store for next iteration
        self.prev_params = Some(x_new);
        self.prev_grads = Some(g_new);

        // Check for step size convergence
        if alpha.abs() < self.config.tolerance_change {
            // Step too small, may have converged
        }

        Ok(f_new)
    }
}

/// Learning rate scheduler trait.
///
/// Implement this trait to create custom learning rate schedules.
pub trait LrScheduler {
    /// Returns the learning rate for the given epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Current epoch number (0-indexed)
    /// * `current_lr` - Current learning rate (may be ignored)
    fn get_lr(&self, epoch: usize, current_lr: f32) -> f32;
}

/// Step decay learning rate scheduler.
///
/// Multiplies the learning rate by `gamma` every `step_size` epochs.
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::{StepLR, LrScheduler};
///
/// let scheduler = StepLR::new(0.1, 10, 0.5);
///
/// assert!((scheduler.get_lr(0, 0.1) - 0.1).abs() < 1e-6);
/// assert!((scheduler.get_lr(10, 0.1) - 0.05).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Decay factor (multiplied each step).
    pub gamma: f32,
    /// Number of epochs between decays.
    pub step_size: usize,
}

impl StepLR {
    /// Creates a new step decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate
    /// * `step_size` - Epochs between decays
    /// * `gamma` - Decay factor
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size,
        }
    }
}

impl LrScheduler for StepLR {
    fn get_lr(&self, epoch: usize, _: f32) -> f32 {
        let n_steps = epoch / self.step_size;
        self.initial_lr * self.gamma.powi(n_steps as i32)
    }
}

/// Cosine annealing learning rate scheduler.
///
/// Smoothly decreases the learning rate from `initial_lr` to `min_lr`
/// following a cosine curve over `t_max` epochs.
///
/// # Formula
///
/// $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t \pi}{T_{max}}))$$
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::{CosineAnnealingLR, LrScheduler};
///
/// let scheduler = CosineAnnealingLR::new(0.1, 100, 0.001);
///
/// // At start: ~0.1
/// let lr_start = scheduler.get_lr(0, 0.1);
///
/// // At end: ~0.001
/// let lr_end = scheduler.get_lr(100, 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    /// Initial (maximum) learning rate.
    pub initial_lr: f32,
    /// Minimum learning rate at the end of annealing.
    pub min_lr: f32,
    /// Total number of epochs for one cycle.
    pub t_max: usize,
}

impl CosineAnnealingLR {
    /// Creates a new cosine annealing scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate
    /// * `t_max` - Total epochs for decay
    /// * `min_lr` - Minimum learning rate
    pub fn new(initial_lr: f32, t_max: usize, min_lr: f32) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn get_lr(&self, epoch: usize, _: f32) -> f32 {
        let epoch = epoch.min(self.t_max);
        let cos_inner = std::f32::consts::PI * epoch as f32 / self.t_max as f32;
        self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + cos_inner.cos())
    }
}

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
