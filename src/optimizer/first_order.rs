//! Adam and SGD parameter updates.

use super::*;

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
/// After Grid Extension changes tensor shapes, call `reinitialize(network)` and
/// `bump_version()` to resize state and update the tracked version.
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

    fn updated_values(
        param: f32,
        grad: f32,
        m: f32,
        v: f32,
        config: &AdamConfig,
        correction: (f32, f32),
    ) -> [f32; 3] {
        let m = config.beta1 * m + (1.0 - config.beta1) * grad;
        let v = config.beta2 * v + (1.0 - config.beta2) * grad * grad;
        let update = config.lr * (m / correction.0) / ((v / correction.1).sqrt() + config.epsilon);
        let decayed = if config.weight_decay > 0.0 {
            param * (1.0 - config.lr * config.weight_decay)
        } else {
            param
        };
        [m, v, decayed - update]
    }

    /// Updates a single parameter tensor using Adam.
    ///
    /// # Update Order (AdamW-style)
    ///
    /// 1. **Moment update**: $m_t$, $v_t$ computed from gradients
    /// 2. **Weight decay**: `param *= 1 - lr * decay` (applied BEFORE gradient step)
    /// 3. **Gradient step**: `param -= lr * m_hat / (sqrt(v_hat) + eps)`
    ///
    /// where `m_hat = m / (1 - beta1^t)` and `v_hat = v / (1 - beta2^t)`.
    ///
    /// Epsilon is added to `sqrt(v_hat)` (the bias-corrected second moment),
    /// matching the PyTorch / original Adam paper convention:
    ///   `theta -= lr * m_hat / (sqrt(v_hat) + eps)`
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

        let correction = (
            1.0 - config.beta1.powi(state.t as i32),
            1.0 - config.beta2.powi(state.t as i32),
        );
        let m = state.m.as_mut_slice();
        let v = state.v.as_mut_slice();
        for i in 0..params.len() {
            let values = Self::updated_values(params[i], grads[i], m[i], v[i], config, correction);
            m[i] = values[0];
            v[i] = values[1];
            params[i] = values[2];
        }
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

impl Adam {
    pub(crate) fn step_with_outcome(
        &mut self,
        network: &mut KanNetwork,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) -> ArkanResult<StepOutcome> {
        let mut parameters = network.try_parameters_mut()?;
        validate_grad_shapes(&parameters, weight_grads, bias_grads)?;
        validate_safety(&self.config.safety, max_grad_norm)?;
        validate_nonnegative(self.config.lr as f64, "learning rate")?;
        validate_nonnegative(self.config.weight_decay as f64, "weight decay")?;
        if !(0.0..1.0).contains(&self.config.beta1)
            || !(0.0..1.0).contains(&self.config.beta2)
            || !self.config.epsilon.is_finite()
            || self.config.epsilon <= 0.0
        {
            return Err(ArkanError::optimizer(
                "Adam requires betas in [0, 1) and finite positive epsilon",
            ));
        }
        validate_shape(parameters.len(), self.layer_states.len())?;
        for ((parameter_weights, parameter_bias), state) in
            parameters.iter().zip(&self.layer_states)
        {
            for (params, tensor) in [
                (parameter_weights, &state.weights),
                (parameter_bias, &state.bias),
            ] {
                validate_shape(params.len(), tensor.m.len())?;
                validate_shape(params.len(), tensor.v.len())?;
                if tensor.t >= i32::MAX as usize {
                    return Err(ArkanError::optimizer("Adam timestep exhausted"));
                }
            }
        }
        let Some((weights, biases)) =
            prepare_gradients(weight_grads, bias_grads, &self.config.safety, max_grad_norm)?
        else {
            return Ok(StepOutcome::Skipped);
        };

        if self.config.safety.fail_on_nan || self.config.safety.skip_step_on_nan {
            for (i, ((parameter_weights, parameter_bias), state)) in
                parameters.iter().zip(&self.layer_states).enumerate()
            {
                for (params, grads, tensor) in [
                    (parameter_weights, &weights[i], &state.weights),
                    (parameter_bias, &biases[i], &state.bias),
                ] {
                    let t = (tensor.t + 1) as i32;
                    let correction = (
                        1.0 - self.config.beta1.powi(t),
                        1.0 - self.config.beta2.powi(t),
                    );
                    for j in 0..params.len() {
                        let values = Self::updated_values(
                            params[j],
                            grads[j],
                            tensor.m.as_slice()[j],
                            tensor.v.as_slice()[j],
                            &self.config,
                            correction,
                        );
                        if check_finite(&values, &self.config.safety, "Adam update or state")? {
                            return Ok(StepOutcome::Skipped);
                        }
                    }
                }
            }
        }

        for (i, layer) in parameters.iter_mut().enumerate() {
            let state = &mut self.layer_states[i];

            // Update parameters
            Self::update_params(layer.weights, &weights[i], &mut state.weights, &self.config);

            Self::update_params(layer.bias, &biases[i], &mut state.bias, &self.config);
        }

        Ok(StepOutcome::Applied)
    }
}

impl Optimizer for Adam {
    fn step(
        &mut self,
        network: &mut KanNetwork,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) -> ArkanResult<()> {
        self.step_with_outcome(network, weight_grads, bias_grads, max_grad_norm)
            .map(|_| ())
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

    fn updated_values(
        param: f32,
        grad: f32,
        velocity: f32,
        config: &SGDConfig,
        decay: f32,
    ) -> [f32; 2] {
        let velocity = config.momentum * velocity + grad;
        let update = if config.nesterov {
            config.momentum * velocity + grad
        } else {
            velocity
        };
        let decayed = if decay > 0.0 {
            param * (1.0 - config.lr * decay)
        } else {
            param
        };
        [velocity, decayed - config.lr * update]
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

impl SGD {
    pub(crate) fn step_with_outcome(
        &mut self,
        network: &mut KanNetwork,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) -> ArkanResult<StepOutcome> {
        let mut parameters = network.try_parameters_mut()?;
        validate_grad_shapes(&parameters, weight_grads, bias_grads)?;
        validate_safety(&self.config.safety, max_grad_norm)?;
        validate_nonnegative(self.config.lr as f64, "learning rate")?;
        validate_nonnegative(self.config.weight_decay as f64, "weight decay")?;
        if !(0.0..1.0).contains(&self.config.momentum) {
            return Err(ArkanError::optimizer("SGD momentum must be in [0, 1)"));
        }
        validate_shape(parameters.len(), self.velocities.len())?;
        for ((parameter_weights, parameter_bias), (vw, vb)) in
            parameters.iter().zip(&self.velocities)
        {
            validate_shape(parameter_weights.len(), vw.len())?;
            validate_shape(parameter_bias.len(), vb.len())?;
        }
        let Some((weights_grads, biases_grads)) =
            prepare_gradients(weight_grads, bias_grads, &self.config.safety, max_grad_norm)?
        else {
            return Ok(StepOutcome::Skipped);
        };
        if self.config.safety.fail_on_nan || self.config.safety.skip_step_on_nan {
            for (i, ((parameter_weights, parameter_bias), (vw, vb))) in
                parameters.iter().zip(&self.velocities).enumerate()
            {
                for (params, grads, velocity, decay) in [
                    (
                        parameter_weights,
                        &weights_grads[i],
                        vw.as_slice(),
                        self.config.weight_decay,
                    ),
                    (parameter_bias, &biases_grads[i], vb.as_slice(), 0.0),
                ] {
                    for j in 0..params.len() {
                        let values = Self::updated_values(
                            params[j],
                            grads[j],
                            velocity[j],
                            &self.config,
                            decay,
                        );
                        if check_finite(&values, &self.config.safety, "SGD update or state")? {
                            return Ok(StepOutcome::Skipped);
                        }
                    }
                }
            }
        }
        for (i, layer) in parameters.iter_mut().enumerate() {
            let (vw, vb) = &mut self.velocities[i];
            for (params, grads, velocity, decay) in [
                (
                    layer.weights,
                    &weights_grads[i],
                    vw.as_mut_slice(),
                    self.config.weight_decay,
                ),
                (layer.bias, &biases_grads[i], vb.as_mut_slice(), 0.0),
            ] {
                for j in 0..params.len() {
                    let values =
                        Self::updated_values(params[j], grads[j], velocity[j], &self.config, decay);
                    velocity[j] = values[0];
                    params[j] = values[1];
                }
            }
        }

        Ok(StepOutcome::Applied)
    }
}

impl Optimizer for SGD {
    fn step(
        &mut self,
        network: &mut KanNetwork,
        weight_grads: &[Vec<f32>],
        bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) -> ArkanResult<()> {
        self.step_with_outcome(network, weight_grads, bias_grads, max_grad_norm)
            .map(|_| ())
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
