//! Adam optimizer for KAN training.

use crate::buffer::AlignedBuffer;
use crate::layer::KanLayer;
use crate::network::KanNetwork;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Adam optimizer state for a single parameter tensor.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdamState {
    /// First moment (mean of gradients).
    pub m: AlignedBuffer,

    /// Second moment (variance of gradients).
    pub v: AlignedBuffer,

    /// Timestep for bias correction.
    pub t: usize,
}

impl AdamState {
    /// Creates a new Adam state for a parameter tensor of given size.
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
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
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
}

/// Per-layer optimizer state.
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

/// Adam optimizer for a KAN network.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Adam {
    /// Configuration.
    pub config: AdamConfig,

    /// Per-layer states.
    pub layer_states: Vec<LayerAdamState>,
}

impl Adam {
    /// Creates a new Adam optimizer for the given network.
    pub fn new(network: &KanNetwork, config: AdamConfig) -> Self {
        let layer_states = network.layers.iter().map(LayerAdamState::new).collect();

        Self {
            config,
            layer_states,
        }
    }

    /// Creates optimizer with default config.
    pub fn default_for(network: &KanNetwork) -> Self {
        Self::new(network, AdamConfig::default())
    }

    /// Resets all optimizer state.
    pub fn reset(&mut self) {
        for state in &mut self.layer_states {
            state.reset();
        }
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

    /// Updates a single parameter tensor using Adam.
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

            // Apply weight decay (decoupled)
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

    /// Performs one optimization step on the entire network.
    ///
    /// `all_weight_grads` and `all_bias_grads` should contain gradients
    /// for all layers concatenated.
    pub fn step(
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

    /// Backward-compatible вызов без клиппинга.
    pub fn step_unclipped(
        &mut self,
        network: &mut KanNetwork,
        all_weight_grads: &[Vec<f32>],
        all_bias_grads: &[Vec<f32>],
    ) {
        self.step(network, all_weight_grads, all_bias_grads, None);
    }
}

impl Clone for Adam {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            layer_states: self.layer_states.clone(),
        }
    }
}

/// SGD optimizer with momentum.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SGD {
    /// Learning rate.
    pub lr: f32,

    /// Momentum coefficient.
    pub momentum: f32,

    /// Weight decay.
    pub weight_decay: f32,

    /// Velocity for each parameter.
    pub velocities: Vec<(AlignedBuffer, AlignedBuffer)>,
}

impl SGD {
    /// Creates a new SGD optimizer.
    pub fn new(network: &KanNetwork, lr: f32, momentum: f32, weight_decay: f32) -> Self {
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
            lr,
            momentum,
            weight_decay,
            velocities,
        }
    }

    /// Creates SGD without momentum.
    pub fn vanilla(network: &KanNetwork, lr: f32) -> Self {
        Self::new(network, lr, 0.0, 0.0)
    }

    /// Performs one optimization step.
    pub fn step(
        &mut self,
        network: &mut KanNetwork,
        all_weight_grads: &[Vec<f32>],
        all_bias_grads: &[Vec<f32>],
        max_grad_norm: Option<f32>,
    ) {
        for (i, layer) in network.layers.iter_mut().enumerate() {
            let (ref mut vw, ref mut vb) = self.velocities[i];

            // Update weights
            let weights = layer.weights.as_mut_slice();
            let weight_grads = &all_weight_grads[i];
            let vw_slice = vw.as_mut_slice();
            let bias_grads = &all_bias_grads[i];

            // Клиппинг по слою, если задан
            let (wg_view, bg_view) = if let Some(max_norm) = max_grad_norm {
                let mut sq: f32 = weight_grads.iter().map(|g| g * g).sum();
                sq += bias_grads.iter().map(|g| g * g).sum::<f32>();
                let norm = sq.sqrt();
                if norm > max_norm && norm > 0.0 {
                    let scale = max_norm / norm;
                    let mut wg = weight_grads.clone();
                    let mut bg = bias_grads.clone();
                    for g in &mut wg {
                        *g *= scale;
                    }
                    for g in &mut bg {
                        *g *= scale;
                    }
                    (wg, bg)
                } else {
                    (weight_grads.clone(), bias_grads.clone())
                }
            } else {
                (weight_grads.clone(), bias_grads.clone())
            };

            for j in 0..weights.len() {
                vw_slice[j] = self.momentum * vw_slice[j] + wg_view[j];
                if self.weight_decay > 0.0 {
                    // decoupled weight decay
                    weights[j] *= 1.0 - self.lr * self.weight_decay;
                }
                weights[j] -= self.lr * vw_slice[j];
            }

            // Update bias
            let bias = layer.bias.as_mut_slice();
            let vb_slice = vb.as_mut_slice();

            for j in 0..bias.len() {
                vb_slice[j] = self.momentum * vb_slice[j] + bg_view[j];
                bias[j] -= self.lr * vb_slice[j];
            }
        }
    }

    /// Backward-compatible вызов без клиппинга.
    pub fn step_unclipped(
        &mut self,
        network: &mut KanNetwork,
        all_weight_grads: &[Vec<f32>],
        all_bias_grads: &[Vec<f32>],
    ) {
        self.step(network, all_weight_grads, all_bias_grads, None);
    }

    /// Resets velocity.
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
            lr: self.lr,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
            velocities: self.velocities.clone(),
        }
    }
}

/// Learning rate scheduler trait.
pub trait LrScheduler {
    /// Gets the learning rate for the given epoch.
    fn get_lr(&self, epoch: usize, current_lr: f32) -> f32;
}

/// Step decay scheduler.
#[derive(Debug, Clone)]
pub struct StepLR {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Decay factor.
    pub gamma: f32,
    /// Step size in epochs.
    pub step_size: usize,
}

impl StepLR {
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

/// Cosine annealing scheduler.
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Minimum learning rate.
    pub min_lr: f32,
    /// Total epochs.
    pub t_max: usize,
}

impl CosineAnnealingLR {
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
        let config = KanConfig::default_poker();
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
        };

        let mut network = KanNetwork::new(config);
        let mut optimizer = Adam::new(&network, AdamConfig::with_lr(0.1));

        // Get initial weight
        let initial_weight = network.layers[0].weights[0];

        // Create gradients (all positive)
        let weight_grads = vec![vec![1.0f32; network.layers[0].weights.len()]];
        let bias_grads = vec![vec![0.5f32; network.layers[0].bias.len()]];

        // Step
        optimizer.step_unclipped(&mut network, &weight_grads, &bias_grads);

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
}
