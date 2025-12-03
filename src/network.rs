//! Full KAN network with multi-layer support.

use crate::buffer::Workspace;
use crate::config::KanConfig;
use crate::layer::KanLayer;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Complete KAN network with multiple layers.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KanNetwork {
    /// Configuration.
    pub config: KanConfig,

    /// Layers: input→hidden[0]→...→hidden[n]→output.
    pub layers: Vec<KanLayer>,

    /// Cached layer dimensions for quick access.
    layer_dims: Vec<usize>,

    /// Дефолтные опции обучения (клиппинг/decay), чтобы не передавать каждый раз.
    pub default_train_options: TrainOptions,
}

/// Настройки обучения для одного шага.
#[derive(Debug, Clone, Copy)]
pub struct TrainOptions {
    /// Максимальная норма градиента (L2) для клиппинга. None — без клиппинга.
    pub max_grad_norm: Option<f32>,
    /// Decoupled weight decay коэффициент. 0 — без декея.
    pub weight_decay: f32,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            max_grad_norm: None,
            weight_decay: 0.0,
        }
    }
}

impl KanNetwork {
    /// Creates a new KAN network from configuration.
    pub fn new(config: KanConfig) -> Self {
        let layer_dims = config.layer_dims();
        let mut layers = Vec::with_capacity(layer_dims.len() - 1);

        for i in 0..layer_dims.len() - 1 {
            let in_dim = layer_dims[i];
            let out_dim = layer_dims[i + 1];
            layers.push(KanLayer::from_config(in_dim, out_dim, &config));
        }

        Self {
            config,
            layers,
            layer_dims,
            default_train_options: TrainOptions::default(),
        }
    }

    /// Creates network from configuration (alias).
    pub fn from_config(config: KanConfig) -> Self {
        Self::new(config)
    }

    /// Установить дефолтные опции обучения.
    pub fn set_default_train_options(&mut self, opts: TrainOptions) {
        self.default_train_options = opts;
    }

    /// Returns the number of layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns total number of trainable parameters.
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    /// Sets normalization statistics for the input layer.
    pub fn set_input_normalization(&mut self, mean: &[f32], std: &[f32]) {
        if !self.layers.is_empty() {
            self.layers[0].set_normalization(mean, std);
        }
    }

    /// Forward pass for a single sample.
    ///
    /// # Arguments
    /// * `input` - Input features [input_dim]
    /// * `output` - Output buffer [output_dim]
    /// * `workspace` - Pre-allocated workspace
    pub fn forward_single(&self, input: &[f32], output: &mut [f32], workspace: &mut Workspace) {
        debug_assert_eq!(input.len(), self.config.input_dim);
        debug_assert_eq!(output.len(), self.config.output_dim);

        if self.layers.is_empty() {
            return;
        }

        // Reserve workspace for batch_size=1
        workspace.reserve(1, &self.config);

        // Calculate max dimension for ping-pong buffers
        let max_dim = self.layer_dims.iter().copied().max().unwrap_or(1);
        workspace.layer_output.resize(max_dim);
        workspace.layer_input.resize(max_dim);

        // Get max basis_aligned across all layers
        let max_basis = self
            .layers
            .iter()
            .map(|l| l.basis_aligned)
            .max()
            .unwrap_or(8);
        workspace.basis_values.resize(max_basis);

        if self.layers.len() == 1 {
            // Single layer: input → output
            let basis_buf =
                &mut workspace.basis_values.as_mut_slice()[..self.layers[0].basis_aligned];
            self.layers[0].forward_single(input, output, basis_buf);
        } else {
            // Multi-layer: use ping-pong buffers
            let mut use_output_as_current = true;

            // First layer: input → layer_output
            {
                let layer = &self.layers[0];
                let out_slice = &mut workspace.layer_output.as_mut_slice()[..layer.out_dim];
                let basis_buf = &mut workspace.basis_values.as_mut_slice()[..layer.basis_aligned];
                layer.forward_single(input, out_slice, basis_buf);
            }

            // Hidden layers: ping-pong
            for i in 1..self.layers.len() - 1 {
                let layer = &self.layers[i];

                // Copy current to z_buffer to avoid borrow issues
                let in_size = layer.in_dim;
                workspace.z_buffer.resize(in_size);
                if use_output_as_current {
                    workspace
                        .z_buffer
                        .as_mut_slice()
                        .copy_from_slice(&workspace.layer_output.as_slice()[..in_size]);
                } else {
                    workspace
                        .z_buffer
                        .as_mut_slice()
                        .copy_from_slice(&workspace.layer_input.as_slice()[..in_size]);
                }

                // Forward to the other buffer
                let out_slice = if use_output_as_current {
                    &mut workspace.layer_input.as_mut_slice()[..layer.out_dim]
                } else {
                    &mut workspace.layer_output.as_mut_slice()[..layer.out_dim]
                };
                let basis_buf = &mut workspace.basis_values.as_mut_slice()[..layer.basis_aligned];
                layer.forward_single(workspace.z_buffer.as_slice(), out_slice, basis_buf);

                use_output_as_current = !use_output_as_current;
            }

            // Last layer: current buffer → output
            {
                let layer = self.layers.last().unwrap();
                let in_size = layer.in_dim;

                // Copy to z_buffer
                workspace.z_buffer.resize(in_size);
                if use_output_as_current {
                    workspace
                        .z_buffer
                        .as_mut_slice()
                        .copy_from_slice(&workspace.layer_output.as_slice()[..in_size]);
                } else {
                    workspace
                        .z_buffer
                        .as_mut_slice()
                        .copy_from_slice(&workspace.layer_input.as_slice()[..in_size]);
                }

                let basis_buf = &mut workspace.basis_values.as_mut_slice()[..layer.basis_aligned];
                layer.forward_single(workspace.z_buffer.as_slice(), output, basis_buf);
            }
        }
    }

    /// Forward pass for a batch of samples.
    ///
    /// # Arguments
    /// * `input` - Input batch [batch_size * input_dim], Row-Major
    /// * `output` - Output buffer [batch_size * output_dim]
    /// * `workspace` - Pre-allocated workspace
    pub fn forward_batch(&self, input: &[f32], output: &mut [f32], workspace: &mut Workspace) {
        let batch_size = input.len() / self.config.input_dim;
        debug_assert_eq!(input.len(), batch_size * self.config.input_dim);
        debug_assert_eq!(output.len(), batch_size * self.config.output_dim);

        if self.layers.is_empty() {
            return;
        }

        workspace.reserve(batch_size, &self.config);

        if self.layers.len() == 1 {
            // Single layer network
            workspace.prepare_forward(batch_size, &self.config);
            self.layers[0].forward_batch(input, output, workspace);
        } else {
            // Multi-layer: ping-pong buffers without heap allocations
            let max_hidden = self
                .layer_dims
                .iter()
                .copied()
                .max()
                .unwrap_or(self.config.input_dim);
            let ping_pong_size = batch_size * max_hidden;
            workspace.layer_output.resize(ping_pong_size);
            workspace.layer_input.resize(ping_pong_size);

            // Staging buffer to avoid aliasing workspace slices during forward_batch
            let mut temp_input = vec![0.0f32; ping_pong_size];

            let mut use_output_as_current = true;

            // First layer: input -> layer_output
            {
                let layer = &self.layers[0];
                let out_size = batch_size * layer.out_dim;
                let mut layer_output_buf = std::mem::take(&mut workspace.layer_output);
                layer_output_buf.resize(out_size);
                let out_slice = &mut layer_output_buf.as_mut_slice()[..out_size];
                layer.forward_batch(input, out_slice, workspace);
                workspace.layer_output = layer_output_buf;
            }

            // Hidden layers: ping-pong
            for i in 1..self.layers.len() - 1 {
                let layer = &self.layers[i];
                let in_size = batch_size * layer.in_dim;
                let out_size = batch_size * layer.out_dim;

                // Copy current buffer into staging to avoid aliasing with mutable workspace borrow
                if use_output_as_current {
                    temp_input[..in_size]
                        .copy_from_slice(&workspace.layer_output.as_slice()[..in_size]);
                } else {
                    temp_input[..in_size]
                        .copy_from_slice(&workspace.layer_input.as_slice()[..in_size]);
                }

                if use_output_as_current {
                    let mut buf = std::mem::take(&mut workspace.layer_input);
                    buf.resize(out_size);
                    let out_slice = &mut buf.as_mut_slice()[..out_size];
                    layer.forward_batch(&temp_input[..in_size], out_slice, workspace);
                    workspace.layer_input = buf;
                } else {
                    let mut buf = std::mem::take(&mut workspace.layer_output);
                    buf.resize(out_size);
                    let out_slice = &mut buf.as_mut_slice()[..out_size];
                    layer.forward_batch(&temp_input[..in_size], out_slice, workspace);
                    workspace.layer_output = buf;
                }
                use_output_as_current = !use_output_as_current;
            }

            // Last layer: current buffer -> output
            {
                let layer = self.layers.last().unwrap();
                let in_size = batch_size * layer.in_dim;
                if use_output_as_current {
                    temp_input[..in_size]
                        .copy_from_slice(&workspace.layer_output.as_slice()[..in_size]);
                } else {
                    temp_input[..in_size]
                        .copy_from_slice(&workspace.layer_input.as_slice()[..in_size]);
                }
                layer.forward_batch(&temp_input[..in_size], output, workspace);
            }
        }
    }

    /// Forward pass for training: stores per-layer normalized inputs and grid indices.
    pub fn forward_batch_training(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
    ) {
        let batch_size = input.len() / self.config.input_dim;
        debug_assert_eq!(input.len(), batch_size * self.config.input_dim);
        debug_assert_eq!(output.len(), batch_size * self.config.output_dim);

        if self.layers.is_empty() {
            return;
        }

        workspace.prepare_training(batch_size, &self.config, &self.layer_dims);

        // Two-vector ping-pong to avoid borrow conflicts; reuses allocations.
        let mut current: Vec<f32> = input.to_vec();
        let mut next: Vec<f32> = Vec::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let out_size = batch_size * layer.out_dim;
            if next.len() < out_size {
                next.resize(out_size, 0.0);
            } else {
                next[..out_size].fill(0.0);
            }

            layer.forward_batch(&current, &mut next[..out_size], workspace);

            // Save normalized inputs and grid indices for backward
            let hist_in =
                &mut workspace.layers_inputs[layer_idx].as_mut_slice()[..batch_size * layer.in_dim];
            hist_in.copy_from_slice(workspace.z_buffer.as_slice());

            let hist_idx =
                &mut workspace.layers_grid_indices[layer_idx][..batch_size * layer.in_dim];
            hist_idx.copy_from_slice(&workspace.grid_indices[..batch_size * layer.in_dim]);

            if layer_idx == self.layers.len() - 1 {
                output.copy_from_slice(&next[..out_size]);
            } else {
                std::mem::swap(&mut current, &mut next);
            }
        }
    }

    /// Multi-layer forward pass using ping-pong buffers.
    ///

    /// Full training step: forward + backward + update.
    ///
    /// Returns the loss value.
    pub fn train_step(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        learning_rate: f32,
        workspace: &mut Workspace,
    ) -> f32 {
        let opts = self.default_train_options;
        self.train_step_with_options(input, target, mask, learning_rate, workspace, &opts)
    }

    /// Полноценный шаг обучения с опциями.
    pub fn train_step_with_options(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        learning_rate: f32,
        workspace: &mut Workspace,
        opts: &TrainOptions,
    ) -> f32 {
        let batch_size = input.len() / self.config.input_dim;
        let output_dim = self.config.output_dim;

        // Forward pass with history capture
        let mut predictions = vec![0.0f32; batch_size * output_dim];
        self.forward_batch_training(input, &mut predictions, workspace);

        // Compute loss and output gradient (mean reduction)
        let (loss, mut grad_next) =
            compute_masked_mse_loss(&predictions, target, mask, batch_size, output_dim);

        workspace.assert_history_batch(batch_size);

        // Контейнеры для глобального клиппинга
        let mut weight_grads_all: Vec<Vec<f32>> = vec![Vec::new(); self.layers.len()];
        let mut bias_grads_all: Vec<Vec<f32>> = vec![Vec::new(); self.layers.len()];
        let mut total_sq_norm: f32 = 0.0;

        // Backward pass through all layers (накапливаем градиенты)
        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[layer_idx];
            let in_dim = layer.in_dim;
            let out_dim = layer.out_dim;

            let grad_out_slice = &grad_next[..batch_size * out_dim];

            let mut weight_grad = vec![0.0f32; layer.weights.len()];
            let mut bias_grad = vec![0.0f32; layer.bias.len()];

            // Temporarily take gradient buffer from workspace to avoid borrow conflicts
            let mut grad_buffer = if layer_idx > 0 {
                Some(std::mem::take(&mut workspace.layer_grads))
            } else {
                None
            };
            if let Some(ref mut buf) = grad_buffer {
                let needed = batch_size * in_dim;
                buf.reserve(needed);
                buf.resize(needed);
                buf.as_mut_slice().iter_mut().for_each(|x| *x = 0.0);
            }

            // Temporarily take history buffers to avoid aliasing with `workspace`
            let layer_input_buf = std::mem::take(&mut workspace.layers_inputs[layer_idx]);
            let layer_grid_buf = std::mem::take(&mut workspace.layers_grid_indices[layer_idx]);

            layer.backward(
                layer_input_buf.as_slice(),
                &layer_grid_buf,
                grad_out_slice,
                grad_buffer
                    .as_mut()
                    .map(|b| &mut b.as_mut_slice()[..batch_size * in_dim]),
                &mut weight_grad,
                &mut bias_grad,
                workspace,
            );

            // Return history buffers
            workspace.layers_inputs[layer_idx] = layer_input_buf;
            workspace.layers_grid_indices[layer_idx] = layer_grid_buf;

            // Накопить норму по параметрам для глобального клиппинга
            total_sq_norm += weight_grad.iter().map(|g| g * g).sum::<f32>();
            total_sq_norm += bias_grad.iter().map(|g| g * g).sum::<f32>();

            weight_grads_all[layer_idx] = weight_grad;
            bias_grads_all[layer_idx] = bias_grad;

            if let Some(buf) = grad_buffer {
                let needed = batch_size * in_dim;
                if grad_next.len() < needed {
                    grad_next.resize(needed, 0.0);
                }
                grad_next[..needed].copy_from_slice(&buf.as_slice()[..needed]);
                grad_next.truncate(needed);

                // Return buffer to workspace for reuse
                workspace.layer_grads = buf;
            } else {
                grad_next.clear();
            }
        }

        // Глобальный клиппинг по всем параметрам, если задан
        if let Some(max_norm) = opts.max_grad_norm {
            let norm = total_sq_norm.sqrt();
            if norm > max_norm && norm > 0.0 {
                let scale = max_norm / norm;
                for wg in &mut weight_grads_all {
                    for g in wg.iter_mut() {
                        *g *= scale;
                    }
                }
                for bg in &mut bias_grads_all {
                    for g in bg.iter_mut() {
                        *g *= scale;
                    }
                }
            }
        }

        // Decoupled weight decay + SGD update (градиенты уже средние по batch/mask)
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if opts.weight_decay > 0.0 {
                let decay = opts.weight_decay;
                for w in layer.weights.iter_mut() {
                    *w *= 1.0 - learning_rate * decay;
                }
            }

            for (w, g) in layer.weights.iter_mut().zip(weight_grads_all[i].iter()) {
                *w -= learning_rate * g;
            }
            for (b, g) in layer.bias.iter_mut().zip(bias_grads_all[i].iter()) {
                *b -= learning_rate * g;
            }
        }

        loss
    }

    /// Creates a workspace sized for this network.
    pub fn create_workspace(&self, max_batch: usize) -> Workspace {
        let mut ws = Workspace::new(&self.config);
        ws.reserve(max_batch, &self.config);
        ws
    }

    /// Saves network to bytes using bincode.
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Loads network from bytes.
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

impl Clone for KanNetwork {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            layers: self.layers.clone(),
            layer_dims: self.layer_dims.clone(),
            default_train_options: self.default_train_options,
        }
    }
}

/// Computes masked MSE loss and gradient.
fn compute_masked_mse_loss(
    predictions: &[f32],
    targets: &[f32],
    mask: Option<&[f32]>,
    batch_size: usize,
    output_dim: usize,
) -> (f32, Vec<f32>) {
    let mut loss = 0.0f32;
    let mut grad = vec![0.0f32; batch_size * output_dim];
    let mut count = 0.0f32;

    for b in 0..batch_size {
        for o in 0..output_dim {
            let idx = b * output_dim + o;
            let m = mask.map(|m| m[idx]).unwrap_or(1.0);

            if m > 0.0 {
                let diff = predictions[idx] - targets[idx];
                loss += m * diff * diff;
                grad[idx] = 2.0 * m * diff;
                count += m;
            }
        }
    }

    if count > 0.0 {
        let inv = 1.0 / count;
        loss *= inv;
        for g in grad.iter_mut() {
            *g *= inv;
        }
    }

    (loss, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = KanConfig::default_poker();
        let network = KanNetwork::new(config);

        // 21 → 64 → 64 → 24: 3 layers
        assert_eq!(network.num_layers(), 3);
        assert_eq!(network.layers[0].in_dim, 21);
        assert_eq!(network.layers[0].out_dim, 64);
        assert_eq!(network.layers[2].out_dim, 24);
    }

    #[test]
    fn test_network_forward_single() {
        let config = KanConfig::default_poker();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input = vec![0.5f32; 21];
        let mut output = vec![0.0f32; 24];

        network.forward_single(&input, &mut output, &mut workspace);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_network_forward_batch() {
        let config = KanConfig::default_poker();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(32);

        let batch_size = 16;
        let input: Vec<f32> = (0..batch_size * 21)
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();
        let mut output = vec![0.0f32; batch_size * 24];

        network.forward_batch(&input, &mut output, &mut workspace);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_network_train_step() {
        let config = KanConfig::default_poker();
        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(16);

        let batch_size = 8;
        let input: Vec<f32> = vec![0.5; batch_size * 21];
        let target: Vec<f32> = vec![0.1; batch_size * 24];

        let loss1 = network.train_step(&input, &target, None, 0.01, &mut workspace);
        let loss2 = network.train_step(&input, &target, None, 0.01, &mut workspace);

        // Loss should decrease after training
        assert!(
            loss2 < loss1,
            "Loss should decrease: {} -> {}",
            loss1,
            loss2
        );
    }

    #[test]
    fn test_network_param_count() {
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8],
            grid_size: 5,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            multithreading_threshold: 1024,
            simd_width: 8,
            init_seed: None,
        };

        let network = KanNetwork::new(config);

        // Layer 1: 4 → 8, basis_aligned=8
        // Weights: 8 * 4 * 8 = 256, Bias: 8 → 264
        // Layer 2: 8 → 2
        // Weights: 2 * 8 * 8 = 128, Bias: 2 → 130
        // Total: 394

        let params = network.param_count();
        assert!(params > 0);
        assert_eq!(params, 264 + 130);
    }

    #[test]
    fn test_single_layer_network() {
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![], // No hidden layers
            grid_size: 5,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            multithreading_threshold: 1024,
            simd_width: 8,
            init_seed: None,
        };

        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(8);

        assert_eq!(network.num_layers(), 1);

        let input = vec![0.5f32; 4];
        let mut output = vec![0.0f32; 2];
        network.forward_single(&input, &mut output, &mut workspace);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_gradcheck_single_layer() {
        // Маленькая сеть 2 -> 1 для численной проверки градиентов
        let config = KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![],
            grid_size: 4,
            spline_order: 2,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            multithreading_threshold: 16,
            simd_width: 4,
            init_seed: None,
        };

        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input = vec![0.25f32, -0.4];
        let target = vec![0.1f32];

        // Прямой проход с записью истории
        let mut preds = vec![0.0f32; 1];
        network.forward_batch_training(&input, &mut preds, &mut workspace);
        let diff = preds[0] - target[0];
        let grad_out = vec![2.0 * diff]; // dMSE/dy для batch=1

        // Аналитические градиенты
        let layer = &network.layers[0];
        let mut weight_grad = vec![0.0f32; layer.weights.len()];
        let mut bias_grad = vec![0.0f32; layer.bias.len()];
        let mut grad_in = vec![0.0f32; input.len()];

        let hist_in = workspace.layers_inputs[0].as_slice().to_vec();
        let hist_idx = workspace.layers_grid_indices[0].clone();

        network.layers[0].backward(
            &hist_in,
            &hist_idx,
            &grad_out,
            Some(&mut grad_in),
            &mut weight_grad,
            &mut bias_grad,
            &mut workspace,
        );

        // Численный градиент по весам
        let eps = 1e-3f32;
        let mut ws_num = Workspace::new(&config);
        ws_num.reserve(1, &config);
        let mut out_buf = vec![0.0f32; 1];

        for idx in 0..network.layers[0].weights.len() {
            let orig = network.layers[0].weights[idx];

            network.layers[0].weights[idx] = orig + eps;
            network.forward_batch(&input, &mut out_buf, &mut ws_num);
            let lp = {
                let d = out_buf[0] - target[0];
                d * d
            };

            network.layers[0].weights[idx] = orig - eps;
            network.forward_batch(&input, &mut out_buf, &mut ws_num);
            let lm = {
                let d = out_buf[0] - target[0];
                d * d
            };

            network.layers[0].weights[idx] = orig;

            let num = (lp - lm) / (2.0 * eps);
            let ana = weight_grad[idx];
            let rel_err = (ana - num).abs() / num.abs().max(1e-4);
            assert!(
                rel_err < 1e-2,
                "gradcheck weight {} failed: ana={} num={} rel_err={}",
                idx,
                ana,
                num,
                rel_err
            );
        }
    }

    #[test]
    fn test_mask_blocks_update() {
        // Маска из нулей должна блокировать обновления
        let config = KanConfig {
            input_dim: 2,
            output_dim: 2,
            hidden_dims: vec![],
            grid_size: 4,
            spline_order: 2,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            multithreading_threshold: 16,
            simd_width: 4,
            init_seed: None,
        };

        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input = vec![0.3f32, -0.2];
        let target = vec![0.5f32, -0.5];
        let mask = vec![0.0f32; 2]; // все выключено

        let before_w = network.layers[0].weights.clone();
        let before_b = network.layers[0].bias.clone();

        network.train_step_with_options(
            &input,
            &target,
            Some(&mask),
            0.1,
            &mut workspace,
            &TrainOptions::default(),
        );

        assert_eq!(before_w, network.layers[0].weights);
        assert_eq!(before_b, network.layers[0].bias);
    }
}
