//! KAN network with multi-layer support and zero-allocation inference.
//!
//! This module provides [`KanNetwork`], the main entry point for using ArKan.
//! It supports both inference and training with configurable options.
//!
//! # Example: Inference
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork};
//!
//! let config = KanConfig::preset();
//! let network = KanNetwork::new(config.clone());
//! let mut workspace = network.create_workspace(1);
//!
//! let input = vec![0.5f32; config.input_dim];
//! let mut output = vec![0.0f32; config.output_dim];
//!
//! // ~30 µs latency, zero allocations
//! network.forward_single(&input, &mut output, &mut workspace);
//! ```
//!
//! # Example: Training
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork, TrainOptions};
//!
//! let config = KanConfig::preset();
//! let mut network = KanNetwork::new(config.clone());
//! let mut workspace = network.create_workspace(64);
//!
//! let inputs = vec![0.5f32; 64 * config.input_dim];
//! let targets = vec![0.1f32; 64 * config.output_dim];
//!
//! // Training with gradient clipping
//! let opts = TrainOptions {
//!     max_grad_norm: Some(1.0),
//!     weight_decay: 0.01,
//! };
//! network.set_default_train_options(opts);
//!
//! let loss = network.train_step(&inputs, &targets, None, 0.001, &mut workspace);
//! ```
//!
//! # Performance
//!
//! | Method | Batch Size | Time | Use Case |
//! |--------|-----------|------|----------|
//! | `forward_single` | 1 | ~15 µs | Real-time play |
//! | `forward_batch` | 1 | ~30 µs | General inference |
//! | `forward_batch` | 64 | ~2 ms | Batch inference |
//! | `train_step` | 64 | ~5 ms | Training |

use crate::buffer::Workspace;
use crate::config::KanConfig;
use crate::error::{ArkanError, ArkanResult};
use crate::layer::KanLayer;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Magic bytes for serialized network files.
///
/// Used to identify ArKan model files and distinguish from other formats.
#[cfg(feature = "serde")]
const SERIALIZATION_MAGIC: &[u8; 5] = b"ARKAN";

/// Current serialization format version.
///
/// Incremented when the format changes in a backwards-incompatible way.
/// - v1: Initial versioned format (ArKan 0.3.0+)
#[cfg(feature = "serde")]
const SERIALIZATION_VERSION: u32 = 1;

/// Complete KAN network with multiple layers.
///
/// This is the main struct for using ArKan. It holds the network configuration,
/// all layers, and provides methods for inference and training.
///
/// # Zero-Allocation Inference
///
/// Both `forward_single` and `forward_batch` perform zero allocations when
/// the [`Workspace`] is properly sized. Create the workspace once with
/// [`create_workspace`](Self::create_workspace) and reuse it.
///
/// # Thread Safety
///
/// The network is `Send + Sync` (when `serde` feature is off). Multiple threads
/// can perform inference on the same network with separate workspaces.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KanNetwork {
    /// Network configuration.
    pub config: KanConfig,

    /// Layers: input→hidden\[0\]→...→hidden\[n\]→output.
    pub layers: Vec<KanLayer>,

    /// Cached layer dimensions for quick access.
    layer_dims: Vec<usize>,

    /// Cached parameter sizes per layer (weights, bias) for train_step.
    layer_param_sizes: Vec<(usize, usize)>,

    /// Default training options (gradient clipping, weight decay).
    pub default_train_options: TrainOptions,
}

/// Training options for a single step.
///
/// These options control gradient clipping and weight decay during training.
/// Set via [`KanNetwork::set_default_train_options`] or pass to
/// [`train_step_with_options`](KanNetwork::train_step_with_options).
///
/// # Example
///
/// ```rust
/// use arkan::TrainOptions;
///
/// let opts = TrainOptions {
///     max_grad_norm: Some(1.0),  // Clip gradients with L2 norm > 1.0
///     weight_decay: 0.01,        // AdamW-style decoupled weight decay
/// };
/// ```
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrainOptions {
    /// Maximum L2 norm for gradient clipping. `None` disables clipping.
    ///
    /// When set, gradients are scaled down if their total L2 norm exceeds
    /// this value. Helps prevent exploding gradients.
    pub max_grad_norm: Option<f32>,

    /// Decoupled weight decay coefficient (AdamW-style).
    ///
    /// Applied as `w = w * (1 - lr * weight_decay)` before the gradient update.
    /// Set to `0.0` to disable.
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
    ///
    /// Initializes all layers with random weights using Xavier initialization.
    /// Use [`KanConfig::init_seed`] for deterministic initialization.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config);
    ///
    /// assert_eq!(network.num_layers(), 3); // 2 hidden + 1 output
    /// ```
    #[must_use = "this creates a new network without modifying anything"]
    pub fn new(config: KanConfig) -> Self {
        let layer_dims = config.layer_dims();
        let mut layers = Vec::with_capacity(layer_dims.len() - 1);

        for i in 0..layer_dims.len() - 1 {
            let in_dim = layer_dims[i];
            let out_dim = layer_dims[i + 1];
            layers.push(KanLayer::from_config(in_dim, out_dim, &config));
        }

        let layer_param_sizes: Vec<(usize, usize)> = layers
            .iter()
            .map(|l| (l.weights.len(), l.bias.len()))
            .collect();

        Self {
            config,
            layers,
            layer_dims,
            layer_param_sizes,
            default_train_options: TrainOptions::default(),
        }
    }

    /// Creates network from configuration (alias for [`new`](Self::new)).
    pub fn from_config(config: KanConfig) -> Self {
        Self::new(config)
    }

    /// Sets default training options for all subsequent `train_step` calls.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork, TrainOptions};
    ///
    /// let mut network = KanNetwork::new(KanConfig::preset());
    /// network.set_default_train_options(TrainOptions {
    ///     max_grad_norm: Some(1.0),
    ///     weight_decay: 0.01,
    /// });
    /// ```
    pub fn set_default_train_options(&mut self, opts: TrainOptions) {
        self.default_train_options = opts;
    }

    /// Returns the number of layers (hidden + output).
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns total number of trainable parameters (weights + biases).
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let network = KanNetwork::new(KanConfig::preset());
    /// println!("Parameters: {}", network.param_count()); // ~56K
    /// ```
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    /// Sets input normalization statistics (mean and std per feature).
    ///
    /// Call this after computing statistics from your training data.
    /// Normalization is applied in the first layer only.
    ///
    /// # Arguments
    ///
    /// * `mean` - Per-feature mean values `[input_dim]`
    /// * `std` - Per-feature standard deviations `[input_dim]`
    pub fn set_input_normalization(&mut self, mean: &[f32], std: &[f32]) {
        if !self.layers.is_empty() {
            self.layers[0].set_normalization(mean, std);
        }
    }

    /// Forward pass for a single sample (optimized for latency).
    ///
    /// This method is optimized for single-sample inference, achieving ~15 µs
    /// latency on the poker config. Use this for real-time applications.
    ///
    /// # Arguments
    ///
    /// * `input` - Input features `[input_dim]`
    /// * `output` - Output buffer `[output_dim]` (will be overwritten)
    /// * `workspace` - Pre-allocated workspace (reuse for zero-alloc)
    ///
    /// # Panics
    ///
    /// Debug-asserts if `input.len() != config.input_dim` or
    /// `output.len() != config.output_dim`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(1);
    ///
    /// let input = vec![0.5f32; config.input_dim];
    /// let mut output = vec![0.0f32; config.output_dim];
    ///
    /// network.forward_single(&input, &mut output, &mut workspace);
    /// ```
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

    /// Forward pass for a batch of samples (zero-allocation).
    ///
    /// Processes multiple samples in parallel, leveraging cache locality
    /// for better throughput than multiple `forward_single` calls.
    ///
    /// # Zero-Allocation Strategy
    ///
    /// This method achieves zero allocations by using **ping-pong buffers**:
    /// - `workspace.layer_output` (buffer A) and `workspace.layer_input` (buffer B)
    ///   are pre-allocated to `batch_size * max_hidden_dim`
    /// - Each layer reads from one buffer and writes to the other
    /// - `std::mem::take` temporarily moves buffers out of workspace to satisfy
    ///   Rust's borrow checker (no aliasing between input/output slices)
    /// - Buffers are returned to workspace at the end for reuse
    ///
    /// # Buffer Flow
    ///
    /// ```text
    /// Layer 0: input → buffer_a
    /// Layer 1: buffer_a → buffer_b
    /// Layer 2: buffer_b → buffer_a
    /// ...     (alternating)
    /// Last:    buffer_X → output
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input` - Input batch `[batch_size * input_dim]`, row-major layout
    /// * `output` - Output buffer `[batch_size * output_dim]` (will be overwritten)
    /// * `workspace` - Pre-allocated workspace (reuse for zero-alloc)
    ///
    /// # Panics
    ///
    /// Debug-asserts if input/output lengths don't match expected dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(64);
    ///
    /// let batch_size = 64;
    /// let input = vec![0.5f32; batch_size * config.input_dim];
    /// let mut output = vec![0.0f32; batch_size * config.output_dim];
    ///
    /// network.forward_batch(&input, &mut output, &mut workspace);
    ///
    /// // Access first sample's output
    /// let first_output = &output[0..config.output_dim];
    /// ```
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
            // Multi-layer: ping-pong buffers without heap allocations.
            // We use two pre-allocated buffers (A and B) that alternate as
            // input/output for successive layers.
            let max_hidden = self
                .layer_dims
                .iter()
                .copied()
                .max()
                .unwrap_or(self.config.input_dim);
            let ping_pong_size = batch_size * max_hidden;

            // Borrow-safety: `std::mem::take` moves buffers out of workspace,
            // allowing us to have mutable references to both without aliasing.
            // This is the key trick for zero-allocation multi-layer forward.
            //
            // Panic safety: We use a scope + explicit return at the end.
            // If panic occurs, buffers are lost but workspace remains valid
            // (empty buffers). Caller should recreate workspace after panic.
            let mut buffer_a = std::mem::take(&mut workspace.layer_output);
            let mut buffer_b = std::mem::take(&mut workspace.layer_input);
            buffer_a.resize(ping_pong_size);
            buffer_b.resize(ping_pong_size);

            // First layer: input -> buffer_a
            {
                let layer = &self.layers[0];
                let out_size = batch_size * layer.out_dim;
                buffer_a.resize(out_size);
                layer.forward_batch(input, &mut buffer_a.as_mut_slice()[..out_size], workspace);
            }

            let mut current_is_a = true;

            // Hidden layers: ping-pong between buffer_a/buffer_b
            for i in 1..self.layers.len() - 1 {
                let layer = &self.layers[i];
                let in_size = batch_size * layer.in_dim;
                let out_size = batch_size * layer.out_dim;

                let (input_buf, output_buf) = if current_is_a {
                    (&buffer_a, &mut buffer_b)
                } else {
                    (&buffer_b, &mut buffer_a)
                };

                output_buf.resize(out_size);
                layer.forward_batch(
                    &input_buf.as_slice()[..in_size],
                    &mut output_buf.as_mut_slice()[..out_size],
                    workspace,
                );

                current_is_a = !current_is_a;
            }

            // Last layer: current buffer -> output
            {
                let layer = self.layers.last().unwrap();
                let in_size = batch_size * layer.in_dim;
                let input_buf = if current_is_a { &buffer_a } else { &buffer_b };
                layer.forward_batch(&input_buf.as_slice()[..in_size], output, workspace);
            }

            // Return buffers to workspace for reuse in subsequent calls.
            // This preserves capacity, avoiding re-allocation on next forward.
            workspace.layer_output = buffer_a;
            workspace.layer_input = buffer_b;
        }
    }

    /// Forward pass with Result return type for better error handling.
    ///
    /// This is the Result-returning version of [`forward_batch`](Self::forward_batch).
    /// Use this when you want explicit error handling instead of panics.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data `[batch_size * input_dim]`
    /// * `output` - Output buffer `[batch_size * output_dim]` (will be overwritten)
    /// * `workspace` - Pre-allocated workspace
    ///
    /// # Errors
    ///
    /// Returns `ArkanError::ShapeMismatch` if input/output lengths don't match expected dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(64);
    ///
    /// let batch_size = 64;
    /// let input = vec![0.5f32; batch_size * config.input_dim];
    /// let mut output = vec![0.0f32; batch_size * config.output_dim];
    ///
    /// network.try_forward_batch(&input, &mut output, &mut workspace)?;
    /// # Ok::<(), arkan::ArkanError>(())
    /// ```
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_forward_batch(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
    ) -> ArkanResult<()> {
        let batch_size = input.len() / self.config.input_dim;

        // Validate input length
        let expected_input_len = batch_size * self.config.input_dim;
        if input.len() != expected_input_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_input_len],
                &[input.len()],
            ));
        }

        // Validate output length
        let expected_output_len = batch_size * self.config.output_dim;
        if output.len() != expected_output_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_output_len],
                &[output.len()],
            ));
        }

        // Delegate to forward_batch (which now has guaranteed valid sizes)
        self.forward_batch(input, output, workspace);
        Ok(())
    }

    /// Forward pass for training: stores per-layer normalized inputs and grid indices.
    ///
    /// This method extends [`forward_batch`](Self::forward_batch) by saving
    /// intermediate values needed for the backward pass:
    ///
    /// - **Normalized inputs** (`workspace.layers_inputs\[layer\]`): the z-values
    ///   after input normalization, used to recompute spline basis derivatives.
    /// - **Grid indices** (`workspace.layers_grid_indices\[layer\]`): the spline
    ///   segment each input falls into, used to index into B-spline weights.
    ///
    /// # Buffer Layout
    ///
    /// Uses the same ping-pong scheme as `forward_batch`, plus:
    /// - `workspace.layers_inputs`: `Vec<AlignedBuffer>` with one buffer per layer
    /// - `workspace.layers_grid_indices`: `Vec<Vec<u32>>` with indices per layer
    ///
    /// These history buffers are prepared by [`Workspace::prepare_training`] and
    /// sized to `batch_size * in_dim` per layer.
    ///
    /// # Zero-Allocation Note
    ///
    /// After warmup (first call with a given batch size), this method performs
    /// zero heap allocations. The history buffers grow monotonically and are
    /// reused across training steps.
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

        let max_hidden = self
            .layer_dims
            .iter()
            .copied()
            .max()
            .unwrap_or(self.config.input_dim);
        let ping_pong_size = batch_size * max_hidden;

        let mut buffer_a = std::mem::take(&mut workspace.layer_output);
        let mut buffer_b = std::mem::take(&mut workspace.layer_input);
        buffer_a.resize(ping_pong_size);
        buffer_b.resize(ping_pong_size);

        let mut current_is_a = true;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let in_size = batch_size * layer.in_dim;
            let out_size = batch_size * layer.out_dim;

            let (input_slice, output_buf) = if layer_idx == 0 {
                (
                    input,
                    if current_is_a {
                        &mut buffer_a
                    } else {
                        &mut buffer_b
                    },
                )
            } else if current_is_a {
                (&buffer_a.as_slice()[..in_size], &mut buffer_b)
            } else {
                (&buffer_b.as_slice()[..in_size], &mut buffer_a)
            };

            output_buf.resize(out_size);
            layer.forward_batch(
                input_slice,
                &mut output_buf.as_mut_slice()[..out_size],
                workspace,
            );

            // Save normalized inputs and grid indices for backward
            let hist_in = &mut workspace.layers_inputs[layer_idx].as_mut_slice()[..in_size];
            hist_in.copy_from_slice(workspace.z_buffer.as_slice());

            let hist_idx = &mut workspace.layers_grid_indices[layer_idx][..in_size];
            hist_idx.copy_from_slice(&workspace.grid_indices[..in_size]);

            if layer_idx == self.layers.len() - 1 {
                output.copy_from_slice(&output_buf.as_slice()[..out_size]);
            }

            current_is_a = !current_is_a;
        }

        workspace.layer_output = buffer_a;
        workspace.layer_input = buffer_b;
    }

    /// Full training step: forward + backward + SGD update.
    ///
    /// Performs a complete training iteration with zero allocations (after warmup).
    /// Uses [`default_train_options`](Self::default_train_options) for gradient
    /// clipping and weight decay.
    ///
    /// # Arguments
    ///
    /// * `input` - Input batch `[batch_size * input_dim]`
    /// * `target` - Target values `[batch_size * output_dim]`
    /// * `mask` - Optional mask `[batch_size * output_dim]` (1.0 = active, 0.0 = ignore)
    /// * `learning_rate` - SGD learning rate
    /// * `workspace` - Pre-allocated workspace
    ///
    /// # Returns
    ///
    /// The MSE loss value for this batch.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let mut network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(64);
    ///
    /// let inputs = vec![0.5f32; 64 * config.input_dim];
    /// let targets = vec![0.1f32; 64 * config.output_dim];
    ///
    /// for epoch in 0..100 {
    ///     let loss = network.train_step(&inputs, &targets, None, 0.001, &mut workspace);
    ///     if epoch % 10 == 0 {
    ///         println!("Epoch {}: loss = {:.4}", epoch, loss);
    ///     }
    /// }
    /// ```
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

    /// Full training step with explicit options.
    ///
    /// Same as [`train_step`](Self::train_step) but allows passing custom
    /// [`TrainOptions`] instead of using the default.
    ///
    /// # Arguments
    ///
    /// * `opts` - Training options (gradient clipping, weight decay)
    ///
    /// See [`train_step`](Self::train_step) for other arguments.
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

        // Ensure workspace has gradient buffers for all layers
        workspace.prepare_grad_buffers(&self.layer_param_sizes);

        // Forward pass with history capture using workspace predictions buffer
        workspace.predictions_buffer.resize(batch_size * output_dim);
        // Take predictions buffer to avoid borrow conflict
        let mut predictions_buf = std::mem::take(&mut workspace.predictions_buffer);
        self.forward_batch_training(input, predictions_buf.as_mut_slice(), workspace);

        // Compute loss and output gradient into workspace buffer
        workspace.grad_output.resize(batch_size * output_dim);
        let loss = compute_masked_mse_loss_into(
            predictions_buf.as_slice(),
            target,
            mask,
            batch_size,
            output_dim,
            workspace.grad_output.as_mut_slice(),
        );
        // Return predictions buffer
        workspace.predictions_buffer = predictions_buf;

        workspace.assert_history_batch(batch_size);

        // =====================================================================
        // Backward pass: compute gradients for all layers (zero-allocation).
        //
        // Buffer scheme:
        // - `staging_buffer`: holds dL/d(layer_output) for the current layer.
        //   After processing layer i, it's overwritten with dL/d(layer_input)
        //   which becomes dL/d(layer_output) for layer i-1.
        // - `layer_grads`: temporary buffer for accumulating input gradients.
        // - `weight_grads[i]`, `bias_grads[i]`: accumulated gradients per layer.
        //
        // Borrow-safety: we use `std::mem::take` to temporarily move buffers
        // out of workspace, then return them after use. This avoids aliasing
        // between the layer's input history and the gradient buffers.
        //
        // Flow for layer i (reverse order, from output to input):
        //   1. Read grad_out from staging_buffer
        //   2. Compute weight/bias gradients using saved inputs (layers_inputs[i])
        //   3. Compute input gradients into layer_grads
        //   4. Copy layer_grads → staging_buffer for next iteration
        // =====================================================================
        let num_layers = self.layers.len();
        let mut total_sq_norm: f32 = 0.0;

        // Zero out gradient buffers before accumulating
        for i in 0..num_layers {
            for g in workspace.weight_grads[i].iter_mut() {
                *g = 0.0;
            }
            for g in workspace.bias_grads[i].iter_mut() {
                *g = 0.0;
            }
        }

        // Backward pass through all layers
        // staging_buffer holds the current layer's output gradient (dL/dy).
        // It's initialized with grad_output and updated each iteration.
        let max_dim = self.layer_dims.iter().copied().max().unwrap_or(output_dim);
        workspace.staging_buffer.resize(batch_size * max_dim);

        // Seed the backward pass with dL/d(network_output)
        let grad_out_size = batch_size * output_dim;
        workspace.staging_buffer.as_mut_slice()[..grad_out_size]
            .copy_from_slice(workspace.grad_output.as_slice());

        for layer_idx in (0..num_layers).rev() {
            let layer = &mut self.layers[layer_idx];
            let in_dim = layer.in_dim;
            let out_dim = layer.out_dim;

            // Take staging buffer with current gradient
            let staging = std::mem::take(&mut workspace.staging_buffer);
            let grad_out_slice = &staging.as_slice()[..batch_size * out_dim];

            // Take weight/bias gradient buffers to avoid borrow conflicts
            let mut weight_grad = std::mem::take(&mut workspace.weight_grads[layer_idx]);
            let mut bias_grad = std::mem::take(&mut workspace.bias_grads[layer_idx]);

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

            // Return gradient buffers
            workspace.weight_grads[layer_idx] = weight_grad;
            workspace.bias_grads[layer_idx] = bias_grad;

            // Return staging buffer
            workspace.staging_buffer = staging;

            if let Some(buf) = grad_buffer {
                let needed = batch_size * in_dim;
                // Copy gradient for next layer into staging_buffer
                workspace.staging_buffer.resize(needed);
                workspace.staging_buffer.as_mut_slice()[..needed]
                    .copy_from_slice(&buf.as_slice()[..needed]);

                // Return buffer to workspace for reuse
                workspace.layer_grads = buf;
            }
        }

        // Глобальный клиппинг по всем параметрам, если задан
        if let Some(max_norm) = opts.max_grad_norm {
            let norm = total_sq_norm.sqrt();
            if norm > max_norm && norm > 0.0 {
                let scale = max_norm / norm;
                for wg in workspace.weight_grads.iter_mut() {
                    for g in wg.iter_mut() {
                        *g *= scale;
                    }
                }
                for bg in workspace.bias_grads.iter_mut() {
                    for g in bg.iter_mut() {
                        *g *= scale;
                    }
                }
            }
        }

        // =====================================================================
        // Parameter update: decoupled weight decay + SGD
        //
        // Order:
        // 1. Weight decay: w *= (1 - lr * decay)  [applied first, only to weights]
        // 2. Gradient step: w -= lr * grad
        //
        // This is "decoupled" weight decay (like AdamW), not L2 regularization.
        // Biases are NOT decayed, following standard practice.
        // =====================================================================
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if opts.weight_decay > 0.0 {
                let decay = opts.weight_decay;
                for w in layer.weights.iter_mut() {
                    *w *= 1.0 - learning_rate * decay;
                }
            }

            for (w, g) in layer
                .weights
                .iter_mut()
                .zip(workspace.weight_grads[i].iter())
            {
                *w -= learning_rate * g;
            }
            for (b, g) in layer.bias.iter_mut().zip(workspace.bias_grads[i].iter()) {
                *b -= learning_rate * g;
            }
        }

        loss
    }

    /// Training step with Result return type for better error handling.
    ///
    /// This is the Result-returning version of [`train_step`](Self::train_step).
    /// Use this when you want explicit error handling instead of panics.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data `[batch_size * input_dim]`
    /// * `target` - Target values `[batch_size * output_dim]`
    /// * `mask` - Optional mask `[batch_size * output_dim]`
    /// * `learning_rate` - Learning rate for SGD update
    /// * `workspace` - Pre-allocated workspace
    ///
    /// # Errors
    ///
    /// Returns `ArkanError::ShapeMismatch` if input/target/mask lengths don't match expected dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let mut network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(64);
    ///
    /// let batch_size = 64;
    /// let inputs = vec![0.5f32; batch_size * config.input_dim];
    /// let targets = vec![0.1f32; batch_size * config.output_dim];
    ///
    /// let loss = network.try_train_step(&inputs, &targets, None, 0.001, &mut workspace)?;
    /// # Ok::<(), arkan::ArkanError>(())
    /// ```
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_train_step(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        learning_rate: f32,
        workspace: &mut Workspace,
    ) -> ArkanResult<f32> {
        let opts = self.default_train_options;
        self.try_train_step_with_options(input, target, mask, learning_rate, workspace, &opts)
    }

    /// Training step with explicit options and Result return type.
    ///
    /// This is the Result-returning version of [`train_step_with_options`](Self::train_step_with_options).
    ///
    /// # Arguments
    ///
    /// * `input` - Input data `[batch_size * input_dim]`
    /// * `target` - Target values `[batch_size * output_dim]`
    /// * `mask` - Optional mask `[batch_size * output_dim]`
    /// * `learning_rate` - Learning rate for SGD update
    /// * `workspace` - Pre-allocated workspace
    /// * `opts` - Training options (gradient clipping, weight decay)
    ///
    /// # Errors
    ///
    /// Returns `ArkanError::ShapeMismatch` if input/target/mask lengths don't match expected dimensions.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_train_step_with_options(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        learning_rate: f32,
        workspace: &mut Workspace,
        opts: &TrainOptions,
    ) -> ArkanResult<f32> {
        let batch_size = input.len() / self.config.input_dim;

        // Validate input length
        let expected_input_len = batch_size * self.config.input_dim;
        if input.len() != expected_input_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_input_len],
                &[input.len()],
            ));
        }

        // Validate target length
        let expected_target_len = batch_size * self.config.output_dim;
        if target.len() != expected_target_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_target_len],
                &[target.len()],
            ));
        }

        // Validate mask length if provided
        if let Some(m) = mask {
            if m.len() != expected_target_len {
                return Err(ArkanError::shape_mismatch(
                    &[expected_target_len],
                    &[m.len()],
                ));
            }
        }

        // Delegate to train_step_with_options (which now has guaranteed valid sizes)
        Ok(self.train_step_with_options(input, target, mask, learning_rate, workspace, opts))
    }

    /// Creates a workspace sized for this network (fallible version).
    ///
    /// The workspace is preallocated for the given maximum batch size.
    /// Reuse this workspace across all forward/backward calls to achieve
    /// zero-allocation inference and training.
    ///
    /// # Arguments
    ///
    /// * `max_batch` - Maximum batch size you'll use with this workspace
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Overflow`] if buffer size calculations overflow.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let network = KanNetwork::new(KanConfig::preset());
    /// let mut workspace = network.try_create_workspace(64)?;
    /// # Ok::<(), arkan::ArkanError>(())
    /// ```
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_create_workspace(&self, max_batch: usize) -> ArkanResult<Workspace> {
        let mut ws = Workspace::new(&self.config);
        ws.try_reserve(max_batch, &self.config)?;
        Ok(ws)
    }

    /// Creates a workspace sized for this network.
    ///
    /// The workspace is preallocated for the given maximum batch size.
    /// Reuse this workspace across all forward/backward calls to achieve
    /// zero-allocation inference and training.
    ///
    /// # Arguments
    ///
    /// * `max_batch` - Maximum batch size you'll use with this workspace
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_create_workspace`](Self::try_create_workspace)
    /// for a fallible version.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let network = KanNetwork::new(KanConfig::preset());
    ///
    /// // Create workspace for batches up to 64
    /// let mut workspace = network.create_workspace(64);
    ///
    /// // Can be used for any batch size <= 64
    /// // Workspace will grow automatically if needed, but that causes allocation
    /// ```
    #[must_use = "this creates a new workspace without modifying anything"]
    pub fn create_workspace(&self, max_batch: usize) -> Workspace {
        self.try_create_workspace(max_batch)
            .expect("KanNetwork::create_workspace: buffer size overflow")
    }

    /// Saves network to bytes using bincode with version header.
    ///
    /// The format includes:
    /// - Magic bytes: `ARKAN` (5 bytes)
    /// - Version: u32 (4 bytes)
    /// - Network data: bincode serialized
    ///
    /// Requires the `serde` feature.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let network = KanNetwork::new(KanConfig::preset());
    /// let bytes = network.to_bytes().unwrap();
    ///
    /// // Bytes start with magic header "ARKAN"
    /// assert_eq!(&bytes[..5], b"ARKAN");
    /// ```
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        use std::io::Write;

        let mut bytes = Vec::new();
        // Magic bytes
        bytes.write_all(SERIALIZATION_MAGIC).map_err(|e| {
            bincode::Error::from(bincode::ErrorKind::Io(e))
        })?;
        // Version
        bytes.write_all(&SERIALIZATION_VERSION.to_le_bytes()).map_err(|e| {
            bincode::Error::from(bincode::ErrorKind::Io(e))
        })?;
        // Network data
        let network_bytes = bincode::serialize(self)?;
        bytes.extend(network_bytes);
        Ok(bytes)
    }

    /// Loads network from bytes with version validation.
    ///
    /// Requires the `serde` feature.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Magic bytes don't match
    /// - Version is incompatible
    /// - Deserialization fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// // Create network and serialize
    /// let original = KanNetwork::new(KanConfig::preset());
    /// let bytes = original.to_bytes().unwrap();
    ///
    /// // Deserialize
    /// let loaded = KanNetwork::from_bytes(&bytes).unwrap();
    /// assert_eq!(loaded.param_count(), original.param_count());
    /// ```
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        const HEADER_SIZE: usize = SERIALIZATION_MAGIC.len() + 4; // magic + version

        if bytes.len() < HEADER_SIZE {
            return Err(bincode::Error::from(bincode::ErrorKind::Custom(
                "Invalid file: too short for header".to_string(),
            )));
        }

        // Check magic bytes
        if &bytes[..SERIALIZATION_MAGIC.len()] != SERIALIZATION_MAGIC {
            return Err(bincode::Error::from(bincode::ErrorKind::Custom(
                "Invalid file: wrong magic bytes (not an ArKan model)".to_string(),
            )));
        }

        // Check version
        let version_bytes: [u8; 4] = bytes[SERIALIZATION_MAGIC.len()..HEADER_SIZE]
            .try_into()
            .unwrap();
        let version = u32::from_le_bytes(version_bytes);

        if version != SERIALIZATION_VERSION {
            return Err(bincode::Error::from(bincode::ErrorKind::Custom(
                format!(
                    "Incompatible model version: expected {}, got {}",
                    SERIALIZATION_VERSION, version
                ),
            )));
        }

        // Deserialize network data
        bincode::deserialize(&bytes[HEADER_SIZE..])
    }

    /// Loads network from bytes without version check (legacy format).
    ///
    /// Use this to load models saved with ArKan < 0.3.0.
    ///
    /// # Warning
    ///
    /// This method is provided for backwards compatibility only.
    /// New code should use [`from_bytes`](Self::from_bytes).
    #[cfg(feature = "serde")]
    pub fn from_bytes_legacy(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

impl Clone for KanNetwork {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            layers: self.layers.clone(),
            layer_dims: self.layer_dims.clone(),
            layer_param_sizes: self.layer_param_sizes.clone(),
            default_train_options: self.default_train_options,
        }
    }
}

/// Computes masked MSE loss and gradient.
#[allow(dead_code)]
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

/// Computes masked MSE loss and writes gradient into provided buffer (zero-allocation).
fn compute_masked_mse_loss_into(
    predictions: &[f32],
    targets: &[f32],
    mask: Option<&[f32]>,
    batch_size: usize,
    output_dim: usize,
    grad: &mut [f32],
) -> f32 {
    let mut loss = 0.0f32;
    let mut count = 0.0f32;

    // Zero out gradient buffer
    grad.iter_mut().for_each(|g| *g = 0.0);

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

    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config);

        // 21 → 64 → 64 → 24: 3 layers
        assert_eq!(network.num_layers(), 3);
        assert_eq!(network.layers[0].in_dim, 21);
        assert_eq!(network.layers[0].out_dim, 64);
        assert_eq!(network.layers[2].out_dim, 24);
    }

    #[test]
    fn test_network_forward_single() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input = vec![0.5f32; 21];
        let mut output = vec![0.0f32; 24];

        network.forward_single(&input, &mut output, &mut workspace);

        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_network_forward_batch() {
        let config = KanConfig::preset();
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
        let config = KanConfig::preset();
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

    #[test]
    fn test_try_forward_batch_ok() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let batch_size = 4;
        let input = vec![0.5f32; batch_size * config.input_dim];
        let mut output = vec![0.0f32; batch_size * config.output_dim];

        let result = network.try_forward_batch(&input, &mut output, &mut workspace);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_forward_batch_input_mismatch() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let batch_size = 4;
        // Wrong input size (missing one element)
        let input = vec![0.5f32; batch_size * config.input_dim - 1];
        let mut output = vec![0.0f32; batch_size * config.output_dim];

        let result = network.try_forward_batch(&input, &mut output, &mut workspace);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ArkanError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_try_forward_batch_output_mismatch() {
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let batch_size = 4;
        let input = vec![0.5f32; batch_size * config.input_dim];
        // Wrong output size
        let mut output = vec![0.0f32; batch_size * config.output_dim - 1];

        let result = network.try_forward_batch(&input, &mut output, &mut workspace);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ArkanError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_try_train_step_ok() {
        let config = KanConfig::preset();
        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let batch_size = 4;
        let input = vec![0.5f32; batch_size * config.input_dim];
        let target = vec![0.1f32; batch_size * config.output_dim];

        let result = network.try_train_step(&input, &target, None, 0.001, &mut workspace);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_try_train_step_input_mismatch() {
        let config = KanConfig::preset();
        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let batch_size = 4;
        // Wrong input size
        let input = vec![0.5f32; batch_size * config.input_dim + 1];
        let target = vec![0.1f32; batch_size * config.output_dim];

        let result = network.try_train_step(&input, &target, None, 0.001, &mut workspace);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ArkanError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_try_train_step_target_mismatch() {
        let config = KanConfig::preset();
        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let batch_size = 4;
        let input = vec![0.5f32; batch_size * config.input_dim];
        // Wrong target size
        let target = vec![0.1f32; batch_size * config.output_dim - 2];

        let result = network.try_train_step(&input, &target, None, 0.001, &mut workspace);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ArkanError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_try_train_step_mask_mismatch() {
        let config = KanConfig::preset();
        let mut network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let batch_size = 4;
        let input = vec![0.5f32; batch_size * config.input_dim];
        let target = vec![0.1f32; batch_size * config.output_dim];
        // Wrong mask size
        let mask = vec![1.0f32; batch_size * config.output_dim + 1];

        let result = network.try_train_step(&input, &target, Some(&mask), 0.001, &mut workspace);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ArkanError::ShapeMismatch { .. }));
    }

    // ==================== Edge-case tests ====================

    #[test]
    fn test_batch_size_zero() {
        // Empty batch should work without panic
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input: Vec<f32> = vec![];
        let mut output: Vec<f32> = vec![];

        // Should not panic
        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.is_empty());
    }

    #[test]
    fn test_batch_size_one() {
        // Single sample batch
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input = vec![0.5f32; config.input_dim];
        let mut output = vec![0.0f32; config.output_dim];

        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_spline_order_2() {
        // Quadratic splines (order 2)
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8],
            grid_size: 5,
            spline_order: 2,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            ..Default::default()
        };

        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let input = vec![0.5f32; 4 * config.input_dim];
        let mut output = vec![0.0f32; 4 * config.output_dim];

        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_spline_order_4() {
        // Quartic splines (order 4)
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8],
            grid_size: 5,
            spline_order: 4,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            ..Default::default()
        };

        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(4);

        let input = vec![0.5f32; 4 * config.input_dim];
        let mut output = vec![0.0f32; 4 * config.output_dim];

        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_no_hidden_layers() {
        // Direct input -> output (single layer)
        let config = KanConfig {
            input_dim: 10,
            output_dim: 5,
            hidden_dims: vec![],
            grid_size: 5,
            spline_order: 3,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 10],
            input_std: vec![1.0; 10],
            ..Default::default()
        };

        let network = KanNetwork::new(config.clone());
        assert_eq!(network.num_layers(), 1);

        let mut workspace = network.create_workspace(2);
        let input = vec![0.5f32; 2 * config.input_dim];
        let mut output = vec![0.0f32; 2 * config.output_dim];

        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_deep_network() {
        // Many hidden layers
        let config = KanConfig {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8, 8, 8, 8, 8], // 5 hidden layers
            grid_size: 3,
            spline_order: 2,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            ..Default::default()
        };

        let network = KanNetwork::new(config.clone());
        assert_eq!(network.num_layers(), 6); // 5 hidden + 1 output

        let mut workspace = network.create_workspace(2);
        let input = vec![0.5f32; 2 * config.input_dim];
        let mut output = vec![0.0f32; 2 * config.output_dim];

        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_extreme_inputs() {
        // Very large/small inputs
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        // Large positive
        let input = vec![100.0f32; config.input_dim];
        let mut output = vec![0.0f32; config.output_dim];
        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));

        // Large negative
        let input = vec![-100.0f32; config.input_dim];
        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));

        // Near zero
        let input = vec![1e-10f32; config.input_dim];
        network.forward_batch(&input, &mut output, &mut workspace);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_workspace_reuse() {
        // Same workspace for different batch sizes
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(64);

        // Small batch
        let input1 = vec![0.5f32; 4 * config.input_dim];
        let mut output1 = vec![0.0f32; 4 * config.output_dim];
        network.forward_batch(&input1, &mut output1, &mut workspace);

        // Larger batch
        let input2 = vec![0.5f32; 32 * config.input_dim];
        let mut output2 = vec![0.0f32; 32 * config.output_dim];
        network.forward_batch(&input2, &mut output2, &mut workspace);

        // Same small batch again
        let mut output3 = vec![0.0f32; 4 * config.output_dim];
        network.forward_batch(&input1, &mut output3, &mut workspace);

        // Results should be reproducible
        assert_eq!(output1, output3);
    }
}
