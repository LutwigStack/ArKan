//! Borrowed training history, backward orchestration and training policy.

use crate::buffer::{checked_buffer_size, checked_buffer_size3, Workspace};
use crate::error::{ArkanError, ArkanResult};
use crate::model::ModelLayout;
use crate::network::KanNetwork;
use crate::optimizer::Optimizer;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub(crate) mod gradients;

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

impl TrainOptions {
    pub(crate) fn validate(&self) -> ArkanResult<()> {
        if self
            .max_grad_norm
            .is_some_and(|norm| !norm.is_finite() || norm <= 0.0)
            || !self.weight_decay.is_finite()
            || self.weight_decay < 0.0
        {
            return Err(ArkanError::cpu("Invalid training options"));
        }
        Ok(())
    }
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
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow or shape validation fails.
    /// Use [`try_train_step_with_options`](Self::try_train_step_with_options)
    /// for a fallible version.
    pub fn train_step_with_options(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        learning_rate: f32,
        workspace: &mut Workspace,
        opts: &TrainOptions,
    ) -> f32 {
        self.try_train_step_with_options(input, target, mask, learning_rate, workspace, opts)
            .expect("train_step_with_options failed")
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
    /// All internal operations use fallible versions with proper overflow checking.
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
    /// Returns `ArkanError::Overflow` if buffer size calculations overflow.
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
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(ArkanError::cpu(
                "Learning rate must be finite and nonnegative",
            ));
        }
        opts.validate()?;
        let loss = self.try_forward_backward_mse(input, target, mask, workspace)?;

        if input.is_empty() {
            return Ok(loss);
        }

        gradients::clip_in_place(
            &mut workspace.weight_grads,
            &mut workspace.bias_grads,
            opts.max_grad_norm,
        );

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
        for (i, layer) in self.try_parameters_mut()?.iter_mut().enumerate() {
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

        Ok(loss)
    }

    /// One-call training step with a standalone [`Optimizer`] (e.g. Adam, SGD).
    ///
    /// Runs forward(training) → MSE backward → `optimizer.step` (unscale, then optional clipping).
    /// This lets you use any optimizer that implements the [`Optimizer`] trait without
    /// manually writing the forward/backward loop.
    ///
    /// # Loss
    ///
    /// MSE loss only. For custom losses use [`Self::try_forward_for_backward`]
    /// and [`ForwardPass::backward`], then `optimizer.step`, or the GPU
    /// `GpuNetwork::train_step_cross_entropy` (requires the `gpu` feature).
    ///
    /// # Weight decay
    ///
    /// `opts.weight_decay` is NOT applied here; pass it to your optimizer's own
    /// config (e.g. `AdamConfig::weight_decay`) to avoid double-counting.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data `[batch_size * input_dim]`
    /// * `target` - Target values `[batch_size * output_dim]`
    /// * `mask` - Optional mask `[batch_size * output_dim]` (1.0 = active, 0.0 = ignore)
    /// * `workspace` - Pre-allocated workspace (must cover the batch size)
    /// * `optimizer` - Any mutable optimizer implementing [`Optimizer`]
    /// * `opts` - Training options; only `max_grad_norm` is used (see weight_decay note above)
    ///
    /// # Returns
    ///
    /// MSE loss for this batch, or an error if shapes are mismatched.
    ///
    /// # Errors
    ///
    /// Returns `ArkanError::ShapeMismatch` if input/target/mask lengths don't match.
    /// Returns `ArkanError::Overflow` if buffer size calculations overflow.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork, TrainOptions, Adam, AdamConfig};
    ///
    /// let config = KanConfig::preset();
    /// let mut network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(64);
    /// let mut adam = Adam::new(&network, AdamConfig::default());
    ///
    /// let inputs = vec![0.5f32; 64 * config.input_dim];
    /// let targets = vec![0.1f32; 64 * config.output_dim];
    ///
    /// let loss = network.train_step_with_optimizer(
    ///     &inputs, &targets, None, &mut workspace, &mut adam, &TrainOptions::default(),
    /// )?;
    /// # Ok::<(), arkan::ArkanError>(())
    /// ```
    #[must_use = "this returns a Result that should be handled"]
    pub fn train_step_with_optimizer(
        &mut self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        workspace: &mut Workspace,
        optimizer: &mut impl Optimizer,
        opts: &TrainOptions,
    ) -> ArkanResult<f32> {
        opts.validate()?;
        let loss = self.try_forward_backward_mse(input, target, mask, workspace)?;
        if !input.is_empty() {
            optimizer.step(
                self,
                &workspace.weight_grads,
                &workspace.bias_grads,
                opts.max_grad_norm,
            )?;
        }
        Ok(loss)
    }

    /// Forward and MSE backward without clipping, decay or parameter updates.
    fn try_forward_backward_mse(
        &self,
        input: &[f32],
        target: &[f32],
        mask: Option<&[f32]>,
        workspace: &mut Workspace,
    ) -> ArkanResult<f32> {
        self.checked_layout()?;
        let batch_size = input.len() / self.config.input_dim;
        let output_dim = self.config.output_dim;

        // Validate input length
        let expected_input_len = checked_buffer_size(batch_size, self.config.input_dim)?;
        if input.len() != expected_input_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_input_len],
                &[input.len()],
            ));
        }

        // Validate target length
        let expected_target_len = checked_buffer_size(batch_size, output_dim)?;
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

        if batch_size == 0 {
            workspace.try_prepare_grad_buffers(self.layout().parameter_sizes())?;
            workspace.zero_grads();
            workspace.try_prepare_training(0, &self.config, self.layout().layer_dims())?;
            return Ok(0.0);
        }

        // Ensure workspace has gradient buffers for all layers
        workspace.try_prepare_grad_buffers(self.layout().parameter_sizes())?;

        // Forward pass with history capture using workspace predictions buffer
        let pred_size = checked_buffer_size(batch_size, output_dim)?;
        workspace.predictions_buffer.try_resize(pred_size)?;

        workspace.grad_output.try_resize(pred_size)?;
        // Take predictions buffer to avoid borrow conflict, restoring it on failure.
        let mut predictions_buf = std::mem::take(&mut workspace.predictions_buffer);
        if let Err(error) =
            self.try_forward_batch_training(input, predictions_buf.as_mut_slice(), workspace)
        {
            workspace.predictions_buffer = predictions_buf;
            return Err(error);
        }

        // Compute loss and output gradient into workspace buffer
        let loss = crate::loss::masked_mse_into(
            predictions_buf.as_slice(),
            target,
            mask,
            workspace.grad_output.as_mut_slice(),
        );
        // Return predictions buffer
        workspace.predictions_buffer = predictions_buf;
        let loss = loss?;

        let grad_output = std::mem::take(&mut workspace.grad_output);
        let result = backward_into(
            self,
            self.layout(),
            batch_size,
            grad_output.as_slice(),
            workspace,
        );
        workspace.grad_output = grad_output;
        result?;

        Ok(loss)
    }
}

/// A completed forward pass that exclusively owns access to its saved history.
///
/// Parameters and normalization cannot change before this pass is consumed:
///
/// ```compile_fail
/// use arkan::{KanConfig, KanNetwork};
/// let mut network = KanNetwork::new(KanConfig::preset());
/// let mut workspace = network.create_workspace(1);
/// let input = vec![0.0; network.config.input_dim];
/// let mut output = vec![0.0; network.config.output_dim];
/// let pass = network.try_forward_for_backward(&input, &mut output, &mut workspace).unwrap();
/// network.layers[0].weights[0] = 0.0;
/// pass.backward(&output).unwrap();
/// ```
///
/// The workspace also remains exclusively borrowed until backward:
///
/// ```compile_fail
/// use arkan::{KanConfig, KanNetwork};
/// let network = KanNetwork::new(KanConfig::preset());
/// let mut workspace = network.create_workspace(1);
/// let input = vec![0.0; network.config.input_dim];
/// let mut output = vec![0.0; network.config.output_dim];
/// let pass = network.try_forward_for_backward(&input, &mut output, &mut workspace).unwrap();
/// workspace.layers_inputs.clear();
/// pass.backward(&output).unwrap();
/// ```
#[must_use = "consume the pass with backward to compute parameter gradients"]
pub struct ForwardPass<'model, 'work> {
    model: &'model KanNetwork,
    layout: &'model ModelLayout,
    workspace: &'work mut Workspace,
    batch_size: usize,
}

/// Parameter gradients borrowed from the workspace after backward.
/// The model borrow has ended, so these slices can be passed to an optimizer.
pub struct Gradients<'work> {
    /// Per-layer weight gradients in the model's coefficient order.
    pub weights: &'work [Vec<f32>],
    /// Per-layer bias gradients.
    pub biases: &'work [Vec<f32>],
}

impl KanNetwork {
    /// Run a training forward pass and retain exclusive access to its history.
    ///
    /// `output` remains available to compute any loss and its output derivative.
    /// The model and workspace remain borrowed until the returned pass is consumed.
    ///
    /// ```
    /// use arkan::{KanConfig, KanNetwork, SGD, SGDConfig, Optimizer};
    /// use arkan::loss::masked_bce_with_logits;
    /// let mut network = KanNetwork::new(KanConfig::preset());
    /// let mut workspace = network.create_workspace(1);
    /// let mut optimizer = SGD::new(&network, SGDConfig::with_lr(0.01));
    /// let input = vec![0.0; network.config.input_dim];
    /// let target = vec![1.0; network.config.output_dim];
    /// let mut output = vec![0.0; target.len()];
    /// let pass = network.try_forward_for_backward(&input, &mut output, &mut workspace)?;
    /// let (_loss, derivative) = masked_bce_with_logits(&output, &target, None);
    /// let gradients = pass.backward(&derivative)?;
    /// optimizer.step(&mut network, gradients.weights, gradients.biases, Some(1.0))?;
    /// # Ok::<(), arkan::ArkanError>(())
    /// ```
    pub fn try_forward_for_backward<'model, 'work>(
        &'model self,
        input: &[f32],
        output: &mut [f32],
        workspace: &'work mut Workspace,
    ) -> ArkanResult<ForwardPass<'model, 'work>> {
        self.try_forward_batch_training(input, output, workspace)?;
        Ok(ForwardPass {
            model: self,
            layout: self.layout(),
            workspace,
            batch_size: input.len() / self.config.input_dim,
        })
    }
}

impl<'work> ForwardPass<'_, 'work> {
    /// Compute parameter gradients for the saved forward pass, without updates
    /// or clipping. Shape errors leave parameter gradients unchanged.
    pub fn backward(self, grad_output: &[f32]) -> ArkanResult<Gradients<'work>> {
        backward_into(
            self.model,
            self.layout,
            self.batch_size,
            grad_output,
            self.workspace,
        )?;
        Ok(Gradients {
            weights: &self.workspace.weight_grads,
            biases: &self.workspace.bias_grads,
        })
    }
}

/// The sole CPU reverse-layer orchestration, shared by custom losses and MSE.
fn backward_into(
    model: &KanNetwork,
    layout: &ModelLayout,
    batch_size: usize,
    grad_output: &[f32],
    workspace: &mut Workspace,
) -> ArkanResult<()> {
    let expected = checked_buffer_size(batch_size, model.config.output_dim)?;
    if grad_output.len() != expected {
        return Err(ArkanError::shape_mismatch(
            &[expected],
            &[grad_output.len()],
        ));
    }
    workspace.check_history_batch(batch_size)?;
    if workspace.layers_inputs.len() != model.layers.len()
        || workspace.layers_grid_indices.len() != model.layers.len()
    {
        return Err(ArkanError::cpu("Training history layer count mismatch"));
    }
    for (index, layer) in model.layers.iter().enumerate() {
        let expected = checked_buffer_size(batch_size, layer.in_dim)?;
        if workspace.layers_inputs[index].len() != expected
            || workspace.layers_grid_indices[index].len() != expected
        {
            return Err(ArkanError::cpu("Training history layer extent mismatch"));
        }
    }
    // Complete all fallible preparation before removing any workspace buffers.
    let max_dim = layout.layer_dims().iter().copied().max().unwrap_or(0);
    let max_extent = checked_buffer_size(batch_size, max_dim)?;
    let max_basis = checked_buffer_size3(batch_size, max_dim, model.config.basis_size_aligned())?;
    workspace.try_prepare_grad_buffers(layout.parameter_sizes())?;
    workspace.staging_buffer.try_resize(max_extent)?;
    workspace.layer_grads.try_resize(max_extent)?;
    workspace.basis_values.try_resize(max_basis)?;
    workspace.basis_derivs.try_resize(max_basis)?;
    let use_parallel =
        cfg!(feature = "parallel") && batch_size >= model.config.multithreading_threshold;
    #[cfg(feature = "parallel")]
    if use_parallel && batch_size > 0 {
        for layer in &model.layers {
            workspace.parallel_backward_scratch.prepare(layer);
        }
    }
    workspace.zero_grads();
    if batch_size == 0 {
        return Ok(());
    }
    workspace.staging_buffer.as_mut_slice()[..expected].copy_from_slice(grad_output);
    for layer_idx in (0..model.layers.len()).rev() {
        let layer = &model.layers[layer_idx];
        // Products are bounded by max_extent above. No fallible operation follows
        // a take; every buffer is restored before processing another layer.
        let in_size = batch_size * layer.in_dim;
        let out_size = batch_size * layer.out_dim;
        let staging = std::mem::take(&mut workspace.staging_buffer);
        let mut weight_grad = std::mem::take(&mut workspace.weight_grads[layer_idx]);
        let mut bias_grad = std::mem::take(&mut workspace.bias_grads[layer_idx]);
        let mut grad_buffer = (layer_idx > 0).then(|| std::mem::take(&mut workspace.layer_grads));
        let saved_input = std::mem::take(&mut workspace.layers_inputs[layer_idx]);
        let saved_spans = std::mem::take(&mut workspace.layers_grid_indices[layer_idx]);
        if use_parallel {
            #[cfg(feature = "parallel")]
            layer.backward_parallel_with_scratch(
                saved_input.as_slice(),
                &saved_spans,
                &staging.as_slice()[..out_size],
                grad_buffer
                    .as_mut()
                    .map(|buffer| &mut buffer.as_mut_slice()[..in_size]),
                &mut weight_grad,
                &mut bias_grad,
                &mut workspace.parallel_backward_scratch,
            );
        } else {
            layer.backward(
                saved_input.as_slice(),
                &saved_spans,
                &staging.as_slice()[..out_size],
                grad_buffer
                    .as_mut()
                    .map(|buffer| &mut buffer.as_mut_slice()[..in_size]),
                &mut weight_grad,
                &mut bias_grad,
                workspace,
            );
        }
        workspace.layers_inputs[layer_idx] = saved_input;
        workspace.layers_grid_indices[layer_idx] = saved_spans;
        workspace.weight_grads[layer_idx] = weight_grad;
        workspace.bias_grads[layer_idx] = bias_grad;
        workspace.staging_buffer = staging;
        if let Some(buffer) = grad_buffer {
            workspace.staging_buffer.as_mut_slice()[..in_size]
                .copy_from_slice(&buffer.as_slice()[..in_size]);
            workspace.layer_grads = buffer;
        }
    }
    Ok(())
}
