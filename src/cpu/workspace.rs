//! Reusable CPU execution buffers and history.

use crate::config::KanConfig;
use crate::error::{ArkanError, ArkanResult};
use crate::memory::{
    checked_buffer_size, checked_buffer_size3, AlignedBuffer, MAX_BUFFER_ELEMENTS,
};
use crate::model::KanNetwork;

/// Preallocated workspace for zero-allocation forward/backward passes.
///
/// The workspace holds all intermediate buffers needed during inference
/// and training. By preallocating these buffers, ArKan avoids heap
/// allocations in the hot path.
///
/// # Buffer Preparation
///
/// Before using the workspace, call the appropriate preparation method:
///
/// | Method | Use Case |
/// |--------|----------|
/// | [`reserve`](Self::reserve) | Ensure capacity for batch size |
/// | [`prepare_forward`](Self::prepare_forward) | Inference: resize z_buffer, basis_values, etc. |
/// | [`prepare_training`](Self::prepare_training) | Training: add history buffers for backward pass |
/// | [`prepare_grad_buffers`](Self::prepare_grad_buffers) | Training: allocate per-layer gradient vectors |
///
/// Typical flow:
/// ```text
/// network.create_workspace(batch_size)   // calls reserve() internally
///   \u2514\u2500> Inference: forward_batch()         // calls prepare_forward() automatically
///   \u2514\u2500> Training:  train_step()            // calls prepare_training() + prepare_grad_buffers()
/// ```
///
/// # Buffer Categories
///
/// **Forward buffers** (used during inference and training):
/// - `z_buffer`: Normalized inputs `[batch, input_dim]`
/// - `basis_values`: B-spline basis values `[batch, input_dim, basis_size]`
/// - `layer_output`, `layer_input`: Ping-pong buffers for layer activations
///
/// **Backward buffers** (training only):
/// - `layers_inputs`: Saved normalized inputs per layer for gradient computation
/// - `layers_grid_indices`: Saved spline segment indices per layer
/// - `staging_buffer`: Current layer's output gradient (ping-pong with `layer_grads`)
/// - `layer_grads`: Accumulated input gradients during backprop
///
/// **Gradient buffers** (training only):
/// - `weight_grads`: Per-layer weight gradients `[layer][weights.len()]`
/// - `bias_grads`: Per-layer bias gradients `[layer][bias.len()]`
/// - `grad_output`: Initial output gradient (dL/dy)
/// - `predictions_buffer`: Forward pass outputs before loss computation
///
/// # Usage
///
/// Create a workspace using [`KanNetwork::create_workspace`](crate::KanNetwork::create_workspace):
///
/// ```rust
/// use arkan::{KanConfig, KanNetwork};
///
/// let network = KanNetwork::new(KanConfig::preset());
/// let mut workspace = network.create_workspace(64);
///
/// // Reuse workspace for all calls
/// ```
///
/// # Thread Safety
///
/// Workspaces are NOT thread-safe. Each thread should have its own workspace.
/// The network itself can be shared (it's read-only during inference).
#[derive(Default)]
pub struct Workspace {
    #[cfg(feature = "parallel")]
    pub(crate) parallel_backward_scratch: crate::cpu::layer::ParallelBackwardScratch,
    /// Normalized inputs: `[Batch, Input]`
    pub z_buffer: AlignedBuffer,

    /// Basis function values: `[Batch, Input, Basis]`
    pub basis_values: AlignedBuffer,

    /// Basis function derivatives: `[Batch, Input, Basis]`
    pub basis_derivs: AlignedBuffer,

    /// Grid indices: `[Batch, Input]`
    ///
    /// The high bit carries [`crate::SPAN_CLAMPED_FLAG`] when the forward pass
    /// clamped that input to the grid range; mask with
    /// [`crate::SPAN_INDEX_MASK`] before using an entry as a span index.
    pub grid_indices: Vec<u32>,

    /// Intermediate layer outputs: `[Batch, MaxHiddenDim]`
    pub layer_output: AlignedBuffer,

    /// Previous layer output (for multi-layer): `[Batch, MaxHiddenDim]`
    pub layer_input: AlignedBuffer,

    // --- Backward pass history ---
    /// Saved normalized inputs per layer: `[Layer][Batch * in_dim_layer]`.
    ///
    /// Recorded during `forward_batch_training` and used in backward pass
    /// to recompute B-spline basis derivatives. Each buffer holds the
    /// z-normalized input values for one layer.
    pub layers_inputs: Vec<AlignedBuffer>,

    /// Saved grid indices per layer: `[Layer][Batch * in_dim_layer]`.
    ///
    /// The spline segment index for each input, recorded during forward.
    /// Used in backward to index into the correct spline weights.
    ///
    /// As with [`Workspace::grid_indices`], the high bit is
    /// [`crate::SPAN_CLAMPED_FLAG`] and must be masked off with
    /// [`crate::SPAN_INDEX_MASK`] before use as an index.
    pub layers_grid_indices: Vec<Vec<u32>>,

    /// Gradient buffer passed between layers during backprop: `[Batch, MaxDim]`.
    ///
    /// During backward pass, this buffer accumulates dL/d(layer_input) which
    /// becomes dL/d(layer_output) for the previous layer.
    pub layer_grads: AlignedBuffer,

    /// Staging buffer for ping-pong gradient propagation: `[Batch, MaxDim]`.
    ///
    /// Holds the current layer's output gradient (dL/dy). After each layer's
    /// backward pass, `layer_grads` is copied here for the next iteration.
    /// This enables zero-allocation backward without double-borrow issues.
    pub staging_buffer: AlignedBuffer,

    /// Predictions buffer for train_step: `[Batch, OutputDim]`.
    ///
    /// Stores forward pass outputs before loss computation. Used with
    /// `std::mem::take` to avoid borrow conflicts with workspace during
    /// forward_batch_training.
    pub predictions_buffer: AlignedBuffer,

    /// Weight gradients per layer: `[Layer][weights.len()]`.
    ///
    /// Accumulated during backward pass. Prepared by `prepare_grad_buffers()`
    /// with sizes matching each layer's weight count.
    pub weight_grads: Vec<Vec<f32>>,

    /// Bias gradients per layer: `[Layer][bias.len()]`.
    ///
    /// Accumulated during backward pass. Prepared by `prepare_grad_buffers()`
    /// with sizes matching each layer's bias count.
    pub bias_grads: Vec<Vec<f32>>,

    /// Output gradient buffer for backprop: `[Batch * output_dim]`.
    ///
    /// Initial gradient dL/d(network_output), computed from loss function.
    /// Seeded by `compute_masked_mse_loss_into()` and propagated backward.
    pub grad_output: AlignedBuffer,

    /// Tracking ---
    /// Current batch capacity
    batch_capacity: usize,

    /// Max dimension across all layers
    max_dim: usize,

    /// Batch size of the last recorded history (for assertions)
    history_batch_size: usize,
}

impl Workspace {
    /// Creates a new workspace for the given config, with no batch capacity yet.
    ///
    /// Call [`reserve`](Self::reserve) - or go through
    /// [`KanNetwork::create_workspace`](crate::KanNetwork::create_workspace), which
    /// does it for you - to size it for a batch. The forward and training paths
    /// also reserve on demand, so the only cost of skipping it is one allocation on
    /// the first call.
    ///
    /// This used to pre-reserve `config.multithreading_threshold` rows, which was
    /// wrong twice over. That field is documented as consulted only with the
    /// `parallel` feature, yet it sized every workspace on every build; and it is
    /// caller-controlled and unbounded, so `multithreading_threshold = 1 << 30` -
    /// the natural way to say "never go parallel" - made `try_create_workspace`
    /// *panic* out of a `Result`-returning function before its own `try_reserve`
    /// could report the overflow. At the default of 128 it silently gave a
    /// `create_workspace(1)` caller 128 rows of buffers.
    pub fn new(config: &KanConfig) -> Self {
        // Nothing here is sized from the config any more. The parameter stays for
        // API compatibility, and because every `reserve` call needs the same one.
        let _ = config;
        Self {
            #[cfg(feature = "parallel")]
            parallel_backward_scratch: crate::cpu::layer::ParallelBackwardScratch::default(),
            z_buffer: AlignedBuffer::new(),
            basis_values: AlignedBuffer::new(),
            basis_derivs: AlignedBuffer::new(),
            grid_indices: Vec::new(),
            layer_output: AlignedBuffer::new(),
            layer_input: AlignedBuffer::new(),
            layers_inputs: Vec::new(),
            layers_grid_indices: Vec::new(),
            layer_grads: AlignedBuffer::new(),
            staging_buffer: AlignedBuffer::new(),
            predictions_buffer: AlignedBuffer::new(),
            weight_grads: Vec::new(),
            bias_grads: Vec::new(),
            grad_output: AlignedBuffer::new(),
            batch_capacity: 0,
            max_dim: 0,
            history_batch_size: 0,
        }
    }

    fn checked_layout(config: &KanConfig) -> ArkanResult<(usize, usize)> {
        if config.input_dim == 0
            || config.output_dim == 0
            || config.hidden_dims.contains(&0)
            || config.grid_size == 0
            || config.grid_size > crate::config::MAX_GRID_SIZE
            || config.spline_order == 0
            || config.spline_order > crate::config::MAX_SPLINE_ORDER
            || !matches!(config.simd_width, 4 | 8 | 16)
        {
            return Err(ArkanError::cpu("Invalid workspace configuration"));
        }
        let max_dim = config
            .hidden_dims
            .iter()
            .copied()
            .chain([config.input_dim, config.output_dim])
            .max()
            .unwrap();
        checked_buffer_size(max_dim, 1)?;
        Ok((max_dim, config.basis_size_aligned()))
    }

    /// Ensures workspace has capacity for the given batch size (fallible version).
    ///
    /// This is the checked version that returns an error on overflow instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Overflow`] if any buffer size calculation overflows.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_reserve(&mut self, batch_size: usize, config: &KanConfig) -> ArkanResult<()> {
        self.try_reserve_buffers(batch_size, config, true)
    }

    fn try_reserve_buffers(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
        reserve_predictions: bool,
    ) -> ArkanResult<()> {
        let (max_dim, basis) = Self::checked_layout(config)?;
        let output_dim = config.output_dim;

        // Check all size calculations for overflow using checked_buffer_size
        // z_buffer: [batch, max_dim] - needs max_dim for hidden layers wider than input
        let z_size = checked_buffer_size(batch_size, max_dim)?;

        // basis_values: [batch, max_dim, basis] - needs max_dim for hidden layers
        let basis_size = checked_buffer_size3(batch_size, max_dim, basis)?;

        // layer buffers: [batch, max_dim]
        let layer_size = checked_buffer_size(batch_size, max_dim)?;

        // predictions_buffer: [batch, output_dim]
        let pred_size = checked_buffer_size(batch_size, output_dim)?;

        // All checks passed, now allocate through checked fallible paths.
        self.z_buffer.try_reserve(z_size)?;
        self.basis_values.try_reserve(basis_size)?;
        self.basis_derivs.try_reserve(basis_size)?;

        if self.grid_indices.len() < z_size {
            self.grid_indices
                .try_reserve(z_size - self.grid_indices.len())
                .map_err(|_| ArkanError::cpu("Grid index allocation failed"))?;
            self.grid_indices.resize(z_size, 0);
        }
        self.layer_output.try_reserve(layer_size)?;
        self.layer_input.try_reserve(layer_size)?;
        self.layer_grads.try_reserve(layer_size)?;
        self.staging_buffer.try_reserve(layer_size)?;
        if reserve_predictions {
            self.predictions_buffer.try_reserve(pred_size)?;
        }
        self.grad_output.try_reserve(pred_size)?;

        self.batch_capacity = self.batch_capacity.max(batch_size);
        self.max_dim = max_dim;

        Ok(())
    }

    /// Ensures workspace has capacity for the given batch size.
    /// Only allocates if batch_size > current capacity.
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_reserve`](Self::try_reserve)
    /// for a fallible version.
    #[inline]
    pub fn reserve(&mut self, batch_size: usize, config: &KanConfig) {
        self.try_reserve(batch_size, config)
            .expect("Workspace::reserve: buffer size overflow")
    }

    /// Prepares workspace for a forward pass with the given batch size (fallible version).
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Overflow`] if any buffer size calculation overflows.
    #[inline]
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_prepare_forward(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
    ) -> ArkanResult<()> {
        self.try_reserve(batch_size, config)?;
        let (max_dim, basis) = Self::checked_layout(config)?;

        // Use checked arithmetic with max_dim
        let z_size = checked_buffer_size(batch_size, max_dim)?;
        let basis_size = checked_buffer_size3(batch_size, max_dim, basis)?;

        self.z_buffer.try_resize(z_size)?;
        self.basis_values.try_resize(basis_size)?;
        self.basis_derivs.try_resize(basis_size)?;
        self.grid_indices.resize(z_size, 0);

        Ok(())
    }

    /// Prepares workspace for a forward pass with the given batch size.
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_prepare_forward`](Self::try_prepare_forward)
    /// for a fallible version.
    #[inline]
    pub fn prepare_forward(&mut self, batch_size: usize, config: &KanConfig) {
        self.try_prepare_forward(batch_size, config)
            .expect("Workspace::prepare_forward: buffer size overflow")
    }

    /// Prepares workspace for training with history tracking (fallible version).
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Overflow`] if any buffer size calculation overflows.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_prepare_training(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
        layer_dims: &[usize],
    ) -> ArkanResult<()> {
        if !layer_dims
            .iter()
            .copied()
            .eq(std::iter::once(config.input_dim)
                .chain(config.hidden_dims.iter().copied())
                .chain(std::iter::once(config.output_dim)))
        {
            return Err(ArkanError::cpu(
                "Training history layout does not match configuration",
            ));
        }
        // Predictions may be temporarily owned by the training caller.
        self.try_reserve_buffers(batch_size, config, false)?;

        let num_layers = layer_dims.len().saturating_sub(1);
        if self.layers_inputs.len() < num_layers {
            self.layers_inputs
                .try_reserve(num_layers - self.layers_inputs.len())
                .map_err(|_| ArkanError::cpu("History allocation failed"))?;
            self.layers_inputs
                .resize_with(num_layers, AlignedBuffer::new);
        }
        if self.layers_grid_indices.len() < num_layers {
            self.layers_grid_indices
                .try_reserve(num_layers - self.layers_grid_indices.len())
                .map_err(|_| ArkanError::cpu("History index allocation failed"))?;
            self.layers_grid_indices.resize_with(num_layers, Vec::new);
        }

        self.layers_inputs.truncate(num_layers);
        self.layers_grid_indices.truncate(num_layers);

        let basis = config.basis_size_aligned();
        let max_in_dim = *layer_dims.iter().max().unwrap_or(&config.input_dim);

        // Ensure history buffers sized per layer with overflow checks
        for (layer_idx, in_dim) in layer_dims.iter().copied().enumerate().take(num_layers) {
            let needed = checked_buffer_size(batch_size, in_dim)?;

            let buf = &mut self.layers_inputs[layer_idx];
            buf.try_reserve(needed)?;
            buf.try_resize(needed)?;

            let indices = &mut self.layers_grid_indices[layer_idx];
            if indices.len() < needed {
                indices
                    .try_reserve(needed - indices.len())
                    .map_err(|_| ArkanError::cpu("History index allocation failed"))?;
                indices.resize(needed, 0);
            } else {
                indices.truncate(needed);
            }
        }

        // Gradient ping-pong buffer
        let grad_size = checked_buffer_size(batch_size, max_in_dim)?;
        self.layer_grads.try_reserve(grad_size)?;
        self.layer_grads.try_resize(grad_size)?;

        // Derivatives buffer (same layout as basis_values)
        let deriv_size = checked_buffer_size3(batch_size, max_in_dim, basis)?;
        self.basis_derivs.try_reserve(deriv_size)?;
        self.basis_derivs.try_resize(deriv_size)?;

        // Training ping-pong buffers.
        //
        // `predictions_buffer` is deliberately NOT sized here. Its only consumer,
        // `KanNetwork::try_forward_backward_mse`, sizes it itself and then
        // `std::mem::take`s it to dodge a borrow conflict before calling into the
        // forward pass — which lands here. Re-reserving it at that point allocates a
        // fresh buffer for the field the caller just emptied, and the caller then
        // overwrites it with the original on the way out, throwing the new one away.
        // That was one heap allocation on every single training step, which is what
        // the "zero-allocation training" claim tripped over. Pinned by
        // tests/allocation_budget.rs.
        let output_dim = config.output_dim;
        let output_size = checked_buffer_size(batch_size, output_dim)?;

        self.grad_output.try_reserve(output_size)?;
        self.grad_output.try_resize(output_size)?;

        self.history_batch_size = batch_size;

        Ok(())
    }

    /// Prepares workspace for training with history tracking.
    ///
    /// Allocates/resizes buffers needed for backward pass:
    /// - `layers_inputs`: one buffer per layer for saved normalized inputs
    /// - `layers_grid_indices`: one vec per layer for saved spline indices
    /// - `layer_grads`, `staging_buffer`: ping-pong gradient buffers
    /// - `basis_derivs`: B-spline derivative values
    /// - `grad_output`: loss gradient buffer
    ///
    /// Note: `predictions_buffer` is NOT sized here — its consumer sizes and owns it.
    /// See the comment at the call site below.
    ///
    /// # Zero-Allocation Guarantee
    ///
    /// After the first call with a given batch size, subsequent calls with
    /// the same or smaller batch size perform zero allocations. Buffers
    /// grow monotonically and are reused.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples in the batch
    /// * `config` - Network configuration
    /// * `layer_dims` - Dimensions of all layers `[input, hidden..., output]`
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_prepare_training`](Self::try_prepare_training)
    /// for a fallible version.
    #[inline]
    pub fn prepare_training(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
        layer_dims: &[usize],
    ) {
        self.try_prepare_training(batch_size, config, layer_dims)
            .expect("Workspace::prepare_training: buffer size overflow")
    }

    /// Prepares gradient buffers for training with layer sizes.
    ///
    /// Allocates `weight_grads` and `bias_grads` vectors with correct sizes
    /// for each layer. These buffers accumulate gradients during backward pass.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` - Tuples of `(weight_count, bias_count)` per layer.
    ///   Obtained from `KanNetwork::layer_param_sizes`.
    ///
    /// # Zero-Allocation Note
    ///
    /// Buffer capacities are retained when logical lengths shrink. With the same
    /// layer count, repeated shapes within those capacities perform zero allocations.
    /// Removing layers releases their buffers; adding them again may allocate.
    ///
    /// # Panics
    ///
    /// May panic on allocation failure. Use [`try_prepare_grad_buffers`](Self::try_prepare_grad_buffers)
    /// for a fallible version.
    #[inline]
    pub fn prepare_grad_buffers(&mut self, layer_sizes: &[(usize, usize)]) {
        self.try_prepare_grad_buffers(layer_sizes)
            .expect("Workspace::prepare_grad_buffers: allocation failed")
    }

    /// Fallible version of [`prepare_grad_buffers`](Self::prepare_grad_buffers).
    ///
    /// Returns `Ok(())` on success, or `ArkanError::Overflow` if buffer size
    /// calculations overflow or exceed `MAX_BUFFER_ELEMENTS`.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` - Tuples of `(weight_count, bias_count)` per layer.
    #[inline]
    pub fn try_prepare_grad_buffers(&mut self, layer_sizes: &[(usize, usize)]) -> ArkanResult<()> {
        let num_layers = layer_sizes.len();

        // Validate sizes won't overflow
        for (w_size, b_size) in layer_sizes {
            if *w_size > MAX_BUFFER_ELEMENTS {
                return Err(ArkanError::overflow(format!(
                    "Weight gradient size {} exceeds MAX_BUFFER_ELEMENTS ({})",
                    w_size, MAX_BUFFER_ELEMENTS
                )));
            }
            if *b_size > MAX_BUFFER_ELEMENTS {
                return Err(ArkanError::overflow(format!(
                    "Bias gradient size {} exceeds MAX_BUFFER_ELEMENTS ({})",
                    b_size, MAX_BUFFER_ELEMENTS
                )));
            }
        }

        // Resize gradient vectors if needed
        if self.weight_grads.len() < num_layers {
            self.weight_grads
                .try_reserve(num_layers - self.weight_grads.len())
                .map_err(|_| ArkanError::cpu("Weight gradient allocation failed"))?;
            self.weight_grads.resize_with(num_layers, Vec::new);
        }
        if self.bias_grads.len() < num_layers {
            self.bias_grads
                .try_reserve(num_layers - self.bias_grads.len())
                .map_err(|_| ArkanError::cpu("Bias gradient allocation failed"))?;
            self.bias_grads.resize_with(num_layers, Vec::new);
        }

        self.weight_grads.truncate(num_layers);
        self.bias_grads.truncate(num_layers);

        // Retain capacity, but expose only the current logical shape.
        for (i, (w_size, b_size)) in layer_sizes.iter().enumerate() {
            let buf = &mut self.weight_grads[i];
            buf.try_reserve(w_size.saturating_sub(buf.len()))
                .map_err(|_| ArkanError::cpu("Weight gradient allocation failed"))?;
            buf.resize(*w_size, 0.0);
            let buf = &mut self.bias_grads[i];
            buf.try_reserve(b_size.saturating_sub(buf.len()))
                .map_err(|_| ArkanError::cpu("Bias gradient allocation failed"))?;
            buf.resize(*b_size, 0.0);
        }

        Ok(())
    }

    /// Zeros all gradient buffers in-place without reallocation.
    ///
    /// Call this at the start of each training step to clear gradients
    /// from the previous iteration. This is more efficient than reallocating
    /// the buffers.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork, Workspace};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(64);
    ///
    /// // After training step, zero grads for next iteration
    /// workspace.zero_grads();
    /// ```
    #[inline]
    pub fn zero_grads(&mut self) {
        for wg in &mut self.weight_grads {
            wg.fill(0.0);
        }
        for bg in &mut self.bias_grads {
            bg.fill(0.0);
        }
    }

    /// Current batch capacity.
    #[inline]
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }

    /// Batch size of the history recorded by the last training forward pass.
    ///
    /// Zero before any `forward_batch_training` / `train_step` call. This is the
    /// row count that [`crate::KanNetwork::clamped_fraction`] reads.
    #[inline]
    pub fn history_batch_size(&self) -> usize {
        self.history_batch_size
    }

    /// Checks that workspace history matches the expected batch size.
    ///
    /// Returns `Ok(())` if `history_batch_size == batch_size`, otherwise
    /// returns `ArkanError::ShapeMismatch`.
    #[inline]
    pub fn check_history_batch(&self, batch_size: usize) -> ArkanResult<()> {
        if self.history_batch_size != batch_size {
            return Err(ArkanError::shape_mismatch(
                &[batch_size],
                &[self.history_batch_size],
            ));
        }
        Ok(())
    }

    /// Checks workspace capacity, returning error if insufficient.
    #[inline]
    pub fn check_capacity(&self, batch_size: usize) -> ArkanResult<()> {
        if batch_size > self.batch_capacity {
            return Err(ArkanError::batch_too_large(batch_size, self.batch_capacity));
        }
        Ok(())
    }
}

/// RAII guard for workspace ping-pong buffers.
///
/// This guard ensures that buffers borrowed from a [`Workspace`] are returned
/// even if a panic occurs during computation. This provides basic exception
/// safety guarantee - the workspace remains valid (though possibly with
/// different buffer contents) after unwinding.
///
/// # Usage
///
/// ```rust
/// use arkan::{KanConfig, Workspace, WorkspaceGuard};
///
/// let config = KanConfig::preset();
/// let mut workspace = Workspace::new(&config);
/// workspace.reserve(64, &config);
///
/// {
///     let mut guard = WorkspaceGuard::new(&mut workspace);
///     let (buffer_a, buffer_b) = guard.buffers_mut();
///
///     // ... do computations ...
///     buffer_a.resize(100);
///
///     // Buffers are returned to workspace when guard is dropped
///     guard.finish();
/// }
///
/// // Workspace has the buffers back
/// assert!(workspace.layer_output.capacity() >= 100);
/// ```
///
/// # Panic Behavior
///
/// If a panic occurs while the guard is active:
/// - The `Drop` implementation will return the buffers to the workspace
/// - The workspace will be in a valid state (buffers have correct capacity)
/// - Buffer contents may be in an intermediate state
pub struct WorkspaceGuard<'a> {
    workspace: &'a mut Workspace,
    buffer_a: Option<AlignedBuffer>,
    buffer_b: Option<AlignedBuffer>,
}

impl<'a> WorkspaceGuard<'a> {
    /// Creates a new guard, taking ownership of the ping-pong buffers.
    #[inline]
    pub fn new(workspace: &'a mut Workspace) -> Self {
        let buffer_a = std::mem::take(&mut workspace.layer_output);
        let buffer_b = std::mem::take(&mut workspace.layer_input);
        Self {
            workspace,
            buffer_a: Some(buffer_a),
            buffer_b: Some(buffer_b),
        }
    }

    /// Returns mutable references to both buffers.
    #[inline]
    pub fn buffers_mut(&mut self) -> (&mut AlignedBuffer, &mut AlignedBuffer) {
        (
            self.buffer_a.as_mut().expect("buffer_a already taken"),
            self.buffer_b.as_mut().expect("buffer_b already taken"),
        )
    }

    /// Returns references to both buffers.
    #[inline]
    pub fn buffers(&self) -> (&AlignedBuffer, &AlignedBuffer) {
        (
            self.buffer_a.as_ref().expect("buffer_a already taken"),
            self.buffer_b.as_ref().expect("buffer_b already taken"),
        )
    }

    /// Explicitly returns buffers to workspace and consumes the guard.
    /// This is the normal completion path (no panic).
    #[inline]
    pub fn finish(mut self) {
        self.return_buffers();
    }

    fn return_buffers(&mut self) {
        if let Some(buf) = self.buffer_a.take() {
            self.workspace.layer_output = buf;
        }
        if let Some(buf) = self.buffer_b.take() {
            self.workspace.layer_input = buf;
        }
    }
}

impl<'a> Drop for WorkspaceGuard<'a> {
    fn drop(&mut self) {
        // Return any buffers that weren't explicitly taken
        self.return_buffers();
    }
}

impl KanNetwork {
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
        self.validate_layout()?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_reserve() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);

        // A fresh workspace holds no batch capacity. It used to pre-reserve
        // `multithreading_threshold` rows - see `Workspace::new`.
        assert_eq!(ws.batch_capacity(), 0);

        // Reserve more
        ws.reserve(1024, &config);
        assert!(ws.batch_capacity() >= 1024);

        // No allocation on smaller batch
        let old_cap = ws.batch_capacity();
        ws.reserve(512, &config);
        assert_eq!(ws.batch_capacity(), old_cap);
    }

    #[test]
    fn test_workspace_prepare_forward() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);

        ws.prepare_forward(64, &config);

        // After fix: workspace uses max_dim (64) for z_buffer and grid_indices
        let max_dim = *config.layer_dims().iter().max().unwrap();
        assert_eq!(ws.z_buffer.len(), 64 * max_dim);
        assert_eq!(ws.grid_indices.len(), 64 * max_dim);
    }

    #[test]
    fn test_workspace_guard_normal_flow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);

        // Get initial capacities
        let initial_output_cap = ws.layer_output.capacity();
        let initial_input_cap = ws.layer_input.capacity();

        {
            let mut guard = WorkspaceGuard::new(&mut ws);
            let (buf_a, buf_b) = guard.buffers_mut();

            // Buffers should have the original capacity
            assert_eq!(buf_a.capacity(), initial_output_cap);
            assert_eq!(buf_b.capacity(), initial_input_cap);

            // Modify buffers
            buf_a.resize(100);
            buf_b.resize(100);

            guard.finish();
        }

        // Buffers returned to workspace
        assert!(ws.layer_output.capacity() >= 100);
        assert!(ws.layer_input.capacity() >= 100);
    }

    #[test]
    fn test_workspace_guard_drop_returns_buffers() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);

        {
            let mut guard = WorkspaceGuard::new(&mut ws);
            let (buf_a, _buf_b) = guard.buffers_mut();
            buf_a.resize(200);
            // Guard dropped without calling finish()
        }

        // Buffers should still be returned
        assert!(ws.layer_output.capacity() >= 200);
        assert!(ws.layer_input.capacity() > 0);
    }

    #[test]
    fn test_workspace_check_capacity() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);

        let cap = ws.batch_capacity();

        // Within capacity - OK
        assert!(ws.check_capacity(32).is_ok());
        assert!(ws.check_capacity(cap).is_ok());

        // Exceeds capacity - Error
        assert!(ws.check_capacity(cap + 1).is_err());
    }

    #[test]
    fn test_try_reserve_success() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        assert!(ws.try_reserve(100, &config).is_ok());
    }

    #[test]
    fn test_try_reserve_overflow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        // Overflow: huge batch × reasonable dims
        let result = ws.try_reserve(usize::MAX / 2, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_prepare_forward_success() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);
        let result = ws.try_prepare_forward(32, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_prepare_forward_overflow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        // Overflow in size calculation
        let result = ws.try_prepare_forward(usize::MAX / 2, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_prepare_training_success() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);
        let dims = config.layer_dims();
        let result = ws.try_prepare_training(32, &config, &dims);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_prepare_training_overflow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        let dims = config.layer_dims();
        // Overflow in gradient size calculation
        let result = ws.try_prepare_training(usize::MAX / 4, &config, &dims);
        assert!(result.is_err());
    }
}
