//! CPU forward execution over a checked model.

use crate::buffer::checked_buffer_size;
use crate::cpu::Workspace;
use crate::error::{ArkanError, ArkanResult};
use crate::model::KanNetwork;
use crate::spline::SPAN_CLAMPED_FLAG;

impl KanNetwork {
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
        self.try_forward_single(input, output, workspace)
            .expect("forward_single failed");
    }

    /// Fallible single-sample forward pass.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_forward_single(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
    ) -> ArkanResult<()> {
        self.validate_layout()?;
        if input.len() != self.config.input_dim {
            return Err(ArkanError::shape_mismatch(
                &[self.config.input_dim],
                &[input.len()],
            ));
        }
        if output.len() != self.config.output_dim {
            return Err(ArkanError::shape_mismatch(
                &[self.config.output_dim],
                &[output.len()],
            ));
        }

        // Reserve workspace for batch_size=1
        workspace.try_reserve(1, &self.config)?;

        // Calculate max dimension for ping-pong buffers
        let max_dim = self
            .layout()
            .layer_dims()
            .iter()
            .copied()
            .max()
            .unwrap_or(1);
        workspace.layer_output.try_resize(max_dim)?;
        workspace.layer_input.try_resize(max_dim)?;

        // Get max basis_aligned across all layers
        let max_basis = self
            .layers
            .iter()
            .map(|l| l.basis_aligned)
            .max()
            .unwrap_or(8);
        workspace.basis_values.try_resize(max_basis)?;

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
                workspace.z_buffer.try_resize(in_size)?;
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
                workspace.z_buffer.try_resize(in_size)?;
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
        Ok(())
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
        self.try_forward_batch(input, output, workspace)
            .expect("forward_batch failed");
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
    /// Returns `ArkanError::Overflow` if buffer size calculations overflow.
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
        self.validate_layout()?;
        let batch_size = input.len() / self.config.input_dim;

        // Validate input length
        let expected_input_len = checked_buffer_size(batch_size, self.config.input_dim)?;
        if input.len() != expected_input_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_input_len],
                &[input.len()],
            ));
        }

        // Validate output length
        let expected_output_len = checked_buffer_size(batch_size, self.config.output_dim)?;
        if output.len() != expected_output_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_output_len],
                &[output.len()],
            ));
        }

        if batch_size == 0 {
            return Ok(());
        }

        workspace.try_reserve(batch_size, &self.config)?;

        if self.layers.len() == 1 {
            workspace.try_prepare_forward(batch_size, &self.config)?;
            self.layers[0].forward_batch(input, output, workspace);
            return Ok(());
        }

        let max_hidden = self
            .layout()
            .layer_dims()
            .iter()
            .copied()
            .max()
            .unwrap_or(self.config.input_dim);
        let ping_pong_size = checked_buffer_size(batch_size, max_hidden)?;

        // Use std::mem::take for borrow-safety (allows mutable refs to both buffers)
        let mut buffer_a = std::mem::take(&mut workspace.layer_output);
        let mut buffer_b = std::mem::take(&mut workspace.layer_input);

        // Wrap in a closure for early return with buffer restoration
        let result = (|| -> ArkanResult<()> {
            buffer_a.try_resize(ping_pong_size)?;
            buffer_b.try_resize(ping_pong_size)?;

            // First layer: input -> buffer_a
            {
                let layer = &self.layers[0];
                let out_size = checked_buffer_size(batch_size, layer.out_dim)?;
                buffer_a.try_resize(out_size)?;
                layer.forward_batch(input, &mut buffer_a.as_mut_slice()[..out_size], workspace);
            }

            let mut current_is_a = true;

            // Hidden layers: ping-pong between buffer_a/buffer_b
            for i in 1..self.layers.len() - 1 {
                let layer = &self.layers[i];
                let in_size = checked_buffer_size(batch_size, layer.in_dim)?;
                let out_size = checked_buffer_size(batch_size, layer.out_dim)?;

                let (input_buf, output_buf) = if current_is_a {
                    (&buffer_a, &mut buffer_b)
                } else {
                    (&buffer_b, &mut buffer_a)
                };

                output_buf.try_resize(out_size)?;
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
                let in_size = checked_buffer_size(batch_size, layer.in_dim)?;
                let input_buf = if current_is_a { &buffer_a } else { &buffer_b };
                layer.forward_batch(&input_buf.as_slice()[..in_size], output, workspace);
            }

            Ok(())
        })();

        // Return buffers to workspace (even on error)
        workspace.layer_output = buffer_a;
        workspace.layer_input = buffer_b;

        result
    }

    /// Parallel forward pass for batch inference.
    ///
    /// **Requires the `parallel` feature.** Without it use
    /// [`forward_batch`](Self::forward_batch), which produces identical output
    /// on a single thread.
    ///
    /// This method processes samples in parallel using rayon, which is faster
    /// for large batches on multi-core CPUs. Each sample gets its own workspace
    /// allocated via thread-local storage.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data `[batch_size * input_dim]`
    /// * `output` - Output buffer `[batch_size * output_dim]`
    ///
    /// # Performance
    ///
    /// - Use for batch_size >= 32 on multi-core systems
    /// - For small batches, use [`forward_batch`](Self::forward_batch) instead
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config.clone());
    ///
    /// let batch_size = 256;
    /// let input = vec![0.5f32; batch_size * config.input_dim];
    /// let mut output = vec![0.0f32; batch_size * config.output_dim];
    ///
    /// network.forward_batch_parallel(&input, &mut output);
    /// ```
    #[cfg(feature = "parallel")]
    pub fn forward_batch_parallel(&self, input: &[f32], output: &mut [f32]) {
        self.validate_layout().expect("Invalid network layout");
        use rayon::prelude::*;
        use std::cell::RefCell;

        let batch_size = input.len() / self.config.input_dim;
        let in_dim = self.config.input_dim;
        let out_dim = self.config.output_dim;

        debug_assert_eq!(input.len(), batch_size * in_dim);
        debug_assert_eq!(output.len(), batch_size * out_dim);

        // Thread-local workspace
        thread_local! {
            static LOCAL_WORKSPACE: RefCell<Option<Workspace>> = const { RefCell::new(None) };
        }

        let config = &self.config;

        // Process samples in parallel, writing directly to output slices
        output
            .par_chunks_mut(out_dim)
            .enumerate()
            .for_each(|(b, out_slice)| {
                let in_start = b * in_dim;
                let in_slice = &input[in_start..in_start + in_dim];

                LOCAL_WORKSPACE.with(|ws_cell| {
                    let mut ws_ref = ws_cell.borrow_mut();
                    if ws_ref.is_none() {
                        *ws_ref = Some(Workspace::new(config));
                    }
                    let workspace = ws_ref.as_mut().unwrap();

                    self.forward_single(in_slice, out_slice, workspace);
                });
            });
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
    /// # Storage and Scheduling
    ///
    /// After warmup for the required capacity, ArKan reuses its execution and
    /// history storage. With `parallel`, the configured multithreading threshold
    /// also schedules cached output accumulation when multiple workers are available.
    /// External-thread calls may allocate Rayon scheduling-queue blocks; warmed
    /// enclosing-pool zero-allocation observations do not guarantee this for every pool.
    pub fn forward_batch_training(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
    ) {
        self.try_forward_batch_training(input, output, workspace)
            .expect("forward_batch_training failed");
    }

    /// Fraction of `(sample, feature)` pairs that the last training forward pass
    /// clamped to `grid_range`, one entry per layer, each in `0.0..=1.0`.
    ///
    /// Reads the [`SPAN_CLAMPED_FLAG`] bits that
    /// [`forward_batch_training`](Self::forward_batch_training) and
    /// [`train_step`](Self::train_step) already record, so it costs one pass over
    /// the span indices and no extra forward pass. Returns all zeros if no
    /// training forward pass has run on `workspace` yet.
    ///
    /// # Why you want this
    ///
    /// `z = clamp((x - mean) / std, lo, hi)` has `dz/dx == 0` outside the grid, so
    /// a layer boundary where *every* pair is clamped passes exactly zero gradient
    /// to everything upstream of it - zero, not small. Nothing bounds a layer's
    /// output to the next layer's `grid_range`, and hidden layers normalize with a
    /// fixed mean 0 / std 1 that never adapts, so a run can walk into that state
    /// from a perfectly healthy start and then sit there: the loss stops moving and
    /// no learning rate brings it back, because there is no gradient to scale.
    ///
    /// A number near 1.0 at any layer is the warning. See
    /// `tests/training_dynamics.rs` for the full characterization and the one
    /// escape that works (decoupled weight decay - it moves weights without a
    /// gradient).
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork};
    ///
    /// let net = KanNetwork::new(KanConfig::preset());
    /// let mut ws = net.create_workspace(4);
    /// let mut out = vec![0.0f32; 4 * 24];
    /// net.forward_batch_training(&vec![0.5f32; 4 * 21], &mut out, &mut ws);
    ///
    /// let clamped = net.clamped_fraction(&ws);
    /// assert_eq!(clamped.len(), net.layers.len());
    /// assert!(clamped.iter().all(|f| (0.0..=1.0).contains(f)));
    /// ```
    #[must_use]
    pub fn clamped_fraction(&self, workspace: &Workspace) -> Vec<f32> {
        let batch = workspace.history_batch_size();
        self.layers
            .iter()
            .enumerate()
            .map(|(li, layer)| {
                let count = batch * layer.in_dim;
                match workspace.layers_grid_indices.get(li) {
                    Some(spans) if count > 0 && spans.len() >= count => {
                        let clamped = spans[..count]
                            .iter()
                            .filter(|s| *s & SPAN_CLAMPED_FLAG != 0)
                            .count();
                        clamped as f32 / count as f32
                    }
                    _ => 0.0,
                }
            })
            .collect()
    }

    /// Fallible version of [`forward_batch_training`](Self::forward_batch_training).
    ///
    /// Returns `ArkanError::ShapeMismatch` if input/output lengths don't match expected dimensions.
    /// Returns `ArkanError::Overflow` if buffer size calculations overflow.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_forward_batch_training(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut Workspace,
    ) -> ArkanResult<()> {
        self.validate_layout()?;
        let batch_size = input.len() / self.config.input_dim;

        let expected_input_len = checked_buffer_size(batch_size, self.config.input_dim)?;
        if input.len() != expected_input_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_input_len],
                &[input.len()],
            ));
        }

        let expected_output_len = checked_buffer_size(batch_size, self.config.output_dim)?;
        if output.len() != expected_output_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_output_len],
                &[output.len()],
            ));
        }

        if batch_size == 0 {
            // Clear recorded history as well as accepting the empty output shape.
            workspace.try_prepare_training(0, &self.config, self.layout().layer_dims())?;
            return Ok(());
        }

        workspace.try_prepare_training(batch_size, &self.config, self.layout().layer_dims())?;
        let parallel_accumulation = cfg!(feature = "parallel")
            && batch_size > 1
            && batch_size >= self.config.multithreading_threshold;

        let max_hidden = self
            .layout()
            .layer_dims()
            .iter()
            .copied()
            .max()
            .unwrap_or(self.config.input_dim);
        let ping_pong_size = checked_buffer_size(batch_size, max_hidden)?;

        // Use std::mem::take for borrow-safety
        let mut buffer_a = std::mem::take(&mut workspace.layer_output);
        let mut buffer_b = std::mem::take(&mut workspace.layer_input);

        // Wrap in a closure for early return with buffer restoration
        let result = (|| -> ArkanResult<()> {
            buffer_a.try_resize(ping_pong_size)?;
            buffer_b.try_resize(ping_pong_size)?;

            // Ping-pong buffer tracking:
            // - After each layer, current_is_a indicates which buffer CONTAINS the output
            // - Next layer reads from that buffer and writes to the other
            let mut current_is_a = false; // Will become true after layer 0 writes to buffer_a

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let in_size = checked_buffer_size(batch_size, layer.in_dim)?;
                let out_size = checked_buffer_size(batch_size, layer.out_dim)?;

                // Determine input source and output destination
                let (input_slice, output_buf): (&[f32], &mut _) = if layer_idx == 0 {
                    // First layer: read from input slice, write to buffer_a
                    (input, &mut buffer_a)
                } else if current_is_a {
                    // Previous layer wrote to buffer_a, read from there, write to buffer_b
                    (&buffer_a.as_slice()[..in_size], &mut buffer_b)
                } else {
                    // Previous layer wrote to buffer_b, read from there, write to buffer_a
                    (&buffer_b.as_slice()[..in_size], &mut buffer_a)
                };

                output_buf.try_resize(out_size)?;
                layer.forward_batch_impl(
                    input_slice,
                    &mut output_buf.as_mut_slice()[..out_size],
                    workspace,
                    parallel_accumulation,
                )?;

                // Save normalized inputs and grid indices for backward
                let hist_in = &mut workspace.layers_inputs[layer_idx].as_mut_slice()[..in_size];
                hist_in.copy_from_slice(workspace.z_buffer.as_slice());

                let hist_idx = &mut workspace.layers_grid_indices[layer_idx][..in_size];
                hist_idx.copy_from_slice(&workspace.grid_indices[..in_size]);

                if layer_idx == self.layers.len() - 1 {
                    output.copy_from_slice(&output_buf.as_slice()[..out_size]);
                }

                // After writing, update current_is_a to indicate where the output now lives
                // Layer 0 always writes to buffer_a, so current_is_a becomes true
                // Subsequent layers toggle: if we just read from A and wrote to B, current_is_a = false
                if layer_idx == 0 {
                    current_is_a = true; // Layer 0 always outputs to buffer_a
                } else {
                    current_is_a = !current_is_a;
                }
            }

            Ok(())
        })();

        // Return buffers to workspace (even on error)
        workspace.layer_output = buffer_a;
        workspace.layer_input = buffer_b;

        result
    }
}
