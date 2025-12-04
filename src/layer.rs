//! KAN Layer Implementation with B-spline basis functions.
//!
//! This module contains the [`KanLayer`] struct which implements a single
//! Kolmogorov-Arnold Network layer using B-spline basis functions.
//!
//! # Mathematical Foundation
//!
//! Each layer computes:
//!
//! $$y_j = \sum_i \sum_k c_{j,i,k} \cdot B_k(x_i)$$
//!
//! where:
//! - $x_i$ is the i-th input (normalized to \[0,1\])
//! - $B_k$ are B-spline basis functions of given order
//! - $c_{j,i,k}$ are learnable spline coefficients
//!
//! # Spline Indexing
//!
//! For order `p` and `n` grid intervals:
//! - Global basis count: `n + p` (NOT `p + 1`!)
//! - Each input point activates exactly `p + 1` consecutive basis functions
//! - Span `s` = floor(x * n), clamped to [p, n+p-1]
//! - Active weights start at index `s - p` in global coefficient array
//!
//! # Example
//!
//! ```rust
//! use arkan::{KanConfig, KanLayer};
//!
//! let config = KanConfig::preset();
//! let layer = KanLayer::new(4, 8, &config);
//!
//! // Single sample forward pass
//! let input = vec![0.1, 0.2, 0.3, 0.4];
//! let mut output = vec![0.0; 8];
//! let mut basis_buf = vec![0.0; layer.basis_aligned()];
//!
//! layer.forward_single(&input, &mut output, &mut basis_buf);
//! ```
//!
//! # SIMD Optimization
//!
//! The layer uses SIMD instructions when available:
//! - 8-wide AVX2 for batch sizes ≥ 8
//! - 4-wide SSE4 for smaller batches
//! - Scalar fallback for non-aligned cases

use crate::buffer::Workspace;
use crate::config::{KanConfig, EPSILON};
use crate::spline::{compute_basis, compute_basis_and_deriv, compute_knots, find_span};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use wide::{f32x4, f32x8};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single KAN layer with learnable spline coefficients.
///
/// Each `KanLayer` transforms an input vector through learnable B-spline
/// functions. Unlike traditional neural networks that use fixed activation
/// functions, KAN layers learn the activation shape itself.
///
/// # Architecture
///
/// ```text
/// Input(in_dim) → Normalize → B-spline Basis → Weighted Sum → Output(out_dim)
/// ```
///
/// # Weight Layout
///
/// Weights are stored as a flat array with indexing:
/// `weights[out_idx * in_dim * global_basis_size + in_idx * global_basis_size + basis_idx]`
///
/// # Example
///
/// ```rust
/// use arkan::{KanConfig, KanLayer};
///
/// let config = KanConfig::preset();
/// let layer = KanLayer::new(4, 8, &config);
///
/// assert_eq!(layer.in_dim, 4);
/// assert_eq!(layer.out_dim, 8);
/// assert_eq!(layer.param_count(), 4 * 8 * (config.grid_size + config.spline_order) + 8);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KanLayer {
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Spline order (degree). Typical values: 2-4.
    pub order: usize,
    /// Number of grid intervals.
    pub grid_size: usize,
    /// Global basis size = `grid_size + order`.
    pub global_basis_size: usize,
    /// Local basis size = `order + 1` (active functions per input).
    pub local_basis_size: usize,
    /// Local basis size aligned to SIMD width for efficient operations.
    pub basis_aligned: usize,
    /// Grid range (min, max) for input normalization.
    pub grid_range: (f32, f32),
    /// Precomputed knot vector (skipped during serialization).
    #[cfg_attr(feature = "serde", serde(skip))]
    knots: Vec<f32>,
    /// Per-input normalization mean.
    pub mean: Vec<f32>,
    /// Per-input normalization std (clamped by [`EPSILON`]).
    pub std: Vec<f32>,
    /// Spline coefficients: `(out_dim, in_dim, global_basis_size)` stored flat.
    pub weights: Vec<f32>,
    /// Bias terms for each output.
    pub bias: Vec<f32>,
    /// SIMD width for aligned operations.
    #[allow(dead_code)]
    simd_width: usize,
}

impl KanLayer {
    /// Creates a new KAN layer with the given dimensions (fallible version).
    ///
    /// This is the fallible version of [`new`](Self::new) that returns an error
    /// instead of panicking on invalid inputs or overflow conditions.
    ///
    /// # Arguments
    ///
    /// * `in_dim` - Input dimension (must be positive)
    /// * `out_dim` - Output dimension (must be positive)
    /// * `config` - Network configuration containing spline parameters
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Config`] if dimensions are zero.
    /// Returns [`ArkanError::Overflow`] if weight count overflows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanLayer};
    ///
    /// let config = KanConfig::preset();
    /// let layer = KanLayer::try_new(10, 20, &config).expect("valid config");
    /// ```
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_new(
        in_dim: usize,
        out_dim: usize,
        config: &KanConfig,
    ) -> Result<Self, crate::ArkanError> {
        use crate::config::ConfigError;
        use crate::ArkanError;
        use std::borrow::Cow;

        if in_dim == 0 {
            return Err(ArkanError::Config(ConfigError::InvalidDimension(
                Cow::Borrowed("input dimension must be positive"),
            )));
        }
        if out_dim == 0 {
            return Err(ArkanError::Config(ConfigError::InvalidDimension(
                Cow::Borrowed("output dimension must be positive"),
            )));
        }

        let order = config.spline_order;
        let grid_size = config.grid_size;
        let simd_width = config.simd_width;
        let grid_range = config.grid_range;

        // Global basis: one B-spline per knot interval that can be active
        let global_basis_size = grid_size
            .checked_add(order)
            .ok_or_else(|| ArkanError::Overflow("grid_size + order overflow".into()))?;

        // Local: only order+1 functions are non-zero at any point
        let local_basis_size = order + 1; // order <= 5, safe
        let basis_aligned = (local_basis_size + simd_width - 1) & !(simd_width - 1);

        // Calculate total weights with overflow checks BEFORE allocating
        let total_weights = out_dim
            .checked_mul(in_dim)
            .and_then(|x| x.checked_mul(global_basis_size))
            .ok_or_else(|| {
                ArkanError::Overflow(format!(
                    "weight count overflow: {} * {} * {}",
                    out_dim, in_dim, global_basis_size
                ))
            })?;

        // Check against practical limits (avoid OOM)
        const MAX_WEIGHTS: usize = 1 << 30; // ~1 billion weights, ~4GB
        if total_weights > MAX_WEIGHTS {
            return Err(ArkanError::Overflow(format!(
                "weight count {} exceeds maximum {}",
                total_weights, MAX_WEIGHTS
            )));
        }

        let knots = compute_knots(grid_size, order, grid_range);

        // Normalization: only apply to input layer (where in_dim matches config)
        // Hidden layers use identity normalization (mean=0, std=1)
        let (mean, std) = if in_dim == config.input_dim && config.input_mean.len() == in_dim {
            (
                config.input_mean.clone(),
                config.input_std.iter().map(|s| s.max(EPSILON)).collect(),
            )
        } else {
            (vec![0.0; in_dim], vec![1.0; in_dim])
        };

        // Initialize weights with small random values (Xavier-like)
        let scale = (2.0 / (in_dim + out_dim) as f32).sqrt() * 0.1;
        let mut rng: SmallRng = if let Some(seed) = config.init_seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };
        let weights = (0..total_weights)
            .map(|_| rng.gen_range(-0.5f32..0.5f32) * scale)
            .collect();

        let bias = vec![0.0; out_dim];

        Ok(Self {
            in_dim,
            out_dim,
            order,
            grid_size,
            global_basis_size,
            local_basis_size,
            basis_aligned,
            grid_range,
            knots,
            mean,
            std,
            weights,
            bias,
            simd_width,
        })
    }

    /// Creates a new KAN layer with the given dimensions.
    ///
    /// Initializes weights using Xavier-like initialization scaled by 0.1.
    /// Bias terms are initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `in_dim` - Input dimension (must be positive)
    /// * `out_dim` - Output dimension (must be positive)
    /// * `config` - Network configuration containing spline parameters
    ///
    /// # Panics
    ///
    /// Panics if `in_dim` or `out_dim` is zero, or if weight count overflows.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanLayer};
    ///
    /// let config = KanConfig::preset();
    /// let layer = KanLayer::new(10, 20, &config);
    /// ```
    #[must_use = "this creates a new layer without modifying anything"]
    pub fn new(in_dim: usize, out_dim: usize, config: &KanConfig) -> Self {
        Self::try_new(in_dim, out_dim, config).expect("KanLayer::new failed")
    }

    /// Creates layer from config (alias for [`new`](Self::new)).
    #[inline]
    #[must_use]
    pub fn from_config(in_dim: usize, out_dim: usize, config: &KanConfig) -> Self {
        Self::new(in_dim, out_dim, config)
    }

    /// Returns the weight index for coefficient c[out_idx, in_idx, basis_idx].
    #[inline]
    fn weight_index(&self, out_idx: usize, in_idx: usize, basis_idx: usize) -> usize {
        (out_idx * self.in_dim + in_idx) * self.global_basis_size + basis_idx
    }

    /// Basis size aligned to SIMD width.
    ///
    /// Use this value to allocate basis function buffers for [`forward_single`](Self::forward_single).
    #[inline]
    pub fn basis_aligned(&self) -> usize {
        self.basis_aligned
    }

    /// Total number of trainable parameters (weights + biases).
    #[inline]
    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Sets normalization parameters (mean and standard deviation).
    ///
    /// # Arguments
    ///
    /// * `mean` - Per-input mean values
    /// * `std` - Per-input standard deviations (clamped to [`EPSILON`])
    ///
    /// # Panics
    ///
    /// Panics if lengths don't match `in_dim`.
    pub fn set_normalization(&mut self, mean: &[f32], std: &[f32]) {
        assert_eq!(mean.len(), self.in_dim);
        assert_eq!(std.len(), self.in_dim);
        self.mean.clear();
        self.mean.extend_from_slice(mean);
        self.std.clear();
        self.std.extend(std.iter().map(|s| s.max(EPSILON)));
    }

    /// Forward pass for a single input sample.
    ///
    /// This is the lowest-level forward function. For batch processing,
    /// use [`forward_batch`](Self::forward_batch) instead.
    ///
    /// # Arguments
    ///
    /// * `input` - Input values of length `in_dim`
    /// * `output` - Output buffer of length `out_dim`
    /// * `basis_buf` - Temporary buffer of length [`basis_aligned()`](Self::basis_aligned)
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanLayer};
    ///
    /// let config = KanConfig::preset();
    /// let layer = KanLayer::new(4, 8, &config);
    ///
    /// let input = vec![0.1, 0.2, 0.3, 0.4];
    /// let mut output = vec![0.0; 8];
    /// let mut basis_buf = vec![0.0; layer.basis_aligned()];
    ///
    /// layer.forward_single(&input, &mut output, &mut basis_buf);
    /// ```
    pub fn forward_single(&self, input: &[f32], output: &mut [f32], basis_buf: &mut [f32]) {
        debug_assert_eq!(input.len(), self.in_dim);
        debug_assert_eq!(output.len(), self.out_dim);
        debug_assert!(basis_buf.len() >= self.basis_aligned);

        // Initialize outputs with bias
        output.copy_from_slice(&self.bias);

        // For each input, compute basis and accumulate
        for (i, raw) in input.iter().enumerate() {
            let z =
                ((*raw - self.mean[i]) / self.std[i]).clamp(self.grid_range.0, self.grid_range.1);

            // Find span and compute basis
            let span = find_span(z, &self.knots, self.order, self.grid_size);
            compute_basis(
                z,
                span,
                &self.knots,
                self.order,
                &mut basis_buf[..self.local_basis_size],
            );

            let start_idx = span - self.order;
            let basis_slice = &basis_buf[..self.local_basis_size];

            // Accumulate for each output
            for (j, out) in output.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for (k, basis_value) in basis_slice.iter().enumerate() {
                    let weight_idx = self.weight_index(j, i, start_idx + k);
                    sum += self.weights[weight_idx] * *basis_value;
                }
                *out += sum;
            }
        }
    }

    /// Forward pass for a batch of samples using preallocated workspace.
    ///
    /// This method uses SIMD acceleration when available and stores
    /// intermediate values in the workspace for potential backward pass.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Flattened input batch: `[batch_size * in_dim]`
    /// * `outputs` - Output buffer: `[batch_size * out_dim]`
    /// * `workspace` - Preallocated buffers (resized automatically if needed)
    ///
    /// # Memory Layout
    ///
    /// - `inputs`: Row-major `[Batch, Input]`
    /// - `outputs`: Row-major `[Batch, Output]`
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_forward_batch`](Self::try_forward_batch)
    /// for a fallible version.
    pub fn forward_batch(&self, inputs: &[f32], outputs: &mut [f32], workspace: &mut Workspace) {
        self.forward_batch_impl(inputs, outputs, workspace)
            .expect("forward_batch: buffer size overflow")
    }

    /// Internal implementation of forward_batch with overflow checking.
    fn forward_batch_impl(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        workspace: &mut Workspace,
    ) -> crate::ArkanResult<()> {
        use crate::buffer::{checked_buffer_size, checked_buffer_size3};

        let batch_size = inputs.len() / self.in_dim;
        debug_assert_eq!(inputs.len(), batch_size * self.in_dim);
        debug_assert_eq!(outputs.len(), batch_size * self.out_dim);

        // Use checked arithmetic to prevent overflow
        let z_needed = checked_buffer_size(batch_size, self.in_dim)?;
        workspace.z_buffer.try_resize(z_needed)?;

        let basis_needed = checked_buffer_size3(batch_size, self.in_dim, self.basis_aligned)?;
        if workspace.basis_values.len() < basis_needed {
            workspace.basis_values.try_resize(basis_needed)?;
        }

        let spans_needed = checked_buffer_size(batch_size, self.in_dim)?;
        if workspace.grid_indices.len() < spans_needed {
            workspace.grid_indices.resize(spans_needed, 0);
        }

        // Compute basis values for all samples
        for b in 0..batch_size {
            let input_start = b * self.in_dim;
            let span_batch_start = b * self.in_dim;
            let basis_batch_start = b * self.in_dim * self.basis_aligned;

            for i in 0..self.in_dim {
                let raw = inputs[input_start + i];
                let z = ((raw - self.mean[i]) / self.std[i])
                    .clamp(self.grid_range.0, self.grid_range.1);
                workspace.z_buffer.as_mut_slice()[input_start + i] = z;
                let span = find_span(z, &self.knots, self.order, self.grid_size);
                workspace.grid_indices[span_batch_start + i] = span as u32;

                let basis_start = basis_batch_start + i * self.basis_aligned;
                let basis_slice = workspace.basis_values.as_mut_slice();
                compute_basis(
                    z,
                    span,
                    &self.knots,
                    self.order,
                    &mut basis_slice[basis_start..basis_start + self.local_basis_size],
                );

                // Zero padding for alignment
                for j in self.local_basis_size..self.basis_aligned {
                    basis_slice[basis_start + j] = 0.0;
                }
            }
        }

        // Accumulate outputs
        self.accumulate_batch(workspace, outputs, batch_size);

        Ok(())
    }

    /// Forward pass for a single input sample (fallible version).
    ///
    /// This is the fallible version of [`forward_single`](Self::forward_single)
    /// that validates buffer sizes and returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::ShapeMismatch`] if buffer sizes don't match expected dimensions.
    #[inline]
    pub fn try_forward_single(
        &self,
        input: &[f32],
        output: &mut [f32],
        basis_buf: &mut [f32],
    ) -> crate::ArkanResult<()> {
        use crate::ArkanError;

        if input.len() != self.in_dim {
            return Err(ArkanError::shape_mismatch(
                &[self.in_dim],
                &[input.len()],
            ));
        }
        if output.len() != self.out_dim {
            return Err(ArkanError::shape_mismatch(
                &[self.out_dim],
                &[output.len()],
            ));
        }
        if basis_buf.len() < self.basis_aligned {
            return Err(ArkanError::shape_mismatch(
                &[self.basis_aligned],
                &[basis_buf.len()],
            ));
        }

        self.forward_single(input, output, basis_buf);
        Ok(())
    }

    /// Forward pass for a batch of samples (fallible version).
    ///
    /// This is the fallible version of [`forward_batch`](Self::forward_batch)
    /// that validates buffer sizes and returns an error instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::ShapeMismatch`] if input/output sizes don't match expected dimensions.
    #[inline]
    pub fn try_forward_batch(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        workspace: &mut Workspace,
    ) -> crate::ArkanResult<()> {
        use crate::ArkanError;
        use crate::buffer::checked_buffer_size;

        if inputs.is_empty() {
            return Ok(());
        }

        if inputs.len() % self.in_dim != 0 {
            return Err(ArkanError::shape_mismatch(
                &[self.in_dim],
                &[inputs.len()],
            ));
        }

        let batch_size = inputs.len() / self.in_dim;
        // Use checked arithmetic for expected_output_len
        let expected_output_len = checked_buffer_size(batch_size, self.out_dim)?;

        if outputs.len() != expected_output_len {
            return Err(ArkanError::shape_mismatch(
                &[expected_output_len],
                &[outputs.len()],
            ));
        }

        // Use internal impl that returns Result
        self.forward_batch_impl(inputs, outputs, workspace)
    }

    /// Accumulates outputs for a batch (internal helper).
    #[inline]
    fn accumulate_batch(&self, workspace: &Workspace, outputs: &mut [f32], batch_size: usize) {
        let basis_slice = workspace.basis_values.as_slice();
        let spans = &workspace.grid_indices;

        for b in 0..batch_size {
            let span_batch_start = b * self.in_dim;
            let basis_batch_start = b * self.in_dim * self.basis_aligned;
            let out_start = b * self.out_dim;

            // Initialize outputs with bias
            let out_slice = &mut outputs[out_start..out_start + self.out_dim];
            out_slice.copy_from_slice(&self.bias);

            for (j, out) in out_slice.iter_mut().enumerate() {
                let sum = match self.simd_width {
                    8 if self.local_basis_size <= 8 && self.in_dim >= 8 => self.accumulate_simd8(
                        basis_slice,
                        spans,
                        basis_batch_start,
                        span_batch_start,
                        j,
                    ),
                    4 if self.local_basis_size <= 4 && self.in_dim >= 4 => self.accumulate_simd4(
                        basis_slice,
                        spans,
                        basis_batch_start,
                        span_batch_start,
                        j,
                    ),
                    _ => {
                        // Scalar fallback
                        let mut s = 0.0f32;
                        for i in 0..self.in_dim {
                            let span = spans[span_batch_start + i] as usize;
                            let start_idx = span - self.order;
                            let basis_start = basis_batch_start + i * self.basis_aligned;

                            for k in 0..self.local_basis_size {
                                let weight_idx = self.weight_index(j, i, start_idx + k);
                                s += self.weights[weight_idx] * basis_slice[basis_start + k];
                            }
                        }
                        s
                    }
                };

                *out += sum;
            }
        }
    }

    /// SIMD-accelerated accumulation (8-wide).
    #[inline]
    fn accumulate_simd8(
        &self,
        basis_values: &[f32],
        spans: &[u32],
        basis_batch_start: usize,
        span_batch_start: usize,
        out_idx: usize,
    ) -> f32 {
        let mut acc = f32x8::splat(0.0);

        // Process 8 inputs at a time
        let chunks = self.in_dim / 8;
        for chunk in 0..chunks {
            let i_base = chunk * 8;

            // For each basis function (k), gather weights for 8 inputs
            for k in 0..self.local_basis_size {
                // Gather basis values for 8 inputs
                let mut basis_arr = [0.0f32; 8];
                let mut weight_arr = [0.0f32; 8];

                for lane in 0..8 {
                    let i = i_base + lane;
                    let span = spans[span_batch_start + i] as usize;
                    let start_idx = span - self.order;
                    let basis_start = basis_batch_start + i * self.basis_aligned;

                    basis_arr[lane] = basis_values[basis_start + k];
                    weight_arr[lane] = self.weights[self.weight_index(out_idx, i, start_idx + k)];
                }

                let basis_vec = f32x8::new(basis_arr);
                let weight_vec = f32x8::new(weight_arr);
                acc += basis_vec * weight_vec;
            }
        }

        // Sum SIMD lanes
        let arr: [f32; 8] = acc.into();
        let mut sum: f32 = arr.iter().sum();

        // Handle remaining inputs (scalar)
        for i in (chunks * 8)..self.in_dim {
            let span = spans[span_batch_start + i] as usize;
            let start_idx = span - self.order;
            let basis_start = basis_batch_start + i * self.basis_aligned;

            for k in 0..self.local_basis_size {
                let weight_idx = self.weight_index(out_idx, i, start_idx + k);
                sum += self.weights[weight_idx] * basis_values[basis_start + k];
            }
        }

        sum
    }

    /// SIMD-accelerated accumulation (4-wide).
    #[inline]
    fn accumulate_simd4(
        &self,
        basis_values: &[f32],
        spans: &[u32],
        basis_batch_start: usize,
        span_batch_start: usize,
        out_idx: usize,
    ) -> f32 {
        let mut acc = f32x4::splat(0.0);

        let chunks = self.in_dim / 4;
        for chunk in 0..chunks {
            let i_base = chunk * 4;

            for k in 0..self.local_basis_size {
                let mut basis_arr = [0.0f32; 4];
                let mut weight_arr = [0.0f32; 4];

                for lane in 0..4 {
                    let i = i_base + lane;
                    let span = spans[span_batch_start + i] as usize;
                    let start_idx = span - self.order;
                    let basis_start = basis_batch_start + i * self.basis_aligned;

                    basis_arr[lane] = basis_values[basis_start + k];
                    weight_arr[lane] = self.weights[self.weight_index(out_idx, i, start_idx + k)];
                }

                let basis_vec = f32x4::new(basis_arr);
                let weight_vec = f32x4::new(weight_arr);
                acc += basis_vec * weight_vec;
            }
        }

        let arr: [f32; 4] = acc.into();
        let mut sum: f32 = arr.iter().sum();

        // Tail
        for i in (chunks * 4)..self.in_dim {
            let span = spans[span_batch_start + i] as usize;
            let start_idx = span - self.order;
            let basis_start = basis_batch_start + i * self.basis_aligned;

            for k in 0..self.local_basis_size {
                let weight_idx = self.weight_index(out_idx, i, start_idx + k);
                sum += basis_values[basis_start + k] * self.weights[weight_idx];
            }
        }

        sum
    }

    /// Returns the total number of trainable parameters.
    ///
    /// Equivalent to [`param_count`](Self::param_count).
    #[inline]
    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Gets all parameters as a flat vector for optimization.
    ///
    /// Returns weights followed by biases.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = self.weights.clone();
        params.extend(&self.bias);
        params
    }

    /// Sets parameters from a flat slice.
    ///
    /// # Panics
    ///
    /// Panics if `params.len() != num_parameters()`.
    pub fn set_parameters(&mut self, params: &[f32]) {
        let expected = self.num_parameters();
        assert_eq!(
            params.len(),
            expected,
            "Parameter count mismatch: expected {}, got {}",
            expected,
            params.len()
        );

        let w_end = self.weights.len();
        self.weights.copy_from_slice(&params[..w_end]);
        self.bias.copy_from_slice(&params[w_end..]);
    }

    /// Backward pass: computes gradients for weights, biases, and optionally inputs.
    ///
    /// # Arguments
    ///
    /// * `normalized_input` - Stored normalized inputs from forward pass: `[batch * in_dim]`
    /// * `grid_indices` - Stored span indices from forward pass: `[batch * in_dim]`
    /// * `grad_output` - Gradient of loss w.r.t. output: `[batch * out_dim]`
    /// * `grad_input` - Optional gradient buffer for input: `[batch * in_dim]`
    /// * `grad_weights` - Gradient buffer for weights: `[weights.len()]`
    /// * `grad_bias` - Gradient buffer for biases: `[bias.len()]`
    /// * `workspace` - Workspace with basis value buffers
    ///
    /// # Note
    ///
    /// This method supports masked training: if `grad_output[i] == 0.0`,
    /// that sample is skipped for efficiency.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        normalized_input: &[f32],
        grid_indices: &[u32],
        grad_output: &[f32],
        mut grad_input: Option<&mut [f32]>,
        grad_weights: &mut [f32],
        grad_bias: &mut [f32],
        workspace: &mut Workspace,
    ) {
        let batch_size = normalized_input.len() / self.in_dim;
        debug_assert_eq!(normalized_input.len(), batch_size * self.in_dim);
        debug_assert_eq!(grid_indices.len(), batch_size * self.in_dim);
        debug_assert_eq!(grad_output.len(), batch_size * self.out_dim);
        debug_assert_eq!(grad_weights.len(), self.weights.len());
        debug_assert_eq!(grad_bias.len(), self.bias.len());

        // Prepare basis buffers
        let basis_needed = batch_size * self.in_dim * self.basis_aligned;
        if workspace.basis_values.len() < basis_needed {
            workspace.basis_values.resize(basis_needed);
        }
        if workspace.basis_derivs.len() < basis_needed {
            workspace.basis_derivs.resize(basis_needed);
        }

        let basis_slice = workspace.basis_values.as_mut_slice();
        let deriv_slice = workspace.basis_derivs.as_mut_slice();

        // Recompute basis values and derivatives for stored inputs
        for b in 0..batch_size {
            let base_offset = b * self.in_dim * self.basis_aligned;
            let input_offset = b * self.in_dim;

            for i in 0..self.in_dim {
                let z = normalized_input[input_offset + i];
                let span = grid_indices[input_offset + i] as usize;
                let basis_offset = base_offset + i * self.basis_aligned;

                compute_basis_and_deriv(
                    z,
                    span,
                    &self.knots,
                    self.order,
                    &mut basis_slice[basis_offset..basis_offset + self.local_basis_size],
                    &mut deriv_slice[basis_offset..basis_offset + self.local_basis_size],
                );

                // Zero padding for alignment to avoid reading stale data
                for k in self.local_basis_size..self.basis_aligned {
                    basis_slice[basis_offset + k] = 0.0;
                    deriv_slice[basis_offset + k] = 0.0;
                }
            }
        }

        if let Some(ref mut gi) = grad_input {
            debug_assert_eq!(gi.len(), batch_size * self.in_dim);
            gi.iter_mut().for_each(|x| *x = 0.0);
        }

        // Accumulate gradients
        for b in 0..batch_size {
            let span_batch_start = b * self.in_dim;
            let basis_batch_start = b * self.in_dim * self.basis_aligned;
            let grad_out_start = b * self.out_dim;

            for j in 0..self.out_dim {
                let g_out = grad_output[grad_out_start + j];
                // Masking safety: zero grad_output short-circuits
                if g_out == 0.0 {
                    continue;
                }

                grad_bias[j] += g_out;

                for i in 0..self.in_dim {
                    let span = grid_indices[span_batch_start + i] as usize;
                    let start_idx = span - self.order;
                    let basis_start = basis_batch_start + i * self.basis_aligned;

                    for k in 0..self.local_basis_size {
                        let weight_idx = self.weight_index(j, i, start_idx + k);
                        let basis_val = basis_slice[basis_start + k];
                        grad_weights[weight_idx] += g_out * basis_val;

                        if let Some(ref mut gi) = grad_input {
                            let deriv = deriv_slice[basis_start + k];
                            let std_inv = 1.0 / self.std[i].max(EPSILON);
                            gi[span_batch_start + i] +=
                                g_out * self.weights[weight_idx] * deriv * std_inv;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(order: usize, grid_size: usize) -> KanConfig {
        KanConfig {
            input_dim: 4,
            output_dim: 8,
            hidden_dims: vec![8, 8],
            spline_order: order,
            grid_size,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 4],
            input_std: vec![1.0; 4],
            multithreading_threshold: 128,
            simd_width: 8,
            init_seed: None,
        }
    }

    #[test]
    fn test_layer_dimensions() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);

        assert_eq!(layer.in_dim, 4);
        assert_eq!(layer.out_dim, 8);
        assert_eq!(layer.order, 3);
        assert_eq!(layer.grid_size, 5);
        assert_eq!(layer.global_basis_size, 8); // 5 + 3
        assert_eq!(layer.local_basis_size, 4); // 3 + 1
        assert_eq!(layer.basis_aligned, 8); // Aligned to simd_width
    }

    #[test]
    fn test_weight_count() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);

        // Weights should be out_dim * in_dim * global_basis_size
        let expected_weights = 8 * 4 * 8; // 256
        assert_eq!(layer.weights.len(), expected_weights);
    }

    #[test]
    fn test_forward_single() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let mut basis_buf = vec![0.0f32; layer.basis_aligned];

        let input = vec![0.0, 0.3, -0.5, 0.7];
        let mut output = vec![0.0; 8];

        layer.forward_single(&input, &mut output, &mut basis_buf);

        // Output should be non-zero after forward pass
        let sum: f32 = output.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Output should be non-zero");
    }

    #[test]
    fn test_forward_batch() {
        let config = make_config(2, 4);
        let layer = KanLayer::new(3, 5, &config);
        let mut workspace = Workspace::default();

        let batch_size = 4;
        let inputs: Vec<f32> = (0..batch_size * 3)
            .map(|i| ((i as f32) / (batch_size * 3) as f32) * 2.0 - 1.0)
            .collect();
        let mut outputs = vec![0.0; batch_size * 5];

        layer.forward_batch(&inputs, &mut outputs, &mut workspace);

        // Each sample should have non-zero output
        for b in 0..batch_size {
            let sample_output = &outputs[b * 5..(b + 1) * 5];
            let sum: f32 = sample_output.iter().map(|x| x.abs()).sum();
            assert!(sum > 0.0, "Sample {} output should be non-zero", b);
        }
    }

    #[test]
    fn test_boundary_spans() {
        // Test that edge cases don't panic
        let config = make_config(3, 5);
        let layer = KanLayer::new(2, 3, &config);
        let mut basis_buf = vec![0.0f32; layer.basis_aligned];

        // Test boundary inputs
        let inputs = vec![-1.0, 1.0]; // Min and max grid range values
        let mut output = vec![0.0; 3];

        layer.forward_single(&inputs, &mut output, &mut basis_buf);
        // Should not panic
    }

    #[test]
    fn test_get_set_parameters() {
        let config = make_config(2, 4);
        let mut layer = KanLayer::new(3, 5, &config);

        let params = layer.get_parameters();
        assert_eq!(params.len(), layer.num_parameters());

        // Modify and set back
        let mut new_params = params.clone();
        new_params[0] = 42.0;
        layer.set_parameters(&new_params);

        assert_eq!(layer.weights[0], 42.0);
    }

    #[test]
    fn test_global_basis_math() {
        // Verify global_basis_size = grid_size + order for various configs
        for order in 1..=4 {
            for grid_size in 2..=8 {
                let config = make_config(order, grid_size);
                let layer = KanLayer::new(2, 2, &config);

                assert_eq!(
                    layer.global_basis_size,
                    grid_size + order,
                    "order={}, grid_size={}",
                    order,
                    grid_size
                );
                assert_eq!(
                    layer.local_basis_size,
                    order + 1,
                    "order={}, grid_size={}",
                    order,
                    grid_size
                );
            }
        }
    }

    #[test]
    fn test_weight_indexing() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);

        // Weight indexing should be consistent
        for out_idx in 0..layer.out_dim {
            for in_idx in 0..layer.in_dim {
                for basis_idx in 0..layer.global_basis_size {
                    let idx = layer.weight_index(out_idx, in_idx, basis_idx);
                    assert!(idx < layer.weights.len());
                }
            }
        }
    }

    #[test]
    fn test_try_new_success() {
        let config = make_config(3, 5);
        let result = KanLayer::try_new(4, 8, &config);
        assert!(result.is_ok());
        let layer = result.unwrap();
        assert_eq!(layer.in_dim, 4);
        assert_eq!(layer.out_dim, 8);
    }

    #[test]
    fn test_try_new_zero_in_dim() {
        let config = make_config(3, 5);
        let result = KanLayer::try_new(0, 8, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::ArkanError::Config(_)));
    }

    #[test]
    fn test_try_new_zero_out_dim() {
        let config = make_config(3, 5);
        let result = KanLayer::try_new(4, 0, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::ArkanError::Config(_)));
    }

    #[test]
    fn test_try_new_overflow() {
        let config = make_config(3, 5);
        // Dimensions that would exceed MAX_WEIGHTS (~1 billion)
        // 100_000 * 100_000 * 8 = 80 billion > MAX_WEIGHTS
        let result = KanLayer::try_new(100_000, 100_000, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::ArkanError::Overflow(_)));
    }

    #[test]
    fn test_try_new_multiplication_overflow() {
        let config = make_config(3, 5);
        // Extreme dimensions that overflow usize multiplication
        let result = KanLayer::try_new(usize::MAX / 2, usize::MAX / 2, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::ArkanError::Overflow(_)));
    }

    #[test]
    fn test_try_forward_single_success() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let mut output = vec![0.0; 8];
        let mut basis_buf = vec![0.0; layer.basis_aligned()];

        let result = layer.try_forward_single(&input, &mut output, &mut basis_buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_forward_single_input_mismatch() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let input = vec![0.1, 0.2, 0.3]; // Wrong size: 3 instead of 4
        let mut output = vec![0.0; 8];
        let mut basis_buf = vec![0.0; layer.basis_aligned()];

        let result = layer.try_forward_single(&input, &mut output, &mut basis_buf);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::ArkanError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_try_forward_single_output_mismatch() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let input = vec![0.1, 0.2, 0.3, 0.4];
        let mut output = vec![0.0; 5]; // Wrong size: 5 instead of 8
        let mut basis_buf = vec![0.0; layer.basis_aligned()];

        let result = layer.try_forward_single(&input, &mut output, &mut basis_buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_forward_batch_success() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let mut workspace = crate::buffer::Workspace::new(&config);

        let inputs = vec![0.1; 4 * 10]; // batch_size = 10
        let mut outputs = vec![0.0; 8 * 10];

        let result = layer.try_forward_batch(&inputs, &mut outputs, &mut workspace);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_forward_batch_input_not_divisible() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let mut workspace = crate::buffer::Workspace::new(&config);

        let inputs = vec![0.1; 4 * 10 + 1]; // Not divisible by in_dim
        let mut outputs = vec![0.0; 8 * 10];

        let result = layer.try_forward_batch(&inputs, &mut outputs, &mut workspace);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_forward_batch_output_mismatch() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let mut workspace = crate::buffer::Workspace::new(&config);

        let inputs = vec![0.1; 4 * 10];
        let mut outputs = vec![0.0; 8 * 5]; // Wrong: expects 8*10

        let result = layer.try_forward_batch(&inputs, &mut outputs, &mut workspace);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_forward_batch_empty() {
        let config = make_config(3, 5);
        let layer = KanLayer::new(4, 8, &config);
        let mut workspace = crate::buffer::Workspace::new(&config);

        let inputs: Vec<f32> = vec![];
        let mut outputs: Vec<f32> = vec![];

        // Empty input should return Ok
        let result = layer.try_forward_batch(&inputs, &mut outputs, &mut workspace);
        assert!(result.is_ok());
    }
}
