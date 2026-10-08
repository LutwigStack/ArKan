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

use crate::buffer::{Workspace, MAX_BUFFER_ELEMENTS};
use crate::config::{KanConfig, EPSILON};
use crate::model::{LayerSpec, Normalization};
use crate::spline::{
    compute_basis, compute_basis_and_deriv, compute_knots, find_span, SPAN_CLAMPED_FLAG,
    SPAN_INDEX_MASK,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use wide::{f32x4, f32x8};

/// Recovers the knot span from a span index stored by the forward pass.
///
/// Stored values carry [`SPAN_CLAMPED_FLAG`] in their high bit, so they must never
/// be used as an index directly.
#[inline]
fn span_of(stored: u32) -> usize {
    (stored & SPAN_INDEX_MASK) as usize
}

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
    /// Precomputed knot vector (recomputed after deserialization).
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

/// Reusable bounded scratch for deterministic parallel backward passes.
#[cfg(feature = "parallel")]
#[derive(Debug, Default)]
pub(crate) struct ParallelBackwardScratch {
    chunks: [ParallelChunk; 8],
}

#[cfg(feature = "parallel")]
#[derive(Debug, Default)]
struct ParallelChunk {
    weights: Vec<f32>,
    bias: Vec<f32>,
    basis: Vec<f32>,
    derivs: Vec<f32>,
}

#[cfg(feature = "parallel")]
impl ParallelBackwardScratch {
    /// Prepare for a validated layer, retaining capacities when layer shapes change.
    pub(crate) fn prepare(&mut self, layer: &KanLayer) {
        for chunk in &mut self.chunks {
            chunk.weights.resize(layer.weights.len(), 0.0);
            chunk.bias.resize(layer.bias.len(), 0.0);
            chunk.basis.resize(layer.in_dim * layer.basis_aligned, 0.0);
            chunk.derivs.resize(layer.in_dim * layer.basis_aligned, 0.0);
        }
    }
}

impl KanLayer {
    /// Creates a new KAN layer with the given dimensions (fallible version).
    ///
    /// This is the fallible version of [`new`](Self::new) that returns an error
    /// instead of panicking on invalid inputs or overflow conditions.
    /// A standalone layer uses the configured input statistics when `in_dim`
    /// matches `config.input_dim`; otherwise it uses identity normalization.
    ///
    /// # Arguments
    ///
    /// * `in_dim` - Input dimension (must be positive)
    /// * `out_dim` - Output dimension (must be positive)
    /// * `config` - Network configuration containing spline parameters
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Config`](crate::ArkanError::Config) for zero dimensions
    /// or invalid spline, SIMD, or normalization configuration.
    /// Returns [`ArkanError::Overflow`](crate::ArkanError::Overflow) if weight count overflows.
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
        Self::try_new_at(in_dim, out_dim, config, 0)
    }

    /// As [`try_new`](Self::try_new), but decorrelates the initialization of a layer
    /// at position `layer_index` in a stack.
    ///
    /// With `config.init_seed = Some(s)`, [`try_new`](Self::try_new) seeds its RNG
    /// with `s` every time, so two layers of the same shape in the same network get
    /// *bit-identical* weights - `hidden_dims: vec![64, 64]` starts from a
    /// layer-to-layer symmetric point, which is not what "deterministic
    /// initialization" leads you to expect. This offsets the seed by `layer_index`,
    /// so `layer_index = 0` reproduces the old weights exactly and each later layer
    /// draws its own stream. `KanNetwork::try_new` passes the position.
    ///
    /// Only position zero uses input normalization when its width matches `config.input_dim`.
    /// Later positions always start with identity normalization.
    ///
    /// With `init_seed = None` initialization is identical to [`try_new`](Self::try_new): that
    /// path already seeds per layer from entropy.
    ///
    /// # Errors
    ///
    /// Same as [`try_new`](Self::try_new).
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_new_at(
        in_dim: usize,
        out_dim: usize,
        config: &KanConfig,
        layer_index: usize,
    ) -> Result<Self, crate::ArkanError> {
        let normalization =
            (layer_index == 0 && in_dim == config.input_dim).then_some(Normalization {
                mean: &config.input_mean,
                std: &config.input_std,
            });
        let spec = LayerSpec::new(in_dim, out_dim, config, normalization)?;
        Self::try_from_spec(spec, config.init_seed, layer_index)
    }

    pub(crate) fn try_from_spec(
        spec: LayerSpec<'_>,
        init_seed: Option<u64>,
        layer_index: usize,
    ) -> crate::ArkanResult<Self> {
        use crate::ArkanError;
        let LayerSpec {
            in_dim,
            out_dim,
            order,
            grid_size,
            grid_range,
            simd_width,
            normalization,
        } = spec;

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
        if total_weights > MAX_BUFFER_ELEMENTS {
            return Err(ArkanError::Overflow(format!(
                "weight count {} exceeds maximum {}",
                total_weights, MAX_BUFFER_ELEMENTS
            )));
        }

        let knots = compute_knots(grid_size, order, grid_range);

        let (mean, std) = match normalization {
            Some(stats) => (
                stats.mean.to_vec(),
                stats.std.iter().map(|s| s.max(EPSILON)).collect(),
            ),
            None => (vec![0.0; in_dim], vec![1.0; in_dim]),
        };

        // Initialize weights with small random values (Xavier-like)
        // KAN needs larger initialization because B-splines have bounded support
        let scale = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let mut rng: SmallRng = if let Some(seed) = init_seed {
            // `seed_from_u64` runs SplitMix64 to fill the state, so consecutive
            // seeds give unrelated streams - offsetting by the layer's position is
            // enough to stop two same-shaped layers being bit-identical.
            SmallRng::seed_from_u64(seed.wrapping_add(layer_index as u64))
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
    /// Panics if lengths don't match `in_dim` or statistics are non-finite.
    pub fn set_normalization(&mut self, mean: &[f32], std: &[f32]) {
        self.try_set_normalization(mean, std)
            .expect("KanLayer::set_normalization failed");
    }

    /// Sets finite normalization statistics, clamping finite standard deviations to EPSILON.
    ///
    /// Returns an error for mismatched lengths or non-finite statistics, without mutation.
    pub fn try_set_normalization(&mut self, mean: &[f32], std: &[f32]) -> crate::ArkanResult<()> {
        use crate::config::{validate_finite_normalization, ConfigError};
        if mean.len() != self.in_dim {
            return Err(ConfigError::MismatchedNormalization("input_mean").into());
        }
        if std.len() != self.in_dim {
            return Err(ConfigError::MismatchedNormalization("input_std").into());
        }
        validate_finite_normalization(mean, std)?;
        self.mean = mean.to_vec();
        self.std = std.iter().map(|s| s.max(EPSILON)).collect();
        Ok(())
    }

    /// Checks public layout metadata without allocating or scanning parameter values.
    pub(crate) fn validate_layout(&self) -> crate::ArkanResult<()> {
        use crate::config::{validate_finite_normalization, ConfigError};
        use std::borrow::Cow;
        let invalid = || ConfigError::InvalidDimension(Cow::Borrowed("inconsistent layer layout"));
        crate::spline::validate_spline(self.grid_size, self.order, self.grid_range)?;
        if !matches!(self.simd_width, 4 | 8 | 16) {
            return Err(ConfigError::InvalidSimdWidth(self.simd_width).into());
        }
        let count = self
            .in_dim
            .checked_mul(self.out_dim)
            .and_then(|n| n.checked_mul(self.global_basis_size));
        if self.in_dim == 0
            || self.out_dim == 0
            || self.global_basis_size != self.grid_size + self.order
            || self.local_basis_size != self.order + 1
            || self.basis_aligned
                != self.local_basis_size.div_ceil(self.simd_width) * self.simd_width
            || count != Some(self.weights.len())
            || self.weights.len() > MAX_BUFFER_ELEMENTS
            || self.bias.len() != self.out_dim
            || self.mean.len() != self.in_dim
            || self.std.len() != self.in_dim
            || self.knots.len() != self.grid_size + 2 * self.order + 1
        {
            return Err(invalid().into());
        }
        validate_finite_normalization(&self.mean, &self.std)?;
        if self.std.iter().any(|&x| x <= 0.0) {
            return Err(ConfigError::NonPositiveInputStd.into());
        }
        let h = (self.grid_range.1 - self.grid_range.0) / self.grid_size as f32;
        if self
            .knots
            .iter()
            .enumerate()
            .any(|(i, &k)| k != self.grid_range.0 + (i as f32 - self.order as f32) * h)
        {
            return Err(invalid().into());
        }
        Ok(())
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
        // ponytail: unlike forward_batch this does not record SPAN_CLAMPED_FLAG -
        // it keeps no history buffer, so no backward pass can consume it. If a
        // single-sample training path ever appears, it needs the flag too.
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
                let unclamped = (raw - self.mean[i]) / self.std[i];
                let z = unclamped.clamp(self.grid_range.0, self.grid_range.1);
                workspace.z_buffer.as_mut_slice()[input_start + i] = z;
                let span = find_span(z, &self.knots, self.order, self.grid_size);
                // Record saturation with the span: dz/dx is 0 here, and backward
                // cannot recover that from the clamped z alone.
                let clamped = if z == unclamped { 0 } else { SPAN_CLAMPED_FLAG };
                workspace.grid_indices[span_batch_start + i] = span as u32 | clamped;

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
    /// Returns [`ArkanError::ShapeMismatch`](crate::ArkanError::ShapeMismatch) if buffer sizes don't match expected dimensions.
    #[inline]
    pub fn try_forward_single(
        &self,
        input: &[f32],
        output: &mut [f32],
        basis_buf: &mut [f32],
    ) -> crate::ArkanResult<()> {
        use crate::ArkanError;

        if input.len() != self.in_dim {
            return Err(ArkanError::shape_mismatch(&[self.in_dim], &[input.len()]));
        }
        if output.len() != self.out_dim {
            return Err(ArkanError::shape_mismatch(&[self.out_dim], &[output.len()]));
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
    /// Returns [`ArkanError::ShapeMismatch`](crate::ArkanError::ShapeMismatch) if input/output sizes don't match expected dimensions.
    #[inline]
    pub fn try_forward_batch(
        &self,
        inputs: &[f32],
        outputs: &mut [f32],
        workspace: &mut Workspace,
    ) -> crate::ArkanResult<()> {
        use crate::buffer::checked_buffer_size;
        use crate::ArkanError;

        if inputs.len() % self.in_dim != 0 {
            return Err(ArkanError::shape_mismatch(&[self.in_dim], &[inputs.len()]));
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
                            let span = span_of(spans[span_batch_start + i]);
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
                    let span = span_of(spans[span_batch_start + i]);
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
            let span = span_of(spans[span_batch_start + i]);
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
                    let span = span_of(spans[span_batch_start + i]);
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
            let span = span_of(spans[span_batch_start + i]);
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
                let span = span_of(grid_indices[input_offset + i]);
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
                    let stored_span = grid_indices[span_batch_start + i];
                    let span = span_of(stored_span);
                    let start_idx = span - self.order;
                    let basis_start = basis_batch_start + i * self.basis_aligned;
                    // dz/dx for the *clamped* normalization: zero where the forward
                    // pass saturated, so a saturated input reports no sensitivity.
                    let dz_dx = if stored_span & SPAN_CLAMPED_FLAG != 0 {
                        0.0
                    } else {
                        1.0 / self.std[i].max(EPSILON)
                    };

                    for k in 0..self.local_basis_size {
                        let weight_idx = self.weight_index(j, i, start_idx + k);
                        let basis_val = basis_slice[basis_start + k];
                        grad_weights[weight_idx] += g_out * basis_val;

                        if let Some(ref mut gi) = grad_input {
                            let deriv = deriv_slice[basis_start + k];
                            gi[span_batch_start + i] +=
                                g_out * self.weights[weight_idx] * deriv * dz_dx;
                        }
                    }
                }
            }
        }
    }

    /// Parallel backward pass with deterministic logical chunks.
    ///
    /// Requires the `parallel` feature. Input gradients are overwritten; weight
    /// and bias gradients accumulate, as in [`backward`](Self::backward).
    /// Up to eight fixed logical chunks reduce in index order, independent of Rayon
    /// scheduling and pool size. Scratch is bounded by eight parameter buffers
    /// plus per-sample basis buffers, with no full-batch input-gradient copies.
    /// This convenience wrapper allocates scratch; network training reuses it.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanLayer, Workspace};
    ///
    /// let config = KanConfig::preset();
    /// let layer = KanLayer::new(4, 8, &config);
    /// let mut workspace = Workspace::new(&config);
    ///
    /// let batch_size = 128;
    /// let normalized_input = vec![0.5f32; batch_size * 4];
    /// let grid_indices = vec![3u32; batch_size * 4];
    /// let grad_output = vec![1.0f32; batch_size * 8];
    /// let mut grad_input = vec![0.0f32; batch_size * 4];
    /// let mut grad_weights = vec![0.0f32; layer.weights.len()];
    /// let mut grad_bias = vec![0.0f32; layer.bias.len()];
    ///
    /// layer.backward_parallel(
    ///     &normalized_input,
    ///     &grid_indices,
    ///     &grad_output,
    ///     Some(&mut grad_input),
    ///     &mut grad_weights,
    ///     &mut grad_bias,
    /// );
    /// ```
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub fn backward_parallel(
        &self,
        normalized_input: &[f32],
        grid_indices: &[u32],
        grad_output: &[f32],
        grad_input: Option<&mut [f32]>,
        grad_weights: &mut [f32],
        grad_bias: &mut [f32],
    ) {
        let mut scratch = ParallelBackwardScratch::default();
        self.backward_parallel_with_scratch(
            normalized_input,
            grid_indices,
            grad_output,
            grad_input,
            grad_weights,
            grad_bias,
            &mut scratch,
        );
    }

    /// Workspace kernel; callers provide buffers and forward history for this layer.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward_parallel_with_scratch(
        &self,
        normalized_input: &[f32],
        grid_indices: &[u32],
        grad_output: &[f32],
        grad_input: Option<&mut [f32]>,
        grad_weights: &mut [f32],
        grad_bias: &mut [f32],
        scratch: &mut ParallelBackwardScratch,
    ) {
        use rayon::prelude::*;
        let batch_size = normalized_input.len() / self.in_dim;
        debug_assert_eq!(normalized_input.len(), batch_size * self.in_dim);
        debug_assert_eq!(grid_indices.len(), normalized_input.len());
        debug_assert_eq!(grad_output.len(), batch_size * self.out_dim);
        debug_assert_eq!(grad_weights.len(), self.weights.len());
        debug_assert_eq!(grad_bias.len(), self.bias.len());
        if let Some(ref gi) = grad_input {
            debug_assert_eq!(gi.len(), normalized_input.len());
        }
        if batch_size == 0 {
            return;
        }
        scratch.prepare(self);
        // Avoid scheduling tiny jobs; the fixed minimum also applies across pool sizes.
        let samples_per_chunk = batch_size.div_ceil(scratch.chunks.len()).max(32);
        let chunk_count = batch_size.div_ceil(samples_per_chunk);
        let chunks = &mut scratch.chunks[..chunk_count];
        let compute = |index: usize, chunk: &mut ParallelChunk, mut gi: Option<&mut [f32]>| {
            chunk.weights.fill(0.0);
            chunk.bias.fill(0.0);
            if let Some(ref mut gi) = gi {
                gi.fill(0.0);
            }
            let first = index * samples_per_chunk;
            let end = (first + samples_per_chunk).min(batch_size);
            for b in first..end {
                let input_offset = b * self.in_dim;
                for i in 0..self.in_dim {
                    let basis_offset = i * self.basis_aligned;
                    compute_basis_and_deriv(
                        normalized_input[input_offset + i],
                        span_of(grid_indices[input_offset + i]),
                        &self.knots,
                        self.order,
                        &mut chunk.basis[basis_offset..basis_offset + self.local_basis_size],
                        &mut chunk.derivs[basis_offset..basis_offset + self.local_basis_size],
                    );
                }
                for j in 0..self.out_dim {
                    let g_out = grad_output[b * self.out_dim + j];
                    if g_out == 0.0 {
                        continue;
                    }
                    chunk.bias[j] += g_out;
                    for i in 0..self.in_dim {
                        let stored = grid_indices[input_offset + i];
                        let start_idx = span_of(stored) - self.order;
                        let basis_start = i * self.basis_aligned;
                        let dz_dx = if stored & SPAN_CLAMPED_FLAG != 0 {
                            0.0
                        } else {
                            1.0 / self.std[i].max(EPSILON)
                        };
                        for k in 0..self.local_basis_size {
                            let weight_idx = self.weight_index(j, i, start_idx + k);
                            chunk.weights[weight_idx] += g_out * chunk.basis[basis_start + k];
                            if let Some(ref mut gi) = gi {
                                gi[(b - first) * self.in_dim + i] += g_out
                                    * self.weights[weight_idx]
                                    * chunk.derivs[basis_start + k]
                                    * dz_dx;
                            }
                        }
                    }
                }
            }
        };
        if let Some(gi) = grad_input {
            chunks
                .par_iter_mut()
                .zip(gi.par_chunks_mut(samples_per_chunk * self.in_dim))
                .enumerate()
                .for_each(|(index, (chunk, gi))| compute(index, chunk, Some(gi)));
        } else {
            chunks
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, chunk)| compute(index, chunk, None));
        }
        // Floating-point sums must always follow the same logical chunk order.
        for chunk in chunks {
            for (out, &value) in grad_weights.iter_mut().zip(&chunk.weights) {
                *out += value;
            }
            for (out, &value) in grad_bias.iter_mut().zip(&chunk.bias) {
                *out += value;
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

    #[cfg(feature = "parallel")]
    #[test]
    fn reusable_parallel_scratch_handles_shapes_batches_and_pool_sizes() {
        let config = make_config(3, 5);
        let layers = [KanLayer::new(4, 8, &config), KanLayer::new(8, 4, &config)];
        let pools = [1, 4].map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        });
        let mut scratch = ParallelBackwardScratch::default();
        let mut workspace = Workspace::new(&config);
        for layer_index in [0, 1, 0, 1] {
            let layer = &layers[layer_index];
            for batch in [1, 65, 513, 17] {
                let input: Vec<f32> = (0..batch * layer.in_dim)
                    .map(|i| (i as f32 * 0.37).sin())
                    .collect();
                let spans: Vec<u32> = input
                    .iter()
                    .map(|&x| find_span(x, &layer.knots, layer.order, layer.grid_size) as u32)
                    .collect();
                let go: Vec<f32> = (0..batch * layer.out_dim)
                    .map(|i| (i as f32 * 0.73).cos())
                    .collect();
                let mut expected = vec![0.0; input.len()];
                let mut ew = vec![0.0; layer.weights.len()];
                let mut eb = vec![0.0; layer.bias.len()];
                layer.backward(
                    &input,
                    &spans,
                    &go,
                    Some(&mut expected),
                    &mut ew,
                    &mut eb,
                    &mut workspace,
                );
                let mut first = None;
                for pool in &pools {
                    let mut gi = vec![9.0; input.len()];
                    let mut gw = vec![0.0; ew.len()];
                    let mut gb = vec![0.0; eb.len()];
                    pool.install(|| {
                        layer.backward_parallel_with_scratch(
                            &input,
                            &spans,
                            &go,
                            Some(&mut gi),
                            &mut gw,
                            &mut gb,
                            &mut scratch,
                        )
                    });
                    assert_eq!(gi, expected);
                    for (&a, &b) in gw.iter().chain(&gb).zip(ew.iter().chain(&eb)) {
                        assert!((a - b).abs() < 1e-4);
                    }
                    if let Some((ref fw, ref fb)) = first {
                        assert_eq!(&gw, fw);
                        assert_eq!(&gb, fb);
                    } else {
                        first = Some((gw.clone(), gb.clone()));
                    }
                    // No-input-gradient calls reuse the same parameter scratch.
                    gw.fill(0.0);
                    gb.fill(0.0);
                    pool.install(|| {
                        layer.backward_parallel_with_scratch(
                            &input,
                            &spans,
                            &go,
                            None,
                            &mut gw,
                            &mut gb,
                            &mut scratch,
                        )
                    });
                    assert_eq!(&(gw, gb), first.as_ref().unwrap());
                }
            }
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

impl KanLayer {
    pub(crate) fn normalization(&self) -> Normalization<'_> {
        Normalization {
            mean: &self.mean,
            std: &self.std,
        }
    }

    #[cfg(feature = "serde")]
    pub(crate) fn wire_simd_width(&self) -> usize {
        self.simd_width
    }

    #[cfg(feature = "serde")]
    pub(crate) fn from_record(data: crate::format::layer::LayerRecord) -> crate::ArkanResult<Self> {
        crate::spline::validate_spline(data.grid_size, data.order, data.grid_range)?;
        // Recompute knots from grid_size, order, and grid_range
        let knots = compute_knots(data.grid_size, data.order, data.grid_range);

        let layer = KanLayer {
            in_dim: data.in_dim,
            out_dim: data.out_dim,
            order: data.order,
            grid_size: data.grid_size,
            global_basis_size: data.global_basis_size,
            local_basis_size: data.local_basis_size,
            basis_aligned: data.basis_aligned,
            grid_range: data.grid_range,
            knots,
            mean: data.mean,
            std: data.std,
            weights: data.weights,
            bias: data.bias,
            simd_width: data.simd_width,
        };
        layer.validate_layout()?;
        if layer
            .weights
            .iter()
            .chain(&layer.bias)
            .any(|x| !x.is_finite())
        {
            return Err(crate::ArkanError::cpu("non-finite layer parameters"));
        }
        Ok(layer)
    }
}
