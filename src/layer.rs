//! KAN Layer Implementation with B-spline basis functions.
//!
//! # Mathematical Foundation
//!
//! Each layer computes: y[j] = Σ_i Σ_k c[j,i,k] * B_k(x[i])
//!
//! where:
//! - x[i] is the i-th input (normalized to [0,1])
//! - B_k are B-spline basis functions of given order
//! - c[j,i,k] are learnable spline coefficients
//!
//! # Spline Indexing (Critical!)
//!
//! For order `p` and `n` grid intervals:
//! - Global basis count: `n + p` (NOT `p + 1`!)
//! - Each input point activates exactly `p + 1` consecutive basis functions
//! - Span `s` = floor(x * n), clamped to [p, n+p-1]
//! - Active weights start at index `s - p` in global coefficient array

use crate::buffer::Workspace;
use crate::config::{KanConfig, EPSILON};
use crate::spline::{compute_basis, compute_knots, find_span};
use wide::{f32x4, f32x8};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A single KAN layer with learnable spline coefficients.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KanLayer {
    /// Input dimension
    pub in_dim: usize,
    /// Output dimension
    pub out_dim: usize,
    /// Spline order (degree)
    pub order: usize,
    /// Number of grid intervals
    pub grid_size: usize,
    /// Global basis size = grid_size + order
    pub global_basis_size: usize,
    /// Local basis size = order + 1 (active functions per input)
    pub local_basis_size: usize,
    /// Local basis size aligned to SIMD width
    pub basis_aligned: usize,
    /// Grid range (min, max)
    pub grid_range: (f32, f32),
    /// Precomputed knot vector
    #[cfg_attr(feature = "serde", serde(skip))]
    knots: Vec<f32>,
    /// Per-input normalization mean
    pub mean: Vec<f32>,
    /// Per-input normalization std (clamped by EPSILON)
    pub std: Vec<f32>,
    /// Spline coefficients: [out_dim][in_dim][global_basis_size]
    /// Stored as flat array for cache efficiency
    pub weights: Vec<f32>,
    /// Bias terms for each output
    pub bias: Vec<f32>,
    /// SIMD width for aligned operations (reserved for future use)
    #[allow(dead_code)]
    simd_width: usize,
}

impl KanLayer {
    /// Creates a new KAN layer with the given dimensions.
    pub fn new(in_dim: usize, out_dim: usize, config: &KanConfig) -> Self {
        assert!(in_dim > 0, "Input dimension must be positive");
        assert!(out_dim > 0, "Output dimension must be positive");

        let order = config.spline_order;
        let grid_size = config.grid_size;
        let simd_width = config.simd_width;
        let grid_range = config.grid_range;

        // Global basis: one B-spline per knot interval that can be active
        let global_basis_size = grid_size + order;

        // Local: only order+1 functions are non-zero at any point
        let local_basis_size = order + 1;
        let basis_aligned = (local_basis_size + simd_width - 1) & !(simd_width - 1);

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
        let total_weights = out_dim * in_dim * global_basis_size;
        let scale = (2.0 / (in_dim + out_dim) as f32).sqrt() * 0.1;
        let weights = (0..total_weights)
            .map(|i| {
                // Simple deterministic "random" for reproducibility
                let hash = ((i as u64 * 2654435761) % 1000) as f32 / 1000.0 - 0.5;
                hash * scale
            })
            .collect();

        let bias = vec![0.0; out_dim];

        Self {
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
        }
    }

    /// Creates layer from config (alias for new).
    pub fn from_config(in_dim: usize, out_dim: usize, config: &KanConfig) -> Self {
        Self::new(in_dim, out_dim, config)
    }

    /// Returns the weight index for coefficient c[out_idx, in_idx, basis_idx].
    #[inline]
    fn weight_index(&self, out_idx: usize, in_idx: usize, basis_idx: usize) -> usize {
        (out_idx * self.in_dim + in_idx) * self.global_basis_size + basis_idx
    }

    /// Basis size aligned to SIMD width.
    #[inline]
    pub fn basis_aligned(&self) -> usize {
        self.basis_aligned
    }

    /// Total number of parameters.
    #[inline]
    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Sets normalization parameters (mean/std).
    pub fn set_normalization(&mut self, mean: &[f32], std: &[f32]) {
        assert_eq!(mean.len(), self.in_dim);
        assert_eq!(std.len(), self.in_dim);
        self.mean.clear();
        self.mean.extend_from_slice(mean);
        self.std.clear();
        self.std.extend(std.iter().map(|s| s.max(EPSILON)));
    }

    /// Forward pass for a single input sample (with simple basis buffer).
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

    /// Forward pass for a batch of samples using workspace.
    pub fn forward_batch(&self, inputs: &[f32], outputs: &mut [f32], workspace: &mut Workspace) {
        let batch_size = inputs.len() / self.in_dim;
        debug_assert_eq!(inputs.len(), batch_size * self.in_dim);
        debug_assert_eq!(outputs.len(), batch_size * self.out_dim);

        // Ensure workspace has enough capacity
        let basis_needed = batch_size * self.in_dim * self.basis_aligned;
        if workspace.basis_values.len() < basis_needed {
            workspace.basis_values.resize(basis_needed);
        }

        let spans_needed = batch_size * self.in_dim;
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
    }

    /// Accumulates outputs for a batch.
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
    pub fn num_parameters(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Gets all parameters as a flat slice for optimization.
    pub fn get_parameters(&self) -> Vec<f32> {
        let mut params = self.weights.clone();
        params.extend(&self.bias);
        params
    }

    /// Sets parameters from a flat slice.
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

    /// Backward pass: computes gradients for weights and bias.
    pub fn backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        _grad_input: Option<&mut [f32]>,
        grad_weights: &mut [f32],
        grad_bias: &mut [f32],
        workspace: &Workspace,
    ) {
        let batch_size = input.len() / self.in_dim;
        debug_assert_eq!(grad_output.len(), batch_size * self.out_dim);
        debug_assert_eq!(grad_weights.len(), self.weights.len());
        debug_assert_eq!(grad_bias.len(), self.bias.len());

        let basis_slice = workspace.basis_values.as_slice();
        let spans = &workspace.grid_indices;

        for b in 0..batch_size {
            let span_batch_start = b * self.in_dim;
            let basis_batch_start = b * self.in_dim * self.basis_aligned;
            let grad_out_start = b * self.out_dim;

            // Gradient w.r.t. bias: just copy grad_output
            for j in 0..self.out_dim {
                grad_bias[j] += grad_output[grad_out_start + j];
            }

            // Gradient w.r.t. weights: outer product of grad_output and basis
            for j in 0..self.out_dim {
                let g_out = grad_output[grad_out_start + j];

                for i in 0..self.in_dim {
                    let span = spans[span_batch_start + i] as usize;
                    let start_idx = span - self.order;
                    let basis_start = basis_batch_start + i * self.basis_aligned;

                    for k in 0..self.local_basis_size {
                        let weight_idx = self.weight_index(j, i, start_idx + k);
                        grad_weights[weight_idx] += g_out * basis_slice[basis_start + k];
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
}
