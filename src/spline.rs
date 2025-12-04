//! B-spline basis computation with SIMD optimization.
//!
//! This module provides the core B-spline computations used by KAN layers:
//! - Knot vector generation
//! - Basis function evaluation (Cox-de Boor algorithm)
//! - Derivative computation for backpropagation
//! - SIMD-accelerated batch operations
//!
//! # B-spline Basics
//!
//! A B-spline of order `p` (degree `p`) is defined by:
//! - A knot vector `t₀ ≤ t₁ ≤ ... ≤ tₘ`
//! - `n = m - p - 1` basis functions
//!
//! The Cox-de Boor recursion computes basis values:
//!
//! $$B_{i,0}(x) = \begin{cases} 1 & t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$
//!
//! $$B_{i,p}(x) = \frac{x - t_i}{t_{i+p} - t_i} B_{i,p-1}(x) + \frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(x)$$
//!
//! # Partition of Unity
//!
//! B-spline basis functions sum to 1 at any point:
//! $\sum_i B_{i,p}(x) = 1$
//!
//! This property is preserved in all computations.
//!
//! # Example
//!
//! ```rust
//! use arkan::spline::{compute_knots, find_span, compute_basis};
//!
//! let knots = compute_knots(5, 3, (0.0, 1.0));
//! let x = 0.5;
//! let span = find_span(x, &knots, 3, 5);
//!
//! let mut basis = [0.0f32; 8];
//! compute_basis(x, span, &knots, 3, &mut basis);
//!
//! // Verify partition of unity
//! let sum: f32 = basis[..4].iter().sum();
//! assert!((sum - 1.0).abs() < 1e-5);
//! ```

use crate::config::{KanConfig, EPSILON, MAX_SPLINE_ORDER};
use wide::f32x8;

/// Maximum supported spline order (degree ≤ 7).
///
/// This is imported from config for consistency. Used for stack allocation
/// in the Cox-de Boor algorithm.
const MAX_ORDER: usize = MAX_SPLINE_ORDER;

/// Computes uniform B-spline knot vector.
///
/// For `grid_size` G and `order` k, generates `G + 2k + 1` knots
/// uniformly spaced within the grid range.
///
/// # Arguments
///
/// * `grid_size` - Number of grid intervals
/// * `order` - Spline order (degree)
/// * `grid_range` - (min, max) values for the grid
///
/// # Returns
///
/// Vector of knot values.
///
/// # Example
///
/// ```rust
/// use arkan::spline::compute_knots;
///
/// let knots = compute_knots(5, 3, (0.0, 1.0));
/// assert_eq!(knots.len(), 5 + 2 * 3 + 1); // 12 knots
/// ```
#[inline]
pub fn compute_knots(grid_size: usize, order: usize, grid_range: (f32, f32)) -> Vec<f32> {
    let (t_min, t_max) = grid_range;
    let n_knots = grid_size + 2 * order + 1;
    let h = (t_max - t_min) / grid_size as f32;

    (0..n_knots)
        .map(|i| t_min + (i as f32 - order as f32) * h)
        .collect()
}

/// Finds the knot span index for a given value in O(1) time.
///
/// Returns index `i` such that `knots[i] <= x < knots[i+1]`.
/// Uses uniform grid assumption for constant-time lookup.
///
/// # Arguments
///
/// * `x` - Input value to locate
/// * `knots` - Knot vector from [`compute_knots`]
/// * `order` - Spline order
/// * `grid_size` - Number of grid intervals
///
/// # Returns
///
/// Span index in range `[order, order + grid_size - 1]`.
#[inline]
pub fn find_span(x: f32, knots: &[f32], order: usize, grid_size: usize) -> usize {
    // Uniform grid, so we can compute span in O(1).
    let t_min = knots[order];
    let t_max = knots[order + grid_size];
    let step = (t_max - t_min).max(EPSILON) / grid_size as f32;
    let x_clamped = x.clamp(t_min, t_max);

    let raw_idx = ((x_clamped - t_min) / step).floor();
    let mut idx = raw_idx as isize;

    // Clamp to valid intervals to avoid OOB at boundaries (x == t_max).
    let max_interval = grid_size as isize - 1;
    if idx < 0 {
        idx = 0;
    } else if idx > max_interval {
        idx = max_interval;
    }

    (idx as usize) + order
}

/// Computes non-vanishing B-spline basis functions at point x.
///
/// Uses the Cox-de Boor algorithm. At any point, exactly `order + 1`
/// basis functions are non-zero.
///
/// # Arguments
///
/// * `x` - Evaluation point
/// * `span` - Knot span index from [`find_span`]
/// * `knots` - Knot vector
/// * `order` - Spline order
/// * `basis_out` - Output buffer (must have length ≥ `order + 1`)
///
/// # Output
///
/// Stores `order + 1` basis values in `basis_out`:
/// - `basis_out[0]` = N_{span-order}(x)
/// - `basis_out[1]` = N_{span-order+1}(x)
/// - ...
/// - `basis_out[order]` = N_{span}(x)
///
/// The values sum to 1.0 (partition of unity).
#[inline]
pub fn compute_basis(x: f32, span: usize, knots: &[f32], order: usize, basis_out: &mut [f32]) {
    debug_assert!(basis_out.len() > order);

    // Temporary storage for de Boor algorithm
    let mut left = [0.0f32; MAX_ORDER + 1];
    let mut right = [0.0f32; MAX_ORDER + 1];

    basis_out[0] = 1.0;

    for j in 1..=order {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;

        let mut saved = 0.0f32;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom.abs() > EPSILON {
                basis_out[r] / denom
            } else {
                0.0
            };
            basis_out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis_out[j] = saved;
    }
}

/// Computes basis functions and their first derivatives.
///
/// Used during backward pass to compute gradients.
///
/// # Arguments
///
/// * `x` - Evaluation point
/// * `span` - Knot span index
/// * `knots` - Knot vector
/// * `order` - Spline order
/// * `basis_out` - Output buffer for basis values (length ≥ `order + 1`)
/// * `deriv_out` - Output buffer for derivatives (length ≥ `order + 1`)
///
/// # Derivative Formula
///
/// $$\frac{d}{dx} B_{i,p}(x) = p \left( \frac{B_{i,p-1}(x)}{t_{i+p} - t_i} - \frac{B_{i+1,p-1}(x)}{t_{i+p+1} - t_{i+1}} \right)$$
#[inline]
pub fn compute_basis_and_deriv(
    x: f32,
    span: usize,
    knots: &[f32],
    order: usize,
    basis_out: &mut [f32],
    deriv_out: &mut [f32],
) {
    debug_assert!(basis_out.len() > order);
    debug_assert!(deriv_out.len() > order);

    // Compute basis functions of order-1 first
    let mut ndu = [[0.0f32; MAX_ORDER + 1]; MAX_ORDER + 1];
    let mut left = [0.0f32; MAX_ORDER + 1];
    let mut right = [0.0f32; MAX_ORDER + 1];

    ndu[0][0] = 1.0;

    for j in 1..=order {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;

        let mut saved = 0.0f32;
        for r in 0..j {
            // Lower triangle
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = if ndu[j][r].abs() > EPSILON {
                ndu[r][j - 1] / ndu[j][r]
            } else {
                0.0
            };

            // Upper triangle
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    // Load basis functions
    for j in 0..=order {
        basis_out[j] = ndu[j][order];
    }

    // Compute derivatives
    let mut a = [[0.0f32; MAX_ORDER + 1]; 2];

    for r in 0..=order {
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = 1.0;

        // Compute 1st derivative
        let mut d = 0.0f32;
        let rk = r as i32 - 1;
        let pk = order as i32 - 1;

        if r >= 1 {
            a[s2][0] = a[s1][0] / ndu[pk as usize + 1][rk as usize];
            d = a[s2][0] * ndu[rk as usize][pk as usize];
        }

        let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
        let j2 = if (r as i32 - 1) <= pk { 1 } else { order - r };

        if j1 <= j2 {
            for j in j1..=j2 {
                let idx = (rk + j as i32) as usize;
                let denom = ndu[pk as usize + 1][idx];
                a[s2][j] = if denom.abs() > EPSILON {
                    (a[s1][j] - a[s1][j - 1]) / denom
                } else {
                    0.0
                };
                d += a[s2][j] * ndu[idx][pk as usize];
            }
        }

        if r < order {
            a[s2][1] = -a[s1][0] / ndu[pk as usize + 1][r];
            d += a[s2][1] * ndu[r][pk as usize];
        }

        deriv_out[r] = d * order as f32;

        // Swap rows
        std::mem::swap(&mut s1, &mut s2);
    }
}

/// Batched normalization with SIMD acceleration.
///
/// Computes: `z = clamp((x - mean) / std, grid_min, grid_max)`
///
/// Processes 8 samples at a time using AVX/SSE SIMD instructions.
///
/// # Arguments
///
/// * `x` - Input values: `[batch_size * input_dim]` (row-major)
/// * `mean` - Per-feature mean: `[input_dim]`
/// * `std` - Per-feature std: `[input_dim]`
/// * `grid_range` - (min, max) clamp values
/// * `z_out` - Output buffer: `[batch_size * input_dim]`
#[inline]
pub fn normalize_batch(
    x: &[f32],    // [batch * input_dim], Row-Major
    mean: &[f32], // [input_dim]
    std: &[f32],  // [input_dim]
    grid_range: (f32, f32),
    z_out: &mut [f32], // [batch * input_dim]
) {
    let input_dim = mean.len();
    let batch_size = x.len() / input_dim;

    let grid_min = f32x8::splat(grid_range.0);
    let grid_max = f32x8::splat(grid_range.1);

    for i in 0..input_dim {
        let m = f32x8::splat(mean[i]);
        let s = f32x8::splat(std[i].max(EPSILON));

        // Process 8 samples at a time
        let mut b = 0;
        while b + 8 <= batch_size {
            // Gather x[b*input_dim + i], x[(b+1)*input_dim + i], ...
            let x_vals = f32x8::new([
                x[b * input_dim + i],
                x[(b + 1) * input_dim + i],
                x[(b + 2) * input_dim + i],
                x[(b + 3) * input_dim + i],
                x[(b + 4) * input_dim + i],
                x[(b + 5) * input_dim + i],
                x[(b + 6) * input_dim + i],
                x[(b + 7) * input_dim + i],
            ]);

            // Normalize and clamp
            let z = ((x_vals - m) / s).max(grid_min).min(grid_max);

            // Scatter to z_out
            let z_arr: [f32; 8] = z.into();
            for k in 0..8 {
                z_out[(b + k) * input_dim + i] = z_arr[k];
            }

            b += 8;
        }

        // Handle remainder
        while b < batch_size {
            let z = ((x[b * input_dim + i] - mean[i]) / std[i].max(EPSILON))
                .clamp(grid_range.0, grid_range.1);
            z_out[b * input_dim + i] = z;
            b += 1;
        }
    }
}

/// Batched basis computation for all inputs in a batch.
///
/// Computes normalized inputs, finds spans, and evaluates basis functions
/// for the entire batch at once.
///
/// # Arguments
///
/// * `z` - Normalized inputs: `[batch_size * input_dim]`
/// * `knots` - Knot vector: `[grid_size + 2*order + 1]`
/// * `config` - Network configuration
/// * `grid_indices` - Output span indices: `[batch_size * input_dim]`
/// * `basis_out` - Output basis values: `[batch_size * input_dim * basis_aligned]`
#[inline]
pub fn compute_basis_batch(
    z: &[f32],     // [batch * input_dim], normalized inputs
    knots: &[f32], // [grid_size + 2*order + 1]
    config: &KanConfig,
    grid_indices: &mut [u32], // [batch * input_dim]
    basis_out: &mut [f32],    // [batch * input_dim * basis_size_aligned]
) {
    let input_dim = config.input_dim;
    let batch_size = z.len() / input_dim;
    let order = config.spline_order;
    let grid_size = config.grid_size;
    let basis_aligned = config.basis_size_aligned();

    for b in 0..batch_size {
        for i in 0..input_dim {
            let x = z[b * input_dim + i];
            let span = find_span(x, knots, order, grid_size);

            grid_indices[b * input_dim + i] = span as u32;

            let basis_offset = (b * input_dim + i) * basis_aligned;
            compute_basis(
                x,
                span,
                knots,
                order,
                &mut basis_out[basis_offset..basis_offset + order + 1],
            );

            // Zero padding for alignment
            for j in (order + 1)..basis_aligned {
                basis_out[basis_offset + j] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_knots() {
        let knots = compute_knots(5, 3, (-1.0, 1.0));
        // G=5, k=3 → 5 + 6 + 1 = 12 knots
        assert_eq!(knots.len(), 12);

        // Check endpoints
        assert!(knots[3] <= -1.0 + 0.01); // First internal knot near grid_min
        assert!(knots[8] >= 1.0 - 0.01); // Last internal knot near grid_max
    }

    #[test]
    fn test_find_span() {
        let knots = compute_knots(5, 3, (0.0, 1.0));
        let order = 3;
        let grid_size = 5;

        // At grid minimum
        let span = find_span(0.0, &knots, order, grid_size);
        assert!(span >= order);

        // At grid maximum
        let span = find_span(1.0, &knots, order, grid_size);
        assert!(span <= grid_size + order - 1);

        // Middle value
        let span = find_span(0.5, &knots, order, grid_size);
        assert!(knots[span] <= 0.5);
        assert!(knots[span + 1] >= 0.5);
    }

    #[test]
    fn test_basis_partition_of_unity() {
        let knots = compute_knots(5, 3, (0.0, 1.0));
        let order = 3;
        let grid_size = 5;

        // Test at several points
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let span = find_span(x, &knots, order, grid_size);
            let mut basis = [0.0f32; 8];
            compute_basis(x, span, &knots, order, &mut basis);

            // Sum should be 1.0 (partition of unity)
            let sum: f32 = basis[..order + 1].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Partition of unity failed at x={}: sum={}",
                x,
                sum
            );

            // All values should be non-negative
            for b in &basis[..order + 1] {
                assert!(*b >= -1e-6, "Negative basis value at x={}", x);
            }
        }
    }

    #[test]
    fn test_normalize_batch() {
        let batch = 16;
        let input_dim = 4;

        let x: Vec<f32> = (0..batch * input_dim).map(|i| i as f32 / 10.0).collect();
        let mean = vec![0.5; input_dim];
        let std = vec![0.3; input_dim];
        let mut z = vec![0.0f32; batch * input_dim];

        normalize_batch(&x, &mean, &std, (-3.0, 3.0), &mut z);

        // Check all values are in range
        for &val in &z {
            assert!(val >= -3.0 && val <= 3.0);
        }
    }
}
