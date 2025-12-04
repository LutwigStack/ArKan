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

    // Use round instead of floor to handle floating point precision issues
    // at grid boundaries. When x is exactly at a knot, (x - t_min) / step
    // should give an integer, but due to f32 precision it might be 0.9999... or 1.0001...
    // Adding 0.5 * EPSILON before floor handles this edge case.
    let ratio = (x_clamped - t_min) / step;
    let raw_idx = (ratio + EPSILON).floor();
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

    // First compute the basis functions of order p (stored in basis_out)
    compute_basis(x, span, knots, order, basis_out);

    if order == 0 {
        // Order 0 basis functions have derivative 0 (piecewise constant)
        for d in deriv_out.iter_mut().take(order + 1) {
            *d = 0.0;
        }
        return;
    }

    // Now compute derivatives using the recurrence relation:
    // B'_{i,p}(x) = p * (B_{i,p-1}(x) / (t_{i+p} - t_i) - B_{i+1,p-1}(x) / (t_{i+p+1} - t_{i+1}))
    //
    // For span s and order p, the active basis functions are B_{s-p,p} through B_{s,p}
    // Their derivatives depend on B_{s-p,p-1} through B_{s+1,p-1}

    let prev_order = order - 1;
    let start_idx = span - order; // First global index: B_{start_idx, order}

    // We need basis functions of order-1 from indices start_idx to span+1
    // That's (order + 2) functions: B_{start_idx, p-1}, ..., B_{span+1, p-1}

    // Compute basis functions of order-1
    // The span for order-1 may differ, but the basis values are the same
    // regardless of which span we use, as long as x is in the support.

    // Build order p-1 basis functions from scratch using De Boor recursion
    let mut basis_prev = [0.0f32; MAX_ORDER + 2];

    // Initialize order 0: B_{j,0}(x) = 1 if t_j <= x < t_{j+1}, else 0
    // We need values for j from start_idx to span+1
    for j in start_idx..=span + 1 {
        let idx = j - start_idx;
        if idx < MAX_ORDER + 2 {
            basis_prev[idx] = if x >= knots[j] && x < knots[j + 1] {
                1.0
            } else {
                0.0
            };
        }
    }

    // Handle the right endpoint (x == knots[span+1] case)
    // The rightmost non-zero basis should be 1 at the right boundary
    if (x - knots[span + 1]).abs() < EPSILON {
        for j in start_idx..=span + 1 {
            let idx = j - start_idx;
            if idx < MAX_ORDER + 2 {
                basis_prev[idx] = 0.0;
            }
        }
        // Set the rightmost basis to 1
        let idx = span - start_idx;
        if idx < MAX_ORDER + 2 {
            basis_prev[idx] = 1.0;
        }
    }

    // Build up to order p-1 using De Boor recursion
    // We need (order + 2) basis functions of order (p-1) for the derivative formula
    // basis_prev[i] = B_{start_idx + i, p-1}(x) for i = 0..=order+1
    for p in 1..=prev_order {
        let mut new_basis = [0.0f32; MAX_ORDER + 2];
        // We need (order + 2 - p) basis functions at level p
        // But that's wrong! We need (order + 2) basis functions at level p-1
        // Actually: at level p, we have (n_knots - p - 1) basis functions total
        // But we only care about those in range [start_idx, span+1-p]
        // For derivatives, we need B_{j,p-1} for j in [start_idx, span+1]
        // That's (span+1 - start_idx + 1) = (span + 1 - (span - order) + 1) = order + 2 functions
        
        // But at level p, the number of functions is reduced by p from level 0
        // Actually, the issue is: we need functions at level prev_order = order - 1
        // At level 0: we have (order + 2) functions initialized
        // At level 1: we need (order + 1) functions
        // At level prev_order = order - 1: we need 3 functions (when order >= 2)
        
        // The correct formula: at level p, we need (order + 2 - p) functions
        // Final level is prev_order, so we need (order + 2 - prev_order) = (order + 2 - (order-1)) = 3 functions
        // But for derivatives of order p, we need (order + 2) basis functions of order (p-1)!
        
        // The bug is: we're computing too few functions at the final level.
        // We need basis_prev[0..=order+1] but we're only computing num_funcs = order + 2 - p
        
        // Fix: always compute (order + 2) functions at every level
        let num_funcs = order + 2; // Always need order+2 functions for derivative computation

        for k in 0..num_funcs {
            let j = start_idx + k; // Global index of this basis function

            // B_{j,p}(x) = (x - t_j)/(t_{j+p} - t_j) * B_{j,p-1}(x)
            //           + (t_{j+p+1} - x)/(t_{j+p+1} - t_{j+1}) * B_{j+1,p-1}(x)
            
            // Make sure we don't go out of bounds on knots
            if j + p + 1 >= knots.len() {
                new_basis[k] = 0.0;
                continue;
            }
            
            let denom1 = knots[j + p] - knots[j];
            let term1 = if denom1.abs() > EPSILON {
                (x - knots[j]) / denom1 * basis_prev[k]
            } else {
                0.0
            };

            let denom2 = knots[j + p + 1] - knots[j + 1];
            let term2 = if denom2.abs() > EPSILON && k + 1 < MAX_ORDER + 2 {
                (knots[j + p + 1] - x) / denom2 * basis_prev[k + 1]
            } else {
                0.0
            };

            new_basis[k] = term1 + term2;
        }

        basis_prev = new_basis;
    }

    // Now basis_prev contains B_{start_idx, p-1}, B_{start_idx+1, p-1}, ..., B_{span+1, p-1}
    // Compute derivatives
    for i in 0..=order {
        let idx = start_idx + i; // Global index of this order-p basis function

        // B'_{idx,p}(x) = p * (B_{idx,p-1}(x)/(t_{idx+p} - t_idx) - B_{idx+1,p-1}(x)/(t_{idx+p+1} - t_{idx+1}))
        let denom1 = knots[idx + order] - knots[idx];
        let term1 = if denom1.abs() > EPSILON {
            basis_prev[i] / denom1
        } else {
            0.0
        };

        let denom2 = knots[idx + order + 1] - knots[idx + 1];
        let term2 = if denom2.abs() > EPSILON {
            basis_prev[i + 1] / denom2
        } else {
            0.0
        };

        deriv_out[i] = (order as f32) * (term1 - term2);
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
