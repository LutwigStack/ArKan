//! B-spline basis computation with SIMD optimization.

use crate::config::{KanConfig, EPSILON};
use wide::f32x8;

/// Computes uniform B-spline knot vector.
/// For grid_size G and order k, we have G+2k+1 knots.
#[inline]
pub fn compute_knots(grid_size: usize, order: usize, grid_range: (f32, f32)) -> Vec<f32> {
    let (t_min, t_max) = grid_range;
    let n_knots = grid_size + 2 * order + 1;
    let h = (t_max - t_min) / grid_size as f32;
    
    (0..n_knots)
        .map(|i| t_min + (i as f32 - order as f32) * h)
        .collect()
}

/// Finds the knot span index for a given value.
/// Returns index i such that knots[i] <= x < knots[i+1].
#[inline]
pub fn find_span(x: f32, knots: &[f32], order: usize, grid_size: usize) -> usize {
    let n = grid_size + order; // Last valid span index
    
    // Handle boundary cases
    if x >= knots[n] {
        return n - 1;
    }
    if x <= knots[order] {
        return order;
    }
    
    // Binary search
    let mut low = order;
    let mut high = n;
    let mut mid = (low + high) / 2;
    
    while x < knots[mid] || x >= knots[mid + 1] {
        if x < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    
    mid
}

/// Computes non-vanishing B-spline basis functions at x.
/// Uses the Cox-de Boor algorithm.
/// 
/// Returns (order+1) values in `basis_out`.
/// The basis functions are N_{span-order}, N_{span-order+1}, ..., N_{span}.
#[inline]
pub fn compute_basis(
    x: f32,
    span: usize,
    knots: &[f32],
    order: usize,
    basis_out: &mut [f32],
) {
    debug_assert!(basis_out.len() >= order + 1);
    
    // Temporary storage for de Boor algorithm
    let mut left = [0.0f32; 8];  // Max order 7
    let mut right = [0.0f32; 8];
    
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

/// Computes basis functions and their derivatives.
/// 
/// Returns:
/// - basis_out: [order+1] basis function values
/// - deriv_out: [order+1] first derivatives
#[inline]
pub fn compute_basis_and_deriv(
    x: f32,
    span: usize,
    knots: &[f32],
    order: usize,
    basis_out: &mut [f32],
    deriv_out: &mut [f32],
) {
    debug_assert!(basis_out.len() >= order + 1);
    debug_assert!(deriv_out.len() >= order + 1);
    
    // Compute basis functions of order-1 first
    let mut ndu = [[0.0f32; 8]; 8]; // Max order 7
    let mut left = [0.0f32; 8];
    let mut right = [0.0f32; 8];
    
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
    let mut a = [[0.0f32; 8]; 2];
    
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
        let j2 = if (r as i32 - 1) <= pk { 1 } else { (order - r) as usize };
        
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
        
        if r <= order - 1 {
            a[s2][1] = -a[s1][0] / ndu[pk as usize + 1][r];
            d += a[s2][1] * ndu[r][pk as usize];
        }
        
        deriv_out[r] = d * order as f32;
        
        // Swap rows
        std::mem::swap(&mut s1, &mut s2);
    }
}

/// Batched normalization: z = clamp((x - mean) / std, grid_min, grid_max)
/// 
/// Processes 8 samples at a time using SIMD.
#[inline]
pub fn normalize_batch(
    x: &[f32],           // [batch * input_dim], Row-Major
    mean: &[f32],        // [input_dim]
    std: &[f32],         // [input_dim]
    grid_range: (f32, f32),
    z_out: &mut [f32],   // [batch * input_dim]
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

/// Batched basis computation for all inputs.
/// 
/// For each sample and input, computes basis functions and stores in basis_out.
#[inline]
pub fn compute_basis_batch(
    z: &[f32],              // [batch * input_dim], normalized inputs
    knots: &[f32],          // [grid_size + 2*order + 1]
    config: &KanConfig,
    grid_indices: &mut [u32],   // [batch * input_dim]
    basis_out: &mut [f32],      // [batch * input_dim * basis_size_aligned]
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
        assert!(knots[8] >= 1.0 - 0.01);  // Last internal knot near grid_max
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
                x, sum
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
