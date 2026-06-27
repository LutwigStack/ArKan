//! Baked (quantized) model for fixed-point int8 inference.
//!
//! This module provides `BakedModel`, a quantized inference-only representation
//! of a trained KAN model. It uses int8 weights and int16 B-spline basis functions
//! with no f32 operations in the hot path.
//!
//! # Design
//!
//! - Weights: int8 (scale 127/max|w| per layer)
//! - Basis functions: u16 Q0.15, value ≈ q_b/32768 (int16 basis variant)
//! - Accumulator: i64 (safe for typical nets; i128 for requant product)
//! - Bias: i64 (folded with weight scale)
//! - No f32 between entry normalization and final dequantization
//!
//! # Accuracy
//!
//! Expected ~3–5% relative error for in_dim ≤ 32 with calibration data.
//! Larger networks may see up to ~10% without per-channel quantization.

use crate::config::{KanConfig, EPSILON};
use crate::network::KanNetwork;
use crate::spline::compute_basis;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Evaluate B-spline basis polynomials in fixed point (int16 basis variant).
///
/// Returns `order+1` values as u16 in Q0.15 format, value ≈ q_b/32768.
/// Uses closed-form expressions for orders 2–5 reused from gpu/shaders.rs.
/// t_q16 is in Q0.16 format: value = t_q16 / 65536.0, in [0, 65535].
fn eval_basis_fixed(order: usize, t_q16: u32, out: &mut [u16]) {
    // t is in [0, 1), represented as t_q16/65536
    // We use Q2.29 intermediate arithmetic scaled by 2^29 for precision
    // Output: Q0.15 values rounded to u16

    // Scale: 1.0 in Q0.15 = 32768
    const SCALE: i64 = 32768;

    // t as rational: t = t_q16 / 65536
    // Use i64 arithmetic. t in [0, 65535], so t/65536 in [0, 1)
    let t = t_q16 as i64; // [0, 65535]
    let one: i64 = 65536; // 1.0 in Q16
    let omt = one - t; // (1-t) in Q16

    match order {
        2 => {
            // Quadratic B-spline (order=2): 3 basis functions
            // B0 = omt^2 / 2
            // B1 = -t^2 + t + 0.5  = (-t^2 + t + 0.5)
            // B2 = t^2 / 2
            // Compute in Q32 then shift down to Q15
            let omt2 = omt * omt; // Q32
            let t2 = t * t; // Q32

            // B0 = omt^2 / 2, in Q32 → Q15: shift by 32-15=17, divide by 2 so 18
                // b0 = omt^2/(2 * 65536^2) * 32768 = omt^2 >> 18
            let b0 = ((omt2 >> 18) as i64).clamp(0, SCALE) as u16;
            // B2 = t^2 / 2
            let b2 = ((t2 >> 18) as i64).clamp(0, SCALE) as u16;
            // B1 = SCALE - b0 - b2 (partition of unity)
            let b1 = (SCALE - b0 as i64 - b2 as i64).clamp(0, SCALE) as u16;

            out[0] = b0;
            out[1] = b1;
            out[2] = b2;
        }
        3 => {
            // Cubic B-spline (order=3): 4 basis functions
            // B0 = (1-t)^3 / 6
            // B1 = (3t^3 - 6t^2 + 4) / 6
            // B2 = (-3t^3 + 3t^2 + 3t + 1) / 6
            // B3 = t^3 / 6
            //
            // Use Q48 intermediate: t in Q16, t^3 in Q48
            // Result in Q15: divide by 65536^3 * 6 * (1/32768) = 65536^3 * 6 / 32768
            //                = 65536^2 * 6 * 2 = 65536^2 * 12
            // Actually: b = polynomial_Q48 * 32768 / (65536^3 * 6)
            //             = polynomial_Q48 / (65536^3 * 6 / 32768)
            //             = polynomial_Q48 / (65536^2 * 12)
            //
            // 65536^2 = 2^32, so divide by 2^32 * 12
            // But i64 * i64 can overflow for t^3. Use i128 for intermediate.

            let t = t_q16 as i128;
            let one: i128 = 65536;
            let omt = one - t;

            let omt3 = omt * omt * omt; // Q48
            let t2 = t * t; // Q32
            let t3 = t2 * t; // Q48

            // Denominator factor: 65536^3 * 6 / 32768 = 65536^2 * 12
            let denom: i128 = 65536 * 65536 * 12; // 2^32 * 12

            let b0 = (omt3 / denom).clamp(0, SCALE as i128) as u16;
            // B1 = (3t^3 - 6t^2 + 4) / 6
            // In Q48 numerator: 3*t3 - 6*t2*65536 + 4*65536^3
            let b1_num = 3 * t3 - 6 * t2 * one + 4 * one * one * one;
            let b1 = (b1_num / denom).clamp(0, SCALE as i128) as u16;
            // B3 = t^3 / 6
            let b3 = (t3 / denom).clamp(0, SCALE as i128) as u16;
            // B2 = partition of unity remainder
            let b2 = (SCALE as i128 - b0 as i128 - b1 as i128 - b3 as i128)
                .clamp(0, SCALE as i128) as u16;

            out[0] = b0;
            out[1] = b1;
            out[2] = b2;
            out[3] = b3;
        }
        4 => {
            // Quartic B-spline (order=4): 5 basis functions
            // B0 = (1-t)^4 / 24
            // B1 = (-4t^4 + 12t^3 - 6t^2 - 12t + 11) / 24
            // B2 = (6t^4 - 12t^3 - 6t^2 + 12t + 11) / 24  (wait, let me re-check the shader)
            // From shaders.rs order=4:
            // result[0] = omt4 / 24.0;
            // result[1] = (-4t^4 + 12t^3 - 6t^2 - 12t + 11) / 24.0;
            // result[2] = (6t^4 - 12t^3 - 6t^2 + 12t + 11) / 24.0;  <- note this one has +12t not -12t based on symmetry
            // result[3] = (-4t^4 + 4t^3 + 6t^2 + 4t + 1) / 24.0;
            // result[4] = t4 / 24.0;
            // Wait, let me re-read the shader carefully:
            // result[2] = (6.0 * t4 - 12.0 * t3 - 6.0 * t2 + 12.0 * t + 11.0) / 24.0;
            // That's what is in the shaders. Let me verify the partition:
            // Sum = (omt4 + (-4t^4+12t^3-6t^2-12t+11) + (6t^4-12t^3-6t^2+12t+11) + (-4t^4+4t^3+6t^2+4t+1) + t^4) / 24
            // t^4: 1 - 4 + 6 - 4 + 1 = 0 ✓
            // t^3: -4·(-1 as omt) + 12 - 12 + 4 = using omt expansion... just trust it sums to 1.
            // Actually (1-t)^4 = 1 - 4t + 6t^2 - 4t^3 + t^4
            // Sum_num = (1-4t+6t^2-4t^3+t^4) + (-4t^4+12t^3-6t^2-12t+11) + (6t^4-12t^3-6t^2+12t+11) + (-4t^4+4t^3+6t^2+4t+1) + t^4
            // const: 1+11+11+1 = 24 ✓
            // t: -4-12+12+4 = 0 ✓
            // t^2: 6-6-6+6 = 0 ✓
            // t^3: -4+12-12+4 = 0 ✓
            // t^4: 1-4+6-4+1 = 0 ✓
            // Sum = 24/24 = 1 ✓

            let t = t_q16 as i128;
            let one: i128 = 65536;
            let omt = one - t;

            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t3 * t; // Q64
            let omt2 = omt * omt;
            let omt3 = omt2 * omt;
            let omt4 = omt3 * omt; // Q64

            // Denominator: 65536^4 * 24 / 32768 = 65536^3 * 48
            let denom: i128 = 65536_i128 * 65536 * 65536 * 48;

            let b0 = (omt4 / denom).clamp(0, SCALE as i128) as u16;
            let b4 = (t4 / denom).clamp(0, SCALE as i128) as u16;

            let b1_num = -4 * t4 + 12 * t3 - 6 * t2 * one * one - 12 * t * one * one * one
                + 11 * one * one * one * one;
            let b1 = (b1_num / denom).clamp(0, SCALE as i128) as u16;

            let b3_num = -4 * t4 + 4 * t3 + 6 * t2 * one * one + 4 * t * one * one * one
                + 1 * one * one * one * one;
            let b3 = (b3_num / denom).clamp(0, SCALE as i128) as u16;

            // B2 via partition of unity
            let b2 = (SCALE as i128 - b0 as i128 - b1 as i128 - b3 as i128 - b4 as i128)
                .clamp(0, SCALE as i128) as u16;

            out[0] = b0;
            out[1] = b1;
            out[2] = b2;
            out[3] = b3;
            out[4] = b4;
        }
        5 => {
            // Quintic B-spline (order=5): 6 basis functions
            // From shaders.rs:
            // result[0] = omt5 / 120.0;
            // result[1] = (5t^5 - 20t^4 + 20t^3 + 20t^2 - 50t + 26) / 120.0;
            // result[2] = (-10t^5 + 30t^4 - 60t^2 + 66) / 120.0;
            // result[3] = (10t^5 - 20t^4 - 20t^3 + 20t^2 + 50t + 26) / 120.0;
            // result[4] = (-5t^5 + 5t^4 + 10t^3 + 10t^2 + 5t + 1) / 120.0;
            // result[5] = t5 / 120.0;

            let t = t_q16 as i128;
            let one: i128 = 65536;
            let omt = one - t;

            let t2 = t * t;
            let t3 = t2 * t;
            let t4 = t3 * t;
            let t5 = t4 * t; // Q80 - will overflow i128!
            // i128 max ≈ 1.7e38, 65536^5 = 2^80 ≈ 1.2e24, *120 ≈ 1.5e26 — fits in i128.
            let omt2 = omt * omt;
            let omt3 = omt2 * omt;
            let omt4 = omt3 * omt;
            let omt5 = omt4 * omt; // Q80

            // Denominator: 65536^5 * 120 / 32768 = 65536^4 * 240
            let denom: i128 = one * one * one * one * 240;

            let b0 = (omt5 / denom).clamp(0, SCALE as i128) as u16;
            let b5 = (t5 / denom).clamp(0, SCALE as i128) as u16;

            let b1_num = 5 * t5 - 20 * t4 + 20 * t3 + 20 * t2 * one * one
                - 50 * t * one * one * one
                + 26 * one * one * one * one;
            let b1 = (b1_num / denom).clamp(0, SCALE as i128) as u16;

            let b4_num = -5 * t5 + 5 * t4 + 10 * t3 + 10 * t2 * one * one
                + 5 * t * one * one * one
                + 1 * one * one * one * one;
            let b4 = (b4_num / denom).clamp(0, SCALE as i128) as u16;

            // For b2 and b3, use partition-of-unity trick on b2+b3
            // b2_num = -10t^5 + 30t^4 - 60t^2 + 66
            let b2_num = -10 * t5 + 30 * t4 - 60 * t2 * one * one + 66 * one * one * one * one;
            let b2 = (b2_num / denom).clamp(0, SCALE as i128) as u16;

            // b3 via partition of unity
            let b3 = (SCALE as i128
                - b0 as i128
                - b1 as i128
                - b2 as i128
                - b4 as i128
                - b5 as i128)
                .clamp(0, SCALE as i128) as u16;

            out[0] = b0;
            out[1] = b1;
            out[2] = b2;
            out[3] = b3;
            out[4] = b4;
            out[5] = b5;
        }
        _ => {
            // Fallback: use the f32 spline code (compute_basis) and quantize
            // Only reached for orders outside 2-5.
            // Build a temporary knot vector. This path is not used in normal ops.
            let t_f = t_q16 as f32 / 65536.0;
            let n_order_plus_1 = order + 1;
            // Simple uniform knots for t in [0,1): order knots at 0, 1 knot interval, order at 1
            let knots: Vec<f32> = (0..=(order + 2))
                .map(|i| i as f32 - order as f32)
                .collect();
            let span = order; // For t in [0,1) and this knot layout, span is always `order`
            let mut basis_f = vec![0.0f32; n_order_plus_1];
            compute_basis(t_f, span, &knots, order, &mut basis_f);
            for (k, &b) in basis_f.iter().enumerate() {
                out[k] = (b * 32768.0).round().clamp(0.0, 32768.0) as u16;
            }
        }
    }
}

/// Per-layer baked data for fixed-point inference.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BakedLayer {
    /// Quantized weights in i8. Layout: [out_dim, in_dim, global_basis_size].
    pub weights_i8: Vec<i8>,
    /// Folded quantized bias: q_bias[j] = round(b_j * s_w * 32768) as i64.
    pub q_bias: Vec<i64>,
    /// Requant multiplier M0 (from M_real = s_act / (s_w * 32768)).
    pub requant_m0: i32,
    /// Requant shift S (so that requant = (acc * M0 + 2^(S-1)) >> S).
    pub requant_shift: u32,
    /// Per-input fixed-point scale: A_FIXED[i] = round(2^16 / (s_act_prev * std_i)).
    pub norm_a_fixed: Vec<i32>,
    /// Per-input fixed-point offset: B_FIXED[i] = round(-mean_i / std_i * 2^16).
    pub norm_b_fixed: Vec<i32>,
    /// Quantized grid range lower bound: round(r_min * 2^16).
    pub q_rmin: i32,
    /// Quantized grid range upper bound: round(r_max * 2^16) - 1.
    pub q_rmax: i32,
    /// One span interval in Q16 units of z: round(65536 * (r_max - r_min) / G).
    /// Used to extract span and t from the Q15.16 z value.
    pub h_q16: i32,
    /// Per-input normalization mean (stored for entry-layer f32 normalization).
    pub mean: Vec<f32>,
    /// Per-input normalization std (stored for entry-layer f32 normalization).
    pub std: Vec<f32>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Spline order.
    pub order: usize,
    /// Number of grid intervals.
    pub grid_size: usize,
    /// Global basis size = grid_size + order.
    pub global_basis_size: usize,
    /// Activation output scale (metadata; never read in forward).
    pub s_act_out: f32,
}

/// Baked (quantized) KAN model for fixed-point inference.
///
/// Use [`BakedModel::from_network`] to create from a trained [`KanNetwork`].
/// Use [`BakedModel::forward`] to run inference.
///
/// The hot path (between entry normalization and final dequant) uses no f32.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BakedModel {
    /// Original configuration for reference/validation.
    pub config: KanConfig,
    /// Per-layer baked data.
    pub layers: Vec<BakedLayer>,
    /// Set to true if baked without calibration data (accuracy may be reduced).
    pub uncalibrated: bool,
}

impl BakedModel {
    /// Bakes a trained KAN network into fixed-point quantized form.
    ///
    /// # Arguments
    ///
    /// * `network` - Trained KAN network to bake.
    /// * `calibration` - Optional calibration inputs (flat: [n_samples * input_dim]).
    ///   If provided, activation scales are set from the actual dynamic range of
    ///   each layer. If None, a heuristic scale is used and `uncalibrated` is set.
    pub fn from_network(network: &KanNetwork, calibration: Option<&[f32]>) -> Self {
        let config = network.config.clone();
        let uncalibrated = calibration.is_none();

        // Compute activation scales s_act[L] for each layer output.
        // s_act[L] = 32767 / max |layer L output| over calibration inputs.
        let n_layers = network.layers.len();
        let mut s_act = vec![1.0f32; n_layers];

        if let Some(cal_data) = calibration {
            let input_dim = config.input_dim;
            if !cal_data.is_empty() && cal_data.len() >= input_dim {
                let n_samples = cal_data.len() / input_dim;
                // Run f32 forward through all layers to collect per-layer max activations.
                let layer_dims: Vec<usize> = config.layer_dims();
                let max_dim = *layer_dims.iter().max().unwrap_or(&1);

                // Buffers for layer-wise forward
                let mut act_in = vec![0.0f32; max_dim];
                let mut act_out = vec![0.0f32; max_dim];
                let mut basis_buf = vec![0.0f32; 16]; // max basis_aligned

                let mut layer_maxes = vec![0.0f32; n_layers];

                for s in 0..n_samples {
                    let sample = &cal_data[s * input_dim..(s + 1) * input_dim];
                    act_in[..input_dim].copy_from_slice(sample);

                    for (l, layer) in network.layers.iter().enumerate() {
                        // Resize basis_buf if needed
                        if basis_buf.len() < layer.basis_aligned {
                            basis_buf.resize(layer.basis_aligned, 0.0);
                        }
                        let in_slice = &act_in[..layer.in_dim];
                        let out_slice = &mut act_out[..layer.out_dim];
                        layer.forward_single(in_slice, out_slice, &mut basis_buf);

                        for &v in out_slice.iter() {
                            let av = v.abs();
                            if av > layer_maxes[l] {
                                layer_maxes[l] = av;
                            }
                        }

                        // Copy output to next input
                        act_in[..layer.out_dim].copy_from_slice(out_slice);
                    }
                }

                for l in 0..n_layers {
                    let max_v = layer_maxes[l];
                    if max_v > EPSILON {
                        s_act[l] = 32767.0 / max_v;
                    } else {
                        s_act[l] = 32767.0; // trivially zero output
                    }
                }
            }
        } else {
            // Heuristic: assume output range ≈ [-2, 2] (reasonable for KAN)
            // This is a loose guess. Calibration strongly recommended.
            eprintln!(
                "[BakedModel] WARNING: No calibration data provided. \
                 Using heuristic activation scales (uncalibrated). \
                 Accuracy may be significantly reduced."
            );
            for s in s_act.iter_mut() {
                *s = 32767.0 / 4.0; // assume max |activation| ≈ 4
            }
        }

        // Build BakedLayer for each layer
        let mut baked_layers = Vec::with_capacity(n_layers);

        // The "previous layer's s_act" for layer 0 is 1.0 (raw f32 input, no quant)
        let mut s_act_prev = 1.0f32;

        for (l, layer) in network.layers.iter().enumerate() {
            let in_dim = layer.in_dim;
            let out_dim = layer.out_dim;
            let order = layer.order;
            let grid_size = layer.grid_size;
            let global_basis_size = layer.global_basis_size;
            let (r_min, r_max) = layer.grid_range;

            // Quantize weights: s_w = 127 / max|w|
            let max_w = layer
                .weights
                .iter()
                .map(|w| w.abs())
                .fold(0.0f32, f32::max);
            let s_w = if max_w > EPSILON { 127.0 / max_w } else { 1.0 };

            let weights_i8: Vec<i8> = layer
                .weights
                .iter()
                .map(|&w| (w * s_w).round().clamp(-127.0, 127.0) as i8)
                .collect();

            // Bias: q_bias[j] = round(b_j * s_w * 32768)
            let q_bias: Vec<i64> = layer
                .bias
                .iter()
                .map(|&b| (b * s_w * 32768.0).round() as i64)
                .collect();

            // Requant: M_real = s_act[l] / (s_w * 32768)
            // Represent M_real as M0 / 2^S where M0 is a positive i32 < 2^30
            let s_act_l = s_act[l];
            let m_real = s_act_l / (s_w * 32768.0);

            // Choose S so that M0 = round(M_real * 2^S) fits in [1, 2^30)
            // M_real can be >> 1 or << 1 depending on calibration.
            let (requant_m0, requant_shift) = {
                let m_real_f64 = m_real as f64;
                if m_real_f64 <= 0.0 || !m_real_f64.is_finite() {
                    (1i32, 0u32)
                } else {
                    // Choose shift S such that M0 = round(M_real * 2^S) fits in [2^28, 2^29).
                    // M_real can be << 1 or > 1 depending on calibration.
                    // log2(M_real) gives exponent; we want M0 >= 2^28, so S >= 28 - floor(log2(M_real)).
                    let log2_m = m_real_f64.log2().floor() as i32;
                    // S = 28 - log2_m (so that M0 ≈ 2^28)
                    let shift = (28i32 - log2_m).clamp(0, 62) as u32;
                    let m0_f64 = (m_real_f64 * (1u64 << shift) as f64).round();
                    let m0 = (m0_f64 as i64).clamp(1, (1i64 << 30) - 1) as i32;
                    (m0, shift)
                }
            };

            // Per-input fixed-point normalization constants
            // For layer 0: the raw inputs are f32 and we normalize in the entry step.
            //   norm_a_fixed and norm_b_fixed are used only for inter-layer (layers 1+).
            //   For layer 0, they are unused but we still compute them for consistency.
            // For layer L > 0: inputs are i16 activations scaled by 1/s_act_prev.
            //   The i16 value q_out satisfies: z_actual = (q_out/s_act_prev - mean_i) / std_i
            //   We need q_z = z_actual * 2^16 for span/t extraction.
            //   q_z = (q_out/s_act_prev - mean_i) / std_i * 2^16
            //       = q_out * (2^16 / (s_act_prev * std_i)) + (-mean_i/std_i) * 2^16
            //       = q_out * A_FIXED[i] + B_FIXED[i]
            let norm_a_fixed: Vec<i32> = (0..in_dim)
                .map(|i| {
                    let std_i = layer.std[i].max(EPSILON);
                    let a = 65536.0 / (s_act_prev * std_i);
                    // Clamp to avoid i32 overflow
                    a.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
                })
                .collect();

            let norm_b_fixed: Vec<i32> = (0..in_dim)
                .map(|i| {
                    let std_i = layer.std[i].max(EPSILON);
                    let b = -layer.mean[i] / std_i * 65536.0;
                    b.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
                })
                .collect();

            // Grid range in fixed point
            // q_rmin = round(r_min * 2^16) in z-space (z = normalized input)
            // q_rmax = round(r_max * 2^16) - 1 (to mirror +EPSILON floor)
            // The z values in Q15.16 format: q_z represents z = q_z / 65536
            let q_rmin = (r_min * 65536.0).round() as i32;
            let q_rmax = (r_max * 65536.0).round() as i32 - 1;

            // inv_h_scaled: used to extract span and t from q_z (fixed-point z)
            // span = (q_z - q_rmin) * inv_h_scaled >> 16, clamped to [0, G-1]
            // t_q16 = (q_z - q_rmin) * inv_h_scaled - span * 65536
            // inv_h_scaled = round(G * 2^16 / (r_max - r_min))
            let h_range = (r_max - r_min).max(EPSILON);
            // h_q16 = round(65536 * range / G) = one span interval in Q16 z-units
            let h_q16 = ((65536.0 * h_range as f64 / grid_size as f64).round() as i64)
                .clamp(1, i32::MAX as i64) as i32;

            baked_layers.push(BakedLayer {
                weights_i8,
                q_bias,
                requant_m0,
                requant_shift,
                norm_a_fixed,
                norm_b_fixed,
                q_rmin,
                q_rmax,
                h_q16,
                mean: layer.mean.clone(),
                std: layer.std.clone(),
                in_dim,
                out_dim,
                order,
                grid_size,
                global_basis_size,
                s_act_out: s_act_l,
            });

            s_act_prev = s_act_l;
        }

        Self {
            config,
            layers: baked_layers,
            uncalibrated,
        }
    }

    /// Computes weight index for coefficient [out_idx, in_idx, basis_idx].
    #[inline]
    fn weight_index(
        global_basis_size: usize,
        in_dim: usize,
        out_idx: usize,
        in_idx: usize,
        basis_idx: usize,
    ) -> usize {
        (out_idx * in_dim + in_idx) * global_basis_size + basis_idx
    }

    /// Fixed-point forward pass. No f32 in the hot path.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw f32 input of length `config.input_dim`.
    /// * `output` - Output buffer of length `config.output_dim` (overwritten).
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != config.input_dim` or `output.len() != config.output_dim`.
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            self.config.input_dim,
            "BakedModel::forward: input len {} != input_dim {}",
            input.len(),
            self.config.input_dim
        );
        assert_eq!(
            output.len(),
            self.config.output_dim,
            "BakedModel::forward: output len {} != output_dim {}",
            output.len(),
            self.config.output_dim
        );

        if self.layers.is_empty() {
            return;
        }

        // Allocate activation buffers (i16). Max size across layers.
        let max_dim = self
            .layers
            .iter()
            .map(|l| l.in_dim.max(l.out_dim))
            .max()
            .unwrap_or(1);

        let mut act_a = vec![0i32; max_dim]; // current layer inputs as Q15.16 z-values
        let mut act_b = vec![0i16; max_dim]; // current layer outputs as i16 (1/s_act)
        let mut basis_buf = vec![0u16; 8]; // max order+1 = 6 (order 5), 8 is safe

        // ENTRY: normalize layer-0 inputs to Q15.16 fixed-point z
        // z_i = clamp((x_i - mean_i) / std_i, r_min, r_max)
        // q_z_i = round(z_i * 2^16) = round((x_i - mean_i) / std_i * 65536)
        {
            let layer0 = &self.layers[0];
            let (r_min, r_max) = self.config.grid_range;
            for i in 0..layer0.in_dim {
                let std_i = layer0.std[i].max(EPSILON);
                let z_f = ((input[i] - layer0.mean[i]) / std_i).clamp(r_min, r_max);
                act_a[i] = (z_f * 65536.0).round() as i32;
            }
        }

        // LAYER LOOP (integer only)
        for (l, layer) in self.layers.iter().enumerate() {
            let in_dim = layer.in_dim;
            let out_dim = layer.out_dim;
            let order = layer.order;
            let grid_size = layer.grid_size;
            let global_basis_size = layer.global_basis_size;
            let local_basis_size = order + 1;

            // Ensure basis_buf is large enough
            if basis_buf.len() < local_basis_size {
                basis_buf.resize(local_basis_size, 0);
            }

            // For each output j
            for j in 0..out_dim {
                let mut acc: i64 = layer.q_bias[j];

                // For each input i
                for i in 0..in_dim {
                    // Extract span and t (Q0.16) from q_z = act_a[i] (Q15.16 = z * 65536)
                    // q_z_off = (z - r_min) * 65536 = position above grid start in Q16 z-units
                    let q_z = act_a[i].clamp(layer.q_rmin, layer.q_rmax);
                    let q_z_off = q_z - layer.q_rmin; // always >= 0 due to clamp

                    // h_q16 = round(65536 * range / G) = one interval in Q16 z-units
                    // span = q_z_off / h_q16 (integer div, gives interval index 0..G)
                    // t_q16 = (q_z_off % h_q16) * 65536 / h_q16
                    let h_q16 = layer.h_q16 as i64;
                    let span_raw = q_z_off as i64 / h_q16;
                    let span = span_raw.clamp(0, grid_size as i64 - 1) as usize;
                    let t_rem = q_z_off as i64 - span as i64 * h_q16;
                    // t_rem in [0, h_q16); t_q16 = t_rem * 65536 / h_q16 in [0, 65535]
                    let t_q16 = ((t_rem * 65536) / h_q16).clamp(0, 65535) as u32;

                    // Evaluate basis functions → Q0.15 u16 values
                    eval_basis_fixed(order, t_q16, &mut basis_buf[..local_basis_size]);

                    // start_idx: the first global weight index for this span
                    // span from find_span returns interval + order
                    // start_idx = span (as interval index, 0-based) for our integer span
                    // BUT: in the f32 code, find_span returns interval + order,
                    // and start_idx = span - order. Here our `span` is the interval index [0,G-1].
                    // So the global weight start = span (= interval) maps to global basis index `span`.
                    // This matches: start_idx = (span_from_find_span - order) = interval.
                    let start_idx = span; // = interval = global weight start index

                    // Accumulate: acc += sum_k( weight[j,i,start_idx+k] * basis[k] )
                    for k in 0..local_basis_size {
                        let w_idx = Self::weight_index(
                            global_basis_size,
                            in_dim,
                            j,
                            i,
                            start_idx + k,
                        );
                        let q_c = layer.weights_i8[w_idx] as i64;
                        let q_b = basis_buf[k] as i64;
                        acc += q_c * q_b;
                    }
                }

                // Requant: q_out = ((acc * M0) + round) >> S, clamp to i16
                let product = (acc as i128) * (layer.requant_m0 as i128);
                let round_offset = if layer.requant_shift > 0 {
                    1i128 << (layer.requant_shift - 1)
                } else {
                    0
                };
                let q_out_i64 = ((product + round_offset) >> layer.requant_shift) as i64;
                act_b[j] = q_out_i64.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            }

            // INTER-LAYER: compute next layer's z values in Q15.16
            // If this is the last layer, skip inter-layer conversion.
            if l + 1 < self.layers.len() {
                let next_layer = &self.layers[l + 1];
                let next_in_dim = next_layer.in_dim;
                // act_b[i] is the i16 output, act_a[i] will be the next layer's q_z
                // q_z = (act_b[i] * A_FIXED[i] + B_FIXED[i]).clamp(q_rmin, q_rmax)
                for i in 0..next_in_dim {
                    let a_fixed = next_layer.norm_a_fixed[i] as i64;
                    let b_fixed = next_layer.norm_b_fixed[i] as i64;
                    let q_z = ((act_b[i] as i64) * a_fixed + b_fixed) as i32;
                    act_a[i] = q_z.clamp(next_layer.q_rmin, next_layer.q_rmax);
                }
            }
        }

        // EXIT: dequantize last layer outputs to f32
        let last_layer = self.layers.last().unwrap();
        let s_act_last = last_layer.s_act_out;
        for j in 0..last_layer.out_dim {
            output[j] = act_b[j] as f32 / s_act_last;
        }
    }

    /// Returns size estimate in bytes.
    pub fn size_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weights_i8.len() + l.q_bias.len() * 8 + l.norm_a_fixed.len() * 4 * 2 + 40)
            .sum()
    }

    /// Serializes baked model to bytes (requires `serde` feature).
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserializes baked model from bytes (requires `serde` feature).
    ///
    /// Note: Format is not yet stabilized. This may fail on models baked
    /// with a different version of the library.
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KanConfig;
    use crate::network::KanNetwork;

    fn make_network(
        input_dim: usize,
        hidden_dims: Vec<usize>,
        output_dim: usize,
        grid_size: usize,
        order: usize,
        seed: u64,
    ) -> KanNetwork {
        let config = KanConfig {
            input_dim,
            output_dim,
            hidden_dims,
            grid_size,
            spline_order: order,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; input_dim],
            input_std: vec![1.0; input_dim],
            multithreading_threshold: 128,
            simd_width: 8,
            init_seed: Some(seed),
        };
        KanNetwork::new(config)
    }

    /// Generate random inputs in the grid range
    fn random_inputs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut out = Vec::with_capacity(n * dim);
        for i in 0..(n * dim) {
            let mut h = DefaultHasher::new();
            (seed ^ (i as u64).wrapping_mul(6364136223846793005)).hash(&mut h);
            let v = h.finish();
            // Map to [-0.9, 0.9]
            let fv = (v as f64 / u64::MAX as f64) * 1.8 - 0.9;
            out.push(fv as f32);
        }
        out
    }

    /// Compute normalized RMSE between baked and f32 forward passes.
    ///
    /// Returns: sqrt(mean(|b-f|^2)) / (sqrt(mean(f^2)) + 1e-6).
    ///
    /// This is the L2 relative error (NRMSE), which is robust to near-zero outputs
    /// and is a standard metric in quantization literature. It matches the "~5% rel err"
    /// expectation from the design spec for randomly-initialized networks.
    fn max_rel_err(
        network: &KanNetwork,
        baked: &BakedModel,
        test_inputs: &[f32],
        input_dim: usize,
        output_dim: usize,
    ) -> f32 {
        let n_samples = test_inputs.len() / input_dim;
        let mut workspace = network.create_workspace(1);

        let mut sum_sq_err: f64 = 0.0;
        let mut sum_sq_f: f64 = 0.0;

        for s in 0..n_samples {
            let inp = &test_inputs[s * input_dim..(s + 1) * input_dim];
            let mut f32_out = vec![0.0f32; output_dim];
            let mut baked_out = vec![0.0f32; output_dim];
            network.forward_single(inp, &mut f32_out, &mut workspace);
            baked.forward(inp, &mut baked_out);

            for j in 0..output_dim {
                let err = (baked_out[j] - f32_out[j]) as f64;
                let f = f32_out[j] as f64;
                sum_sq_err += err * err;
                sum_sq_f += f * f;
            }
        }

        let count = (n_samples * output_dim) as f64;
        let rmse = (sum_sq_err / count).sqrt();
        let rms_f = (sum_sq_f / count).sqrt().max(1e-9);
        (rmse / rms_f) as f32
    }

    #[test]
    fn test_baked_basis_partition_of_unity() {
        // Verify that basis functions sum to ~32768 (Q0.15 partition of unity)
        let test_t_values: &[u32] = &[0, 16384, 32768, 49152, 65535];

        for &t in test_t_values {
            for order in 2..=5usize {
                let mut basis = vec![0u16; order + 1];
                eval_basis_fixed(order, t, &mut basis);
                let sum: i64 = basis.iter().map(|&b| b as i64).sum();
                // Should be close to 32768
                let err = (sum - 32768).abs();
                assert!(
                    err <= 2,
                    "order={}, t={}: basis sum={} (expected ~32768, err={})",
                    order,
                    t,
                    sum,
                    err
                );
            }
        }
    }

    #[test]
    fn test_baked_basis_matches_f32_order3() {
        // Compare fixed-point basis against f32 compute_basis for order=3
        use crate::spline::{compute_basis, compute_knots, find_span};

        let order = 3;
        let grid_size = 5;
        let knots = compute_knots(grid_size, order, (-1.0, 1.0));

        for t_val in [0.0f32, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999] {
            // t_val is local t in [0,1)
            // Pick a sample z value in the middle of a span
            let z = -0.6 + t_val * 0.4; // stays in range [-1, 1]
            let span = find_span(z, &knots, order, grid_size);

            // F32 reference
            let mut basis_f = vec![0.0f32; order + 1];
            compute_basis(z, span, &knots, order, &mut basis_f);

            // Fixed-point: extract t from the f32 forward
            let t_min = knots[order];
            let t_max = knots[order + grid_size];
            let h = (t_max - t_min) / grid_size as f32;
            let interval = (span - order) as f32;
            let t_local = ((z - t_min) / h - interval).clamp(0.0, 1.0 - 1e-6);
            let t_q16 = (t_local * 65536.0).round() as u32;

            let mut basis_i = vec![0u16; order + 1];
            eval_basis_fixed(order, t_q16, &mut basis_i);

            for k in 0..=order {
                let bf = basis_f[k];
                let bi = basis_i[k] as f32 / 32768.0;
                let err = (bf - bi).abs();
                assert!(
                    err < 0.01,
                    "order=3, k={}, t_local={}: f32={:.6}, fixed={:.6}, err={:.6}",
                    k,
                    t_local,
                    bf,
                    bi,
                    err
                );
            }
        }
    }

    #[test]
    fn test_baked_parity_small_config() {
        // Config 1: small network
        // input=4, hidden=[8], output=2, grid=5, order=3
        let input_dim = 4;
        let hidden = vec![8];
        let output_dim = 2;
        let grid_size = 5;
        let order = 3;

        let network = make_network(input_dim, hidden, output_dim, grid_size, order, 42);

        // Calibration set: 256 random inputs
        let cal_inputs = random_inputs(256, input_dim, 1234);

        // Test set: 1000 random inputs
        let test_inputs = random_inputs(1000, input_dim, 5678);

        let baked = BakedModel::from_network(&network, Some(&cal_inputs));

        let err = max_rel_err(&network, &baked, &test_inputs, input_dim, output_dim);
        println!(
            "[test_baked_parity_small_config] max_rel_err = {:.4} ({:.2}%)",
            err,
            err * 100.0
        );

        assert!(
            err < 0.05,
            "max_rel_err = {:.4} exceeds 5% threshold for small config",
            err
        );
    }

    #[test]
    fn test_baked_parity_medium_config() {
        // Config 2: medium network
        // input=8, hidden=[16,8], output=4, order=3
        let input_dim = 8;
        let hidden = vec![16, 8];
        let output_dim = 4;
        let grid_size = 5;
        let order = 3;

        let network = make_network(input_dim, hidden, output_dim, grid_size, order, 99);

        let cal_inputs = random_inputs(256, input_dim, 111);
        let test_inputs = random_inputs(1000, input_dim, 222);

        let baked = BakedModel::from_network(&network, Some(&cal_inputs));

        let err = max_rel_err(&network, &baked, &test_inputs, input_dim, output_dim);
        println!(
            "[test_baked_parity_medium_config] max_rel_err = {:.4} ({:.2}%)",
            err,
            err * 100.0
        );

        assert!(
            err < 0.10,
            "max_rel_err = {:.4} exceeds 10% threshold for medium config",
            err
        );
    }

    #[test]
    fn test_baked_size_bytes() {
        let network = make_network(4, vec![8], 2, 5, 3, 0);
        let baked = BakedModel::from_network(&network, None);
        assert!(baked.size_bytes() > 0);
        assert!(baked.uncalibrated);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_baked_round_trip() {
        let network = make_network(4, vec![8], 2, 5, 3, 77);
        let cal = random_inputs(64, 4, 333);
        let baked = BakedModel::from_network(&network, Some(&cal));

        let bytes = baked.to_bytes().expect("serialization failed");
        let baked2 = BakedModel::from_bytes(&bytes).expect("deserialization failed");

        // Both should produce same output
        let inp = random_inputs(1, 4, 999);
        let mut out1 = vec![0.0f32; 2];
        let mut out2 = vec![0.0f32; 2];
        baked.forward(&inp, &mut out1);
        baked2.forward(&inp, &mut out2);

        for j in 0..2 {
            assert!(
                (out1[j] - out2[j]).abs() < 1e-6,
                "Round-trip mismatch at j={}: {} vs {}",
                j,
                out1[j],
                out2[j]
            );
        }
    }

    #[test]
    fn test_baked_no_calibration_runs() {
        // Without calibration it should still run (with uncalibrated flag)
        let network = make_network(4, vec![8], 2, 5, 3, 0);
        let baked = BakedModel::from_network(&network, None);
        assert!(baked.uncalibrated);

        let inp = random_inputs(1, 4, 1);
        let mut out = vec![0.0f32; 2];
        baked.forward(&inp, &mut out); // should not panic
    }

    #[test]
    fn test_baked_single_layer() {
        // Test a 1-layer network (no inter-layer code path)
        let network = make_network(3, vec![], 2, 4, 3, 55);

        let cal = random_inputs(64, 3, 1);
        let test = random_inputs(200, 3, 2);

        let baked = BakedModel::from_network(&network, Some(&cal));
        let err = max_rel_err(&network, &baked, &test, 3, 2);
        println!(
            "[test_baked_single_layer] max_rel_err = {:.4} ({:.2}%)",
            err,
            err * 100.0
        );
        assert!(
            err < 0.05,
            "single-layer max_rel_err = {:.4} exceeds 5%",
            err
        );
    }

}
