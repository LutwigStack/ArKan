//! Baked (quantized) model for fixed-point int8 inference.
//!
//! This module provides `BakedModel`, a quantized inference-only representation
//! of a trained KAN model. It uses int8 weights and int16 B-spline basis functions
//! with no f32 operations in the hot path.
//!
//! # Design
//!
//! - Weights: int8 (scale 127/max|w| per **output channel** — per-channel quantization)
//! - Basis functions: u16 Q0.15, value ≈ q_b/32768 (int16 basis variant)
//! - Accumulator: i64 (safe for typical nets; i128 for requant product)
//! - Bias: i64 (folded with per-channel weight scale and basis scale)
//! - Inter-layer activations: i32 with target range ~2^28 (wider than i16 to reduce
//!   inter-layer error amplification)
//! - Activation calibration: 99.9th-percentile clip to stop outliers wasting range
//! - Requant: per-output-channel M0[j]/shift[j] derived from s_act/(s_w[j]·32768)
//! - No f32 between entry normalization and final dequantization
//!
//! # Accuracy (post per-channel quantization, WS02)
//!
//! NRMSE: single-layer 0.60%, 1-hidden 0.64%, 2-hidden 1.29% (all within gate).
//! Worst-case on significant outputs: single-layer 9.2%, 1-hidden 8.7%, 2-hidden 115%.
//! The 2-hidden tail (~115%) reflects inter-layer activation requant noise amplification,
//! not weight quantization — a residual int8 precision floor for 3-layer deep configs.
//! The NRMSE aggregate is suitable for ranking/selection; per-output absolute accuracy
//! in deep nets requires int16 weights or per-channel activation scales.

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
///
/// Every term of a numerator must be at the same Q scale as the leading power
/// (Q(16*order)); see the per-order INVARIANT comments. Note that
/// `test_baked_basis_partition_of_unity` cannot police this, because one
/// coefficient per order is derived as `32768 - sum(others)` and so absorbs any
/// error in the others. `test_baked_basis_matches_f32_all_orders` is the check
/// that actually can.
///
// ponytail: truncating i128 division (floor, not round-to-nearest) plus one
// coefficient derived as the remainder. Ceiling: up to `order` Q0.15 LSBs of
// skew (~1.5e-4) pile onto that one coefficient. Round-to-nearest would halve
// the per-coefficient error but break the exact-32768 sum lemma that downstream
// overflow proofs rely on, so it stays floor.
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

            // INVARIANT: every term below must sit at Q64, the scale of the
            // leading power t4. A term built from t^k carries Q(16k), so it needs
            // (4-k) factors of `one` to reach Q64. Mixing scales here is exactly
            // the P0 bug that made this order err by 0.208 (the `t3` terms were
            // left at Q48).
            let b1_num = -4 * t4 + 12 * t3 * one - 6 * t2 * one * one - 12 * t * one * one * one
                + 11 * one * one * one * one;
            let b1 = (b1_num / denom).clamp(0, SCALE as i128) as u16;

            let b3_num = -4 * t4
                + 4 * t3 * one
                + 6 * t2 * one * one
                + 4 * t * one * one * one
                + one * one * one * one;
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
            let t5 = t4 * t; // Q80, < 2^80; see the magnitude bound below.
            let omt2 = omt * omt;
            let omt3 = omt2 * omt;
            let omt4 = omt3 * omt;
            let omt5 = omt4 * omt; // Q80

            // Denominator: 65536^5 * 120 / 32768 = 65536^4 * 240
            let denom: i128 = one * one * one * one * 240;

            let b0 = (omt5 / denom).clamp(0, SCALE as i128) as u16;
            let b5 = (t5 / denom).clamp(0, SCALE as i128) as u16;

            // INVARIANT: every term below must sit at Q80, the scale of the
            // leading power t5. A term built from t^k carries Q(16k), so it needs
            // (5-k) factors of `one` to reach Q80. Mixing scales here is exactly
            // the P0 bug that made this order err by 0.775.
            //
            // Magnitude bound: |t| < 2^16 and one = 2^16, so every term is
            // < |coef| * 2^80 and every numerator (and every left-to-right
            // partial sum) is < (sum of |coef|) * 2^80 <= 166 * 2^80 < 2^87.4.
            // i128 holds up to 2^127 - 1, so there is >= 2^39 of headroom.
            let b1_num = 5 * t5 - 20 * t4 * one + 20 * t3 * one * one + 20 * t2 * one * one * one
                - 50 * t * one * one * one * one
                + 26 * one * one * one * one * one;
            let b1 = (b1_num / denom).clamp(0, SCALE as i128) as u16;

            let b4_num = -5 * t5
                + 5 * t4 * one
                + 10 * t3 * one * one
                + 10 * t2 * one * one * one
                + 5 * t * one * one * one * one
                + one * one * one * one * one;
            let b4 = (b4_num / denom).clamp(0, SCALE as i128) as u16;

            // b2_num = -10t^5 + 30t^4 - 60t^2 + 66
            let b2_num = -10 * t5 + 30 * t4 * one - 60 * t2 * one * one * one
                + 66 * one * one * one * one * one;
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
    /// Each output channel j uses its own scale s_w[j] = 127/max|w[j,*,*]|.
    pub weights_i8: Vec<i8>,
    /// Folded quantized bias per output channel j:
    /// q_bias[j] = round(b_j * s_w[j] * 32768) as i64.
    pub q_bias: Vec<i64>,
    /// Per-output-channel requant multiplier M0[j]
    /// (from M_real[j] = s_act / (s_w[j] * 32768)).
    pub requant_m0: Vec<i32>,
    /// Per-output-channel requant shift S[j]
    /// (so that requant[j] = (acc * M0[j] + 2^(S[j]-1)) >> S[j]).
    pub requant_shift: Vec<u32>,
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
    /// Target range for i32 inter-layer activations.
    /// Using ~2^28 gives 28 bits of dynamic range for typical values,
    /// vs only 15 bits with the old i16 scheme. This is the key lever
    /// against inter-layer error amplification.
    const ACT_TARGET: f64 = 268_435_456.0; // 2^28

    /// Bakes a trained KAN network into fixed-point quantized form.
    ///
    /// # Arguments
    ///
    /// * `network` - Trained KAN network to bake.
    /// * `calibration` - Optional calibration inputs (flat: [n_samples * input_dim]).
    ///   If provided, activation scales are set using 99.9th-percentile clipping
    ///   (stops outliers from wasting the i32 dynamic range). If None, a heuristic
    ///   scale is used and `uncalibrated` is set.
    pub fn from_network(network: &KanNetwork, calibration: Option<&[f32]>) -> Self {
        let config = network.config.clone();
        let uncalibrated = calibration.is_none();

        // Compute activation scales s_act[L] for each layer output.
        // s_act[L] = ACT_TARGET / p99.9(|layer L output|) over calibration inputs.
        // Activations are clamped to ±ACT_TARGET at requant time so outliers saturate
        // instead of stealing dynamic range from the bulk of values.
        let n_layers = network.layers.len();
        let mut s_act = vec![1.0f32; n_layers];

        if let Some(cal_data) = calibration {
            let input_dim = config.input_dim;
            if !cal_data.is_empty() && cal_data.len() >= input_dim {
                let n_samples = cal_data.len() / input_dim;
                // Run f32 forward through all layers to collect per-layer activation magnitudes.
                let layer_dims: Vec<usize> = config.layer_dims();
                let max_dim = *layer_dims.iter().max().unwrap_or(&1);

                // Buffers for layer-wise forward
                let mut act_in = vec![0.0f32; max_dim];
                let mut act_out = vec![0.0f32; max_dim];
                let mut basis_buf = vec![0.0f32; 16]; // max basis_aligned

                // Collect all activation magnitudes per layer for percentile computation
                let mut layer_mags: Vec<Vec<f32>> = vec![Vec::new(); n_layers];

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
                            layer_mags[l].push(v.abs());
                        }

                        // Copy output to next input
                        act_in[..layer.out_dim].copy_from_slice(out_slice);
                    }
                }

                // Compute 99.9th percentile per layer and set s_act accordingly.
                for l in 0..n_layers {
                    let mags = &mut layer_mags[l];
                    if mags.is_empty() {
                        s_act[l] = Self::ACT_TARGET as f32;
                        continue;
                    }
                    // Sort to find percentile (NaN-safe: treat NaN as large)
                    mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
                    // 99.9th percentile index
                    let idx = ((mags.len() - 1) as f64 * 0.999) as usize;
                    let p999 = mags[idx];
                    if p999 > EPSILON {
                        s_act[l] = (Self::ACT_TARGET / p999 as f64) as f32;
                    } else {
                        s_act[l] = Self::ACT_TARGET as f32; // trivially zero output
                    }
                }
            }
        } else {
            // Heuristic: assume output range ≈ [-4, 4] (reasonable for KAN)
            // This is a loose guess. Calibration strongly recommended.
            eprintln!(
                "[BakedModel] WARNING: No calibration data provided. \
                 Using heuristic activation scales (uncalibrated). \
                 Accuracy may be significantly reduced."
            );
            for s in s_act.iter_mut() {
                *s = (Self::ACT_TARGET / 4.0) as f32; // assume max |activation| ≈ 4
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

            // Per-output-channel weight quantization (WS02).
            // s_w[j] = 127 / max over (i,k) of |coeff[j, i, k]|
            // Channels with all-zero weights get scale 1.0 (identity, avoids div-by-zero).
            let s_act_l = s_act[l];
            let basis_scale = 32768.0f64; // Q0.15 basis scale

            // Compute per-channel max|w| and scales.
            let coeff_per_channel = in_dim * global_basis_size; // weights per output channel
            let mut s_w_per_channel = vec![1.0f32; out_dim];
            for j in 0..out_dim {
                let start = j * coeff_per_channel;
                let end = start + coeff_per_channel;
                let max_w = layer.weights[start..end]
                    .iter()
                    .map(|w| w.abs())
                    .fold(0.0f32, f32::max);
                s_w_per_channel[j] = if max_w > EPSILON { 127.0 / max_w } else { 1.0 };
            }

            // Quantize weights using per-channel scale.
            let mut weights_i8: Vec<i8> = Vec::with_capacity(layer.weights.len());
            for j in 0..out_dim {
                let sw_j = s_w_per_channel[j];
                let start = j * coeff_per_channel;
                for &w in &layer.weights[start..start + coeff_per_channel] {
                    weights_i8.push((w * sw_j).round().clamp(-127.0, 127.0) as i8);
                }
            }

            // Bias per channel: q_bias[j] = round(b_j * s_w[j] * basis_scale)
            let q_bias: Vec<i64> = layer
                .bias
                .iter()
                .enumerate()
                .map(|(j, &b)| (b as f64 * s_w_per_channel[j] as f64 * basis_scale).round() as i64)
                .collect();

            // Per-channel requant: M_real[j] = s_act[l] / (s_w[j] * basis_scale)
            // Represent M_real[j] as M0[j] / 2^S[j] where M0[j] in [2^28, 2^29).
            let mut requant_m0: Vec<i32> = Vec::with_capacity(out_dim);
            let mut requant_shift: Vec<u32> = Vec::with_capacity(out_dim);
            for j in 0..out_dim {
                let sw_j = s_w_per_channel[j] as f64;
                let m_real = s_act_l as f64 / (sw_j * basis_scale);
                let (m0, shift) = if m_real <= 0.0 || !m_real.is_finite() {
                    (1i32, 0u32)
                } else {
                    let log2_m = m_real.log2().floor() as i32;
                    let s = (28i32 - log2_m).clamp(0, 62) as u32;
                    let m0_f = (m_real * (1u64 << s) as f64).round();
                    let m0 = (m0_f as i64).clamp(1, (1i64 << 30) - 1) as i32;
                    (m0, s)
                };
                requant_m0.push(m0);
                requant_shift.push(shift);
            }

            // Per-input fixed-point normalization constants
            // For layer 0: the raw inputs are f32 and we normalize in the entry step.
            //   norm_a_fixed and norm_b_fixed are used only for inter-layer (layers 1+).
            //   For layer 0, they are unused but we still compute them for consistency.
            // For layer L > 0: inputs are i32 activations scaled by 1/s_act_prev.
            //   The i32 value q_out satisfies: z_actual = (q_out/s_act_prev - mean_i) / std_i
            //   We need q_z = z_actual * 2^16 for span/t extraction.
            //   q_z = (q_out/s_act_prev - mean_i) / std_i * 2^16
            //       = q_out * (2^16 / (s_act_prev * std_i)) + (-mean_i/std_i) * 2^16
            //       = q_out * A_FIXED[i] + B_FIXED[i]
            // Note: s_act_prev is now ~ACT_TARGET/p999 (much larger than 32767),
            // so A_FIXED will be smaller (often < 1 in f32, stored as i32 fraction).
            // To preserve precision, we scale A_FIXED by 2^16 (stored as i32 fraction
            // of 2^16), and divide by 2^16 in the hot path using i64 arithmetic.
            // This avoids losing sub-1 precision in A_FIXED for i32 activations.
            // Encoding: norm_a_fixed[i] = round(2^32 / (s_act_prev * std_i))
            // Hot path: q_z = (q_in * A_FIXED + B_FIXED_shifted) >> 16
            // where B_FIXED_shifted = round(-mean_i/std_i * 2^32)
            // NOTE: we still store the old scale (2^16 / ...) and shift by 0 when
            // s_act_prev == 1.0 (layer 0, which is unused anyway).
            // For layers > 0 with wide i32 activations, we use Q16 scaling:
            //   A_FIXED = round(2^32 / (s_act_prev * std_i))
            //   B_FIXED = round(-mean_i / std_i * 2^32)
            //   q_z = (q_in * A_FIXED + B_FIXED) >> 16
            // This maps i32 activations in range ±ACT_TARGET to q_z in Q15.16.
            let norm_a_fixed: Vec<i32> = (0..in_dim)
                .map(|i| {
                    let std_i = layer.std[i].max(EPSILON) as f64;
                    let s_prev = s_act_prev as f64;
                    // Use 2^32 scale for i32 activations; for layer 0 (s_prev=1.0),
                    // this is unused but we compute it for consistency.
                    let a = (1u64 << 32) as f64 / (s_prev * std_i);
                    // Clamp to i32 range (a can be very small for large s_act_prev)
                    a.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
                })
                .collect();

            let norm_b_fixed: Vec<i32> = (0..in_dim)
                .map(|i| {
                    let std_i = layer.std[i].max(EPSILON) as f64;
                    // Match the 2^32 scaling used in norm_a_fixed (then >> 16 in hot path).
                    let b = -layer.mean[i] as f64 / std_i * (1u64 << 32) as f64;
                    b.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
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

        // Allocate activation buffers. Max size across layers.
        let max_dim = self
            .layers
            .iter()
            .map(|l| l.in_dim.max(l.out_dim))
            .max()
            .unwrap_or(1);

        let mut act_a = vec![0i32; max_dim]; // current layer inputs as Q15.16 z-values
        let mut act_b = vec![0i32; max_dim]; // current layer outputs as i32 (scaled by s_act)
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

                // Per-channel requant: q_out[j] = ((acc * M0[j]) + round) >> S[j].
                // Using i32 output (wider than old i16) to preserve inter-layer precision.
                // Saturate to ACT_TARGET so that the few outliers above the 99.9th
                // percentile clipping point don't corrupt the fixed-point scale.
                let m0_j = layer.requant_m0[j] as i128;
                let shift_j = layer.requant_shift[j];
                let product = (acc as i128) * m0_j;
                let round_offset = if shift_j > 0 { 1i128 << (shift_j - 1) } else { 0 };
                let q_out_i64 = ((product + round_offset) >> shift_j) as i64;
                const ACT_CLAMP: i64 = 268_435_456; // 2^28 = ACT_TARGET
                act_b[j] = q_out_i64.clamp(-ACT_CLAMP, ACT_CLAMP) as i32;
            }

            // INTER-LAYER: compute next layer's z values in Q15.16
            // If this is the last layer, skip inter-layer conversion.
            if l + 1 < self.layers.len() {
                let next_layer = &self.layers[l + 1];
                let next_in_dim = next_layer.in_dim;
                // act_b[i] is now an i32 output scaled by s_act.
                // A_FIXED[i] = round(2^32 / (s_act_prev * std_i))  (stored in norm_a_fixed)
                // B_FIXED[i] = round(-mean_i / std_i * 2^32)        (stored in norm_b_fixed)
                // q_z = (q_in * A_FIXED + B_FIXED) >> 16
                // This maps i32 activations → Q15.16 z value.
                // q_in * A_FIXED can be up to 2^28 * 2^31 = 2^59 — fits in i64.
                for i in 0..next_in_dim {
                    let a_fixed = next_layer.norm_a_fixed[i] as i64;
                    let b_fixed = next_layer.norm_b_fixed[i] as i64;
                    let q_z = (((act_b[i] as i64) * a_fixed + b_fixed) >> 16) as i32;
                    act_a[i] = q_z.clamp(next_layer.q_rmin, next_layer.q_rmax);
                }
            }
        }

        // EXIT: dequantize last layer outputs to f32
        // act_b[j] is i32 scaled by s_act_last (ACT_TARGET / p99.9).
        // output[j] = act_b[j] / s_act_last
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
            .map(|l| {
                l.weights_i8.len()          // i8 weights
                + l.q_bias.len() * 8         // i64 biases
                + l.requant_m0.len() * 4     // i32 per-channel M0
                + l.requant_shift.len() * 4  // u32 per-channel shift
                + l.norm_a_fixed.len() * 4 * 2  // i32 norm constants
                + 40                         // fixed metadata
            })
            .sum()
    }

    /// Current binary format version written by [`BakedModel::to_bytes`].
    ///
    /// Bump this whenever the bincode layout changes in a breaking way.
    #[cfg(feature = "serde")]
    const FORMAT_VERSION: u32 = 1;

    /// Serializes the baked model to a self-describing byte vector.
    ///
    /// Layout:
    /// ```text
    /// [0..12]  magic   — MAGIC_BAKED (b"KAN_BAKED_v1")
    /// [12..16] version — u32 little-endian format version (currently 1)
    /// [16..]   body    — bincode-encoded BakedModel
    /// ```
    ///
    /// Requires the `serde` feature.
    #[cfg(feature = "serde")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        use crate::MAGIC_BAKED;

        let body = bincode::serialize(self)?;
        let mut out = Vec::with_capacity(MAGIC_BAKED.len() + 4 + body.len());
        out.extend_from_slice(MAGIC_BAKED);
        out.extend_from_slice(&Self::FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Deserializes a baked model from bytes produced by [`BakedModel::to_bytes`].
    ///
    /// Returns a clear `Err` — never panics — for:
    /// - Input shorter than the 16-byte header.
    /// - Wrong magic bytes (not an ArKan baked-model file).
    /// - Wrong format version (produced by a different library version).
    /// - Corrupt bincode body.
    ///
    /// Requires the `serde` feature.
    #[cfg(feature = "serde")]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        use crate::MAGIC_BAKED;

        let header_len = MAGIC_BAKED.len() + 4; // 12 + 4 = 16

        if bytes.len() < header_len {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "BakedModel::from_bytes: input too short ({} bytes, need at least {})",
                bytes.len(),
                header_len
            ))));
        }

        let (magic_bytes, rest) = bytes.split_at(MAGIC_BAKED.len());
        if magic_bytes != MAGIC_BAKED.as_ref() {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "BakedModel::from_bytes: wrong magic bytes (got {:?}, expected {:?}). \
                 Is this an ArKan baked-model file?",
                magic_bytes,
                MAGIC_BAKED
            ))));
        }

        let version = u32::from_le_bytes(rest[..4].try_into().unwrap());
        if version != Self::FORMAT_VERSION {
            return Err(Box::new(bincode::ErrorKind::Custom(format!(
                "BakedModel::from_bytes: format version mismatch (got {}, expected {}). \
                 Re-bake the model with the current library version.",
                version,
                Self::FORMAT_VERSION
            ))));
        }

        bincode::deserialize(&rest[4..])
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

    /// Cox-de Boor ground truth for the *uniform* B-spline basis on t in [0,1).
    ///
    /// Knots are the integers, span = order, x = order + t — the canonical
    /// uniform setup the closed forms in `eval_basis_fixed` (and in
    /// gpu/shaders.rs) are derived from. `x` is exact in f32 (≤3 integer bits
    /// plus 16 fractional bits), so this reference carries no scaling error.
    fn f32_basis_ref(order: usize, t_q16: u32, out: &mut [f32]) {
        let knots: Vec<f32> = (0..=(2 * order + 1)).map(|i| i as f32).collect();
        let x = order as f32 + t_q16 as f32 / 65536.0;
        compute_basis(x, order, &knots, order, out);
    }

    #[test]
    fn test_baked_basis_matches_f32_all_orders() {
        // Every order 2..=5, not just order 3. A correct fixed-point basis is
        // within a couple of Q0.15 LSBs (~1e-4) of the f32 reference, so TOL is
        // set an order of magnitude above that and ~200x below the Q-scale bug
        // it replaced (order 4 erred by 0.208, order 5 by 0.775).
        //
        // This also pins the "every numerator >= 0" premise of the sum lemma:
        // the multi-term numerators have a true minimum of 1/120 ≈ 0.0083 over
        // t in [0,1] — 8x TOL — so a numerator that went negative would clamp to
        // 0 and fail here. The other two numerators are the monomials t^p and
        // (1-t)^p, non-negative by construction.
        const TOL: f32 = 0.001;

        // Report every order before failing, so one run localizes the fault.
        let mut failures = Vec::new();
        for order in 2..=5usize {
            let mut worst = 0.0f32;
            let mut worst_at = (0u32, 0usize);
            let mut basis_f = vec![0.0f32; order + 1];
            let mut basis_i = vec![0u16; order + 1];

            for t_q16 in (0..65536u32).step_by(64).chain([65535]) {
                f32_basis_ref(order, t_q16, &mut basis_f);
                eval_basis_fixed(order, t_q16, &mut basis_i);
                for k in 0..=order {
                    let err = (basis_f[k] - basis_i[k] as f32 / 32768.0).abs();
                    if err > worst {
                        worst = err;
                        worst_at = (t_q16, k);
                    }
                }
            }

            println!(
                "[basis order={}] max abs err vs f32 = {:.6} at t_q16={}, k={}",
                order, worst, worst_at.0, worst_at.1
            );
            if worst >= TOL {
                failures.push(format!(
                    "order={order}: max abs err {worst:.6} >= {TOL} (worst at t_q16={}, k={})",
                    worst_at.0, worst_at.1
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("; "));

        // Exact-integer regression pins for the two vectors the P0 bug report
        // named. These are floor(32768 * B_k), with the last-computed
        // coefficient (k=3 for order 5, k=2 for order 4) taking the remainder,
        // so they sit up to ~1.3 LSB off a round-to-nearest reference. The
        // broken code returned [273,0,0,32495,0,0] and [85,4437,22359,5802,85].
        let mut b5 = [0u16; 6];
        eval_basis_fixed(5, 0, &mut b5);
        assert_eq!(b5, [273, 7099, 18022, 7101, 273, 0], "order 5 at t_q16=0");
        let mut b4 = [0u16; 5];
        eval_basis_fixed(4, 32768, &mut b4);
        assert_eq!(b4, [85, 6485, 19628, 6485, 85], "order 4 at t_q16=32768");
    }

    #[test]
    fn test_baked_basis_sum_lemma_exhaustive() {
        // Sum lemma, load-bearing for downstream overflow proofs: for every
        // representable t_q16 and every order 2..=5 the basis is non-negative
        // (u16, so by type) and sums to EXACTLY 32768.
        //
        // The exact sum is not cosmetic: the last coefficient is computed as
        // 32768 - sum(others), clamped to [0, 32768]. If sum(others) ever
        // exceeded 32768 the clamp would fire and the total would come out
        // above 32768, so `sum == 32768` is the check that the clamp is dead
        // code on this input domain.
        for order in 2..=5usize {
            let mut basis = vec![0u16; order + 1];
            for t_q16 in 0..65536u32 {
                eval_basis_fixed(order, t_q16, &mut basis);
                let sum: u32 = basis.iter().map(|&b| b as u32).sum();
                assert_eq!(
                    sum, 32768,
                    "order={order}, t_q16={t_q16}: basis sum={sum} (basis={basis:?})"
                );
                for &b in basis.iter() {
                    assert!(
                        b <= 32768,
                        "order={order}, t_q16={t_q16}: coeff {b} exceeds 32768"
                    );
                }
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

        // Verify header is present: magic (12) + version (4)
        assert!(bytes.len() >= 16, "serialized output too short");
        assert_eq!(&bytes[..12], b"KAN_BAKED_v1", "magic bytes wrong");
        assert_eq!(&bytes[12..16], &1u32.to_le_bytes(), "version bytes wrong");

        let baked2 = BakedModel::from_bytes(&bytes).expect("deserialization failed");

        // Round-trip must produce byte-identical inference results.
        let inp = random_inputs(1, 4, 999);
        let mut out1 = vec![0.0f32; 2];
        let mut out2 = vec![0.0f32; 2];
        baked.forward(&inp, &mut out1);
        baked2.forward(&inp, &mut out2);

        for j in 0..2 {
            assert_eq!(
                out1[j].to_bits(),
                out2[j].to_bits(),
                "Round-trip produced non-identical inference at j={}: {} vs {}",
                j,
                out1[j],
                out2[j]
            );
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_baked_from_bytes_rejects_wrong_magic() {
        let network = make_network(3, vec![], 1, 4, 3, 1);
        let baked = BakedModel::from_network(&network, None);
        let mut bytes = baked.to_bytes().expect("serialization failed");

        // Corrupt magic bytes
        bytes[0] = b'X';
        bytes[1] = b'X';
        bytes[2] = b'X';

        let result = BakedModel::from_bytes(&bytes);
        assert!(result.is_err(), "should reject wrong magic");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("magic") || err_msg.contains("ArKan"),
            "error message should mention magic/ArKan: {err_msg}"
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_baked_from_bytes_rejects_wrong_version() {
        let network = make_network(3, vec![], 1, 4, 3, 2);
        let baked = BakedModel::from_network(&network, None);
        let mut bytes = baked.to_bytes().expect("serialization failed");

        // Overwrite the version field (bytes 12..16) with an unknown version
        let bad_version: u32 = 999;
        bytes[12..16].copy_from_slice(&bad_version.to_le_bytes());

        let result = BakedModel::from_bytes(&bytes);
        assert!(result.is_err(), "should reject wrong version");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("version") || err_msg.contains("999"),
            "error message should mention version: {err_msg}"
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_baked_from_bytes_rejects_truncated_input() {
        // Too short to even contain a header
        let short = b"KAN_BA";
        let result = BakedModel::from_bytes(short);
        assert!(result.is_err(), "should reject truncated input");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("short") || err_msg.contains("bytes"),
            "error message should mention shortness: {err_msg}"
        );

        // Empty input
        let result2 = BakedModel::from_bytes(&[]);
        assert!(result2.is_err(), "should reject empty input");
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_baked_from_bytes_rejects_foreign_bytes() {
        // Completely foreign data (e.g. JSON)
        let foreign = b"{\"model\": \"not a baked model\"}";
        let result = BakedModel::from_bytes(foreign);
        assert!(result.is_err(), "should reject foreign bytes");
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
