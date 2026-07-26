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
//! - Activations saturate only at the i32 they are stored in. The 99.9th
//!   percentile sets the SCALE, not a clip — see `forward` for why clipping there
//!   only ever added error
//! - Activation calibration: 99.9th percentile sets s_act so outliers do not waste range
//! - Requant: per-output-channel `M0[j]`/`shift[j]` derived from `s_act/(s_w[j]·32768)`
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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Evaluate B-spline basis polynomials in fixed point (int16 basis variant).
///
/// Returns `order+1` values as u16 in Q0.15 format, value ≈ q_b/32768.
/// Uses closed-form expressions for orders 2–5 reused from gpu/shaders.rs.
/// Orders outside 2..=5 have no fixed-point form and are rejected by
/// [`BakedModel::from_network`], so this function is only ever called with 2..=5.
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
            let b0 = (omt2 >> 18).clamp(0, SCALE) as u16;
            // B2 = t^2 / 2
            let b2 = (t2 >> 18).clamp(0, SCALE) as u16;
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
            let b2 = (SCALE as i128 - b0 as i128 - b1 as i128 - b3 as i128).clamp(0, SCALE as i128)
                as u16;

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
            let b3 =
                (SCALE as i128 - b0 as i128 - b1 as i128 - b2 as i128 - b4 as i128 - b5 as i128)
                    .clamp(0, SCALE as i128) as u16;

            out[0] = b0;
            out[1] = b1;
            out[2] = b2;
            out[3] = b3;
            out[4] = b4;
            out[5] = b5;
        }
        // The old `_` arm fell back to the f32 `compute_basis` with a knot vector
        // of length `order + 3`, while `compute_basis` indexes up to `knots[2*order]`
        // — an out-of-bounds panic inside spline.rs for order >= 3, first hit at
        // order 6 (orders 2..=5 never reach here). It also quantized with `.round()`,
        // which can round UP and so break the "basis sums to exactly 32768" lemma
        // the requant/overflow proofs depend on. `from_network` now rejects those
        // orders up front, so there is nothing left to fall back to.
        other => panic!(
            "eval_basis_fixed: spline order {other} has no fixed-point basis (baked \
             supports 2..=5). BakedModel::from_network rejects it at bake time, so this \
             is only reachable from a BakedModel deserialized from a file baked before \
             that check existed — re-bake it."
        ),
    }
}

/// Extracts the grid interval index and the position within that interval from a
/// fixed-point z value.
///
/// `q_z` must already be clamped to `[q_rmin, q_rmax]`. Returns `(span, t_q16)`
/// with `span <= grid_size - 1` and `t_q16 <= 65535`.
///
/// # Both upper clamps are load-bearing, NOT dead code
///
/// At the top of the grid range `q_z_off` is an exact (or near-exact) multiple of
/// `h_q16`, so `span_raw` reaches `grid_size`:
/// - `grid_range = (-3, 3)`, G=5 (library default): `q_z_off = 393215 = 5 * 78643`
/// - `grid_range = (-1, 1)`, G=5 (every baked test): `q_z_off = 131071`, `h_q16 = 26214`
///
/// Unclamped, `start_idx + order == global_basis_size`, so the weight read in
/// [`BakedModel::forward`] runs one element past each `(j, i)` weight block — a
/// cross-channel read for every block but the last, and a genuine out-of-bounds
/// index for the last one (verified: `index out of bounds: the len is 48 but the
/// index is 48`). The `- 1` in `q_rmax` does not prevent it, so the span clamp is
/// memory safety, not tidiness.
///
/// Clamping the span down then leaves `t_rem == h_q16` (or a hair above), so
/// `t_q16` reaches 65536 / 65538 for those two configs (and ~131072 for a narrow
/// range with a large grid). The second clamp keeps `eval_basis_fixed` inside its
/// documented `[0, 65535]` domain, where its closed forms are valid.
///
/// `test_span_t_clamps_fire_at_grid_top` pins both.
#[inline]
fn extract_span_t(q_z: i32, q_rmin: i32, h_q16: i32, grid_size: usize) -> (usize, u32) {
    // q_z_off = (z - r_min) * 65536 = position above grid start in Q16 z-units.
    // Always >= 0 because the caller clamped q_z to [q_rmin, q_rmax].
    let q_z_off = (q_z - q_rmin) as i64;
    let h = h_q16 as i64; // one grid interval in Q16 z-units, >= 1 by construction
    let span = (q_z_off / h).clamp(0, grid_size as i64 - 1) as usize;
    let t_rem = q_z_off - span as i64 * h;
    let t_q16 = ((t_rem * 65536) / h).clamp(0, 65535) as u32;
    (span, t_q16)
}

/// Per-layer baked data for fixed-point inference.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BakedLayer {
    /// Quantized weights in i8. Layout: [out_dim, in_dim, global_basis_size].
    /// Each output channel `j` uses its own scale `s_w[j] = 127/max|w[j,*,*]|`.
    pub weights_i8: Vec<i8>,
    /// Folded quantized bias per output channel j:
    /// `q_bias[j] = round(b_j * s_w[j] * 32768) as i64`.
    pub q_bias: Vec<i64>,
    /// Per-output-channel requant multiplier `M0[j]`
    /// (from `M_real[j] = s_act / (s_w[j] * 32768)`).
    pub requant_m0: Vec<i32>,
    /// Per-output-channel requant shift `S[j]`
    /// (so that `requant[j] = (acc * M0[j] + 2^(S[j]-1)) >> S[j]`).
    pub requant_shift: Vec<u32>,
    /// Per-input fixed-point scale:
    /// `A_FIXED[i] = round(2^(16 + norm_shift) / (s_act_prev * std_i))`, in
    /// `[1, 2^30 - 1]`. Consumed as `(q_in * A_FIXED) >> norm_shift`.
    pub norm_a_fixed: Vec<i32>,
    /// Per-input fixed-point offset: `B_FIXED[i] = round(-mean_i / std_i * 2^16)`,
    /// i.e. a plain Q15.16 z offset added *after* the `norm_shift` right-shift.
    pub norm_b_fixed: Vec<i32>,
    /// Right-shift applied to `q_in * A_FIXED` in the inter-layer step. Chosen per
    /// layer so the largest `A_FIXED` fills the i32 rather than landing on a
    /// single-digit integer. See `from_network` for the derivation.
    pub norm_shift: u32,
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
    ///
    /// # Panics
    ///
    /// Panics if any layer's `spline_order` is outside `2..=5`. Baked inference
    /// only has fixed-point basis polynomials for those orders, even though
    /// [`KanConfig::validate`](crate::KanConfig::validate) accepts
    /// `1..=MAX_SPLINE_ORDER` (7) for the f32 CPU path. Use
    /// [`KanNetwork::forward_single`](crate::KanNetwork::forward_single) for other orders.
    pub fn from_network(network: &KanNetwork, calibration: Option<&[f32]>) -> Self {
        let config = network.config.clone();
        let uncalibrated = calibration.is_none();

        // Baked inference has closed-form fixed-point basis polynomials for orders
        // 2..=5 only (see `eval_basis_fixed`). Orders 6/7 pass `KanConfig::validate`
        // (MAX_SPLINE_ORDER = 7, sized for the f32 CPU path) but used to blow up as an
        // out-of-bounds index deep inside spline.rs on the first `forward` call.
        // ponytail: bake-time panic rather than making `from_network` return a Result
        // — it returns `Self` and is called from tests, benches and the example, so a
        // Result is a breaking API change for orders nobody ships (untested and
        // unmeasured for baked). Ceiling: unrecoverable at bake time, but the message
        // names the cause instead of pointing at spline.rs.
        if let Some(bad) = network
            .layers
            .iter()
            .map(|l| l.order)
            .find(|o| !(2..=5).contains(o))
        {
            panic!(
                "BakedModel::from_network: spline_order {bad} is not supported by baked \
                 (fixed-point) inference; supported range is 2..=5. KanConfig::validate \
                 accepts 1..={} (MAX_SPLINE_ORDER) for the f32 CPU path only. Re-configure \
                 with spline_order in 2..=5, or use KanNetwork::forward for f32 inference.",
                crate::config::MAX_SPLINE_ORDER
            );
        }

        // Compute activation scales s_act[L] for each layer output.
        // s_act[L] = ACT_TARGET / p99.9(|layer L output|) over calibration inputs.
        // Using the percentile rather than the max stops a handful of outliers from
        // stealing dynamic range from the bulk of values. Values above it are NOT
        // clipped — see the saturation comment in `forward`.
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
            for (s_w, chunk) in s_w_per_channel
                .iter_mut()
                .zip(layer.weights.chunks_exact(coeff_per_channel))
            {
                let max_w = chunk.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
                *s_w = if max_w > EPSILON { 127.0 / max_w } else { 1.0 };
            }

            // Quantize weights using per-channel scale.
            let mut weights_i8: Vec<i8> = Vec::with_capacity(layer.weights.len());
            for (&sw_j, chunk) in s_w_per_channel
                .iter()
                .zip(layer.weights.chunks_exact(coeff_per_channel))
            {
                for &w in chunk {
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
            for &sw_j in &s_w_per_channel {
                let sw_j = sw_j as f64;
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

            // Per-input fixed-point normalization constants.
            //
            // For layer 0 the raw inputs are f32 and `forward` normalizes them in the
            // entry step, so these are computed but never read. For layer L > 0 the
            // inputs are i32 activations scaled by 1/s_act_prev, and
            //   q_z = z * 2^16 = ((q_in/s_act_prev) - mean_i)/std_i * 2^16
            //       = q_in * a_z[i] + b_z[i],  where
            //   a_z[i] = 2^16 / (s_act_prev * std_i),  b_z[i] = -mean_i/std_i * 2^16.
            //
            // `a_z` is a small fraction — s_act_prev is ~2^28/p99.9, so with std = 1
            // (every hidden layer) a_z = p99.9 * 2^-12, around 1e-4. The old encoding
            // stored `round(a_z * 2^16)` and shifted by a hardcoded 16, which put
            // A_FIXED on the integers 7..9 on ordinary freshly-built nets (4x8x2,
            // 8x16x4, 16x32x8, 32x64x16 measured 8, 7, 8, 9): THREE bits of mantissa,
            // a systematic 5.6-7.1% error on the inter-layer z scale, compounding with
            // depth. Worse, p99.9 < std/32 made A_FIXED round to 0 and collapsed the
            // next layer's inputs to a constant — verified reachable, see
            // `test_norm_a_fixed_survives_a_tiny_previous_layer`.
            //
            // So the shift is chosen per layer instead of hardcoded: pick the largest
            // `norm_shift` that keeps every A_FIXED inside i32, which lands the biggest
            // one just under 2^30 and gives it ~30 bits instead of 3.
            //
            // ponytail: ONE shift per layer, sized off the largest a_z, not one per
            // input. Ceiling: an input whose std is 2^k above the layer's smallest
            // gets 30-k bits instead of 30. Every hidden layer has std = 1 for all
            // inputs (KanLayer::new), so k = 0 unless a caller has hand-set
            // per-input normalization on a hidden layer; even k = 20 still beats the
            // 3 bits this replaces by a factor of 128.
            let norm_shift: u32 = {
                let a_z_max = (0..in_dim)
                    .map(|i| 65536.0 / (s_act_prev as f64 * layer.std[i].max(EPSILON) as f64))
                    .fold(0.0f64, f64::max);
                if a_z_max > 0.0 && a_z_max.is_finite() {
                    // Target A_FIXED in [2^29, 2^30) for the largest a_z.
                    (29i32 - a_z_max.log2().floor() as i32).clamp(0, 62) as u32
                } else {
                    0
                }
            };

            // A_FIXED[i] = round(a_z[i] * 2^norm_shift), clamped to [1, 2^30 - 1].
            //
            // The upper clamp bounds the hot-path product (see `forward`). The lower
            // clamp makes A_FIXED == 0 — the silent-collapse mode — impossible by
            // construction rather than by luck. It only binds when
            // a_z < 2^-norm_shift <= 2^-62, i.e. s_act_prev * std > 2^78, i.e.
            // p99.9(|previous layer output|) < 2^-50 ~ 1e-15. At that point the
            // previous layer really is numerically zero and A_FIXED = 1 yields
            // q_z = q_in >> 62 = 0, which is the right answer, not a collapse.
            let norm_a_fixed: Vec<i32> = (0..in_dim)
                .map(|i| {
                    let std_i = layer.std[i].max(EPSILON) as f64;
                    let a = 65536.0 / (s_act_prev as f64 * std_i) * (1u64 << norm_shift) as f64;
                    (a.round() as i64).clamp(1, (1i64 << 30) - 1) as i32
                })
                .collect();

            // B_FIXED[i] = round(b_z[i]) — a plain Q15.16 z offset, added AFTER the
            // shift. Keeping it at Q16 rather than at 2^(16+norm_shift) is what stops
            // it overflowing now that norm_shift can reach 62; its rounding error is
            // half a Q16 tick, ~2e-5 of a grid interval at grid_range (-1,1), G = 5.
            let norm_b_fixed: Vec<i32> = (0..in_dim)
                .map(|i| {
                    let std_i = layer.std[i].max(EPSILON) as f64;
                    let b = -layer.mean[i] as f64 / std_i * 65536.0;
                    b.round().clamp(i32::MIN as f64, i32::MAX as f64) as i32
                })
                .collect();

            // Grid range in fixed point
            // q_rmin = round(r_min * 2^16) in z-space (z = normalized input)
            // q_rmax = round(r_max * 2^16) - 1 (to mirror +EPSILON floor)
            // The z values in Q15.16 format: q_z represents z = q_z / 65536
            let q_rmin = (r_min * 65536.0).round() as i32;
            // `.max(q_rmin)`: for a range narrower than ~1.5e-5 (e.g. the
            // validate-approved grid_range = (0.0, 0.000005)) both endpoints round to
            // the same Q16 tick and the `- 1` puts q_rmax *below* q_rmin, which made
            // `q_z.clamp(q_rmin, q_rmax)` in `forward` panic with "min > max".
            // Collapsing to a single-tick interval keeps the clamp well-formed; such a
            // range is degenerate anyway (every input maps to span 0, t 0).
            let q_rmax = ((r_max * 65536.0).round() as i32 - 1).max(q_rmin);

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
                norm_shift,
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

            // Requantized activations saturate at the i32 they are stored in, and
            // nothing tighter.
            //
            // They used to saturate at ACT_TARGET (2^28) on every layer, which — since
            // the exit scale is s_act_last = 2^28 / p99.9 — is exactly p99.9 of the
            // calibration set. On the OUTPUT layer that capped every value the model
            // could ever return: on a calibrated 4->2 net, 2 of the 2000 *calibration*
            // samples already sat above the ceiling, worst case f32 -0.520944 vs baked
            // -0.471458, 9.50% error from clipping alone.
            //
            // On a HIDDEN layer the stated justification was that an outlier would
            // otherwise wreck the next layer's fixed-point z scale. Measured, it does
            // not: `q_z` is clamped to [q_rmin, q_rmax] in the inter-layer step below,
            // and that IS the grid range — the same range the f32 path clamps z to
            // (`KanLayer::forward_single`). Both paths saturate an outlier the same
            // way. ACT_TARGET was a *second*, tighter saturation at p99.9 that the f32
            // path does not have, so it could only add error, and it was most of the
            // remaining tail: on 8->[16,8]->4 the worst case on >=1 sigma outputs went
            // 25.2 / 51.8 / 33.7 / 29.5% -> 7.9 / 3.1 / 3.4 / 6.4% for orders 2/3/4/5
            // when it was dropped.
            //
            // The calibration percentile still sets the *scale* (`s_act`), which is
            // what stops outliers wasting dynamic range. Only the clip is gone.
            const ACT_LO: i64 = i32::MIN as i64;
            const ACT_HI: i64 = i32::MAX as i64;

            // For each output j
            for (j, act_out) in act_b[..out_dim].iter_mut().enumerate() {
                let mut acc: i64 = layer.q_bias[j];

                // For each input i
                for (i, &q_z_raw) in act_a[..in_dim].iter().enumerate() {
                    // Extract span and t (Q0.16) from q_z = act_a[i] (Q15.16 = z * 65536).
                    // The clamps inside `extract_span_t` are what keep the weight read
                    // below in bounds — see its doc comment.
                    let q_z = q_z_raw.clamp(layer.q_rmin, layer.q_rmax);
                    let (span, t_q16) = extract_span_t(q_z, layer.q_rmin, layer.h_q16, grid_size);

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
                    for (k, &q_b) in basis_buf[..local_basis_size].iter().enumerate() {
                        let w_idx =
                            Self::weight_index(global_basis_size, in_dim, j, i, start_idx + k);
                        let q_c = layer.weights_i8[w_idx] as i64;
                        acc += q_c * q_b as i64;
                    }
                }

                // Per-channel requant: q_out[j] = ((acc * M0[j]) + round) >> S[j].
                // Using i32 output (wider than old i16) to preserve inter-layer precision.
                let m0_j = layer.requant_m0[j] as i128;
                let shift_j = layer.requant_shift[j];
                let product = (acc as i128) * m0_j;
                let round_offset = if shift_j > 0 {
                    1i128 << (shift_j - 1)
                } else {
                    0
                };
                let q_out_i64 = ((product + round_offset) >> shift_j) as i64;
                *act_out = q_out_i64.clamp(ACT_LO, ACT_HI) as i32;
            }

            // INTER-LAYER: compute next layer's z values in Q15.16
            // If this is the last layer, skip inter-layer conversion.
            if l + 1 < self.layers.len() {
                let next_layer = &self.layers[l + 1];
                let next_in_dim = next_layer.in_dim;
                // act_b[i] is an i32 output scaled by s_act; map it to a Q15.16 z:
                //   q_z = ((q_in * A_FIXED[i] + round) >> SH) + B_FIXED[i]
                // with A_FIXED = round(2^(16+SH) / (s_act_prev * std_i)),
                //      B_FIXED = round(-mean_i / std_i * 2^16),
                //      SH      = next_layer.norm_shift.
                //
                // Overflow bound. act_b[i] is an i32, so |act_b[i]| <= 2^31.
                // from_network clamps A_FIXED to [1, 2^30 - 1] and SH to [0, 62], so
                //   |q_in * A_FIXED| < 2^31 * 2^30 = 2^61,
                //   round = 2^(SH-1) <= 2^61,
                //   |sum| < 2^62 < i64::MAX.
                // The final clamp is done in i64 against two i32 bounds, so the `as
                // i32` cannot truncate even when B_FIXED pushes the sum out of range.
                // Note that clamp is to the GRID RANGE, which is what makes the
                // ACT_TARGET saturation above unnecessary: an activation far past
                // p99.9 lands outside [q_rmin, q_rmax] and saturates here instead,
                // exactly as the f32 path saturates z.
                let sh = next_layer.norm_shift;
                let round = if sh > 0 { 1i64 << (sh - 1) } else { 0 };
                for i in 0..next_in_dim {
                    let a_fixed = next_layer.norm_a_fixed[i] as i64;
                    let b_fixed = next_layer.norm_b_fixed[i] as i64;
                    let q_z = (((act_b[i] as i64) * a_fixed + round) >> sh) + b_fixed;
                    act_a[i] = q_z.clamp(next_layer.q_rmin as i64, next_layer.q_rmax as i64) as i32;
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
                + 44 // fixed metadata (incl. u32 norm_shift)
            })
            .sum()
    }

    /// Current binary format version written by [`BakedModel::to_bytes`].
    ///
    /// Bump this whenever the bincode layout changes in a breaking way.
    #[cfg(feature = "serde")]
    const FORMAT_VERSION: u32 = 2;

    /// Serializes the baked model to a self-describing byte vector.
    ///
    /// Layout:
    /// ```text
    /// [0..12]  magic   — MAGIC_BAKED (b"KAN_BAKED_v1")
    /// [12..16] version — u32 little-endian format version (currently 2)
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
                magic_bytes, MAGIC_BAKED
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
    use crate::spline::compute_basis;

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

            // Exhaustive over the whole t domain, not a stride. A step_by(64) sweep
            // only ever visits residues {0, 63} mod 64, and mutation testing showed
            // five defects gated on the other 62 residues surviving it (worst true
            // error 0.34) while still passing this test, the partition-of-unity test
            // and the sum lemma. 65536 x 4 orders costs ~0.01s in release.
            for t_q16 in 0..65536u32 {
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

    /// The output layer must not saturate at the calibration set's 99.9th percentile.
    ///
    /// `ACT_CLAMP` (2^28 = `ACT_TARGET`) used to be applied to every layer, and the
    /// exit scale is `s_act_last = 2^28 / p99.9`, so `|output[j]| <= p99.9` held by
    /// construction — the model could not return a larger magnitude no matter what
    /// the input was. Calibration and test draw from the *same* distribution here
    /// and the ceiling is still crossed, which is the point: this is not an
    /// out-of-distribution scenario.
    #[test]
    fn test_output_layer_not_capped_at_calibration_p999() {
        let network = make_network(4, vec![], 2, 5, 3, 77);
        let cal = random_inputs(2000, 4, 333);
        let baked = BakedModel::from_network(&network, Some(&cal));
        // The hard ceiling the unconditional clamp imposed on every returned value.
        let ceiling = BakedModel::ACT_TARGET as f32 / baked.layers[0].s_act_out;

        let test = random_inputs(4000, 4, 444);
        let mut workspace = network.create_workspace(1);
        let mut f32_out = vec![0.0f32; 2];
        let mut baked_out = vec![0.0f32; 2];
        let mut n_above = 0usize;
        let mut worst_rel = 0.0f32;
        let mut max_baked = 0.0f32;

        for s in 0..4000 {
            let inp = &test[s * 4..(s + 1) * 4];
            network.forward_single(inp, &mut f32_out, &mut workspace);
            baked.forward(inp, &mut baked_out);
            for j in 0..2 {
                max_baked = max_baked.max(baked_out[j].abs());
                if f32_out[j].abs() > ceiling {
                    n_above += 1;
                    worst_rel = worst_rel.max((baked_out[j] - f32_out[j]).abs() / f32_out[j].abs());
                }
            }
        }

        println!(
            "[output clamp] ceiling={ceiling:.6}: {n_above} of 8000 f32 outputs above it, \
             max |baked| = {max_baked:.6}, worst rel err {:.2}%",
            worst_rel * 100.0
        );
        assert!(
            n_above > 0,
            "fixture no longer crosses the p99.9 ceiling ({ceiling}), so it guards nothing — \
             pick a config that does rather than deleting this test"
        );
        // The structural falsifier: under the old clamp `max_baked <= ceiling` held
        // for every input, so this is a binary check, not a tolerance.
        assert!(
            max_baked > ceiling,
            "no baked output exceeded the calibration p99.9 ceiling ({ceiling}); max |baked| = \
             {max_baked} — the output layer is still saturating"
        );
        // Loose but far below the 5.00% the clamp produced on this fixture.
        assert!(
            worst_rel < 0.01,
            "output layer accuracy above the ceiling regressed: worst rel err {:.2}% over \
             {n_above} outputs above {ceiling}",
            worst_rel * 100.0
        );
    }

    /// `norm_a_fixed` must carry real mantissa, not 3 bits.
    ///
    /// The old encoding was `round(2^32 / (s_act_prev * std_i))` consumed with a
    /// hardcoded `>> 16`. On these four ordinary freshly-built nets that landed on
    /// the integers 8, 7, 8, 9 — a systematic 5.6-7.1% error on the inter-layer z
    /// scale, at every hop. The per-layer shift puts A_FIXED in [2^29, 2^30), so
    /// its rounding error is at most 0.5/2^29 < 1e-9.
    #[test]
    fn test_norm_a_fixed_uses_the_full_i32() {
        for (in_dim, hidden) in [(4usize, 8usize), (8, 16), (16, 32), (32, 64)] {
            let out_dim = in_dim / 2;
            let network = make_network(in_dim, vec![hidden], out_dim, 5, 3, 7);
            let cal = random_inputs(2000, in_dim, 1234);
            let baked = BakedModel::from_network(&network, Some(&cal));

            for (li, layer) in baked.layers.iter().enumerate() {
                for (i, &a) in layer.norm_a_fixed.iter().enumerate() {
                    assert!(
                        (1 << 28..1 << 30).contains(&a),
                        "{in_dim}x{hidden}x{out_dim} layer {li} input {i}: norm_a_fixed = {a} \
                         (shift {}), want [2^28, 2^30) — the old encoding produced 7..9 here",
                        layer.norm_shift
                    );
                }
            }
        }
    }

    /// `norm_a_fixed == 0` collapsed the next layer's inputs to a constant, silently.
    ///
    /// Reachability, since "can that happen?" was open: under the old encoding
    /// `A_FIXED = round(2^32 / (s_act_prev * std_i)) = round(16 * p99.9 / std_i)`,
    /// so any layer whose outputs have `p99.9 < std/32` rounds it to 0. Scaling one
    /// layer's weights by 1e-3 is enough — measured: every `norm_a_fixed` of the
    /// next layer became 0 and the model returned the SAME pair of values
    /// (0.12060832, 0.3845482) for every input while f32 varied.
    ///
    /// The fix makes 0 unreachable by construction (the shift chases the magnitude,
    /// and A_FIXED is clamped to >= 1), so this pins behaviour, not just the value.
    #[test]
    fn test_norm_a_fixed_survives_a_tiny_previous_layer() {
        let mut network = make_network(4, vec![8], 2, 5, 3, 7);
        for w in network.layers[0].weights.iter_mut() {
            *w *= 0.001;
        }
        for b in network.layers[0].bias.iter_mut() {
            *b *= 0.001;
        }

        let cal = random_inputs(2000, 4, 1234);
        let baked = BakedModel::from_network(&network, Some(&cal));
        for (i, &a) in baked.layers[1].norm_a_fixed.iter().enumerate() {
            assert!(
                a > 0,
                "norm_a_fixed[{i}] = {a}: a zero scale collapses layer 1's inputs to a constant"
            );
        }

        let test = random_inputs(64, 4, 42);
        let mut workspace = network.create_workspace(1);
        let mut f32_out = vec![0.0f32; 2];
        let mut baked_out = vec![0.0f32; 2];
        let mut distinct = std::collections::BTreeSet::new();
        let mut worst_rel = 0.0f32;
        for s in 0..64 {
            let inp = &test[s * 4..(s + 1) * 4];
            network.forward_single(inp, &mut f32_out, &mut workspace);
            baked.forward(inp, &mut baked_out);
            distinct.insert(baked_out[0].to_bits());
            worst_rel = worst_rel.max((baked_out[0] - f32_out[0]).abs() / f32_out[0].abs());
        }

        println!(
            "[tiny prev layer] shift={}, {} distinct baked outputs over 64 inputs, worst rel err \
             {:.3}%",
            baked.layers[1].norm_shift,
            distinct.len(),
            worst_rel * 100.0
        );
        assert!(
            distinct.len() > 32,
            "baked output collapsed: only {} distinct values over 64 inputs",
            distinct.len()
        );
        assert!(
            worst_rel < 0.01,
            "worst rel err {:.3}% — layer 1 is not tracking f32",
            worst_rel * 100.0
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
        assert_eq!(
            &bytes[12..16],
            &BakedModel::FORMAT_VERSION.to_le_bytes(),
            "version bytes wrong"
        );

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

    /// Builds a single-layer network with an explicit grid_range (`make_network`
    /// hardcodes (-1, 1)).
    fn make_network_ranged(grid_range: (f32, f32), grid_size: usize, order: usize) -> KanNetwork {
        let input_dim = 3;
        KanNetwork::new(KanConfig {
            input_dim,
            output_dim: 2,
            hidden_dims: vec![],
            grid_size,
            spline_order: order,
            grid_range,
            input_mean: vec![0.0; input_dim],
            input_std: vec![1.0; input_dim],
            ..Default::default()
        })
    }

    /// The upper clamps in `extract_span_t` look provably dead and are not: they
    /// are the only thing keeping the weight read in `forward` in bounds.
    ///
    /// Pins the pre-clamp values (so the "can't happen" reading stays falsified)
    /// *and* the post-clamp invariants, so a rewrite that drops either clamp fails
    /// here instead of reading past `weights_i8`.
    #[test]
    fn test_span_t_clamps_fire_at_grid_top() {
        let grid_size = 5;
        let order = 3;
        // (grid_range, expected h_q16, expected pre-clamp t_q16)
        let cases: [((f32, f32), i32, i64); 2] = [
            ((-3.0, 3.0), 78643, 65536), // library default KanConfig
            ((-1.0, 1.0), 26214, 65538), // every baked test in this crate
        ];

        for (range, want_h_q16, want_t_q16_raw) in cases {
            let network = make_network_ranged(range, grid_size, order);
            let baked = BakedModel::from_network(&network, None);
            let l = &baked.layers[0];
            assert_eq!(l.h_q16, want_h_q16, "grid_range={range:?}: h_q16");

            // Extreme input: z pinned at the very top of the grid range.
            let q_z_off = (l.q_rmax - l.q_rmin) as i64;
            let h = l.h_q16 as i64;

            // (1) Pre-clamp span reaches grid_size, and the weight index it would
            //     produce for the last (j, i) block is out of bounds. If this ever
            //     stops holding, the test below guards nothing and must be revisited
            //     rather than deleted.
            let span_raw = q_z_off / h;
            assert_eq!(
                span_raw, grid_size as i64,
                "grid_range={range:?}: span_raw must reach grid_size (q_z_off={q_z_off}, h={h})"
            );
            assert_eq!(
                span_raw as usize + order,
                l.global_basis_size,
                "grid_range={range:?}: unclamped start_idx+order must equal global_basis_size"
            );
            let unclamped_w_idx = ((l.out_dim - 1) * l.in_dim + (l.in_dim - 1))
                * l.global_basis_size
                + span_raw as usize
                + order;
            assert!(
                unclamped_w_idx >= l.weights_i8.len(),
                "grid_range={range:?}: unclamped weight index {unclamped_w_idx} must be out of \
                 bounds of weights_i8 (len {})",
                l.weights_i8.len()
            );

            // (2) Once span is clamped down, t_rem reaches h_q16, so t_q16 leaves the
            //     [0, 65535] domain that eval_basis_fixed's closed forms assume.
            let t_rem = q_z_off - (grid_size as i64 - 1) * h;
            assert_eq!(
                (t_rem * 65536) / h,
                want_t_q16_raw,
                "grid_range={range:?}: pre-clamp t_q16"
            );

            // (3) The invariants a refactor must preserve.
            let (span, t_q16) = extract_span_t(l.q_rmax, l.q_rmin, l.h_q16, grid_size);
            assert!(
                span < grid_size,
                "grid_range={range:?}: span {span} exceeds grid_size-1 ({})",
                grid_size - 1
            );
            assert!(
                t_q16 <= 65535,
                "grid_range={range:?}: t_q16 {t_q16} exceeds 65535"
            );

            // (4) Black-box: forward at an input that saturates the grid range must
            //     not read out of bounds.
            let mut out = vec![0.0f32; 2];
            baked.forward(&[1e9, 1e9, 1e9], &mut out);
        }
    }

    /// Orders 6/7 pass `KanConfig::validate` but have no fixed-point basis. They
    /// used to surface as `index out of bounds: the len is 9 but the index is 9`
    /// from inside spline.rs on the first `forward`; the bake must reject them with
    /// a message that names the cause.
    #[test]
    #[should_panic(expected = "spline_order 6 is not supported by baked")]
    fn test_bake_rejects_spline_order_6() {
        let network = make_network_ranged((-1.0, 1.0), 5, 6);
        assert!(
            network.config.validate().is_ok(),
            "order 6 must stay valid for the f32 CPU path"
        );
        let _ = BakedModel::from_network(&network, None);
    }

    #[test]
    #[should_panic(expected = "spline_order 7 is not supported by baked")]
    fn test_bake_rejects_spline_order_7() {
        let network = make_network_ranged((-1.0, 1.0), 5, 7);
        assert!(
            network.config.validate().is_ok(),
            "order 7 must stay valid for the f32 CPU path"
        );
        let _ = BakedModel::from_network(&network, None);
    }

    #[test]
    fn test_bake_and_forward_all_supported_orders() {
        for order in 2..=5usize {
            let network = make_network(3, vec![], 2, 5, order, 4);
            let baked = BakedModel::from_network(&network, None);
            let mut out = vec![0.0f32; 2];
            baked.forward(&[0.4, -0.7, 0.9], &mut out);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "order {order}: non-finite output {out:?}"
            );
        }
    }

    #[test]
    fn test_bake_narrow_grid_range_no_clamp_panic() {
        // This range passes KanConfig::validate but used to bake to q_rmin = 0,
        // q_rmax = -1 and panic in forward's clamp with "min > max".
        let network = make_network_ranged((0.0, 0.000005), 5, 3);
        let baked = BakedModel::from_network(&network, None);
        let l = &baked.layers[0];
        assert!(
            l.q_rmax >= l.q_rmin,
            "q_rmax ({}) must not fall below q_rmin ({})",
            l.q_rmax,
            l.q_rmin
        );
        assert!(l.h_q16 >= 1, "h_q16 must stay >= 1, got {}", l.h_q16);

        let mut out = vec![0.0f32; 2];
        baked.forward(&[0.1, 0.2, 0.3], &mut out); // must not panic
        assert!(out.iter().all(|v| v.is_finite()), "non-finite {out:?}");
    }
}
