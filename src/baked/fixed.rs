//! Fixed-point basis and grid arithmetic.

/// Evaluate B-spline basis polynomials in fixed point (int16 basis variant).
///
/// Returns `order+1` values as u16 in Q0.15 format, value ≈ q_b/32768.
/// Uses closed-form expressions for orders 2–5 reused from gpu/shaders.rs.
/// Orders outside 2..=5 have no fixed-point form and are rejected by
/// [`BakedModel::from_network`](crate::baked::BakedModel::from_network), so this function is only ever called with 2..=5.
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
pub(super) fn eval_basis_fixed(order: usize, t_q16: u32, out: &mut [u16]) {
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
/// Use the whole extent: rounding each interval separately accumulates grid drift.
/// Widen before adding the exclusive upper tick, which may exceed i32::MAX.
#[inline]
pub(super) fn extract_span_t(q_z: i32, q_rmin: i32, q_rmax: i32, grid_size: usize) -> (usize, u32) {
    let extent = i64::from(q_rmax) + 1 - i64::from(q_rmin);
    // The quantized-grid single-tick fallback has extent 1 and zero offset.
    // Validated grids have G <= 64, so even the full i32 range times G * 65536 fits i64.
    let position = (i64::from(q_z) - i64::from(q_rmin)) * grid_size as i64;
    let span = (position / extent).clamp(0, grid_size as i64 - 1) as usize;
    let remainder = position - span as i64 * extent;
    let t_q16 = ((remainder * 65536) / extent).clamp(0, 65535) as u32;
    (span, t_q16)
}
