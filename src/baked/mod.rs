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
//! - Inter-layer z scale: `norm_a_fixed` with a per-layer `norm_shift`, so it fills
//!   the i32 (~30 bits) instead of landing on a single-digit integer
//! - Requant: per-output-channel `M0[j]`/`shift[j]` derived from `s_act/(s_w[j]·32768)`
//! - No f32 between entry normalization and final dequantization
//!
//! # Accuracy
//!
//! Measured by `tests/baked_parity.rs` (`cargo test --release --test baked_parity --
//! --nocapture`), random-init nets, `grid_range = (-1, 1)`, 256 calibration and 2000
//! test samples in `[-0.9, 0.9]`.
//!
//! | Config | NRMSE | worst @0.1σ | @0.5σ | @1.0σ |
//! |---|---|---|---|---|
//! | 4→2 | 0.34% | 9.2% | 2.3% | 1.2% |
//! | `4→[8]→2` | 0.17% | 0.8% | 0.8% | 0.8% |
//! | `8→[16,8]→4` | 0.57% | 47.0% | 10.5% | 5.0% |
//! | `8→[16,8]→4`, orders 2–5 | 0.37–0.59% | 3.1–61.3% | 3.1–14.2% | 3.1–7.9% |
//!
//! The 0.1σ column divides a few-LSB absolute error by a near-noise reference and
//! explodes by construction; it is a diagnostic, not a quality number. The ≥1σ
//! column is the one that decides whether a caller can read an output as a
//! quantity, and `baked_parity` gates it at 15%.

use crate::config::{KanConfig, EPSILON};
use crate::error::{ArkanError, ArkanResult};
use crate::network::KanNetwork;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod calibration;
mod fixed;
mod inference;

use fixed::{eval_basis_fixed, extract_span_t};

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
pub struct BakedModel {
    /// Original configuration for reference/validation.
    pub config: KanConfig,
    /// Per-layer baked data.
    pub layers: Vec<BakedLayer>,
    /// Set to true if baked without calibration data (accuracy may be reduced).
    pub uncalibrated: bool,
}

/// Reusable scratch storage for single-sample baked inference.
/// Create with [`BakedModel::create_workspace`], then reuse across calls.
#[derive(Debug, Clone)]
pub struct BakedWorkspace {
    act_a: Vec<i32>,
    act_b: Vec<i32>,
    spans: Vec<usize>,
    bases: Vec<[u16; 6]>,
}

fn checked_weights(input: usize, output: usize, basis: usize) -> ArkanResult<usize> {
    input
        .checked_mul(output)
        .and_then(|n| n.checked_mul(basis))
        .ok_or_else(|| ArkanError::overflow("baked weight shape"))
}

fn quantized_grid(range: (f32, f32), grid: usize) -> ArkanResult<(i32, i32, i32)> {
    let lo = (range.0 as f64 * 65536.0).round();
    let hi = (range.1 as f64 * 65536.0).round();
    let h = (65536.0 * (range.1 - range.0).max(EPSILON) as f64 / grid as f64)
        .round()
        .max(1.0);
    if grid == 0
        || !lo.is_finite()
        || !hi.is_finite()
        || !h.is_finite()
        || lo < i32::MIN as f64
        || lo > i32::MAX as f64
        || hi < i32::MIN as f64
        || hi > i32::MAX as f64 + 1.0
        || h > i32::MAX as f64
    {
        return Err(ArkanError::cpu(
            "baked grid range or interval is outside Q15.16 representation",
        ));
    }
    Ok((lo as i32, ((hi as i64 - 1).max(lo as i64)) as i32, h as i32))
}

impl BakedModel {
    /// Checks executable shapes, fixed-point bounds and metadata.
    /// Call again after mutating the public model fields.
    pub fn validate(&self) -> ArkanResult<()> {
        self.config.validate()?;
        let dims = self.config.layer_dims();
        let grid = quantized_grid(self.config.grid_range, self.config.grid_size)?;
        if self.layers.len() != self.config.num_layers() {
            return Err(ArkanError::cpu("baked layer count disagrees with config"));
        }
        for (i, l) in self.layers.iter().enumerate() {
            let weights = checked_weights(l.in_dim, l.out_dim, l.global_basis_size)?;
            if l.in_dim != dims[i]
                || l.out_dim != dims[i + 1]
                || l.in_dim == 0
                || l.out_dim == 0
                || l.order != self.config.spline_order
                || !(2..=5).contains(&l.order)
                || l.grid_size != self.config.grid_size
                || l.global_basis_size != l.grid_size + l.order
                || l.weights_i8.len() != weights
                || l.q_bias.len() != l.out_dim
                || l.requant_m0.len() != l.out_dim
                || l.requant_shift.len() != l.out_dim
                || l.norm_a_fixed.len() != l.in_dim
                || l.norm_b_fixed.len() != l.in_dim
                || l.mean.len() != l.in_dim
                || l.std.len() != l.in_dim
                || (l.q_rmin, l.q_rmax, l.h_q16) != grid
            {
                return Err(ArkanError::cpu(format!(
                    "baked layer {i}: invalid executable shape or grid metadata"
                )));
            }
            // Each input's basis sums to 32768, bounding every accumulation prefix.
            let acc_bound = (l.in_dim as i128) * 128 * 32768;
            if l.q_bias
                .iter()
                .any(|b| (*b as i128).abs() + acc_bound > i64::MAX as i128)
                || l.norm_shift > 62
                || l.requant_shift.iter().any(|s| *s > 62)
                || l.norm_a_fixed
                    .iter()
                    .any(|a| !(1..=(1 << 30) - 1).contains(a))
                || l.requant_m0
                    .iter()
                    .any(|m| !(1..=(1 << 30) - 1).contains(m))
                || !l.mean.iter().all(|v| v.is_finite())
                || !l.std.iter().all(|v| v.is_finite() && *v > 0.0)
                || !l.s_act_out.is_finite()
                || l.s_act_out <= 0.0
            {
                return Err(ArkanError::cpu(format!(
                    "baked layer {i}: invalid fixed-point scale, shift or accumulator bounds"
                )));
            }
        }
        Ok(())
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

    /// Estimates counted inference payload bytes (not total resident/serialized size).
    /// Excludes mean/std, config buffers, vector capacity and allocation overhead.
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
    pub(crate) const FORMAT_VERSION: u32 = 2;
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

    #[test]
    fn test_fixed_span_exact_boundary_values() {
        // Literal interval/fraction pins: h=3 gives thirds 21845 and 43690.
        // Include both saturation edges, wide offsets, and a negative q_z whose
        // offset is positive. No second implementation is used as the oracle.
        let cases: &[(i32, i32, i32, usize, usize, u32)] = &[
            (-11, -10, 3, 5, 0, 0),
            (-10, -10, 3, 5, 0, 0),
            (-9, -10, 3, 5, 0, 21845),
            (-8, -10, 3, 5, 0, 43690),
            (-7, -10, 3, 5, 1, 0),
            (-6, -10, 3, 5, 1, 21845),
            (-5, -10, 3, 5, 1, 43690),
            (-4, -10, 3, 5, 2, 0),
            (-3, -10, 3, 5, 2, 21845),
            (-2, -10, 3, 5, 2, 43690),
            (-1, -10, 3, 5, 3, 0),
            (0, -10, 3, 5, 3, 21845),
            (1, -10, 3, 5, 3, 43690),
            (2, -10, 3, 5, 4, 0),
            (3, -10, 3, 5, 4, 21845),
            (4, -10, 3, 5, 4, 43690),
            (5, -10, 3, 5, 4, 65535),
            (6, -10, 3, 5, 4, 65535),
            (2, 0, 3, 1, 0, 43690),
            (3, 0, 3, 1, 0, 65535),
            (63, 0, 1, 64, 63, 0),
            (64, 0, 1, 64, 63, 65535),
            (65, 0, 1, 64, 63, 65535),
            (65536, 0, 1, 64, 63, 65535),
            (4128767, 0, 65536, 64, 62, 65535),
            (4128768, 0, 65536, 64, 63, 0),
            (4128769, 0, 65536, 64, 63, 1),
            (4194303, 0, 65536, 64, 63, 65535),
            (4194304, 0, 65536, 64, 63, 65535),
            (4194305, 0, 65536, 64, 63, 65535),
            (196607, -196608, 78643, 5, 4, 65535),
            (65535, -65536, 26214, 5, 4, 65535),
            (i32::MIN, i32::MIN, 1, 64, 0, 0),
            (i32::MAX, i32::MAX, i32::MAX, 1, 0, 0),
            (i32::MIN, i32::MAX, 1, 64, 0, 0),
            (i32::MIN, i32::MAX, i32::MAX, 64, 0, 0),
            (i32::MAX, i32::MIN, 1, 1, 0, 65535),
            (i32::MAX, i32::MIN, 1, 64, 63, 65535),
            (i32::MAX, i32::MIN, i32::MAX, 1, 0, 65535),
            (i32::MAX, i32::MIN, i32::MAX, 64, 2, 0),
            (2147483646, 0, i32::MAX, 64, 0, 65535),
            (i32::MAX, 0, i32::MAX, 64, 1, 0),
            (i32::MAX, -1, i32::MAX, 64, 1, 0),
        ];
        for &(q_z, q_rmin, h_q16, grid_size, expected_span, expected_t) in cases {
            assert_eq!(
                extract_span_t(q_z, q_rmin, h_q16, grid_size),
                (expected_span, expected_t),
                "q_z={q_z}, q_rmin={q_rmin}, h={h_q16}, grid={grid_size}"
            );
        }
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
