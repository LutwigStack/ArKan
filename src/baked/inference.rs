//! Inference using caller-owned fixed-point scratch.

use super::*;

impl BakedModel {
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
        let mut workspace = self.create_workspace();
        self.forward_with_workspace(input, output, &mut workspace);
    }

    /// Allocates scratch buffers once for this model's largest layer.
    pub fn create_workspace(&self) -> BakedWorkspace {
        let max_dim = self
            .layers
            .iter()
            .map(|l| l.in_dim.max(l.out_dim))
            .max()
            .unwrap_or(0);
        let max_input = self.layers.iter().map(|l| l.in_dim).max().unwrap_or(0);
        BakedWorkspace {
            act_a: vec![0; max_dim],
            act_b: vec![0; max_dim],
            spans: vec![0; max_input],
            bases: vec![[0; 6]; max_input],
        }
    }

    /// Runs single-sample inference without allocating.
    ///
    /// Panics on input/output length mismatch or insufficient workspace capacity.
    /// The workspace may be reused with models whose layers fit its buffers.
    /// Public model fields must retain their validated bake/import invariants.
    pub fn forward_with_workspace(
        &self,
        input: &[f32],
        output: &mut [f32],
        workspace: &mut BakedWorkspace,
    ) {
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

        let BakedWorkspace {
            act_a,
            act_b,
            spans,
            bases,
        } = workspace;
        assert!(
            self.layers.iter().all(|l| {
                l.in_dim <= act_a.len()
                    && l.out_dim <= act_b.len()
                    && l.in_dim <= spans.len()
                    && l.in_dim <= bases.len()
            }),
            "BakedModel::forward_with_workspace: insufficient workspace capacity"
        );

        // ENTRY: normalize layer-0 inputs to Q15.16 fixed-point z
        // z_i = clamp((x_i - mean_i) / std_i, r_min, r_max)
        // q_z_i = round(z_i * 2^16) = round((x_i - mean_i) / std_i * 65536)
        {
            let layer0 = &self.layers[0];
            let (r_min, r_max) = self.config.grid_range;
            for i in 0..layer0.in_dim {
                let std_i = layer0.std[i];
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

            // Span and basis depend on the input, and are shared by all outputs.
            for i in 0..in_dim {
                let q_z = act_a[i].clamp(layer.q_rmin, layer.q_rmax);
                let (span, t) = extract_span_t(q_z, layer.q_rmin, layer.h_q16, grid_size);
                spans[i] = span;
                eval_basis_fixed(order, t, &mut bases[i][..local_basis_size]);
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

                // Preserve input and coefficient accumulation order.
                for i in 0..in_dim {
                    let start_idx = spans[i];
                    for (k, &q_b) in bases[i][..local_basis_size].iter().enumerate() {
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
                let q_out = (product + round_offset) >> shift_j;
                *act_out = q_out.clamp(ACT_LO as i128, ACT_HI as i128) as i32;
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
}
