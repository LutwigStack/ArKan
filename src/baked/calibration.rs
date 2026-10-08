//! Checked CPU conversion and activation calibration.

use super::*;

impl BakedModel {
    /// Target range for i32 inter-layer activations.
    /// Using ~2^28 gives 28 bits of dynamic range for typical values,
    /// vs only 15 bits with the old i16 scheme. This is the key lever
    /// against inter-layer error amplification.
    pub(super) const ACT_TARGET: f64 = 268_435_456.0; // 2^28

    /// Bakes a trained KAN network into fixed-point quantized form.
    ///
    /// # Arguments
    ///
    /// * `network` - Trained KAN network to bake.
    /// * `calibration` - Optional calibration inputs (flat: [n_samples * input_dim]).
    ///   If provided, activation scales use the 99.9th percentile (without clipping
    ///   outputs), so outliers do not waste the i32 dynamic range. If None, a heuristic
    ///   scale is used and `uncalibrated` is set. Empty calibration uses that same
    ///   heuristic; nonempty incomplete or nonfinite calibration is rejected.
    ///
    /// # Panics
    ///
    /// Panics on invalid model/calibration data or an unrepresentable Q15.16 grid
    /// (use [`Self::try_from_network`] for errors), including any layer whose
    /// `spline_order` is outside `2..=5`. Baked inference
    /// only has fixed-point basis polynomials for those orders, even though
    /// [`KanConfig::validate`](crate::KanConfig::validate) accepts
    /// `1..=MAX_SPLINE_ORDER` (7) for the f32 CPU path. Use
    /// [`KanNetwork::forward_single`](crate::KanNetwork::forward_single) for other orders.
    pub fn from_network(network: &KanNetwork, calibration: Option<&[f32]>) -> Self {
        Self::try_from_network(network, calibration).expect("BakedModel::from_network failed")
    }

    /// Fallible bake with shape, finite-value and Q15.16 domain validation.
    ///
    /// Empty calibration uses the same uncalibrated heuristic as `None`.
    /// Nonempty calibration must contain complete samples and finite values.
    /// Grid endpoints and interval spacing must fit Q15.16; sub-tick ranges
    /// retain the single-tick fallback, with reduced precision.
    pub fn try_from_network(
        network: &KanNetwork,
        calibration: Option<&[f32]>,
    ) -> ArkanResult<Self> {
        network.config.validate()?;
        let config = network.config.clone();
        let dims = config.layer_dims();
        if network.layers.len() != config.num_layers() {
            return Err(ArkanError::cpu(
                "baked network layer count disagrees with config",
            ));
        }
        for (i, layer) in network.layers.iter().enumerate() {
            let weights = checked_weights(layer.in_dim, layer.out_dim, layer.global_basis_size)?;
            if !(2..=5).contains(&layer.order) {
                return Err(ArkanError::cpu(format!(
                    "spline_order {} is not supported by baked inference; supported range is 2..=5",
                    layer.order
                )));
            }
            layer.validate_layout()?;
            if layer.in_dim != dims[i]
                || layer.out_dim != dims[i + 1]
                || layer.in_dim == 0
                || layer.out_dim == 0
                || layer.order != config.spline_order
                || layer.grid_size != config.grid_size
                || layer.global_basis_size != layer.grid_size + layer.order
                || layer.local_basis_size != layer.order + 1
                || layer.basis_aligned < layer.local_basis_size
                || layer.grid_range != config.grid_range
                || layer.weights.len() != weights
                || layer.bias.len() != layer.out_dim
                || layer.mean.len() != layer.in_dim
                || layer.std.len() != layer.in_dim
            {
                return Err(ArkanError::cpu(format!(
                    "baked layer {i}: invalid shape or spline_order; supported orders are 2..=5"
                )));
            }
            if !layer
                .weights
                .iter()
                .chain(&layer.bias)
                .chain(&layer.mean)
                .all(|v| v.is_finite())
                || !layer.std.iter().all(|v| v.is_finite() && *v > 0.0)
            {
                return Err(ArkanError::cpu(format!(
                    "baked layer {i}: parameters must be finite and std positive"
                )));
            }
            quantized_grid(layer.grid_range, layer.grid_size)?;
        }
        let calibration = calibration.filter(|values| !values.is_empty());
        if let Some(values) = calibration {
            if values.len() % config.input_dim != 0 || !values.iter().all(|v| v.is_finite()) {
                return Err(ArkanError::cpu(
                    "baked calibration requires complete finite samples",
                ));
            }
        }
        let uncalibrated = calibration.is_none();

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
                            if !v.is_finite() {
                                return Err(ArkanError::cpu(
                                    "baked calibration produced a nonfinite activation",
                                ));
                            }
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
                    let idx = ((mags.len() - 1) as f64 * 0.999) as usize;
                    let (_, p999, _) = mags.select_nth_unstable_by(idx, f32::total_cmp);
                    let p999 = *p999;
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

            let (q_rmin, q_rmax, h_q16) = quantized_grid((r_min, r_max), grid_size)?;

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

        let model = Self {
            config,
            layers: baked_layers,
            uncalibrated,
        };
        model.validate()?;
        Ok(model)
    }
}
