//! Metamorphic relations and edge configurations - no oracle, no fixtures.
//!
//! A metamorphic relation says how the output *must* change when the input
//! changes in a known way. It needs no reference implementation and no expected
//! values, so unlike the PyTorch fixtures it can be checked at any configuration:
//! scale the last layer's weights by `k` and the output scales by `k`, permute the
//! input features and the matching weight blocks and nothing observable moves,
//! zero every weight and the output is exactly the bias.
//!
//! The edge-configuration half covers what nothing else exercises: `in_dim = 1`,
//! `out_dim = 1`, `grid_size` at both ends of its supported range, `simd_width`
//! 4/8/16, `input_std` from 1e-3 to 1e3, inputs exactly at the `grid_range`
//! endpoints, and non-finite inputs (whose behaviour is documented here because
//! nothing else documents it).

use arkan::config::MAX_SPLINE_ORDER;
use arkan::spline::{compute_basis, compute_knots, find_span};
use arkan::{KanConfig, KanLayer, KanNetwork, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Every order `KanConfig::validate` accepts, 1 to [`MAX_SPLINE_ORDER`].
///
/// Order 1 was excluded here for no reason at all: a metamorphic relation - scaling,
/// permutation, zeroing, `forward_single` vs `forward_batch`, `simd_width` invariance,
/// clamping past the endpoints - says nothing about smoothness, so the degeneracy of a
/// piecewise-linear basis is irrelevant to every test in this file. It only meant that
/// `spline_order = 1`, which `validate()` accepts, reached the layer code in exactly
/// one test ([`every_supported_grid_size_and_order_builds_and_runs`], which only checks
/// that the output is finite and not constant).
const ORDERS: std::ops::RangeInclusive<usize> = 1..=MAX_SPLINE_ORDER;

fn config(in_dim: usize, hidden: Vec<usize>, out_dim: usize, order: usize, gs: usize) -> KanConfig {
    KanConfig {
        input_dim: in_dim,
        output_dim: out_dim,
        hidden_dims: hidden,
        spline_order: order,
        grid_size: gs,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; in_dim],
        input_std: vec![1.0; in_dim],
        init_seed: Some(7),
        ..Default::default()
    }
}

/// Fills every layer with non-degenerate weights and biases.
fn randomize(net: &mut KanNetwork, seed: u64, scale: f32) {
    let mut rng = SmallRng::seed_from_u64(seed);
    for layer in net.layers.iter_mut() {
        for w in layer.weights.iter_mut() {
            *w = rng.gen_range(-1.0..1.0) * scale;
        }
        for b in layer.bias.iter_mut() {
            *b = rng.gen_range(-0.3..0.3) * scale;
        }
    }
}

fn seeded(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn run_batch(net: &KanNetwork, input: &[f32]) -> Vec<f32> {
    let batch = input.len() / net.config.input_dim;
    let mut ws = net.create_workspace(batch);
    let mut out = vec![0.0f32; batch * net.config.output_dim];
    net.forward_batch(input, &mut out, &mut ws);
    out
}

// ===========================================================================
// Metamorphic relations
// ===========================================================================

#[test]
fn all_zero_weights_output_exactly_the_last_bias() {
    // Every spline term is `weight * basis`; with all weights zero the output is
    // the last layer's bias whatever the input does, including inputs far outside
    // the grid range. Exact equality, not a tolerance: 0 * anything finite is 0.
    for order in ORDERS {
        for gs in [1usize, 5, 64] {
            let mut net = KanNetwork::new(config(3, vec![5, 4], 3, order, gs));
            randomize(&mut net, 2, 1.0);
            for layer in net.layers.iter_mut() {
                for w in layer.weights.iter_mut() {
                    *w = 0.0;
                }
            }
            let bias = net.layers.last().unwrap().bias.clone();
            let input = seeded(4 * 3, -50.0, 50.0, 21);
            let out = run_batch(&net, &input);
            for (i, v) in out.iter().enumerate() {
                assert_eq!(
                    *v,
                    bias[i % 3],
                    "order={order} grid={gs} index {i}: zero weights must give the bias"
                );
            }
        }
    }
}

#[test]
fn scaling_the_last_layer_scales_the_output() {
    // With the last layer's bias zeroed, its output is linear in its weights, so
    // `w -> k*w` must scale every output by exactly `k`.
    let mut worst = 0.0f32;
    for order in ORDERS {
        for k in [0.5f32, 2.0, -3.0, 1e3] {
            let mut net = KanNetwork::new(config(3, vec![5], 3, order, 5));
            randomize(&mut net, 3, 1.0);
            for b in net.layers.last_mut().unwrap().bias.iter_mut() {
                *b = 0.0;
            }
            let input = seeded(4 * 3, -0.9, 0.9, 31);
            let base = run_batch(&net, &input);

            let mut scaled = net.clone();
            for w in scaled.layers.last_mut().unwrap().weights.iter_mut() {
                *w *= k;
            }
            let got = run_batch(&scaled, &input);

            for i in 0..got.len() {
                let want = k * base[i];
                let rel = (got[i] - want).abs() / want.abs().max(1e-6);
                worst = worst.max(rel);
                assert!(
                    rel < 1e-4,
                    "order={order} k={k} index {i}: expected {want}, got {}",
                    got[i]
                );
            }
        }
    }
    println!("last-layer scaling: worst relative error = {worst:e}");
}

#[test]
fn permuting_inputs_and_their_weight_blocks_changes_nothing() {
    // The input axis of layer 0's coefficients, plus `mean`/`std`, carries all the
    // per-feature identity. Permute the features and those three together and the
    // network must be the same function.
    let perm = [3usize, 0, 4, 1, 2];
    let in_dim = perm.len();
    let mut worst = 0.0f32;

    for order in ORDERS {
        let mut cfg = config(in_dim, vec![6], 3, order, 5);
        // Per-feature normalization, so the permutation has something to move.
        cfg.input_mean = vec![0.1, -0.2, 0.3, 0.0, 0.5];
        cfg.input_std = vec![0.5, 1.5, 0.8, 2.0, 1.0];
        let mut net = KanNetwork::new(cfg);
        randomize(&mut net, 4, 1.0);

        let batch = 4;
        let input = seeded(batch * in_dim, -1.5, 1.5, 41);
        let base = run_batch(&net, &input);

        let mut permuted = net.clone();
        {
            let gbs = net.layers[0].global_basis_size;
            let src_w = net.layers[0].weights.clone();
            let src_mean = net.layers[0].mean.clone();
            let src_std = net.layers[0].std.clone();
            let l0 = &mut permuted.layers[0];
            for j in 0..l0.out_dim {
                for (i, &p) in perm.iter().enumerate() {
                    for k in 0..gbs {
                        l0.weights[(j * in_dim + i) * gbs + k] = src_w[(j * in_dim + p) * gbs + k];
                    }
                }
            }
            for (i, &p) in perm.iter().enumerate() {
                l0.mean[i] = src_mean[p];
                l0.std[i] = src_std[p];
            }
        }
        let mut permuted_input = vec![0.0f32; input.len()];
        for b in 0..batch {
            for (i, &p) in perm.iter().enumerate() {
                permuted_input[b * in_dim + i] = input[b * in_dim + p];
            }
        }
        let got = run_batch(&permuted, &permuted_input);

        for i in 0..got.len() {
            let gap = (got[i] - base[i]).abs();
            worst = worst.max(gap);
            assert!(
                gap < 1e-4,
                "order={order} index {i}: permutation changed the output, {} vs {}",
                got[i],
                base[i]
            );
        }
    }
    println!("input permutation: worst absolute error = {worst:e}");
}

#[test]
fn permuting_the_last_layer_rows_permutes_the_outputs() {
    // Output `j` reads only row `j` of the last layer's coefficients and `bias[j]`,
    // so permuting those rows must permute the outputs and nothing else. Exact:
    // the same additions happen in the same order, just relabelled.
    let perm = [2usize, 0, 1];
    for order in ORDERS {
        let mut net = KanNetwork::new(config(3, vec![5], 3, order, 5));
        randomize(&mut net, 5, 1.0);
        let input = seeded(4 * 3, -0.9, 0.9, 51);
        let base = run_batch(&net, &input);

        let mut permuted = net.clone();
        {
            let last = net.layers.last().unwrap();
            let (in_dim, gbs) = (last.in_dim, last.global_basis_size);
            let src_w = last.weights.clone();
            let src_b = last.bias.clone();
            let out_layer = permuted.layers.last_mut().unwrap();
            for (j, &p) in perm.iter().enumerate() {
                out_layer.bias[j] = src_b[p];
                for i in 0..in_dim {
                    for k in 0..gbs {
                        out_layer.weights[(j * in_dim + i) * gbs + k] =
                            src_w[(p * in_dim + i) * gbs + k];
                    }
                }
            }
        }
        let got = run_batch(&permuted, &input);
        for b in 0..4 {
            for (j, &p) in perm.iter().enumerate() {
                assert_eq!(
                    got[b * 3 + j],
                    base[b * 3 + p],
                    "order={order} sample {b} output {j}"
                );
            }
        }
    }
}

#[test]
fn normalization_is_equivalent_to_pre_normalizing_the_input() {
    // `forward(x; mean=m, std=s)` must equal `forward((x - m) / s; mean=0, std=1)`
    // with identical coefficients. This is the only relation here that exercises
    // extreme `std`, where the fixtures' `std = 1` says nothing.
    let in_dim = 4;
    let mut worst = 0.0f32;
    for order in ORDERS {
        for (mean, std) in [(0.0f32, 1e-3f32), (0.0, 1e3), (2.5, 0.05), (-4.0, 7.0)] {
            let mut cfg = config(in_dim, vec![5], 3, order, 5);
            cfg.input_mean = vec![mean; in_dim];
            cfg.input_std = vec![std; in_dim];
            let mut net = KanNetwork::new(cfg);
            randomize(&mut net, 6, 1.0);

            // Same coefficients, identity normalization.
            let mut identity = net.clone();
            identity.layers[0].set_normalization(&vec![0.0; in_dim], &vec![1.0; in_dim]);

            let batch = 4;
            let z = seeded(batch * in_dim, -1.4, 1.4, 61);
            let x: Vec<f32> = z.iter().map(|v| v * std + mean).collect();

            let from_raw = run_batch(&net, &x);
            let from_normalized = run_batch(&identity, &z);
            for i in 0..from_raw.len() {
                // `x = z*std + mean` then `(x - mean)/std` is not exact in f32, so
                // this is a tolerance, not equality. It is tight because the spline
                // is Lipschitz on the grid.
                let gap = (from_raw[i] - from_normalized[i]).abs();
                worst = worst.max(gap);
                assert!(
                    gap < 5e-3,
                    "order={order} mean={mean} std={std:e} index {i}: {} vs {}",
                    from_raw[i],
                    from_normalized[i]
                );
            }
        }
    }
    println!("normalization equivalence: worst absolute error = {worst:e}");
}

#[test]
fn a_single_layer_equals_the_explicit_basis_weighted_sum() {
    // Recomputation of `output[j] = bias[j] + sum_i sum_k
    // w[j, i, span_i - order + k] * B_k(z_i)` straight from the public spline API,
    // in f64. This is what pins the weight indexing and the SIMD accumulation
    // paths: `accumulate_batch` picks 8-wide, 4-wide or scalar depending on
    // `simd_width` and `in_dim`, and all three must agree with the definition.
    //
    // It is *not* independent of the basis: it calls the same `compute_basis` the
    // forward pass does, so any error in the basis values moves both sides equally and
    // cancels. Verified - a swapped-channel order-1 fast path inside `compute_basis`
    // leaves this test passing while `every_basis_channel_matches_an_independent_cox_de_boor`
    // in `tests/spline_properties.rs` fails. That f64 Cox-de Boor reference is where
    // basis values are pinned; here, only the indexing and accumulation around them.
    let mut worst = 0.0f64;
    let mut worst_at = String::new();

    for order in ORDERS {
        for simd_width in [4usize, 8, 16] {
            for in_dim in [1usize, 3, 4, 7, 8, 9, 16, 21] {
                let out_dim = 3;
                let mut cfg = config(in_dim, vec![], out_dim, order, 5);
                cfg.simd_width = simd_width;
                cfg.grid_range = (-2.0, 3.0);
                cfg.input_mean = vec![0.25; in_dim];
                cfg.input_std = vec![1.5; in_dim];
                cfg.validate().expect("config must be valid");

                let mut layer = KanLayer::new(in_dim, out_dim, &cfg);
                let mut rng = SmallRng::seed_from_u64(71 + order as u64);
                for w in layer.weights.iter_mut() {
                    *w = rng.gen_range(-1.0..1.0);
                }
                for b in layer.bias.iter_mut() {
                    *b = rng.gen_range(-0.5..0.5);
                }

                let batch = 5;
                let input = seeded(batch * in_dim, -6.0, 8.0, 81 + in_dim as u64);
                let mut ws = Workspace::default();
                let mut got = vec![0.0f32; batch * out_dim];
                layer.forward_batch(&input, &mut got, &mut ws);

                let knots = compute_knots(cfg.grid_size, order, cfg.grid_range);
                let gbs = cfg.grid_size + order;
                for b in 0..batch {
                    for j in 0..out_dim {
                        let mut want = f64::from(layer.bias[j]);
                        for i in 0..in_dim {
                            let z = ((input[b * in_dim + i] - cfg.input_mean[i])
                                / cfg.input_std[i])
                                .clamp(cfg.grid_range.0, cfg.grid_range.1);
                            let span = find_span(z, &knots, order, cfg.grid_size);
                            let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                            compute_basis(z, span, &knots, order, &mut basis[..=order]);
                            for (k, &bv) in basis[..=order].iter().enumerate() {
                                let w = layer.weights[(j * in_dim + i) * gbs + (span - order + k)];
                                want += f64::from(w) * f64::from(bv);
                            }
                        }
                        let err = (want - f64::from(got[b * out_dim + j])).abs();
                        if err > worst {
                            worst = err;
                            worst_at = format!(
                                "order={order} simd_width={simd_width} in_dim={in_dim} \
                                 sample={b} out={j}: want={want} got={}",
                                got[b * out_dim + j]
                            );
                        }
                    }
                }
            }
        }
    }

    println!("explicit basis-weighted sum: worst error = {worst:e} at {worst_at}");
    assert!(
        worst < 1e-4,
        "forward_batch disagrees with the definition: {worst:e} at {worst_at}"
    );
}

#[test]
fn simd_width_is_a_performance_knob_not_a_semantic_one() {
    // 4, 8 and 16 select different accumulation kernels and different basis
    // strides. They must compute the same function up to f32 reassociation.
    let mut worst = 0.0f32;
    for order in ORDERS {
        for in_dim in [3usize, 8, 9, 21] {
            let mut outs = Vec::new();
            for simd_width in [4usize, 8, 16] {
                let mut cfg = config(in_dim, vec![8, 6], 3, order, 5);
                cfg.simd_width = simd_width;
                let mut net = KanNetwork::new(cfg);
                randomize(&mut net, 91, 1.0);
                outs.push(run_batch(&net, &seeded(4 * in_dim, -0.9, 0.9, 101)));
            }
            for (i, &reference) in outs[0].iter().enumerate() {
                for (w, other) in [(8usize, outs[1][i]), (16, outs[2][i])] {
                    let gap = (reference - other).abs();
                    worst = worst.max(gap);
                    assert!(
                        gap < 1e-4,
                        "order={order} in_dim={in_dim} index {i}: simd_width 4 vs {w} changed \
                         the result: {reference} vs {other}"
                    );
                }
            }
        }
    }
    println!("simd_width invariance: worst absolute difference = {worst:e}");
}

// ===========================================================================
// forward_single vs forward_batch
// ===========================================================================

/// Nudges every element one ULP toward `+inf`.
fn ulp_up(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|x| {
            if *x == 0.0 {
                f32::from_bits(1)
            } else if *x > 0.0 {
                f32::from_bits(x.to_bits() + 1)
            } else {
                f32::from_bits(x.to_bits() - 1)
            }
        })
        .collect()
}

/// **`forward_single` and `forward_batch` are not bit-identical, and no fixed
/// tolerance is the right way to say how close they are.**
///
/// The two paths perform the same additions in different groupings.
/// `forward_single` accumulates per input feature (`out += sum_k w*B` for each
/// `i`); `accumulate_batch` accumulates the whole `(i, k)` product set into one
/// register before adding the bias, and at `in_dim >= 8` with `simd_width = 8` it
/// does so in eight lanes. They coincide bit-for-bit when the groupings happen to
/// match (`in_dim == 1`, and `in_dim == 8` at `simd_width == 8`, where the lane
/// sum reproduces the per-feature order) and differ by ~1 ULP otherwise.
///
/// One ULP at the input is not one ULP at the output. A stack of KAN layers has a
/// large Lipschitz constant - at `grid_size = 16` a single basis derivative is
/// `~order/h = 16`, multiplied by `in_dim` per layer - so a deep network amplifies
/// its own float noise. Measured on a 7-layer, `in_dim = 8`, `grid_size = 16` net:
/// perturbing **one input by one ULP** moves `forward_batch`'s output by 0.048,
/// while `forward_single` differs from `forward_batch` by 0.105. The paths are not
/// disagreeing; the network is chaotic and neither answer is more correct.
///
/// So the bound asserted here is the network's *own* sensitivity, measured per
/// configuration by re-running `forward_batch` on a one-ULP-nudged input. A real
/// divergence - wrong weight indexing, a mis-wired ping-pong buffer - is `O(1)`
/// while that floor stays at `1e-7` for the shallow configurations, so it is still
/// caught.
#[test]
fn forward_single_matches_forward_batch_to_within_the_networks_own_noise() {
    let mut worst_ratio = 0.0f32;
    let mut worst_at = String::new();
    let mut worst_shallow = 0.0f32;

    for order in ORDERS {
        for gs in [1usize, 5, 16] {
            for hidden in [vec![], vec![9], vec![9, 7], vec![8, 8, 8, 8], vec![8; 6]] {
                for in_dim in [1usize, 3, 8, 9] {
                    let mut net = KanNetwork::new(config(in_dim, hidden.clone(), 4, order, gs));
                    randomize(&mut net, 111, 1.0);
                    let batch = 6;
                    let input = seeded(batch * in_dim, -0.9, 0.9, 121);
                    let batched = run_batch(&net, &input);

                    // The network's response to a one-ULP input perturbation: the
                    // scale at which its own arithmetic stops being meaningful.
                    let nudged = run_batch(&net, &ulp_up(&input));
                    let noise = batched
                        .iter()
                        .zip(&nudged)
                        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                    let bound = 1e-5 + 8.0 * noise;

                    let mut ws = net.create_workspace(1);
                    for b in 0..batch {
                        let mut single = vec![0.0f32; 4];
                        net.forward_single(
                            &input[b * in_dim..(b + 1) * in_dim],
                            &mut single,
                            &mut ws,
                        );
                        for j in 0..4 {
                            let gap = (single[j] - batched[b * 4 + j]).abs();
                            if hidden.len() <= 2 {
                                worst_shallow = worst_shallow.max(gap);
                            }
                            let ratio = gap / bound;
                            if ratio > worst_ratio {
                                worst_ratio = ratio;
                                worst_at = format!(
                                    "order={order} grid={gs} hidden={hidden:?} in_dim={in_dim} \
                                     sample={b} out={j}: single={} batch={} gap={gap:e} \
                                     one_ulp_noise={noise:e}",
                                    single[j],
                                    batched[b * 4 + j]
                                );
                            }
                            assert!(
                                gap <= bound,
                                "single/batch gap {gap:e} exceeds the network's own one-ULP \
                                 noise floor {bound:e} at {order} {gs} {hidden:?} {in_dim} \
                                 sample={b} out={j}: single={} batch={}",
                                single[j],
                                batched[b * 4 + j]
                            );
                        }
                    }
                }
            }
        }
    }

    println!(
        "forward_single vs forward_batch: worst gap/noise ratio = {worst_ratio:.3} at {worst_at}"
    );
    // Up to two hidden layers the network is well conditioned and the gap really is
    // just reassociation, so it can be bounded absolutely as well.
    println!("shallow (<= 2 hidden layers) worst absolute gap = {worst_shallow:e}");
    assert!(
        worst_shallow < 1e-4,
        "shallow networks should agree to reassociation only, got {worst_shallow:e}"
    );
}

// ===========================================================================
// Edge configurations
// ===========================================================================

#[test]
fn minimal_and_maximal_dimensions_work() {
    // in_dim = 1, out_dim = 1, one layer, grid_size at both ends of 1..=64.
    for order in ORDERS {
        for gs in [1usize, 64] {
            let cfg = config(1, vec![], 1, order, gs);
            cfg.validate().expect("minimal config must validate");
            let mut net = KanNetwork::new(cfg);
            randomize(&mut net, 131, 1.0);
            let input = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0];
            let out = run_batch(&net, &input);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "order={order} grid={gs}: non-finite output {out:?}"
            );
            // A one-input, one-output layer is a scalar spline; with grid_size = 1
            // it is a single polynomial piece, so it cannot oscillate.
            assert_eq!(out.len(), 5);
        }
    }

    // Deep and narrow: eight layers of width 1 must still produce finite output.
    let cfg = config(1, vec![1; 7], 1, 3, 5);
    let mut net = KanNetwork::new(cfg);
    randomize(&mut net, 132, 1.0);
    let out = run_batch(&net, &[-0.7f32, 0.0, 0.7]);
    assert!(out.iter().all(|v| v.is_finite()), "deep narrow: {out:?}");
}

#[test]
fn inputs_at_and_beyond_the_grid_endpoints_are_clamped_not_extrapolated() {
    // Exactly at an endpoint the basis is one-sided; beyond it the forward map is
    // constant. Both must be finite, and everything past the endpoint must equal
    // the endpoint value exactly - that is what makes the saturated gradient zero.
    for order in ORDERS {
        for gr in [(-1.0f32, 1.0f32), (0.0, 1.0), (0.5, 2.5), (-5.0, 5.0)] {
            let mut cfg = config(3, vec![4], 2, order, 5);
            cfg.grid_range = gr;
            let mut net = KanNetwork::new(cfg);
            randomize(&mut net, 141, 1.0);

            let at_low = run_batch(&net, &[gr.0; 3]);
            let at_high = run_batch(&net, &[gr.1; 3]);
            assert!(at_low.iter().all(|v| v.is_finite()));
            assert!(at_high.iter().all(|v| v.is_finite()));

            for over in [1e-3f32, 1.0, 1e3, 1e9] {
                let below = run_batch(&net, &[gr.0 - over; 3]);
                let above = run_batch(&net, &[gr.1 + over; 3]);
                assert_eq!(
                    below, at_low,
                    "order={order} range={gr:?}: input {over} below the grid is not clamped"
                );
                assert_eq!(
                    above, at_high,
                    "order={order} range={gr:?}: input {over} above the grid is not clamped"
                );
            }
        }
    }
}

/// **Documented behaviour for non-finite inputs.** Nothing in the crate states
/// this, so it is pinned here rather than left to be discovered.
///
/// - `+inf` / `-inf`: `clamp` maps them to `grid_max` / `grid_min`, so an infinite
///   input behaves *exactly* like an input parked on that endpoint. No panic, no
///   NaN, and the gradient is zero because the feature reads as saturated.
/// - `NaN`: propagates. `clamp` returns NaN, `find_span`'s `NaN as isize` saturates
///   to 0 so the span is the first one (in range, no out-of-bounds read), and the
///   basis is NaN, so every output of *that sample* is NaN.
/// - Contamination is per-sample in a batch: a NaN in sample 0 does not touch
///   sample 1. It is *not* per-feature - one NaN feature poisons the whole sample,
///   because every output sums over every input.
#[test]
fn non_finite_inputs_have_defined_behaviour() {
    for order in ORDERS {
        let mut net = KanNetwork::new(config(3, vec![4], 2, order, 5));
        randomize(&mut net, 151, 1.0);

        let input = vec![
            f32::NAN,
            0.1,
            0.2, // sample 0: NaN in feature 0
            f32::INFINITY,
            0.1,
            0.2, // sample 1
            f32::NEG_INFINITY,
            0.1,
            0.2, // sample 2
            0.5,
            0.1,
            0.2, // sample 3: clean control
        ];
        let out = run_batch(&net, &input);

        // Sample 0: NaN propagates to every output of that sample.
        assert!(
            out[0].is_nan() && out[1].is_nan(),
            "order={order}: NaN input did not propagate: {:?}",
            &out[0..2]
        );
        // Samples 1-3: unaffected by the NaN in sample 0.
        assert!(
            out[2..].iter().all(|v| v.is_finite()),
            "order={order}: NaN leaked across samples: {out:?}"
        );

        // Infinities behave exactly like the corresponding clamped endpoint.
        let clamped = {
            let mut c = input.clone();
            c[3] = net.config.grid_range.1;
            c[6] = net.config.grid_range.0;
            run_batch(&net, &c)
        };
        assert_eq!(
            out[2..6],
            clamped[2..6],
            "order={order}: +/-inf does not match the clamped endpoint"
        );

        // `forward_single` agrees on the NaN convention.
        let mut ws = net.create_workspace(1);
        let mut single = vec![0.0f32; 2];
        net.forward_single(&input[0..3], &mut single, &mut ws);
        assert!(
            single.iter().all(|v| v.is_nan()),
            "order={order}: forward_single disagrees on NaN: {single:?}"
        );
    }
}

/// `normalize_batch` processes eight samples per SIMD step and the remainder with
/// scalar code. Position in the batch must not change the answer.
///
/// It used to: `f32x8::max`/`min` return the other operand for a NaN lane, so with
/// `grid_range = (-1, 1)` a batch of 13 NaNs came back as eight `-1.0`s followed by
/// five NaNs. Fixed in `src/spline.rs` by restoring NaN after the clamp.
#[test]
fn normalize_batch_does_not_depend_on_position_in_the_batch() {
    use arkan::spline::normalize_batch;

    for grid_range in [(-1.0f32, 1.0f32), (0.0, 1.0), (0.5, 2.5), (-5.0, 5.0)] {
        for std in [1e-6f32, 1e-3, 1.0, 1e3] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.4, 123.0] {
                // 13 = one full SIMD step of 8 plus a 5-wide scalar remainder.
                let batch = 13;
                let x = vec![value; batch];
                let mut z = vec![0.0f32; batch];
                normalize_batch(&x, &[0.25], &[std], grid_range, &mut z);

                let reference = ((value - 0.25) / std).clamp(grid_range.0, grid_range.1);
                for (b, got) in z.iter().enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        reference.to_bits(),
                        "range={grid_range:?} std={std:e} value={value} slot {b} \
                         (SIMD head is 0..8, scalar tail is 8..13): got {got}, \
                         f32::clamp gives {reference}"
                    );
                }
            }
        }
    }
}

#[test]
fn extreme_input_std_stays_finite() {
    // `input_std` is only bounded below by EPSILON, so both ends are reachable.
    for std in [1e-6f32, 1e-3, 1.0, 1e3, 1e6] {
        let mut cfg = config(3, vec![4], 2, 3, 5);
        cfg.input_std = vec![std; 3];
        cfg.validate()
            .unwrap_or_else(|e| panic!("std={std:e} rejected: {e}"));
        let mut net = KanNetwork::new(cfg);
        randomize(&mut net, 161, 1.0);

        // Inputs spanning several `std` either side of the mean.
        let input: Vec<f32> = (0..9).map(|i| (i as f32 - 4.0) * std * 0.5).collect();
        let out = run_batch(&net, &input);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "std={std:e}: non-finite output {out:?}"
        );
    }
}

#[test]
fn every_supported_grid_size_and_order_builds_and_runs() {
    // The full supported box, end to end: build, validate, forward, and check that
    // the network is not accidentally constant (which is how a collapsed basis
    // would show up).
    for order in 1..=MAX_SPLINE_ORDER {
        for gs in [1usize, 2, 3, 5, 8, 16, 32, 63, 64] {
            let cfg = config(3, vec![4], 2, order, gs);
            cfg.validate()
                .unwrap_or_else(|e| panic!("order={order} grid={gs} rejected: {e}"));
            let mut net = KanNetwork::new(cfg);
            randomize(&mut net, 171, 1.0);
            let input = seeded(8 * 3, -0.95, 0.95, 181);
            let out = run_batch(&net, &input);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "order={order} grid={gs}: non-finite output"
            );
            let spread = out.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b))
                - out.iter().fold(f32::INFINITY, |a, b| a.min(*b));
            assert!(
                spread > 1e-3,
                "order={order} grid={gs}: output is effectively constant (spread {spread:e}), \
                 which is what a collapsed basis looks like"
            );
        }
    }
}

/// `init_seed: Some(s)` must mean "reproducible", not "every layer is the same
/// layer".
///
/// Every `KanLayer` used to build its own `SmallRng` from the *shared*
/// `config.init_seed` and draw its whole weight vector from it, so two layers with
/// the same `(in_dim, out_dim, basis_size)` came out bit-identical - a seeded run
/// started from a layer-to-layer symmetric point. `hidden_dims: vec![64, 64]` is
/// enough to hit it. `KanConfig::preset()` happens to have no two same-shaped
/// layers, which is why nothing noticed.
#[test]
fn seeded_init_does_not_clone_identically_shaped_layers() {
    let mut cfg = config(8, vec![8, 8, 8], 8, 3, 5);
    cfg.init_seed = Some(1234);
    let net = KanNetwork::new(cfg.clone());
    assert_eq!(net.layers.len(), 4, "all four layers are 8x8 here");

    for i in 0..net.layers.len() {
        for j in i + 1..net.layers.len() {
            assert_ne!(
                net.layers[i].weights, net.layers[j].weights,
                "layers {i} and {j} have identical weights - the seed is not being \
                 varied per layer"
            );
        }
    }

    // Still deterministic: same seed, same network, bit for bit.
    let again = KanNetwork::new(cfg.clone());
    for (i, (a, b)) in net.layers.iter().zip(&again.layers).enumerate() {
        assert_eq!(a.weights, b.weights, "layer {i} is not reproducible");
    }

    // And a different seed is a different network.
    cfg.init_seed = Some(4321);
    let other = KanNetwork::new(cfg);
    assert_ne!(net.layers[0].weights, other.layers[0].weights);
}
