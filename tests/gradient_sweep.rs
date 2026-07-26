//! Finite-difference gradient checks, swept over the configuration space.
//!
//! # Why this is the highest-value file here
//!
//! A central difference through the *real* forward pass is an oracle that needs no
//! reference data: it works at any `(order, grid_size, grid_range, mean, std,
//! depth)` we can construct. `tests/clamp_gradient_parity.rs` already does this
//! for one axis - the normalization clamp - and it is what caught that bug. This
//! generalizes it to the axes the PyTorch fixtures never move:
//!
//! | axis                | fixtures        | here                          |
//! |---------------------|-----------------|-------------------------------|
//! | `spline_order`      | 3, 4            | 2..=7                         |
//! | `grid_size`         | 5               | 1, 2, 5, 8, 16                |
//! | `grid_range`        | (-1, 1)         | + (0,1), (-5,5), (0.5,2.5)    |
//! | `input_mean`/`_std` | 0 / 1           | + 0.05, 0.3, 5.0 and offsets  |
//! | depth               | 1-2 layers      | 1 and 4 layers                |
//! | saturation          | never clamped   | fully and partially clamped   |
//!
//! # Finite-difference hygiene
//!
//! Two things make a central difference lie, and both are handled explicitly
//! rather than absorbed into a loose tolerance:
//!
//! 1. **Straddling a kink.** At the clamp boundary the loss is only piecewise
//!    differentiable. Network-level probes therefore record which activations are
//!    clamped and skip any probe that changes that pattern.
//! 2. **Step size in the wrong space.** The gradient lives in raw-`x` space but the
//!    spline segments live in `z = (x - mean) / std` space. A step that is safely
//!    sub-segment at `std = 1` straddles several segments at `std = 0.05`, so every
//!    `grad_input` step is scaled by `std`.
//!
//! Tolerances are `abs + rel * |finite_diff|`. The absolute floor covers `f32`
//! cancellation in the difference quotient; the relative term covers `O(h^2)`
//! truncation where the loss is strongly curved. Both are far tighter than the
//! errors any real gradient bug produces - the clamp bug reported a nonzero
//! gradient where the truth was exactly zero.

use arkan::config::MAX_SPLINE_ORDER;
use arkan::{KanConfig, KanLayer, KanNetwork, TrainOptions, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const ORDERS: std::ops::RangeInclusive<usize> = 2..=MAX_SPLINE_ORDER;
const GRID_SIZES: [usize; 5] = [1, 2, 5, 8, 16];
const RANGES: [(f32, f32); 4] = [(-1.0, 1.0), (0.0, 1.0), (-5.0, 5.0), (0.5, 2.5)];

// ===========================================================================
// Layer-level harness
// ===========================================================================

/// `L = sum_{b,j} g_out[b,j] * y[b,j]`, accumulated in `f64`. Its gradient w.r.t.
/// the raw inputs is exactly what `backward` reports in `grad_input` when seeded
/// with `grad_output = g_out`, and w.r.t. the weights what it reports in
/// `grad_weights`.
fn directional_loss(layer: &KanLayer, inputs: &[f32], g_out: &[f32], ws: &mut Workspace) -> f64 {
    let batch = inputs.len() / layer.in_dim;
    let mut out = vec![0.0f32; batch * layer.out_dim];
    layer.forward_batch(inputs, &mut out, ws);
    out.iter()
        .zip(g_out)
        .map(|(y, g)| f64::from(*y) * f64::from(*g))
        .sum()
}

struct LayerGrads {
    input: Vec<f32>,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

/// Drives `backward` exactly the way `KanNetwork` does: forward stores `z` and the
/// span/clamp flags, backward consumes them.
fn analytic_grads(layer: &KanLayer, inputs: &[f32], g_out: &[f32]) -> LayerGrads {
    let batch = inputs.len() / layer.in_dim;
    let mut ws = Workspace::default();
    let mut out = vec![0.0f32; batch * layer.out_dim];
    layer.forward_batch(inputs, &mut out, &mut ws);

    let z: Vec<f32> = ws.z_buffer.as_slice()[..batch * layer.in_dim].to_vec();
    let spans: Vec<u32> = ws.grid_indices[..batch * layer.in_dim].to_vec();

    let mut g = LayerGrads {
        input: vec![0.0; batch * layer.in_dim],
        weights: vec![0.0; layer.weights.len()],
        bias: vec![0.0; layer.bias.len()],
    };
    layer.backward(
        &z,
        &spans,
        g_out,
        Some(&mut g.input),
        &mut g.weights,
        &mut g.bias,
        &mut ws,
    );
    g
}

/// A layer with non-degenerate weights and biases. Freshly initialized weights are
/// ~0.1, small enough that a wrong gradient can hide under the tolerance floor.
fn dense_layer(cfg: &KanConfig, in_dim: usize, out_dim: usize, seed: u64) -> KanLayer {
    let mut layer = KanLayer::new(in_dim, out_dim, cfg);
    let mut rng = SmallRng::seed_from_u64(seed);
    for w in layer.weights.iter_mut() {
        *w = rng.gen_range(-1.0..1.0);
    }
    for b in layer.bias.iter_mut() {
        *b = rng.gen_range(-0.5..0.5);
    }
    layer
}

fn layer_config(
    in_dim: usize,
    out_dim: usize,
    order: usize,
    grid_size: usize,
    grid_range: (f32, f32),
    mean: f32,
    std: f32,
) -> KanConfig {
    KanConfig {
        input_dim: in_dim,
        output_dim: out_dim,
        hidden_dims: vec![],
        spline_order: order,
        grid_size,
        grid_range,
        input_mean: vec![mean; in_dim],
        input_std: vec![std; in_dim],
        init_seed: Some(20260726),
        ..Default::default()
    }
}

/// Checks `grad_input`, `grad_weights` and `grad_bias` for one layer against
/// central differences, and pushes a line per mismatch onto `failures`.
fn check_layer(
    label: &str,
    layer: &KanLayer,
    inputs: &[f32],
    g_out: &[f32],
    failures: &mut Vec<String>,
) {
    let batch = inputs.len() / layer.in_dim;
    let ana = analytic_grads(layer, inputs, g_out);
    let mut ws = Workspace::default();
    let width = layer.grid_range.1 - layer.grid_range.0;

    // --- grad_input ---
    let mut probe = inputs.to_vec();
    for idx in 0..inputs.len() {
        // Step measured in z-space (1/1000 of the grid range) and converted back
        // to raw-x space, so it stays sub-segment whatever `std` is.
        let h = width * 1e-3 * layer.std[idx % layer.in_dim];
        let x = inputs[idx];
        probe[idx] = x + h;
        let plus = directional_loss(layer, &probe, g_out, &mut ws);
        probe[idx] = x - h;
        let minus = directional_loss(layer, &probe, g_out, &mut ws);
        probe[idx] = x;
        let fd = (plus - minus) / (2.0 * f64::from(h));
        let gap = (fd - f64::from(ana.input[idx])).abs();
        if gap > 3e-3 + 3e-2 * fd.abs() {
            failures.push(format!(
                "  [{label}] grad_input[{idx}] x={x:e} analytic={} finite_diff={fd} gap={gap:e}",
                ana.input[idx]
            ));
        }
    }

    // --- grad_weights, sampled across the whole coefficient block ---
    let hw = 1e-2f32;
    let mut probe_layer = layer.clone();
    let n = layer.weights.len();
    for widx in (0..n).step_by((n / 37).max(1)) {
        let orig = layer.weights[widx];
        probe_layer.weights[widx] = orig + hw;
        let plus = directional_loss(&probe_layer, inputs, g_out, &mut ws);
        probe_layer.weights[widx] = orig - hw;
        let minus = directional_loss(&probe_layer, inputs, g_out, &mut ws);
        probe_layer.weights[widx] = orig;
        let fd = (plus - minus) / (2.0 * f64::from(hw));
        let gap = (fd - f64::from(ana.weights[widx])).abs();
        if gap > 3e-3 + 3e-2 * fd.abs() {
            failures.push(format!(
                "  [{label}] grad_weights[{widx}] analytic={} finite_diff={fd} gap={gap:e}",
                ana.weights[widx]
            ));
        }
    }

    // --- grad_bias: exactly sum_b g_out[b, j], no finite difference needed ---
    for j in 0..layer.out_dim {
        let want: f64 = (0..batch)
            .map(|b| f64::from(g_out[b * layer.out_dim + j]))
            .sum();
        let gap = (want - f64::from(ana.bias[j])).abs();
        if gap > 1e-4 {
            failures.push(format!(
                "  [{label}] grad_bias[{j}] analytic={} expected={want} gap={gap:e}",
                ana.bias[j]
            ));
        }
    }
}

fn seeded(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

// ===========================================================================
// Layer-level sweeps
// ===========================================================================

#[test]
fn layer_gradients_across_order_grid_and_range() {
    let (in_dim, out_dim, batch) = (5, 3, 4);
    let mut failures = Vec::new();
    let mut cases = 0;

    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
                // in_dim != config.input_dim would give identity normalization; here
                // they match, so this is the input-layer path with mean=0/std=1.
                let cfg = layer_config(in_dim, out_dim, order, grid_size, gr, 0.0, 1.0);
                let layer =
                    dense_layer(&cfg, in_dim, out_dim, 31 * order as u64 + grid_size as u64);
                let pad = 0.08 * (gr.1 - gr.0);
                let inputs = seeded(batch * in_dim, gr.0 + pad, gr.1 - pad, 1000 + order as u64);
                let g_out = seeded(batch * out_dim, -1.0, 1.0, 2000 + grid_size as u64);
                check_layer(
                    &format!("order={order} grid={grid_size} range={gr:?}"),
                    &layer,
                    &inputs,
                    &g_out,
                    &mut failures,
                );
                cases += 1;
            }
        }
    }

    println!("layer gradient sweep: {cases} configurations");
    assert!(
        failures.is_empty(),
        "{} gradient mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn layer_gradients_with_nontrivial_normalization() {
    // The fixtures all use mean=0/std=1, so the `1 / std` factor in `grad_input`
    // and the `- mean` shift are effectively untested by them. Both very small and
    // very large `std` are included: `std` scales the gradient by `1 / std`, so an
    // error there is invisible at `std = 1`.
    let (in_dim, out_dim, batch) = (4, 3, 4);
    let norms: [(f32, f32); 6] = [
        (0.0, 0.25),
        (1.5, 2.0),
        (-0.7, 0.05),
        (3.0, 10.0),
        (0.0, 1e-3),
        (0.0, 1e3),
    ];
    let mut failures = Vec::new();

    for order in ORDERS {
        for gr in RANGES {
            for (mean, std) in norms {
                let cfg = layer_config(in_dim, out_dim, order, 5, gr, mean, std);
                let layer = dense_layer(&cfg, in_dim, out_dim, 13 * order as u64 + 5);
                assert_eq!(layer.std, vec![std; in_dim], "normalization did not apply");

                // Raw x chosen so z = (x - mean) / std lands inside the grid range.
                let pad = 0.08 * (gr.1 - gr.0);
                let inputs: Vec<f32> = seeded(
                    batch * in_dim,
                    gr.0 + pad,
                    gr.1 - pad,
                    77 + order as u64 * 3,
                )
                .into_iter()
                .map(|z| z * std + mean)
                .collect();
                let g_out = seeded(batch * out_dim, -1.0, 1.0, 88 + order as u64);
                check_layer(
                    &format!("order={order} range={gr:?} mean={mean} std={std:e}"),
                    &layer,
                    &inputs,
                    &g_out,
                    &mut failures,
                );
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} gradient mismatches with non-trivial normalization:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn saturated_inputs_have_exactly_zero_input_gradient() {
    // Generalizes `clamp_gradient_parity` off (-1, 1) and off order 3. For a
    // saturated feature `dz/dx` is exactly 0, so the forward pass is exactly flat
    // in that coordinate and the analytic gradient must be exactly 0 - not small,
    // zero. `grad_weights` is *not* zero there and is checked against the
    // finite difference as usual.
    let (in_dim, out_dim, batch) = (6, 3, 3);
    let mut failures = Vec::new();

    for order in ORDERS {
        for gr in RANGES {
            let width = gr.1 - gr.0;
            let cfg = layer_config(in_dim, out_dim, order, 5, gr, 0.0, 1.0);
            let layer = dense_layer(&cfg, in_dim, out_dim, 17 + order as u64);

            // Every feature strictly outside the grid range, alternating sides.
            let inputs: Vec<f32> = (0..batch * in_dim)
                .map(|i| {
                    let over = width * (0.3 + 0.11 * (i % 5) as f32);
                    if i % 2 == 0 {
                        gr.1 + over
                    } else {
                        gr.0 - over
                    }
                })
                .collect();
            let g_out = seeded(batch * out_dim, -1.0, 1.0, 29 + order as u64);

            let ana = analytic_grads(&layer, &inputs, &g_out);
            for (idx, g) in ana.input.iter().enumerate() {
                if *g != 0.0 {
                    failures.push(format!(
                        "  [order={order} range={gr:?}] grad_input[{idx}] = {g} on a saturated \
                         feature (x={}), must be exactly 0",
                        inputs[idx]
                    ));
                }
            }

            // Independent confirmation that the truth really is zero: the forward
            // pass must not move at all.
            let mut ws = Workspace::default();
            let base = directional_loss(&layer, &inputs, &g_out, &mut ws);
            let mut probe = inputs.clone();
            for idx in 0..inputs.len() {
                probe[idx] = inputs[idx] + width * 0.01;
                let moved = directional_loss(&layer, &probe, &g_out, &mut ws);
                probe[idx] = inputs[idx];
                assert_eq!(
                    moved, base,
                    "fixture is not actually saturated at idx={idx} (order={order}, range={gr:?})"
                );
            }

            check_layer(
                &format!("saturated order={order} range={gr:?}"),
                &layer,
                &inputs,
                &g_out,
                &mut failures,
            );
        }
    }

    assert!(
        failures.is_empty(),
        "{} saturated-gradient failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn partially_saturated_inputs_keep_the_live_features_correct() {
    // The dangerous case is a mix: getting the clamp right by zeroing everything
    // would pass the fully saturated test and fail this one.
    let (in_dim, out_dim, batch) = (6, 3, 3);
    let mut failures = Vec::new();

    for order in ORDERS {
        for gr in RANGES {
            let width = gr.1 - gr.0;
            let cfg = layer_config(in_dim, out_dim, order, 5, gr, 0.0, 1.0);
            let layer = dense_layer(&cfg, in_dim, out_dim, 41 + order as u64);

            let inputs: Vec<f32> = (0..batch * in_dim)
                .map(|i| {
                    if i % 2 == 0 {
                        // strictly inside, off the knots
                        gr.0 + width * (0.13 + 0.17 * (i % 4) as f32)
                    } else {
                        gr.1 + width * (0.4 + 0.1 * (i % 3) as f32)
                    }
                })
                .collect();
            let g_out = seeded(batch * out_dim, -1.0, 1.0, 53 + order as u64);

            let ana = analytic_grads(&layer, &inputs, &g_out);
            let live = ana.input.iter().filter(|g| **g != 0.0).count();
            assert!(
                live > 0,
                "fixture has no live features left (order={order}, range={gr:?})"
            );
            for (idx, g) in ana.input.iter().enumerate() {
                if idx % 2 == 1 && *g != 0.0 {
                    failures.push(format!(
                        "  [order={order} range={gr:?}] saturated grad_input[{idx}] = {g}"
                    ));
                }
            }

            check_layer(
                &format!("mixed order={order} range={gr:?}"),
                &layer,
                &inputs,
                &g_out,
                &mut failures,
            );
        }
    }

    assert!(
        failures.is_empty(),
        "{} mixed-saturation failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[cfg(feature = "parallel")]
#[test]
fn backward_parallel_matches_backward_across_orders() {
    let (in_dim, out_dim, batch) = (9, 4, 5);
    for order in ORDERS {
        for gr in RANGES {
            let cfg = layer_config(in_dim, out_dim, order, 5, gr, 0.0, 1.0);
            let layer = dense_layer(&cfg, in_dim, out_dim, 61 + order as u64);
            // Deliberately straddling: some features clamp, some do not.
            let inputs: Vec<f32> =
                seeded(batch * in_dim, gr.0 - 1.0, gr.1 + 1.0, 71 + order as u64);
            let g_out = seeded(batch * out_dim, -1.0, 1.0, 73 + order as u64);

            let seq = analytic_grads(&layer, &inputs, &g_out);

            let mut ws = Workspace::default();
            let mut out = vec![0.0f32; batch * out_dim];
            layer.forward_batch(&inputs, &mut out, &mut ws);
            let z: Vec<f32> = ws.z_buffer.as_slice()[..batch * in_dim].to_vec();
            let spans: Vec<u32> = ws.grid_indices[..batch * in_dim].to_vec();
            let mut par = LayerGrads {
                input: vec![0.0; batch * in_dim],
                weights: vec![0.0; layer.weights.len()],
                bias: vec![0.0; layer.bias.len()],
            };
            layer.backward_parallel(
                &z,
                &spans,
                &g_out,
                Some(&mut par.input),
                &mut par.weights,
                &mut par.bias,
            );

            for (i, (a, b)) in seq.input.iter().zip(&par.input).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "grad_input[{i}] diverges at order={order} range={gr:?}: {a} vs {b}"
                );
            }
            for (i, (a, b)) in seq.weights.iter().zip(&par.weights).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "grad_weights[{i}] diverges at order={order} range={gr:?}: {a} vs {b}"
                );
            }
        }
    }
}

// ===========================================================================
// Network-level harness: deep networks, real train_step gradients
// ===========================================================================

fn mse(net: &KanNetwork, input: &[f32], target: &[f32], ws: &mut Workspace) -> f64 {
    let batch = input.len() / net.config.input_dim;
    let mut pred = vec![0.0f32; batch * net.config.output_dim];
    net.forward_batch(input, &mut pred, ws);
    pred.iter()
        .zip(target)
        .map(|(p, t)| {
            let d = f64::from(*p) - f64::from(*t);
            d * d
        })
        .sum::<f64>()
        / pred.len() as f64
}

/// Which `(layer, sample, feature)` triples the forward pass clamped. The loss is
/// only piecewise differentiable in the weights; a probe that moves this pattern
/// straddles a kink and its central difference means nothing.
fn clamp_pattern(net: &KanNetwork, input: &[f32], ws: &mut Workspace) -> Vec<bool> {
    let batch = input.len() / net.config.input_dim;
    let mut pred = vec![0.0f32; batch * net.config.output_dim];
    net.forward_batch_training(input, &mut pred, ws);
    let mut pattern = Vec::new();
    for (l, layer) in net.layers.iter().enumerate() {
        for i in 0..batch * layer.in_dim {
            pattern.push(ws.layers_grid_indices[l][i] & arkan::spline::SPAN_CLAMPED_FLAG != 0);
        }
    }
    pattern
}

/// Central-difference step for network weights.
const NET_FD_STEP: f32 = 1e-3;

/// Refinement factor used to tell truncation error apart from a wrong gradient.
///
/// A central difference carries `O(h^2)` truncation, which is large where
/// saturation makes the loss sharply curved: a 4-layer saturating net can show 4%
/// at `h = 3e-3`. Shrinking `h` by this factor cuts that by ~16x. A gradient that
/// is genuinely wrong does not move at all. So a mismatch is only reported if it
/// *survives* the refinement - which is stricter than picking a lucky `h`, not
/// looser: the clamp bug reported a nonzero gradient where the truth was exactly
/// zero, and no `h` makes that agree.
const NET_FD_REFINE: f32 = 4.0;

struct NetProbe {
    checked: usize,
    skipped: usize,
    clamped: usize,
    total_features: usize,
    failures: Vec<String>,
}

fn probe_network(
    label: &str,
    cfg: &KanConfig,
    weight_scale: f32,
    batch: usize,
    seed: u64,
) -> NetProbe {
    let mut net = KanNetwork::new(cfg.clone());
    let mut rng = SmallRng::seed_from_u64(seed);
    for layer in net.layers.iter_mut() {
        for w in layer.weights.iter_mut() {
            *w = rng.gen_range(-1.0..1.0) * weight_scale;
        }
        for b in layer.bias.iter_mut() {
            *b = rng.gen_range(-0.2..0.2) * weight_scale;
        }
    }

    let gr = cfg.grid_range;
    let pad = 0.1 * (gr.1 - gr.0);
    let input: Vec<f32> = (0..batch * cfg.input_dim)
        .map(|i| {
            let z = rng.gen_range(gr.0 + pad..gr.1 - pad);
            z * cfg.input_std[i % cfg.input_dim] + cfg.input_mean[i % cfg.input_dim]
        })
        .collect();
    let target: Vec<f32> = (0..batch * cfg.output_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let opts = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };
    let mut ws = net.create_workspace(batch);
    // learning_rate = 0: run the real training step purely to fill the gradient
    // buffers, leaving `net` itself untouched as the finite-difference base point.
    let mut driver = net.clone();
    driver.train_step_with_options(&input, &target, None, 0.0, &mut ws, &opts);
    let grad_w: Vec<Vec<f32>> = ws.weight_grads.clone();
    let grad_b: Vec<Vec<f32>> = ws.bias_grads.clone();

    let base_pattern = clamp_pattern(&net, &input, &mut ws);
    let mut probe = NetProbe {
        checked: 0,
        skipped: 0,
        clamped: base_pattern.iter().filter(|c| **c).count(),
        total_features: base_pattern.len(),
        failures: Vec::new(),
    };

    let compare = |kind: &str,
                   l: usize,
                   idx: usize,
                   analytic: f32,
                   perturbed: &dyn Fn(f32) -> KanNetwork,
                   orig: f32,
                   ws: &mut Workspace,
                   p: &mut NetProbe| {
        // Returns None when a probe crosses a clamp boundary, where the loss is
        // not differentiable and the difference quotient is meaningless.
        let fd_at = |h: f32, ws: &mut Workspace| -> Option<f64> {
            let plus_net = perturbed(orig + h);
            if clamp_pattern(&plus_net, &input, ws) != base_pattern {
                return None;
            }
            let plus = mse(&plus_net, &input, &target, ws);
            let minus_net = perturbed(orig - h);
            if clamp_pattern(&minus_net, &input, ws) != base_pattern {
                return None;
            }
            let minus = mse(&minus_net, &input, &target, ws);
            Some((plus - minus) / (2.0 * f64::from(h)))
        };
        let tol = |fd: f64| 3e-3 + 3e-2 * fd.abs();

        let Some(fd) = fd_at(NET_FD_STEP, ws) else {
            p.skipped += 1;
            return;
        };
        p.checked += 1;
        if (fd - f64::from(analytic)).abs() <= tol(fd) {
            return;
        }
        // Suspect. Refine and see whether the disagreement is h-dependent.
        let Some(fine) = fd_at(NET_FD_STEP / NET_FD_REFINE, ws) else {
            p.skipped += 1;
            p.checked -= 1;
            return;
        };
        let gap = (fine - f64::from(analytic)).abs();
        if gap > tol(fine) {
            p.failures.push(format!(
                "  [{label}] layer{l} {kind}[{idx}] analytic={analytic} \
                 finite_diff(h)={fd} finite_diff(h/{NET_FD_REFINE})={fine} gap={gap:e}"
            ));
        }
    };

    for l in 0..net.num_layers() {
        let n = net.layers[l].weights.len();
        for widx in (0..n).step_by((n / 25).max(1)) {
            let orig = net.layers[l].weights[widx];
            let base = net.clone();
            let make = |v: f32| {
                let mut p = base.clone();
                p.layers[l].weights[widx] = v;
                p
            };
            compare(
                "w",
                l,
                widx,
                grad_w[l][widx],
                &make,
                orig,
                &mut ws,
                &mut probe,
            );
        }
        let nb = net.layers[l].bias.len();
        for (bidx, &analytic_b) in grad_b[l].iter().enumerate().take(nb.min(6)) {
            let orig = net.layers[l].bias[bidx];
            let base = net.clone();
            let make = |v: f32| {
                let mut p = base.clone();
                p.layers[l].bias[bidx] = v;
                p
            };
            compare("b", l, bidx, analytic_b, &make, orig, &mut ws, &mut probe);
        }
    }

    println!(
        "[{label}] checked={} skipped={} clamped={}/{}",
        probe.checked, probe.skipped, probe.clamped, probe.total_features
    );
    probe
}

#[allow(clippy::too_many_arguments)]
fn network_config(
    in_dim: usize,
    hidden: Vec<usize>,
    out_dim: usize,
    order: usize,
    grid_size: usize,
    gr: (f32, f32),
    mean: f32,
    std: f32,
) -> KanConfig {
    KanConfig {
        input_dim: in_dim,
        output_dim: out_dim,
        hidden_dims: hidden,
        spline_order: order,
        grid_size,
        grid_range: gr,
        input_mean: vec![mean; in_dim],
        input_std: vec![std; in_dim],
        init_seed: Some(9),
        ..Default::default()
    }
}

#[test]
fn deep_network_gradients_across_order_and_range() {
    // Four layers: 4 -> 6 -> 6 -> 5 -> 3. Every layer's weight gradient is built
    // from the next layer's `grad_input`, so an error there compounds with depth -
    // the fixtures top out at two layers.
    let mut failures = Vec::new();
    let mut checked = 0;
    for order in ORDERS {
        for gr in RANGES {
            let cfg = network_config(4, vec![6, 6, 5], 3, order, 5, gr, 0.0, 1.0);
            let probe = probe_network(
                &format!("deep order={order} range={gr:?}"),
                &cfg,
                0.15,
                4,
                100 + order as u64,
            );
            checked += probe.checked;
            failures.extend(probe.failures);
        }
    }
    assert!(checked > 2000, "not enough stable probes: {checked}");
    assert!(
        failures.is_empty(),
        "{} deep-network gradient mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn deep_network_gradients_with_nontrivial_normalization() {
    let mut failures = Vec::new();
    for order in ORDERS {
        for (mean, std) in [(0.0f32, 0.3f32), (2.0, 5.0), (-1.0, 0.05)] {
            let cfg = network_config(4, vec![6, 5], 3, order, 5, (-1.0, 1.0), mean, std);
            let probe = probe_network(
                &format!("norm order={order} mean={mean} std={std}"),
                &cfg,
                0.2,
                4,
                200 + order as u64,
            );
            failures.extend(probe.failures);
        }
    }
    assert!(
        failures.is_empty(),
        "{} normalized-network gradient mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn deep_network_gradients_when_hidden_activations_saturate() {
    // Weights scaled up until the hidden activations leave the grid range, which is
    // where the clamp bug lived: layer 0's weight gradients are built from the
    // hidden layer's `grad_input`.
    let mut failures = Vec::new();
    let mut saturated_any = false;
    for order in ORDERS {
        let cfg = network_config(4, vec![8, 8, 6], 3, order, 5, (-1.0, 1.0), 0.0, 1.0);
        let probe = probe_network(
            &format!("saturating order={order}"),
            &cfg,
            2.5,
            4,
            300 + order as u64,
        );
        assert!(
            probe.clamped > 0,
            "fixture for order={order} does not saturate, so this test proves nothing"
        );
        saturated_any = true;
        failures.extend(probe.failures);
    }
    assert!(saturated_any);
    assert!(
        failures.is_empty(),
        "{} saturating-network gradient mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn network_gradients_at_grid_size_extremes() {
    let mut failures = Vec::new();
    for grid_size in [1usize, 2, 64] {
        for order in [2usize, 3, 7] {
            let cfg = network_config(3, vec![5], 2, order, grid_size, (-1.0, 1.0), 0.0, 1.0);
            let probe = probe_network(
                &format!("grid={grid_size} order={order}"),
                &cfg,
                0.2,
                4,
                400 + grid_size as u64 * 10 + order as u64,
            );
            failures.extend(probe.failures);
        }
    }
    assert!(
        failures.is_empty(),
        "{} gradient mismatches at grid-size extremes:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
