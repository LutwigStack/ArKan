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
//! | axis                | fixtures        | here                             |
//! |---------------------|-----------------|----------------------------------|
//! | `spline_order`      | 3, 4            | 1..=7                            |
//! | `grid_size`         | 5               | 1, 2, 5, 8, 16                   |
//! | `grid_range`        | (-1, 1)         | + (0,1), (-5,5), (0.5,2.5), narrow |
//! | `input_mean`/`_std` | 0 / 1           | + 0.05, 0.3, 5.0 and offsets     |
//! | depth               | 1-2 layers      | 1 and 4 layers                   |
//! | saturation          | never clamped   | fully and partially clamped      |
//!
//! # Finite-difference hygiene
//!
//! Three things make a central difference lie, and all three are handled explicitly
//! rather than absorbed into a loose tolerance:
//!
//! 1. **Straddling a kink.** At a clamp boundary the loss is only piecewise
//!    differentiable; at a knot it is only `C^(order-1)`, and at order 1 the slope
//!    itself jumps. Every probe here is rejected if it changes any span index or
//!    clamp flag, at layer and at network level.
//! 2. **Step size in the wrong space.** The gradient lives in raw-`x` space but the
//!    spline segments live in `z = (x - mean) / std` space. A step that is safely
//!    sub-segment at `std = 1` straddles several segments at `std = 0.05`, so every
//!    `grad_input` step is scaled by `std`.
//!
//! 3. **Truncation traded for breadth.** Tolerances are `abs + rel * |finite_diff|`,
//!    and the first version of this file used `3e-3 + 3e-2 * |fd|`. That cannot see
//!    any multiplicative gradient error below ~3%, so "zero mismatches anywhere"
//!    was a claim about a 3-5% band, not about correctness. Every difference
//!    quotient here is now Richardson-extrapolated - `D(h) = f' + c h^2 + O(h^4)`,
//!    so `(4 D(h/2) - D(h)) / 3` cancels the `h^2` term - and probes that cross a
//!    knot or a clamp boundary, where that expansion does not hold, are skipped
//!    rather than absorbed. See [`LAYER_ABS`] and [`NET_ABS`] for the resulting
//!    detection floor and how it was measured; the two `*_tolerance_can_see_*`
//!    tests assert it instead of asserting nothing.
//!
//! The remaining floors are `f32` round-off in the forward pass divided by the step
//! size, not truncation. Where a configuration genuinely cannot be tightened - a
//! narrow grid combined with a large `input_std`, where `z` has no `f32` resolution
//! left - it is named at the point of exclusion rather than papered over.

use arkan::config::MAX_SPLINE_ORDER;
use arkan::spline::{compute_knots, find_span};
use arkan::{KanConfig, KanLayer, KanNetwork, TrainOptions, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Every order `KanConfig::validate` accepts. Order 1 is a supported configuration
/// with a C^0 basis: the layer output's slope jumps at every knot, so a central
/// difference that straddles one converges to the average of two one-sided slopes.
/// [`check_layer`] and [`probe_network`] skip exactly those probes; nothing else
/// about order 1 needs special treatment.
const ORDERS: std::ops::RangeInclusive<usize> = 1..=MAX_SPLINE_ORDER;
const GRID_SIZES: [usize; 5] = [1, 2, 5, 8, 16];

/// The fifth entry is a *narrow* grid: `h` is 2e-7 at `grid_size = 5`, below the
/// absolute `EPSILON` guard that used to collapse the basis to zero. The basis
/// tests in `tests/spline_properties.rs` cover those grids; nothing covered them
/// through a real gradient, so reinstating the guard in the derivative path was
/// invisible here.
const RANGES: [(f32, f32); 5] = [
    (-1.0, 1.0),
    (0.0, 1.0),
    (-5.0, 5.0),
    (0.5, 2.5),
    (0.0, 1e-6),
];

/// Ranges usable with an extreme `input_std`, i.e. everything except the narrow
/// grid.
///
/// **Recorded floor.** A narrow grid and a large `std` are incompatible in `f32`,
/// not merely inconvenient: landing `z = (x - mean) / std` inside a 1e-6-wide grid
/// at `std = 1e3, mean = 3` needs `x ~ 3.000001`, where the ULP is 2.4e-7. The
/// subtraction keeps ~2 significant digits of `z`, and a finite-difference step
/// small enough to stay inside the grid is a few ULP of `x`. No tolerance makes that
/// probe meaningful, so the combination is excluded rather than tolerated.
fn ordinary_ranges() -> impl Iterator<Item = (f32, f32)> {
    RANGES.into_iter().take(4)
}

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

/// `sum |g_out * y|` at the base point: the magnitude the `f32` roundings inside the
/// loss are proportional to. `|L|` itself can be far smaller because those terms
/// cancel, while the noise does not shrink with it.
fn loss_scale(layer: &KanLayer, inputs: &[f32], g_out: &[f32], ws: &mut Workspace) -> f64 {
    let batch = inputs.len() / layer.in_dim;
    let mut out = vec![0.0f32; batch * layer.out_dim];
    layer.forward_batch(inputs, &mut out, ws);
    out.iter()
        .zip(g_out)
        .map(|(y, g)| (f64::from(*y) * f64::from(*g)).abs())
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

/// Plain central difference of `f` at `x`.
fn central(mut f: impl FnMut(f32) -> f64, x: f32, h: f32) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * f64::from(h))
}

/// Richardson-extrapolated central difference of `f` at `x`.
///
/// A central difference satisfies `D(h) = f'(x) + c h^2 + O(h^4)` wherever `f` is
/// smooth enough, so `(4 D(h/2) - D(h)) / 3` cancels the `h^2` term. Costs four
/// evaluations instead of two.
///
/// It is only worth anything where the expansion holds, so every caller refuses to
/// probe across a kink: the input probes skip any probe that changes a span or clamps,
/// the network probes skip any probe that changes the span/clamp pattern anywhere.
/// The weight probes do not use it - the layer output is exactly *linear* in a
/// coefficient, so a plain central difference is already exact there and
/// extrapolating would only amplify round-off.
fn richardson(mut f: impl FnMut(f32) -> f64, x: f32, h: f32) -> f64 {
    (4.0 * central(&mut f, x, h * 0.5) - central(&mut f, x, h)) / 3.0
}

/// Finite-difference step for `grad_input`, as a fraction of the *knot spacing*.
///
/// The step used to be 1e-3 of the whole grid *range*, which at `grid_size = 1` is
/// 500x smaller than this, and that is what forced the old 3% tolerance. Inside one
/// knot interval the loss is a degree-`order` polynomial in `z`, so with Richardson
/// there is almost no truncation error to trade against, and the binding error is
/// `f32` round-off, `eps * |L| / (h * |dL/dz|)` - which *shrinks* as the step grows.
/// Measured on the narrow grid `(0, 1e-6)`, where this matters most: the old step left
/// the difference quotient disagreeing with the analytic gradient by 0.4-1.8%, which
/// was the entire sensitivity of the old check; this one brings the same probes to
/// 1e-5..4e-3, and the well-conditioned ranges to 2e-5.
///
/// The ceiling on the step is the knot interval itself, which is why it is a fraction
/// of the spacing and not of the range. At 5% the "probe crosses a knot" skip costs a
/// few percent of the probes; every sweep prints how many it checked.
const Z_STEP_FRACTION: f32 = 0.05;

/// Absolute floor of the layer-level tolerance: what a gradient of ~0 may disagree by.
///
/// `f32` round-off, not truncation - the loss is accumulated in `f64` but from `f32`
/// outputs, so it carries ~1e-7 of relative noise and the difference quotient divides
/// that by the step. Configurations whose noise is larger than this get a floor
/// computed for them ([`fd_roundoff_floor`], [`step_quantization_floor`]) rather than
/// widening this constant for everybody.
const LAYER_ABS: f64 = 1e-3;

/// Relative term of the layer-level tolerance: the multiplicative gradient error this
/// can resolve wherever the gradient is large enough for this term to dominate.
///
/// It cannot go far below 1e-4. The analytic `grad_input` is an `f32` sum over `(j, k)`
/// of `g_out * w * B'(z)` whose terms are of size `order / h` and whose channels sum to
/// zero exactly, so on a fine or narrow grid the result is a heavily cancelling sum:
/// measured 9e-6 relative at `grid_size = 16` on `(0, 1e-6)`, where the gradient is
/// 2.8e7 and the terms are ~1e9. Nothing the difference quotient does can fix the
/// analytic side.
///
/// Measured worst `gap / tolerance` over the four layer sweeps: 0.373 (order 1,
/// `(0, 1)`, `mean = -0.7`, `std = 0.05`), 0.155, 0.089, 0.001.
/// `layer_tolerance_can_see_a_tenth_of_a_percent_gradient_error` measures what this
/// buys: a 0.1% error is flagged on 361 of 2300 probes and a 0.01% error on 35, against
/// nothing below 3% for the `3e-3 + 3e-2 * |fd|` it replaces.
const LAYER_REL: f64 = 1e-4;

fn layer_tol(fd: f64) -> f64 {
    LAYER_ABS + LAYER_REL * fd.abs()
}

/// Round-off floor of the difference *quotient* itself: `eps * scale / h`.
///
/// The loss is `O(1)` whatever the grid is, so a small step makes the signal
/// `L(x+h) - L(x-h)` small next to the terms it is a difference of, and `f32`
/// cancellation surfaces divided by `h`. The factor 5 covers Richardson - which
/// weights the noisier half-step by 8/3 - and the handful of roundings per term.
///
/// **This is the term that binds on the narrow grids, and the only one that does.**
/// Verified by zeroing it: `layer_gradients_across_order_grid_and_range` and the
/// sensitivity test both fail, and nothing else does. `(0, 1e-6)` at `grid_size = 8`
/// gives a step of 3e-9 in `x` and a floor of ~2e2, against a measured gap of 48 on a
/// gradient of 1.1e4; the same formula on `(-1, 1)` at `grid_size = 5` gives 2e-4. It
/// is also why [`Z_STEP_FRACTION`] is as large as the knot interval allows - halving
/// the step doubles this.
fn fd_roundoff_floor(scale: f64, h: f32) -> f64 {
    5.0 * F32_EPS * scale / f64::from(h)
}

/// Floor from the step not being the step: `|fd| * eps * max(|x|, |mean|) / h`.
///
/// `x + h` rounds, and `z = (x - mean) / std` rounds again, so the *effective* step
/// in `z` differs from `h / std` by about `eps * max(|x|, |mean|) / std`. Dividing by
/// `h / std` makes that a relative error on the difference quotient.
///
/// It is negligible at `mean = 0` and it is what limits the normalized sweep: at
/// `mean = -0.7, std = 0.05` the subtraction `x - mean` cancels to ~0.026, keeping four
/// digits, and a 5e-4 step in `x` carries 1.7e-4 of relative error. Measured: that
/// configuration's worst gap is 1.1e-4 relative (2.5e-2 on a gradient of 227), against
/// 2e-5 for the same order and range at `mean = 0`. Without this term the sweep fails
/// at `gap / tolerance = 1.04`; with it, [`LAYER_REL`] stays at 1e-4 for the probes
/// that deserve it instead of being widened 3x for all of them.
fn step_quantization_floor(fd: f64, x: f32, mean: f32, h: f32) -> f64 {
    fd.abs() * F32_EPS * f64::from(x.abs().max(mean.abs())) / f64::from(h)
}

/// `f32::EPSILON`, spelled out because every floor here is a multiple of it.
const F32_EPS: f64 = 1.2e-7;

/// Collects mismatches, and the margin the tolerance actually had.
///
/// Printing the worst `gap / tolerance` is the difference between "no mismatches"
/// and "no mismatches, with the tolerance 1.6x from firing": the first says nothing
/// about how much room a real error has to hide in.
#[derive(Default)]
struct Check {
    worst_gap: f64,
    worst_gap_at: String,
    worst_ratio: f64,
    worst_at: String,
    probes: usize,
    failures: Vec<String>,
}

impl Check {
    fn record(&mut self, gap: f64, tol: f64, at: impl Fn() -> String) {
        self.probes += 1;
        if gap > self.worst_gap {
            self.worst_gap = gap;
            self.worst_gap_at = at();
        }
        if gap / tol > self.worst_ratio {
            self.worst_ratio = gap / tol;
            self.worst_at = at();
        }
        if gap > tol {
            self.failures.push(at());
        }
    }

    /// Folds another sweep's result in, keeping the worst margin of the two.
    fn absorb(&mut self, other: Check) {
        self.probes += other.probes;
        if other.worst_gap > self.worst_gap {
            self.worst_gap = other.worst_gap;
            self.worst_gap_at = other.worst_gap_at;
        }
        if other.worst_ratio > self.worst_ratio {
            self.worst_ratio = other.worst_ratio;
            self.worst_at = other.worst_at;
        }
        self.failures.extend(other.failures);
    }

    fn assert_clean(&self, what: &str) {
        println!(
            "{what}: {} probes
  worst |fd - analytic| = {:e} at {}
               worst gap/tolerance = {:.3} at {}",
            self.probes, self.worst_gap, self.worst_gap_at, self.worst_ratio, self.worst_at
        );
        assert!(
            self.failures.is_empty(),
            "{} {what} mismatches:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

/// Checks `grad_input`, `grad_weights` and `grad_bias` for one layer against
/// Richardson-extrapolated central differences.
fn check_layer(label: &str, layer: &KanLayer, inputs: &[f32], g_out: &[f32], check: &mut Check) {
    check_layer_scaled(label, layer, inputs, g_out, 1.0, check);
}

/// [`check_layer`], with the analytic gradient multiplied by `inject`.
///
/// `inject != 1.0` is a deliberately wrong gradient, used to measure what this check
/// can actually see. That number is the honest content of a negative result, and it
/// cannot be obtained by reading the tolerance off the source.
fn check_layer_scaled(
    label: &str,
    layer: &KanLayer,
    inputs: &[f32],
    g_out: &[f32],
    inject: f32,
    check: &mut Check,
) {
    let batch = inputs.len() / layer.in_dim;
    let ana = analytic_grads(layer, inputs, g_out);
    let mut ws = Workspace::default();
    let width = layer.grid_range.1 - layer.grid_range.0;
    let knots = compute_knots(layer.grid_size, layer.order, layer.grid_range);
    let scale = loss_scale(layer, inputs, g_out, &mut ws);

    // --- grad_input ---
    let mut probe = inputs.to_vec();
    for idx in 0..inputs.len() {
        let i = idx % layer.in_dim;
        // Step measured in z-space and converted back to raw-x space, so it is the
        // same step in `z` whatever `std` is. See [`Z_STEP_FRACTION`].
        let h = Z_STEP_FRACTION * (width / layer.grid_size as f32) * layer.std[i];
        let x = inputs[idx];
        // Refuse to probe across a kink. Two kinds exist in `z`:
        //  - a knot, where the slope jumps at order 1 and a higher derivative jumps
        //    above it, so the `h^2` expansion Richardson relies on does not hold;
        //  - the clamp, beyond which the forward pass is flat.
        // Either way the difference quotient is not an estimate of anything, so the
        // probe is skipped rather than absorbed into a wider tolerance. `probes` is
        // asserted per sweep so this cannot quietly skip everything.
        let z_at = |v: f32| (v - layer.mean[i]) / layer.std[i];
        let kinked = |v: f32| {
            let z = z_at(v);
            z <= layer.grid_range.0 || z >= layer.grid_range.1
        };
        let span_at = |v: f32| {
            let z = z_at(v).clamp(layer.grid_range.0, layer.grid_range.1);
            find_span(z, &knots, layer.order, layer.grid_size)
        };
        if kinked(x - h) || kinked(x + h) || span_at(x - h) != span_at(x + h) {
            continue;
        }
        let fd = richardson(
            |v| {
                probe[idx] = v;
                let loss = directional_loss(layer, &probe, g_out, &mut ws);
                probe[idx] = x;
                loss
            },
            x,
            h,
        );
        let analytic = f64::from(ana.input[idx] * inject);
        let gap = (fd - analytic).abs();
        // Tolerance = the four mechanisms that limit this probe, each named and
        // measured above. No term is a fudge factor and none of them is free to grow.
        let tol = layer_tol(fd)
            + fd_roundoff_floor(scale, h)
            + step_quantization_floor(fd, x, layer.mean[i], h);
        check.record(gap, tol, || {
            format!("  [{label}] grad_input[{idx}] x={x:e} analytic={analytic} finite_diff={fd} gap={gap:e}")
        });
    }

    // --- grad_weights, sampled across the whole coefficient block ---
    // The output is exactly linear in a coefficient, so the central difference has no
    // truncation error at all and a *large* step is strictly better: it pushes the
    // f32 round-off floor down. 0.25 rather than the 1e-2 used before.
    let hw = 0.25f32;
    let mut probe_layer = layer.clone();
    let n = layer.weights.len();
    for widx in (0..n).step_by((n / 37).max(1)) {
        let orig = layer.weights[widx];
        let fd = central(
            |v| {
                probe_layer.weights[widx] = v;
                let loss = directional_loss(&probe_layer, inputs, g_out, &mut ws);
                probe_layer.weights[widx] = orig;
                loss
            },
            orig,
            hw,
        );
        let analytic = f64::from(ana.weights[widx] * inject);
        let gap = (fd - analytic).abs();
        // No `z_floor`: `grad_weights` is `sum_b g_out * B`, with no `1/h` factor and
        // nothing to cancel. No Richardson either, so `fd_roundoff_floor` without its
        // factor for it - and `hw = 0.25` makes it ~1e-6 anyway.
        check.record(
            gap,
            layer_tol(fd) + fd_roundoff_floor(scale, hw) / 5.0,
            || {
                format!(
                "  [{label}] grad_weights[{widx}] analytic={analytic} finite_diff={fd} gap={gap:e}"
            )
            },
        );
    }

    // --- grad_bias: exactly sum_b g_out[b, j], no finite difference needed ---
    for j in 0..layer.out_dim {
        let want: f64 = (0..batch)
            .map(|b| f64::from(g_out[b * layer.out_dim + j]))
            .sum();
        let analytic = f64::from(ana.bias[j] * inject);
        let gap = (want - analytic).abs();
        check.record(gap, 1e-4, || {
            format!("  [{label}] grad_bias[{j}] analytic={analytic} expected={want} gap={gap:e}")
        });
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
    let mut check = Check::default();
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
                    &mut check,
                );
                cases += 1;
            }
        }
    }

    println!("layer gradient sweep: {cases} configurations");
    check.assert_clean("layer gradient");
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
    let mut check = Check::default();

    // `ordinary_ranges()`, not `RANGES`: see its doc comment for why the narrow grid
    // and `std = 1e3` cannot both be honoured in f32.
    for order in ORDERS {
        for gr in ordinary_ranges() {
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
                    &mut check,
                );
            }
        }
    }

    check.assert_clean("normalized layer gradient");
}

#[test]
fn saturated_inputs_have_exactly_zero_input_gradient() {
    // Generalizes `clamp_gradient_parity` off (-1, 1) and off order 3. For a
    // saturated feature `dz/dx` is exactly 0, so the forward pass is exactly flat
    // in that coordinate and the analytic gradient must be exactly 0 - not small,
    // zero. `grad_weights` is *not* zero there and is checked against the
    // finite difference as usual.
    let (in_dim, out_dim, batch) = (6, 3, 3);
    let mut check = Check::default();

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
                    check.failures.push(format!(
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
                &mut check,
            );
        }
    }

    check.assert_clean("saturated-gradient");
}

#[test]
fn partially_saturated_inputs_keep_the_live_features_correct() {
    // The dangerous case is a mix: getting the clamp right by zeroing everything
    // would pass the fully saturated test and fail this one.
    let (in_dim, out_dim, batch) = (6, 3, 3);
    let mut check = Check::default();

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
                    check.failures.push(format!(
                        "  [order={order} range={gr:?}] saturated grad_input[{idx}] = {g}"
                    ));
                }
            }

            check_layer(
                &format!("mixed order={order} range={gr:?}"),
                &layer,
                &inputs,
                &g_out,
                &mut check,
            );
        }
    }

    check.assert_clean("mixed-saturation");
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

/// The stored span index and clamp flag for every `(layer, sample, feature)` triple.
///
/// The loss is only piecewise smooth in the weights, and there are two kinds of
/// break: a clamp boundary, where it is not differentiable at all, and a knot, where
/// the `order`-th derivative jumps - enough to invalidate the `h^2` expansion
/// Richardson extrapolation rests on, and at order 1 enough to invalidate the
/// difference quotient itself. Both show up as a change in this vector, so comparing
/// the whole thing costs nothing over comparing the clamp flags alone and rejects
/// both. It used to compare only the flags.
fn span_pattern(net: &KanNetwork, input: &[f32], ws: &mut Workspace) -> Vec<u32> {
    let batch = input.len() / net.config.input_dim;
    let mut pred = vec![0.0f32; batch * net.config.output_dim];
    net.forward_batch_training(input, &mut pred, ws);
    let mut pattern = Vec::new();
    for (l, layer) in net.layers.iter().enumerate() {
        pattern.extend_from_slice(&ws.layers_grid_indices[l][..batch * layer.in_dim]);
    }
    pattern
}

/// Central-difference step for network weights.
///
/// The loss is not polynomial in a weight two layers down, so this cannot be pushed
/// up the way the layer-level step was: truncation and the span-pattern skip rate
/// both grow with it. 3e-3 was measured to skip 3x more probes than 1e-3 while
/// improving nothing, since with Richardson the round-off floor - `eps * |L| / h`,
/// ~6e-4 here - is what binds.
const NET_FD_STEP: f32 = 4e-3;

/// Absolute floor of the network-level tolerance.
///
/// Deliberately small, because the real absolute floor is *measured per probe* rather
/// than assumed - see [`net_tol`]. A single constant cannot serve both halves of this
/// file: the well-conditioned sweeps have gradients of ~3e-4 with a difference quotient
/// good to 1.5e-5, while the saturating sweep reaches gradients of 292 with a
/// difference quotient good to 1.6e-2. One number covering the second blinds the
/// first: at the 1.5e-3 this started as, a 1% error was flagged on 13 of 893
/// well-conditioned probes, where the per-probe floor flags 242.
const NET_ABS: f64 = 2e-5;

/// Relative term of the network-level tolerance. Measured worst `gap / tolerance` over
/// the three network sweeps: 0.584, 0.247, 0.198.
///
/// It cannot go as low as [`LAYER_REL`]: a deep saturating network's loss has genuine
/// `O(h^2)` curvature that Richardson only reduces, and `train_step`'s gradient is an
/// `f32` accumulation through four layers with nothing to subtract off. 3e-3 rather
/// than the 3e-2 it replaces.
const NET_REL: f64 = 3e-3;

/// Relative term against the *layer's* gradient scale, not the probe's own value.
///
/// **The floor that only an f64 backward pass could lower.** `train_step`'s gradient
/// is an `f32` accumulation through the whole depth, and a saturating network
/// amplifies its own round-off exactly the way it amplifies input noise - the same
/// effect `forward_single_matches_forward_batch_to_within_the_networks_own_noise`
/// measures in `tests/network_metamorphic.rs`, where one ULP of input moves the output
/// by 0.048. Each layer's `grad_input` multiplies by `~in_dim * |w| * order / h`
/// (~150 for the saturating fixture) while the gradient itself stays `O(1)`, so a
/// coefficient whose gradient is far below its layer's scale is the tail of a heavily
/// cancelling sum.
///
/// Measured on the saturating sweep: layer 0 has `max |grad_w| = 36`, and the five
/// probes whose analytic value is ~1e-4..3e-3 disagree with a stable difference
/// quotient by 1e-4..4e-4 - i.e. ~1e-5 of the layer scale, independent of the probe's
/// own value, which is what a round-off explanation predicts and a systematic error
/// does not. 3e-5 leaves 2.7x margin. On the well-conditioned sweeps the layer scale
/// is ~0.05 and this term is ~1e-6, so it costs nothing there.
const NET_ANA_REL: f64 = 1e-4;

/// Safety factor on the measured spread.
///
/// The spread is a *lower* bound on the remaining error while the difference quotient
/// is still outside its asymptotic regime, which is the usual reason adaptive
/// Richardson schemes carry one. Measured largest `gap / spread` over every network
/// sweep is 4.7, so 5 is at the observed edge and [`NET_ANA_REL`] carries the rest.
const SPREAD_SAFETY: f64 = 5.0;

/// Tolerance for one network probe: the fixed part, plus three times the difference
/// quotient's own measured uncertainty, plus the analytic side's own round-off.
///
/// `spread` is `|R(h) - R(h/2)|`, the disagreement between two Richardson estimates
/// built from central differences at `h`, `h/2` and `h/4`. It is the standard
/// self-estimate of a finite difference, and here it does the job a hand-picked
/// absolute floor cannot: the loss's round-off is `eps * (sum of |terms|)`, and a
/// 4-layer net with `weight_scale = 2.5` sums ~20 in magnitude to produce a
/// prediction of ~2, so the noise is ~10x what `eps * |L|` predicts and varies by an
/// order of magnitude between configurations. Measured: spread 5e-7..3e-5 on the deep
/// and normalized sweeps, up to 7e-3 on the saturating one. Both then get a tolerance
/// matched to what their own arithmetic supports.
fn net_tol(fd: f64, spread: f64, layer_scale: f64) -> f64 {
    NET_ABS + NET_REL * fd.abs() + SPREAD_SAFETY * spread + NET_ANA_REL * layer_scale
}

struct NetProbe {
    checked: usize,
    skipped: usize,
    clamped: usize,
    total_features: usize,
    check: Check,
}

fn probe_network(
    label: &str,
    cfg: &KanConfig,
    weight_scale: f32,
    batch: usize,
    seed: u64,
) -> NetProbe {
    probe_network_scaled(label, cfg, weight_scale, batch, seed, 1.0)
}

/// [`probe_network`], with every analytic gradient multiplied by `inject`. See
/// [`check_layer_scaled`].
fn probe_network_scaled(
    label: &str,
    cfg: &KanConfig,
    weight_scale: f32,
    batch: usize,
    seed: u64,
    inject: f32,
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

    let base_pattern = span_pattern(&net, &input, &mut ws);
    let mut probe = NetProbe {
        checked: 0,
        skipped: 0,
        clamped: base_pattern
            .iter()
            .filter(|s| *s & arkan::spline::SPAN_CLAMPED_FLAG != 0)
            .count(),
        total_features: base_pattern.len(),
        check: Check::default(),
    };

    let compare = |kind: &str,
                   l: usize,
                   idx: usize,
                   analytic: f32,
                   layer_scale: f64,
                   perturbed: &dyn Fn(f32) -> KanNetwork,
                   orig: f32,
                   ws: &mut Workspace,
                   p: &mut NetProbe| {
        // Returns None when a probe crosses a knot or a clamp boundary, where the
        // loss is not smooth enough for the difference quotient to estimate anything.
        let fd_at = |h: f32, ws: &mut Workspace| -> Option<f64> {
            let plus_net = perturbed(orig + h);
            if span_pattern(&plus_net, &input, ws) != base_pattern {
                return None;
            }
            let plus = mse(&plus_net, &input, &target, ws);
            let minus_net = perturbed(orig - h);
            if span_pattern(&minus_net, &input, ws) != base_pattern {
                return None;
            }
            let minus = mse(&minus_net, &input, &target, ws);
            Some((plus - minus) / (2.0 * f64::from(h)))
        };

        // Three steps, always - not "on suspicion". The h/4 refinement this replaces
        // only ever ran on probes that had already failed, so it made reported
        // mismatches trustworthy and did nothing at all for sensitivity: a gradient
        // wrong by 1% still agreed with the coarse difference inside a 3% tolerance
        // and was never refined. Here h, h/2 and h/4 give two Richardson estimates,
        // the finer one is the answer, and their disagreement is the error bar.
        let (Some(d1), Some(d2), Some(d4)) = (
            fd_at(NET_FD_STEP, ws),
            fd_at(NET_FD_STEP * 0.5, ws),
            fd_at(NET_FD_STEP * 0.25, ws),
        ) else {
            p.skipped += 1;
            return;
        };
        let coarse_richardson = (4.0 * d2 - d1) / 3.0;
        let fd = (4.0 * d4 - d2) / 3.0;
        let spread = (fd - coarse_richardson).abs();
        p.checked += 1;
        let analytic = f64::from(analytic * inject);
        let gap = (fd - analytic).abs();
        p.check.record(gap, net_tol(fd, spread, layer_scale), || {
            format!(
                "  [{label}] layer{l} {kind}[{idx}] analytic={analytic} \
                 richardson(h)={coarse_richardson} richardson(h/2)={fd} spread={spread:e} \
                 gap={gap:e}"
            )
        });
    };

    for l in 0..net.num_layers() {
        // The scale the layer's own f32 round-off is proportional to; see
        // [`NET_ANA_REL`].
        let w_scale = f64::from(grad_w[l].iter().fold(0.0f32, |m, g| m.max(g.abs())));
        let b_scale = f64::from(grad_b[l].iter().fold(0.0f32, |m, g| m.max(g.abs())));
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
                w_scale,
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
            compare(
                "b", l, bidx, analytic_b, b_scale, &make, orig, &mut ws, &mut probe,
            );
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
    let mut check = Check::default();
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
            check.absorb(probe.check);
        }
    }
    // Was `> 2000` when a probe was rejected only for crossing a clamp boundary.
    // Rejecting knot crossings as well costs probes, and order 1 and the narrow grid
    // add configurations where nearly everything is rejected; what matters is that
    // the sweep as a whole still lands thousands of usable probes.
    assert!(checked > 2000, "not enough stable probes: {checked}");
    check.assert_clean("deep-network gradient");
}

#[test]
fn deep_network_gradients_with_nontrivial_normalization() {
    let mut check = Check::default();
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
            check.absorb(probe.check);
        }
    }
    check.assert_clean("normalized-network gradient");
}

#[test]
fn deep_network_gradients_when_hidden_activations_saturate() {
    // Weights scaled up until the hidden activations leave the grid range, which is
    // where the clamp bug lived: layer 0's weight gradients are built from the
    // hidden layer's `grad_input`.
    let mut check = Check::default();
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
        // A saturating net is where the span pattern is most fragile, so pin that
        // rejecting unstable probes has not emptied the test out.
        assert!(
            probe.checked > 20,
            "order={order} left only {} usable probes",
            probe.checked
        );
        saturated_any = true;
        check.absorb(probe.check);
    }
    assert!(saturated_any);
    check.assert_clean("saturating-network gradient");
}

#[test]
fn network_gradients_at_grid_size_extremes() {
    let mut check = Check::default();
    for grid_size in [1usize, 2, 64] {
        for order in [1usize, 2, 3, 7] {
            let cfg = network_config(3, vec![5], 2, order, grid_size, (-1.0, 1.0), 0.0, 1.0);
            let probe = probe_network(
                &format!("grid={grid_size} order={order}"),
                &cfg,
                0.2,
                4,
                400 + grid_size as u64 * 10 + order as u64,
            );
            check.absorb(probe.check);
        }
    }
    check.assert_clean("grid-size-extreme gradient");
}

// ===========================================================================
// What the tolerances can actually see
// ===========================================================================

/// The layer sweeps' detection floor, asserted rather than left to the reader's
/// imagination.
///
/// A negative result ("no mismatches anywhere") only means something if the check
/// could have failed. This scales every analytic gradient by `1 + d` and requires the
/// same sweep to notice. Measured flag counts on these fixtures, out of 2300 probes:
///
/// | `d`  | flagged |
/// |------|---------|
/// | 0    | 0       |
/// | 1e-4 | 35      |
/// | 3e-4 | 116     |
/// | 1e-3 | 361     |
/// | 1e-2 | 1391    |
///
/// So the floor is ~1e-4 for the most favourable probes and ~1e-3 broadly. The
/// `3e-3 + 3e-2 * |fd|` tolerance this replaces could not see 3%.
#[test]
fn layer_tolerance_can_see_a_tenth_of_a_percent_gradient_error() {
    let mut clean = Check::default();
    let mut dirty = Check::default();
    for order in ORDERS {
        for gr in RANGES {
            let cfg = layer_config(5, 3, order, 5, gr, 0.0, 1.0);
            let layer = dense_layer(&cfg, 5, 3, 31 * order as u64 + 5);
            let pad = 0.08 * (gr.1 - gr.0);
            let inputs = seeded(4 * 5, gr.0 + pad, gr.1 - pad, 1000 + order as u64);
            let g_out = seeded(4 * 3, -1.0, 1.0, 2005);
            let label = format!("calibration order={order} range={gr:?}");
            check_layer(&label, &layer, &inputs, &g_out, &mut clean);
            check_layer_scaled(&label, &layer, &inputs, &g_out, 1.001, &mut dirty);
        }
    }
    println!(
        "layer sensitivity: {} probes, {} flagged at +0.1%, {} flagged unperturbed",
        dirty.probes,
        dirty.failures.len(),
        clean.failures.len()
    );
    clean.assert_clean("calibration baseline");
    assert!(
        dirty.failures.len() > 150,
        "a 0.1% gradient error was flagged on only {} of {} probes - the layer          tolerance has gone blind",
        dirty.failures.len(),
        dirty.probes
    );
}

/// The network sweeps' detection floor. Same idea as
/// [`layer_tolerance_can_see_a_tenth_of_a_percent_gradient_error`], two orders of
/// magnitude coarser, and that difference is the honest content of the network-level
/// negative results.
///
/// Measured flag counts on the deep fixture, out of 893 probes: 0 at `d = 0`, 0 at
/// 1e-3, 0 at 3e-3, 242 at 1e-2, 361 at 3e-2. The floor is between 0.3% and 1%, set by
/// `f32`
/// round-off in a 4-layer backward pass - see [`NET_ANA_REL`] - not by anything the
/// difference quotient could improve.
#[test]
fn network_tolerance_can_see_a_one_percent_gradient_error() {
    let mut clean = Check::default();
    let mut dirty = Check::default();
    for order in ORDERS {
        let cfg = network_config(4, vec![6, 6, 5], 3, order, 5, (-1.0, 1.0), 0.0, 1.0);
        let label = format!("calibration order={order}");
        clean.absorb(probe_network(&label, &cfg, 0.15, 4, 100 + order as u64).check);
        dirty.absorb(probe_network_scaled(&label, &cfg, 0.15, 4, 100 + order as u64, 1.01).check);
    }
    println!(
        "network sensitivity: {} probes, {} flagged at +1%, {} flagged unperturbed",
        dirty.probes,
        dirty.failures.len(),
        clean.failures.len()
    );
    clean.assert_clean("calibration baseline");
    assert!(
        dirty.failures.len() > 100,
        "a 1% gradient error was flagged on only {} of {} probes - the network          tolerance has gone blind",
        dirty.failures.len(),
        dirty.probes
    );
}
