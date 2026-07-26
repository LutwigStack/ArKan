//! Oracle-free property tests for the B-spline basis, swept over the whole
//! configuration space instead of one point in it.
//!
//! # Why this file exists
//!
//! The PyTorch reference fixtures (`tests/reference_data/*.json`) all sit at the
//! *same* point: `grid_range = (-1, 1)`, `grid_size = 5`, inputs strictly inside
//! the range, `spline_order` 3 or 4. They prove "we match PyTorch there". They say
//! nothing about whether a configuration exists where we are wrong.
//!
//! Everything here needs no oracle: partition of unity, non-negativity, the
//! span-bracketing invariant, and a per-channel comparison against a naive
//! textbook Cox-de Boor recursion written independently in `f64`. All of it works
//! at every `(order, grid_size, grid_range, x)` we can construct.
//!
//! # Why per-channel, not just the sum
//!
//! `src/baked.rs` documents that its fixed-point partition-of-unity test cannot
//! catch coefficient errors, because one channel is computed as
//! `32768 - sum(others)` and absorbs any error in the rest. A sum check is a weak
//! check in general: it is invariant under moving mass between channels. Every
//! test here that could be written as a sum check is written per-channel instead.

use arkan::config::{EPSILON, MAX_GRID_SIZE, MAX_SPLINE_ORDER};
use arkan::spline::{compute_basis, compute_basis_and_deriv, compute_knots, find_span};

/// Orders under test: the whole range `KanConfig::validate` accepts, 1 to
/// [`MAX_SPLINE_ORDER`]. Order 0 is rejected by `validate` and has no derivative.
///
/// Order 1 used to be excluded here on the grounds that its basis is only C^0, so
/// a central difference across a knot converges to the average of two different
/// one-sided limits. That argument is real but it does not apply: the
/// finite-difference derivative test probes at [`interior_points`], which sit at
/// 0.09..0.91 of a knot interval and step by `h * 1e-3`, so no probe ever crosses a
/// knot. Every other property here - partition of unity, non-negativity,
/// per-channel Cox-de Boor, `find_span` bracketing, `sum(B') = 0` - holds at order 1
/// unconditionally. Excluding it left `spline_order = 1`, a supported and
/// `validate()`-accepted configuration, with no per-channel coverage at all.
const ORDERS: std::ops::RangeInclusive<usize> = 1..=MAX_SPLINE_ORDER;

/// Grid sizes spanning the supported range 1..=[`MAX_GRID_SIZE`], including both
/// endpoints and both parities around the SIMD width.
const GRID_SIZES: [usize; 12] = [1, 2, 3, 4, 5, 7, 8, 16, 17, 32, 63, MAX_GRID_SIZE];

/// Grid ranges: symmetric, asymmetric-positive, asymmetric-straddling, wide, and
/// narrow. The fixtures only ever use `(-1, 1)`.
const RANGES: [(f32, f32); 7] = [
    (-1.0, 1.0),
    (0.0, 1.0),
    (-5.0, 5.0),
    (-3.0, 3.0),
    (0.5, 2.5),
    (-0.25, 4.0),
    (-10.0, -2.0),
];

/// Every `(grid_size, grid_range)` pair under test: the full `GRID_SIZES x RANGES`
/// product, plus the four narrow grids of [`NARROW_RANGES`], which carry their own
/// `grid_size` because what makes them narrow is the knot *spacing*.
///
/// The derivative tests iterate this rather than `GRID_SIZES x RANGES`. They used
/// not to, and that is why an absolute-EPSILON guard reintroduced into
/// `compute_basis_and_deriv`'s denominators - the exact bug that was fixed in
/// `compute_basis` - went undetected: the narrow grids were reachable only by tests
/// that never call the derivative.
fn all_grids() -> impl Iterator<Item = (usize, (f32, f32))> {
    GRID_SIZES
        .into_iter()
        .flat_map(|g| RANGES.into_iter().map(move |r| (g, r)))
        .chain(NARROW_RANGES.into_iter().map(|(r, g)| (g, r)))
}

/// Naive Cox-de Boor recursion in `f64`, written straight from the definition.
///
/// Independent of the library's iterative de Boor: different algorithm, different
/// precision, indexed globally instead of relative to a span. This is the
/// "independent computation" every basis channel is checked against.
fn ref_basis(j: isize, p: usize, x: f64, knots: &[f64]) -> f64 {
    let n = knots.len() as isize;
    if p == 0 {
        if j < 0 || j + 1 >= n {
            return 0.0;
        }
        let (a, b) = (knots[j as usize], knots[j as usize + 1]);
        return f64::from(x >= a && x < b);
    }
    if j < 0 || j + p as isize + 1 >= n {
        return 0.0;
    }
    let ju = j as usize;
    let d1 = knots[ju + p] - knots[ju];
    let t1 = if d1 > 0.0 {
        (x - knots[ju]) / d1 * ref_basis(j, p - 1, x, knots)
    } else {
        0.0
    };
    let d2 = knots[ju + p + 1] - knots[ju + 1];
    let t2 = if d2 > 0.0 {
        (knots[ju + p + 1] - x) / d2 * ref_basis(j + 1, p - 1, x, knots)
    } else {
        0.0
    };
    t1 + t2
}

/// One ULP above (`up`) or below `x`. `f32::next_up`/`next_down` stabilized well
/// after this crate's MSRV of 1.73, so the bit walk is done by hand. Only ever
/// called on finite knot values.
fn ulp_step(x: f32, up: bool) -> f32 {
    if x == 0.0 {
        let tiny = f32::from_bits(1);
        return if up { tiny } else { -tiny };
    }
    let toward_infinity = up == (x > 0.0);
    let bits = x.to_bits();
    f32::from_bits(if toward_infinity { bits + 1 } else { bits - 1 })
}

/// How far a basis channel at `x` can legitimately sit outside `[0, 1]`, or away
/// from the textbook value, purely because of `find_span`'s knot snap.
///
/// `find_span` computes `(x - t_min) / h` in `f32` and floors it with a small
/// positive fudge, so when that ratio rounds up to an integer it hands back the
/// interval *starting* at the next knot even though `x` can sit up to one ULP below
/// it. `compute_basis` then evaluates that interval's polynomial pieces a hair
/// outside their interval. A channel's slope is at most `order / h`, so the
/// resulting error is at most `order * ULP(x) / h` - roughly
/// `order * 1.2e-7 * |x| / h`, i.e. it grows with how many knot widths the grid sits
/// away from zero, not with `h` alone.
///
/// Order 1 is where this became visible, because there the bound is tight: on
/// `(0.5, 2.5)` at `grid_size = 63` the measured error is 3.76e-6 and the companion
/// channel is exactly that negative, against `ULP(1.26) / h = 3.75e-6`. At order 2
/// and up the same points are an order of magnitude inside the flat tolerance, which
/// is why a flat tolerance was enough while order 1 was excluded.
fn snap_floor(x: f32, order: usize, h: f32) -> f64 {
    let mag = x.abs();
    f64::from(order as f32 * (ulp_step(mag, true) - mag) / h)
}

/// Evaluation points: every interior knot exactly, one ULP either side of it, a
/// small offset either side, both grid endpoints, and a dense uniform sweep.
///
/// The forward pass always clamps to `grid_range` before calling `find_span`, so
/// points are clamped here too - that is the domain the basis is ever asked about.
fn sample_points(gr: (f32, f32), knots: &[f32], order: usize, grid_size: usize) -> Vec<f32> {
    let mut pts = Vec::with_capacity(5 * (grid_size + 1) + 64);
    let width = gr.1 - gr.0;
    for k in knots.iter().take(order + grid_size + 1).skip(order) {
        pts.push(*k);
        pts.push(ulp_step(*k, true));
        pts.push(ulp_step(*k, false));
        pts.push(k + width * 1e-4);
        pts.push(k - width * 1e-4);
    }
    pts.push(gr.0);
    pts.push(gr.1);
    for s in 0..=61 {
        pts.push(gr.0 + width * (s as f32 / 61.0));
    }
    pts.into_iter().map(|x| x.clamp(gr.0, gr.1)).collect()
}

/// Points strictly inside a knot interval, at several fractional offsets, so a
/// central difference never straddles a knot.
fn interior_points(knots: &[f32], order: usize, grid_size: usize) -> Vec<f32> {
    let mut pts = Vec::new();
    for i in order..order + grid_size {
        let (a, b) = (knots[i], knots[i + 1]);
        for f in [0.09f32, 0.27, 0.5, 0.73, 0.91] {
            pts.push(a + (b - a) * f);
        }
    }
    pts
}

#[test]
fn partition_of_unity_holds_across_the_configuration_space() {
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    let mut cases = 0usize;

    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
                let knots = compute_knots(grid_size, order, gr);
                for x in sample_points(gr, &knots, order, grid_size) {
                    let span = find_span(x, &knots, order, grid_size);
                    let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                    compute_basis(x, span, &knots, order, &mut basis[..=order]);
                    let sum: f32 = basis[..=order].iter().sum();
                    cases += 1;
                    let err = (sum - 1.0).abs();
                    if err > worst {
                        worst = err;
                        worst_at =
                            format!("order={order} grid_size={grid_size} range={gr:?} x={x:e}");
                    }
                }
            }
        }
    }

    println!("partition of unity: {cases} points, worst |sum - 1| = {worst:e} at {worst_at}");
    assert!(
        worst < 1e-5,
        "B-spline basis does not sum to 1: worst |sum - 1| = {worst:e} at {worst_at}"
    );
}

/// Both entry points that produce basis values - `compute_basis` and the
/// `basis_out` half of `compute_basis_and_deriv` - against the independent `f64`
/// recursion.
///
/// `compute_basis_and_deriv` is included here rather than in its own test because
/// the obvious own test is worthless. It used to exist: it asserted that
/// `compute_basis_and_deriv`'s `basis_out` equals `compute_basis`'s output, with
/// `assert_eq!(worst, 0.0)`. `compute_basis_and_deriv`'s first statement *is*
/// `compute_basis(x, span, knots, order, basis_out)` and it never writes `basis_out`
/// again, so that assertion was guaranteed by the implementation's structure and
/// could not fail. Verified: the left/right swap in `compute_basis` that this test
/// catches left the delegation test reporting `worst = 0.0`. An assertion the
/// implementation guarantees is not a test - which is the same lesson as the
/// fixed-point partition-of-unity check in `src/baked.rs` that passed while order 4
/// was off by 0.208. Checked against the independent reference, the contract
/// ("backward's weight gradient comes from these values") is actually pinned.
#[test]
fn every_basis_channel_matches_an_independent_cox_de_boor() {
    let mut worst = 0.0f64;
    let mut worst_ratio = 0.0f64;
    let mut worst_at = String::new();
    let mut cases = 0usize;

    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
                let h = (gr.1 - gr.0) / grid_size as f32;
                let knots = compute_knots(grid_size, order, gr);
                let knots64: Vec<f64> = knots.iter().map(|&k| f64::from(k)).collect();
                for x in sample_points(gr, &knots, order, grid_size) {
                    // The reference uses the half-open convention `[t_i, t_{i+1})`, so
                    // it is identically zero at the right end of the domain. The
                    // library deliberately extends the last segment closed; that
                    // difference is a convention, not an error, so skip that point.
                    if x >= gr.1 {
                        continue;
                    }
                    let span = find_span(x, &knots, order, grid_size);
                    let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                    compute_basis(x, span, &knots, order, &mut basis[..=order]);
                    let mut with_deriv = [0.0f32; MAX_SPLINE_ORDER + 1];
                    let mut deriv = [0.0f32; MAX_SPLINE_ORDER + 1];
                    compute_basis_and_deriv(
                        x,
                        span,
                        &knots,
                        order,
                        &mut with_deriv[..=order],
                        &mut deriv[..=order],
                    );

                    let tol = 5e-6 + snap_floor(x, order, h);
                    for k in 0..=order {
                        let j = span - order + k;
                        let want = ref_basis(j as isize, order, f64::from(x), &knots64);
                        for (entry, got) in
                            [("compute_basis", basis[k]), ("and_deriv", with_deriv[k])]
                        {
                            let err = (want - f64::from(got)).abs();
                            cases += 1;
                            worst = worst.max(err);
                            if err / tol > worst_ratio {
                                worst_ratio = err / tol;
                                worst_at = format!(
                                    "{entry}: order={order} grid_size={grid_size} range={gr:?} \
                                     x={x:e} channel={k} (global {j}): want={want} got={got} \
                                     err={err:e} tol={tol:e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "per-channel basis: {cases} channels, worst error = {worst:e}, \
         worst error/tolerance = {worst_ratio:.3} at {worst_at}"
    );
    assert!(
        worst_ratio < 1.0,
        "a basis channel disagrees with the textbook recursion by more than the \
         f32 knot-snap floor: {worst_at}"
    );
}

#[test]
fn basis_values_are_never_negative() {
    let mut worst = 0.0f32;
    let mut worst_ratio = 0.0f64;
    let mut worst_at = String::new();

    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
                let h = (gr.1 - gr.0) / grid_size as f32;
                let knots = compute_knots(grid_size, order, gr);
                for x in sample_points(gr, &knots, order, grid_size) {
                    let span = find_span(x, &knots, order, grid_size);
                    let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                    compute_basis(x, span, &knots, order, &mut basis[..=order]);
                    // A channel may be negative by at most the knot-snap floor: the
                    // snapped interval's polynomial, evaluated one ULP outside it, is
                    // slightly negative at the end where it should reach exactly 0.
                    let allowed = 1e-6 + snap_floor(x, order, h);
                    for (k, &b) in basis[..=order].iter().enumerate() {
                        worst = worst.min(b);
                        let ratio = -f64::from(b) / allowed;
                        if ratio > worst_ratio {
                            worst_ratio = ratio;
                            worst_at = format!(
                                "order={order} grid={grid_size} range={gr:?} x={x:e} k={k}: \
                                 b={b:e} allowed={allowed:e}"
                            );
                        }
                    }
                }
            }
        }
    }

    println!(
        "most negative basis value = {worst:e}; worst negativity/allowance = \
         {worst_ratio:.3} at {worst_at}"
    );
    assert!(
        worst_ratio < 1.0,
        "basis went negative by more than the f32 knot-snap floor: {worst_at}"
    );
}

#[test]
fn find_span_brackets_x_and_stays_in_range() {
    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
                let knots = compute_knots(grid_size, order, gr);
                for x in sample_points(gr, &knots, order, grid_size) {
                    let span = find_span(x, &knots, order, grid_size);
                    assert!(
                        span >= order && span < order + grid_size,
                        "span {span} out of [{order}, {}) for order={order} grid={grid_size} \
                         range={gr:?} x={x:e}",
                        order + grid_size
                    );
                    // Bracketing, with one knot spacing of slack for the f32 error in
                    // `knots[i] = t_min + (i - order) * h`.
                    let h = (gr.1 - gr.0) / grid_size as f32;
                    assert!(
                        x >= knots[span] - h * 1e-3 && x <= knots[span + 1] + h * 1e-3,
                        "span {span} does not bracket x={x:e}: [{}, {}] for order={order} \
                         grid={grid_size} range={gr:?}",
                        knots[span],
                        knots[span + 1]
                    );
                }
            }
        }
    }
}

#[test]
fn every_derivative_channel_matches_a_central_difference() {
    // Central difference of the *independent* f64 reference basis, taken strictly
    // inside a knot interval so the probe never straddles a knot - which is also
    // what makes this valid at order 1, whose derivative jumps at every knot and
    // nowhere else. Relative tolerance is against `order / h`, the natural scale of
    // a B-spline derivative.
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    let mut cases = 0usize;

    for order in ORDERS {
        for (grid_size, gr) in all_grids() {
            let knots = compute_knots(grid_size, order, gr);
            let knots64: Vec<f64> = knots.iter().map(|&k| f64::from(k)).collect();
            let h = f64::from(gr.1 - gr.0) / grid_size as f64;
            let step = h * 1e-3;

            for x in interior_points(&knots, order, grid_size) {
                let span = find_span(x, &knots, order, grid_size);
                let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                let mut deriv = [0.0f32; MAX_SPLINE_ORDER + 1];
                compute_basis_and_deriv(
                    x,
                    span,
                    &knots,
                    order,
                    &mut basis[..=order],
                    &mut deriv[..=order],
                );

                let xf = f64::from(x);
                for (k, &got) in deriv[..=order].iter().enumerate() {
                    let j = (span - order + k) as isize;
                    let plus = ref_basis(j, order, xf + step, &knots64);
                    let minus = ref_basis(j, order, xf - step, &knots64);
                    let want = (plus - minus) / (2.0 * step);
                    // Scale-free: derivatives live on the 1/h scale.
                    let err = (want - f64::from(got)).abs() * h / order as f64;
                    cases += 1;
                    if err > worst {
                        worst = err;
                        worst_at = format!(
                            "order={order} grid={grid_size} range={gr:?} x={x:e} channel={k}: \
                             finite_diff={want} analytic={got}"
                        );
                    }
                }
            }
        }
    }

    println!(
        "per-channel derivative: {cases} channels, worst scaled error = {worst:e} at {worst_at}"
    );
    // Measured worst over the whole sweep, narrow grids and order 1 included, is
    // 2.1e-7 - the tolerance is not what limits this check, the f64 reference's own
    // `O(step^2)` truncation is.
    assert!(
        worst < 1e-5,
        "a derivative channel disagrees with the finite difference of the basis: \
         {worst:e} at {worst_at}"
    );
}

#[test]
fn derivatives_sum_to_zero() {
    // Corollary of partition of unity: d/dx sum_i B_i(x) = d/dx 1 = 0. Weak on its
    // own (see the module docs) but it is the one property that also holds at the
    // exact knots the finite-difference check has to avoid.
    let mut worst = 0.0f32;
    let mut worst_at = String::new();

    for order in ORDERS {
        for (grid_size, gr) in all_grids() {
            let knots = compute_knots(grid_size, order, gr);
            let h = (gr.1 - gr.0) / grid_size as f32;
            for x in sample_points(gr, &knots, order, grid_size) {
                let span = find_span(x, &knots, order, grid_size);
                let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                let mut deriv = [0.0f32; MAX_SPLINE_ORDER + 1];
                compute_basis_and_deriv(
                    x,
                    span,
                    &knots,
                    order,
                    &mut basis[..=order],
                    &mut deriv[..=order],
                );
                let sum: f32 = deriv[..=order].iter().sum();
                let scaled = sum.abs() * h / order as f32;
                if scaled > worst {
                    worst = scaled;
                    worst_at = format!("order={order} grid={grid_size} range={gr:?} x={x:e}");
                }
            }
        }
    }

    println!("derivative sum: worst scaled |sum| = {worst:e} at {worst_at}");
    // Measured worst is 5e-8. It was 1e-4 while order 1 was excluded and the
    // derivative window could disagree with `find_span` by a whole `1/h`; deriving
    // the order-0 term from `span` (see `src/spline.rs`) made this exact.
    assert!(worst < 1e-6, "derivatives do not sum to zero at {worst_at}");
}

// ===========================================================================
// NARROW GRIDS - the absolute-EPSILON collapse, now fixed.
// ===========================================================================

/// Grid ranges whose knot spacing `h` is at or below the old absolute `EPSILON`
/// guard of 1e-6. All of them are perfectly representable in `f32` - `h` is many
/// orders of magnitude above the ULP at those magnitudes - and all of them used to
/// come back as an all-zero basis.
///
/// Two separate absolute-1e-6 guards were responsible, and the second one hid
/// behind the first:
///  - `compute_basis` skipped every Cox-de Boor denominator with
///    `denom.abs() <= EPSILON`. On these grids that is *every* denominator, so the
///    recursion collapsed to zero: `KanLayer::forward_*` returned exactly `bias`
///    and every gradient through the layer was 0, with nothing reported.
///  - `find_span` floored the grid *width* at `EPSILON` before dividing, so on the
///    `(1e-30, 2e-30)` grid every x landed in the first interval and `compute_basis`
///    extrapolated (per-channel values up to 2e2, still summing to 1).
///
/// `(1e6, 1e6 + 1.0)` at `grid_size = 64` is deliberately *not* here: there `h`
/// really is below the f32 ULP, the knots collide, and no arithmetic fix exists.
/// `KanConfig::validate` rejects it - see
/// [`config_rejects_a_grid_whose_knots_collide_in_f32`].
const NARROW_RANGES: [((f32, f32), usize); 4] = [
    ((0.0, 5e-6), 5),    // h = 1e-6, exactly the old EPSILON
    ((0.0, 1e-5), 64),   // h = 1.5625e-7
    ((0.0, 1e-6), 5),    // h = 2e-7
    ((1e-30, 2e-30), 5), // h = 2e-31
];

/// Partition of unity on grids narrower than the old absolute guard.
///
/// Regression pin for the collapse described on [`NARROW_RANGES`]: before the fix
/// this reported `|sum - 1| = 1.0` exactly - the basis was all zeros - on all 28
/// (range, order) pairs. Re-measured after order 1 joined [`ORDERS`], by putting the
/// `denom > EPSILON` guard back: still all 28.
#[test]
fn partition_of_unity_survives_narrow_grid_ranges() {
    let mut failures = Vec::new();
    for (gr, grid_size) in NARROW_RANGES {
        for order in ORDERS {
            let knots = compute_knots(grid_size, order, gr);
            let h = (gr.1 - gr.0) / grid_size as f32;
            assert!(h <= EPSILON, "fixture must have a sub-EPSILON knot gap");
            let mut worst = 0.0f32;
            for s in 0..=17 {
                let x = gr.0 + (gr.1 - gr.0) * (s as f32 / 17.0);
                let span = find_span(x, &knots, order, grid_size);
                let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                compute_basis(x, span, &knots, order, &mut basis[..=order]);
                let sum: f32 = basis[..=order].iter().sum();
                worst = worst.max((sum - 1.0).abs());
            }
            if worst > 1e-5 {
                failures.push(format!(
                    "  range={gr:?} grid_size={grid_size} order={order} h={h:e}: \
                     worst |sum - 1| = {worst:e}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "basis collapses on {} narrow grids that KanConfig::validate accepts:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The sum check above is invariant under moving mass between channels, so the
/// narrow grids get the same per-channel treatment as the ordinary ones: every
/// channel against the independent `f64` Cox-de Boor.
///
/// This is what proves the fix restored the *right* basis rather than merely a
/// normalized one. It is also the test that caught the `find_span` half of the
/// bug: with only the `compute_basis` guard fixed, `(1e-30, 2e-30)` still had
/// per-channel errors up to 2e2 while its sum read 1.0 to within 4.5e-5.
#[test]
fn every_narrow_grid_basis_channel_matches_the_reference() {
    let mut worst = 0.0f64;
    let mut worst_at = String::new();

    for (gr, grid_size) in NARROW_RANGES {
        for order in ORDERS {
            let knots = compute_knots(grid_size, order, gr);
            let k64: Vec<f64> = knots.iter().map(|&k| f64::from(k)).collect();
            for x in sample_points(gr, &knots, order, grid_size) {
                let span = find_span(x, &knots, order, grid_size);
                let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                compute_basis(x, span, &knots, order, &mut basis[..=order]);
                for (i, &b) in basis[..=order].iter().enumerate() {
                    let j = span as isize - order as isize + i as isize;
                    let err = (f64::from(b) - ref_basis(j, order, f64::from(x), &k64)).abs();
                    if err > worst {
                        worst = err;
                        worst_at = format!(
                            "range={gr:?} grid_size={grid_size} order={order} x={x:e} channel={i}"
                        );
                    }
                }
            }
        }
    }

    assert!(
        worst < 1e-5,
        "worst per-channel basis error {worst:e} at {worst_at}"
    );
}

/// The one grid no arithmetic fix reaches: `h = 1.5625e-2` at a magnitude where
/// the f32 ULP is 0.0625, so consecutive knots are literally the same number.
/// It used to pass `validate` and produce a network that was a constant.
#[test]
fn config_rejects_a_grid_whose_knots_collide_in_f32() {
    let (gr, grid_size, order) = ((1e6f32, 1e6f32 + 1.0), 64usize, 3usize);

    let knots = compute_knots(grid_size, order, gr);
    assert!(
        knots.windows(2).any(|w| w[1] == w[0]),
        "fixture must actually collide in f32"
    );

    let config = arkan::KanConfig {
        grid_size,
        spline_order: order,
        grid_range: gr,
        ..arkan::KanConfig::default()
    };
    assert!(
        matches!(
            config.validate(),
            Err(arkan::config::ConfigError::InvalidGridRange)
        ),
        "a grid with colliding knots must be rejected, not silently zeroed"
    );
}
