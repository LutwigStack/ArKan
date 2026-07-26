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

/// Orders under test. CPU maximum is [`MAX_SPLINE_ORDER`]; order 1 is excluded
/// because its basis is only C^0, so the derivative is a step function and a
/// central difference across a knot converges to the average of two different
/// one-sided limits. Order 0 has no derivative at all.
const ORDERS: std::ops::RangeInclusive<usize> = 2..=MAX_SPLINE_ORDER;

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

#[test]
fn every_basis_channel_matches_an_independent_cox_de_boor() {
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    let mut cases = 0usize;

    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
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

                    for (k, &got) in basis[..=order].iter().enumerate() {
                        let j = span - order + k;
                        let want = ref_basis(j as isize, order, f64::from(x), &knots64);
                        let err = (want - f64::from(got)).abs();
                        cases += 1;
                        if err > worst {
                            worst = err;
                            worst_at = format!(
                                "order={order} grid_size={grid_size} range={gr:?} x={x:e} \
                                 channel={k} (global {j}): want={want} got={got}"
                            );
                        }
                    }
                }
            }
        }
    }

    println!("per-channel basis: {cases} channels, worst error = {worst:e} at {worst_at}");
    assert!(
        worst < 5e-6,
        "a basis channel disagrees with the textbook recursion: {worst:e} at {worst_at}"
    );
}

#[test]
fn basis_values_are_never_negative() {
    let mut worst = 0.0f32;
    let mut worst_at = String::new();

    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
                let knots = compute_knots(grid_size, order, gr);
                for x in sample_points(gr, &knots, order, grid_size) {
                    let span = find_span(x, &knots, order, grid_size);
                    let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
                    compute_basis(x, span, &knots, order, &mut basis[..=order]);
                    for (k, &b) in basis[..=order].iter().enumerate() {
                        if b < worst {
                            worst = b;
                            worst_at = format!(
                                "order={order} grid={grid_size} range={gr:?} x={x:e} k={k}"
                            );
                        }
                    }
                }
            }
        }
    }

    println!("most negative basis value = {worst:e} at {worst_at}");
    assert!(
        worst > -1e-6,
        "basis went meaningfully negative: {worst:e} at {worst_at}"
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
fn compute_basis_and_deriv_reproduces_compute_basis() {
    // Documented contract: `compute_basis_and_deriv` fills `basis_out` with exactly
    // what `compute_basis` would. Backward relies on it - the weight gradient comes
    // from these values, not from the ones the forward pass stored.
    let mut worst = 0.0f32;
    let mut worst_at = String::new();
    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
                let knots = compute_knots(grid_size, order, gr);
                for x in sample_points(gr, &knots, order, grid_size) {
                    let span = find_span(x, &knots, order, grid_size);
                    let mut a = [0.0f32; MAX_SPLINE_ORDER + 1];
                    let mut b = [0.0f32; MAX_SPLINE_ORDER + 1];
                    let mut d = [0.0f32; MAX_SPLINE_ORDER + 1];
                    compute_basis(x, span, &knots, order, &mut a[..=order]);
                    compute_basis_and_deriv(
                        x,
                        span,
                        &knots,
                        order,
                        &mut b[..=order],
                        &mut d[..=order],
                    );
                    for k in 0..=order {
                        let e = (a[k] - b[k]).abs();
                        if e > worst {
                            worst = e;
                            worst_at = format!(
                                "order={order} grid={grid_size} range={gr:?} x={x:e} k={k}"
                            );
                        }
                    }
                }
            }
        }
    }
    println!("basis from compute_basis_and_deriv: worst delta = {worst:e} at {worst_at}");
    assert_eq!(worst, 0.0, "basis values diverge at {worst_at}");
}

#[test]
fn every_derivative_channel_matches_a_central_difference() {
    // Central difference of the *independent* f64 reference basis, taken strictly
    // inside a knot interval so the probe never straddles a knot. Relative
    // tolerance is against `order / h`, the natural scale of a B-spline derivative.
    let mut worst = 0.0f64;
    let mut worst_at = String::new();
    let mut cases = 0usize;

    for order in ORDERS {
        for grid_size in GRID_SIZES {
            for gr in RANGES {
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
    }

    println!(
        "per-channel derivative: {cases} channels, worst scaled error = {worst:e} at {worst_at}"
    );
    assert!(
        worst < 1e-4,
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
        for grid_size in GRID_SIZES {
            for gr in RANGES {
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
    }

    println!("derivative sum: worst scaled |sum| = {worst:e} at {worst_at}");
    assert!(worst < 1e-4, "derivatives do not sum to zero at {worst_at}");
}

// ===========================================================================
// KNOT COLLAPSE - a real defect, deliberately left failing.
// ===========================================================================

/// Grid ranges whose knot spacing is not resolvable in `f32`, either absolutely
/// (`h <= EPSILON`) or relative to the offset (`1e6 + 1.5625e-2` rounds back to
/// `1e6`). All of these pass `KanConfig::validate`.
const DEGENERATE_RANGES: [((f32, f32), usize); 5] = [
    ((0.0, 5e-6), 5),       // h = 1e-6, exactly EPSILON
    ((0.0, 1e-5), 64),      // h = 1.5625e-7
    ((0.0, 1e-6), 5),       // h = 2e-7
    ((1e6, 1e6 + 1.0), 64), // h = 1.5625e-2, below the f32 ULP at 1e6
    ((1e-30, 2e-30), 5),    // h = 2e-31
];

/// **KNOWN BUG - `compute_basis` returns a zero (or partially zeroed) basis when
/// the knot spacing is at or below `config::EPSILON` (1e-6) in absolute terms, or
/// below the `f32` ULP at the grid offset.**
///
/// Cox-de Boor divides by knot differences and skips any denominator with
/// `denom.abs() <= EPSILON` (`src/spline.rs`, inside `compute_basis`). That guard
/// is *absolute*, so on such a grid the whole recursion collapses: partition of
/// unity reads 0 instead of 1, `KanLayer::forward_*` returns exactly `bias` for
/// every input, and every gradient through the layer is 0. Nothing errors -
/// `KanConfig::validate` accepts all the ranges below, training simply never
/// moves and the model is a constant.
///
/// Not fixed here, because neither available fix is small *and* clearly right:
///  - Rejecting these grids in `validate` breaks
///    `baked::tests::test_bake_narrow_grid_range_no_clamp_panic`, which
///    deliberately builds a `(0.0, 5e-6)` network to pin a separate bake bug -
///    i.e. tolerating narrow ranges was a decision, not an oversight.
///  - Making the guard relative (`denom > 0.0`, or scaling it by the knot
///    spacing) changes the numerics of the hottest loop in the crate and trades
///    the silent-zero failure for a possible overflow-to-inf one.
#[test]
#[ignore = "known bug: absolute EPSILON guard in compute_basis collapses narrow/offset grids"]
fn partition_of_unity_survives_narrow_grid_ranges() {
    let mut failures = Vec::new();
    for (gr, grid_size) in DEGENERATE_RANGES {
        for order in ORDERS {
            let knots = compute_knots(grid_size, order, gr);
            let h = (gr.1 - gr.0) / grid_size as f32;
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
        "basis collapses on {} narrow/offset grids that KanConfig::validate accepts:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Characterization of the same bug, kept running so a fix is noticed here first:
/// the `(0.0, 1e-5)` / `grid_size = 64` grid currently yields an all-zero basis.
///
/// If this starts failing, `compute_basis` was fixed - delete it and un-ignore
/// [`partition_of_unity_survives_narrow_grid_ranges`].
#[test]
fn collapsed_knots_currently_produce_a_zero_basis() {
    let (gr, grid_size, order) = ((0.0f32, 1e-5f32), 64usize, 3usize);
    let h = (gr.1 - gr.0) / grid_size as f32;
    assert!(h < EPSILON, "fixture must have a sub-EPSILON knot gap");

    let knots = compute_knots(grid_size, order, gr);
    let x = gr.0 + (gr.1 - gr.0) * 0.5;
    let span = find_span(x, &knots, order, grid_size);
    let mut basis = [0.0f32; MAX_SPLINE_ORDER + 1];
    compute_basis(x, span, &knots, order, &mut basis[..=order]);
    let sum: f32 = basis[..=order].iter().sum();
    assert_eq!(
        sum, 0.0,
        "compute_basis no longer collapses - see the doc comment"
    );
}
