//! Does the network actually *learn*, and can training walk into a state it
//! cannot walk out of?
//!
//! # Why this file exists
//!
//! Everything else in `tests/` checks a single call: forward matches PyTorch,
//! backward matches a finite difference, the baked model matches the f32 model.
//! All of it is evaluated at step 0, on a freshly initialized network. Nothing
//! runs a training loop and looks at the result, so nothing here would notice a
//! library that computes a perfect gradient and still never converges.
//!
//! Two things are pinned:
//!
//! 1. **The happy path is genuinely happy.** [`fits_a_sinusoid_far_better_than_the_mean`]
//!    fails if training stops beating the trivial "predict the mean" baseline by a
//!    wide margin. That is the check that catches "gradients are right but the
//!    update is wrong" and "the loss plateaus instantly".
//!
//! 2. **The clamp is an absorbing state.** `z = clamp((x - mean) / std, ...)` has
//!    `dz/dx == 0` outside the grid, and the backward pass correctly drops those
//!    `grad_input`s (`SPAN_CLAMPED_FLAG`, since the clamp/gradient fix). Nothing
//!    bounds a KAN layer's *output* to the grid range, and hidden layers get
//!    identity normalization, so a layer's activations can grow past the range
//!    while training. Once **every** `(sample, feature)` pair at a boundary is
//!    clamped, every upstream gradient is exactly zero and the network is frozen
//!    into a constant function for the rest of the run. Lowering the learning rate
//!    afterwards cannot recover it, because there is no gradient left to follow.
//!
//! `tests/hidden_layer_saturation.rs` pins the *initialization-time* version of
//! trap 2 (a badly chosen `grid_range` starves hidden layers from step 0). This
//! file pins the version that arrives *during* training from a config that looked
//! fine at step 0, which is the one you cannot see coming.

use arkan::{Adam, AdamConfig, KanConfig, KanNetwork, TrainOptions, SPAN_CLAMPED_FLAG};

/// Deterministic xorshift, so this file needs no `rand` dev-dependency behaviour
/// to stay reproducible across `rand` versions.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // 24 mantissa bits -> [0, 1)
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

struct Data {
    x: Vec<f32>,
    y: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Data {
    fn n(&self) -> usize {
        self.x.len() / self.in_dim
    }
    fn batch(&self, start: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
        (
            self.x[start * self.in_dim..(start + len) * self.in_dim].to_vec(),
            self.y[start * self.out_dim..(start + len) * self.out_dim].to_vec(),
        )
    }
}

/// y = sin(pi * x), x uniform on [-1, 1].
fn sinusoid(n: usize, seed: u64) -> Data {
    let mut rng = Rng(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let v = rng.uniform(-1.0, 1.0);
        x.push(v);
        y.push((std::f32::consts::PI * v).sin());
    }
    Data {
        x,
        y,
        in_dim: 1,
        out_dim: 1,
    }
}

/// y = sin(pi*a) * cos(pi*b): needs a hidden layer, so it exercises a layer
/// boundary where activations can leave the grid range.
fn sincos(n: usize, seed: u64) -> Data {
    let mut rng = Rng(seed);
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let a = rng.uniform(-1.0, 1.0);
        let b = rng.uniform(-1.0, 1.0);
        x.push(a);
        x.push(b);
        y.push((std::f32::consts::PI * a).sin() * (std::f32::consts::PI * b).cos());
    }
    Data {
        x,
        y,
        in_dim: 2,
        out_dim: 1,
    }
}

fn config(in_dim: usize, out_dim: usize, hidden: Vec<usize>, grid_range: (f32, f32)) -> KanConfig {
    KanConfig {
        input_dim: in_dim,
        output_dim: out_dim,
        hidden_dims: hidden,
        grid_size: 8,
        spline_order: 3,
        grid_range,
        input_mean: vec![0.0; in_dim],
        input_std: vec![1.0; in_dim],
        multithreading_threshold: 128,
        simd_width: 8,
        init_seed: Some(42),
    }
}

fn mse(net: &KanNetwork, d: &Data) -> f32 {
    let n = d.n();
    let mut ws = net.create_workspace(n);
    let mut p = vec![0.0f32; n * d.out_dim];
    net.forward_batch(&d.x, &mut p, &mut ws);
    p.iter()
        .zip(d.y.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / p.len() as f32
}

/// MSE of the "predict the training mean" model, evaluated on `d`.
fn mean_baseline(train: &Data, d: &Data) -> f32 {
    let od = train.out_dim;
    let mut mean = vec![0.0f32; od];
    for chunk in train.y.chunks(od) {
        for (m, v) in mean.iter_mut().zip(chunk) {
            *m += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= train.n() as f32;
    }
    let acc: f32 =
        d.y.chunks(od)
            .flat_map(|chunk| chunk.iter().zip(&mean).map(|(v, m)| (v - m) * (v - m)))
            .sum();
    acc / (d.n() * od) as f32
}

/// Percentage of `(sample, feature)` pairs clamped at each layer's input.
fn saturation_percent(net: &KanNetwork, x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let n = x.len() / in_dim;
    let mut ws = net.create_workspace(n);
    let mut out = vec![0.0f32; n * out_dim];
    net.forward_batch_training(x, &mut out, &mut ws);
    net.layers
        .iter()
        .enumerate()
        .map(|(li, l)| {
            let cnt = n * l.in_dim;
            let clamped = ws.layers_grid_indices[li][..cnt]
                .iter()
                .filter(|s| *s & SPAN_CLAMPED_FLAG != 0)
                .count();
            100.0 * clamped as f32 / cnt as f32
        })
        .collect()
}

/// Fixed-order minibatch training with Adam. Deterministic.
fn train_adam(net: &mut KanNetwork, d: &Data, lr: f32, epochs: usize, batch: usize) {
    let mut adam = Adam::new(net, AdamConfig::with_lr(lr));
    let mut ws = net.create_workspace(batch);
    let opts = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };
    let nb = d.n() / batch;
    for _ in 0..epochs {
        for b in 0..nb {
            let (bx, by) = d.batch(b * batch, batch);
            net.train_step_with_optimizer(&bx, &by, None, &mut ws, &mut adam, &opts)
                .expect("train step");
        }
    }
}

// ===========================================================================
// 1. The happy path really is happy.
// ===========================================================================

#[test]
fn fits_a_sinusoid_far_better_than_the_mean() {
    let train = sinusoid(1024, 0x1234_5678);
    let test = sinusoid(256, 0x9876_5432);

    let mut net = KanNetwork::new(config(1, 1, vec![8], (-1.5, 1.5)));
    let before = mse(&net, &test);
    train_adam(&mut net, &train, 0.01, 120, 64);
    let after = mse(&net, &test);
    let baseline = mean_baseline(&train, &test);

    println!("sin(pi x): before={before:.6} after={after:.8} mean-baseline={baseline:.6}");
    assert!(after.is_finite(), "loss went non-finite: {after}");
    // Measured ~2e-6 against a 0.50 baseline, i.e. ~250000x. 1000x leaves three
    // orders of magnitude of headroom while still failing loudly if training
    // degrades to "moves a bit" or "plateaus instantly".
    assert!(
        after < baseline / 1000.0,
        "KAN did not beat the mean predictor by 1000x: test MSE {after:.6e} vs baseline {baseline:.6e}"
    );
}

#[test]
fn fits_a_two_dimensional_product_far_better_than_the_mean() {
    // sin*cos is not a sum of univariate functions, so this needs the hidden layer
    // to actually be doing something.
    let train = sincos(1024, 0x0BAD_F00D);
    let test = sincos(512, 0x5EED_1234);

    let mut net = KanNetwork::new(config(2, 1, vec![12, 12], (-1.5, 1.5)));
    train_adam(&mut net, &train, 0.01, 40, 64);
    let after = mse(&net, &test);
    let baseline = mean_baseline(&train, &test);

    println!("sin*cos: after={after:.8} mean-baseline={baseline:.6}");
    assert!(after.is_finite(), "loss went non-finite: {after}");
    assert!(
        after < baseline / 100.0,
        "KAN did not beat the mean predictor by 100x: test MSE {after:.6e} vs baseline {baseline:.6e}"
    );
}

// ===========================================================================
// 2. The clamp is an absorbing state.
// ===========================================================================

#[test]
fn a_saturated_layer_boundary_zeroes_every_upstream_gradient() {
    // Construct the failure directly instead of hoping a training run wanders
    // into it: drive layer 0's outputs far outside the grid range, then confirm
    // layer 0 receives exactly zero gradient.
    let mut net = KanNetwork::new(config(2, 1, vec![16, 16], (-1.5, 1.5)));
    // Every coefficient equal to C makes layer 0's output exactly `in_dim * C`
    // for every sample, because the basis is a partition of unity. 100 is far
    // outside the (-1.5, 1.5) grid, so layer 1 sees nothing but clamped inputs.
    for w in net.layers[0].weights.iter_mut() {
        *w = 100.0;
    }

    let d = sincos(64, 1);
    let sat = saturation_percent(&net, &d.x, 2, 1);
    println!("forced saturation per layer: {sat:?}");
    assert_eq!(
        sat[1], 100.0,
        "expected layer 1's input to be fully clamped, got {:.1}%",
        sat[1]
    );

    let mut ws = net.create_workspace(64);
    let mut opt = Adam::new(&net, AdamConfig::with_lr(0.0));
    net.train_step_with_optimizer(
        &d.x,
        &d.y,
        None,
        &mut ws,
        &mut opt,
        &TrainOptions::default(),
    )
    .expect("train step");

    let nonzero = ws.weight_grads[0].iter().filter(|g| **g != 0.0).count();
    assert_eq!(
        nonzero,
        0,
        "layer 0 should have exactly zero gradient behind a fully clamped boundary, \
         got {nonzero}/{} nonzero entries",
        ws.weight_grads[0].len()
    );
}

#[test]
fn a_saturated_layer_boundary_is_not_recoverable_by_training() {
    // Same forced saturation, but now let the optimizer run and check that
    // nothing upstream ever moves. This is the part that makes the collapse
    // fatal rather than merely slow: there is no learning rate small enough,
    // no schedule patient enough, because the gradient is zero and not small.
    let train = sincos(512, 0x0BAD_F00D);
    let test = sincos(256, 0x5EED_1234);
    let baseline = mean_baseline(&train, &test);

    let mut net = KanNetwork::new(config(2, 1, vec![12, 12], (-1.5, 1.5)));
    for w in net.layers[0].weights.iter_mut() {
        *w = 100.0;
    }
    let before: Vec<f32> = net.layers[0].weights.clone();

    train_adam(&mut net, &train, 1e-3, 30, 64);

    let after = mse(&net, &test);
    println!("saturated net after 30 epochs: test MSE {after:.6} (mean baseline {baseline:.6})");
    assert_eq!(
        net.layers[0].weights, before,
        "layer 0 moved, so its gradient was not zero after all"
    );
    assert!(
        after > 0.9 * baseline,
        "a fully saturated network learned something ({after:.6} vs baseline {baseline:.6}) - \
         if the clamp gained a nonzero out-of-range slope, or hidden layers gained \
         running normalization, update this test and the guidance in \
         tests/hidden_layer_saturation.rs and docs/ARCHITECTURE.md"
    );

    // ... and it is a constant function, not just a bad one.
    let mut ws = net.create_workspace(test.n());
    let mut preds = vec![0.0f32; test.n()];
    net.forward_batch(&test.x, &mut preds, &mut ws);
    let spread = preds.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - preds.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        spread < 1e-6,
        "saturated network should output a constant, got spread {spread:.3e}"
    );
}

/// Demonstration, not an invariant: an ordinary training run at an aggressive
/// (but not absurd) learning rate walks itself into the collapse above.
///
/// `#[ignore]` for two reasons. It is slow, and unlike the constructed tests it
/// depends on the *dynamics* landing in the absorbing state rather than merely
/// near it — the exact epoch varies with width, batch order and float
/// accumulation. Run it with `cargo test -- --ignored` when touching the clamp,
/// normalization, or the optimizer.
#[test]
#[ignore = "slow (~40s debug) and depends on training dynamics, not just kernels"]
fn an_ordinary_training_run_can_walk_into_the_collapse() {
    let train = sincos(2048, 0x0BAD_F00D);
    let test = sincos(512, 0x5EED_1234);
    let baseline = mean_baseline(&train, &test);

    let mut net = KanNetwork::new(config(2, 1, vec![16, 16], (-1.5, 1.5)));

    // Step 0 looks perfectly healthy: nothing is clamped anywhere. Every
    // initialization-time check, including tests/hidden_layer_saturation.rs,
    // would pass on this network.
    let sat0 = saturation_percent(&net, &train.x[..256 * 2], 2, 1);
    println!("saturation at init: {sat0:?}");
    assert!(
        sat0.iter().all(|s| *s < 1.0),
        "this config must start unsaturated for the test to mean anything, got {sat0:?}"
    );

    train_adam(&mut net, &train, 0.1, 80, 64);

    let collapsed = mse(&net, &test);
    let sat = saturation_percent(&net, &train.x[..256 * 2], 2, 1);
    println!("after 80 epochs at lr=0.1: test MSE {collapsed:.6}, saturation {sat:?}");
    assert!(
        sat.last().copied().unwrap() >= 99.9,
        "expected the last layer boundary to be fully saturated, got {sat:?}"
    );
    assert!(
        collapsed > 0.9 * baseline,
        "expected the collapsed network to be no better than the mean predictor, \
         got {collapsed:.6} vs baseline {baseline:.6}"
    );
}

/// The axes the PyTorch fixtures hold fixed, checked for *convergence* rather
/// than for a matching forward pass. `#[ignore]`: trains ~112 networks.
///
/// Note on the budget, and on why `grid_size = 2` is not in the sweep.
/// Convergence speed is not uniform here: the wider `grid_range` is relative to
/// the data, the fewer knot spans the samples touch, and the smaller each
/// coefficient's gradient is. Measured on this task (data spanning `[-1, 1]`),
/// `grid_size = 2` needs ~50x the epochs at `grid_range = (-6, 6)` that it needs
/// at `(-1, 1)` — 2.6x better than the mean after 60 epochs, 33000x after 3000.
/// It converges; it is just slow, and at `grid_size = 2` with `grid_range =
/// (-3, 3)` the whole dataset lives inside a single knot span, which makes "did
/// it converge" a capacity question rather than a correctness one. The sweep
/// starts at `grid_size = 3`, where every combination clears the bar in 300
/// epochs. The bar is a modest 20x so this fails on "training does nothing",
/// not on "training is slow".
///
/// `spline_order = 1` is excluded for the same reason: `KanConfig::validate`
/// accepts it but documents it as a degenerate step function, and at
/// `grid_size = 3, grid_range = (-3, 3)` it reaches only 2.6x the mean predictor
/// no matter how long it trains, because one linear piece cannot be a sinusoid.
#[test]
#[ignore = "slow: trains 96 networks"]
fn every_spline_order_and_grid_range_still_converges() {
    let train = sinusoid(1024, 0x1234_5678);
    let test = sinusoid(256, 0x9876_5432);
    let baseline = mean_baseline(&train, &test);
    let mut failures = Vec::new();
    let mut worst = (f32::INFINITY, String::new());

    for order in 2..=7usize {
        for grid in [3usize, 5, 8, 16] {
            for range in [(-1.0f32, 1.0f32), (-1.5, 1.5), (-3.0, 3.0), (-3.0, 1.0)] {
                let mut cfg = config(1, 1, vec![8], range);
                cfg.grid_size = grid;
                cfg.spline_order = order;
                let mut net = KanNetwork::new(cfg);
                train_adam(&mut net, &train, 0.01, 300, 64);
                let m = mse(&net, &test);
                let ratio = baseline / m;
                let tag = format!("order={order} grid={grid} range={range:?}");
                let ok = m.is_finite() && ratio > 20.0;
                println!(
                    "{tag}: test MSE {m:.3e} ({ratio:.0}x better than mean){}",
                    if ok { "" } else { "  <-- FAIL" }
                );
                if ratio < worst.0 {
                    worst = (ratio, tag.clone());
                }
                if !ok {
                    failures.push(format!("{tag} mse={m:.3e}"));
                }
            }
        }
    }
    println!("worst ratio: {:.0}x at {}", worst.0, worst.1);
    assert!(
        failures.is_empty(),
        "configurations that failed to beat the mean predictor by 20x:\n  {}",
        failures.join("\n  ")
    );
}
