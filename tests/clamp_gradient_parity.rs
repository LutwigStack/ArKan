//! Finite-difference gradient check for `grad_input` around the normalization clamp.
//!
//! # What this pins down
//!
//! The forward pass computes `z = clamp((x - mean) / std, grid_min, grid_max)`.
//! For a saturated input the output does not depend on `x` at all, so the true
//! `dz/dx` (and therefore `dL/dx`) is exactly **zero**. The backward pass, however,
//! scales the spline derivative by `1 / std` unconditionally, which is `dz/dx` for
//! the *unclamped* map only.
//!
//! Every test here compares the analytic `grad_input` from
//! [`KanLayer::backward`] / [`KanLayer::backward_parallel`] against a central
//! difference taken through the *actual* [`KanLayer::forward_batch`].
//!
//! # Harness soundness
//!
//! `control_*` tests place all inputs strictly inside the grid range, where the
//! clamp never fires. Those must pass regardless of the clamp handling - if they
//! fail, the bug is elsewhere and bigger than the clamp.
//!
//! # Why inputs are kept away from knots and from the exact boundary
//!
//! At a knot the spline is only C^(p-1), and exactly at `grid_min`/`grid_max` the
//! derivative is one-sided; a central difference straddles the kink and matches
//! neither side. Both are properties of the finite-difference probe, not of the
//! gradient, so control points are chosen away from those locations.

use arkan::{KanConfig, KanLayer, KanNetwork, TrainOptions, Workspace};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Central-difference step. Large enough that `L(x+h) - L(x-h)` survives f32
/// cancellation, small enough to stay inside one spline segment.
const FD_STEP: f32 = 2e-3;

fn config(in_dim: usize, order: usize, grid_size: usize) -> KanConfig {
    KanConfig {
        input_dim: in_dim,
        output_dim: 4,
        hidden_dims: vec![],
        spline_order: order,
        grid_size,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; in_dim],
        input_std: vec![1.0; in_dim],
        init_seed: Some(20250726),
        ..Default::default()
    }
}

/// Scalar functional of the layer output: `L = sum_{b,j} g_out[b,j] * y[b,j]`.
/// Its gradient w.r.t. the raw inputs is exactly what `backward` reports in
/// `grad_input` when seeded with `grad_output = g_out`.
fn directional_loss(layer: &KanLayer, inputs: &[f32], g_out: &[f32], ws: &mut Workspace) -> f32 {
    let batch = inputs.len() / layer.in_dim;
    let mut out = vec![0.0f32; batch * layer.out_dim];
    layer.forward_batch(inputs, &mut out, ws);
    out.iter().zip(g_out).map(|(y, g)| y * g).sum()
}

/// Analytic `dL/dx` from the layer's own backward pass, driven exactly the way
/// `KanNetwork` drives it: forward stores `z` + spans, backward consumes them.
fn analytic_grad_input(
    layer: &KanLayer,
    inputs: &[f32],
    g_out: &[f32],
    parallel: bool,
) -> Vec<f32> {
    let batch = inputs.len() / layer.in_dim;
    let mut ws = Workspace::default();
    let mut out = vec![0.0f32; batch * layer.out_dim];
    layer.forward_batch(inputs, &mut out, &mut ws);

    // Copy the saved history out of the workspace (backward needs &mut workspace).
    let z: Vec<f32> = ws.z_buffer.as_slice()[..batch * layer.in_dim].to_vec();
    let spans: Vec<u32> = ws.grid_indices[..batch * layer.in_dim].to_vec();

    let mut grad_input = vec![0.0f32; batch * layer.in_dim];
    let mut grad_w = vec![0.0f32; layer.weights.len()];
    let mut grad_b = vec![0.0f32; layer.bias.len()];

    if parallel {
        layer.backward_parallel(
            &z,
            &spans,
            g_out,
            Some(&mut grad_input),
            &mut grad_w,
            &mut grad_b,
        );
    } else {
        layer.backward(
            &z,
            &spans,
            g_out,
            Some(&mut grad_input),
            &mut grad_w,
            &mut grad_b,
            &mut ws,
        );
    }
    grad_input
}

/// Numeric `dL/dx` by central difference through the real forward pass.
fn fd_grad_input(layer: &KanLayer, inputs: &[f32], g_out: &[f32]) -> Vec<f32> {
    let mut ws = Workspace::default();
    let mut fd = vec![0.0f32; inputs.len()];
    let mut probe = inputs.to_vec();
    for idx in 0..inputs.len() {
        let x = inputs[idx];
        probe[idx] = x + FD_STEP;
        let plus = directional_loss(layer, &probe, g_out, &mut ws);
        probe[idx] = x - FD_STEP;
        let minus = directional_loss(layer, &probe, g_out, &mut ws);
        probe[idx] = x;
        fd[idx] = (plus - minus) / (2.0 * FD_STEP);
    }
    fd
}

fn seeded_grad_output(batch: usize, out_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..batch * out_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

/// Compares analytic vs numeric `grad_input` and returns the worst absolute gap.
fn compare(label: &str, layer: &KanLayer, inputs: &[f32], g_out: &[f32]) -> f32 {
    let fd = fd_grad_input(layer, inputs, g_out);
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    let mut failures = Vec::new();

    for parallel in [false, true] {
        let ana = analytic_grad_input(layer, inputs, g_out, parallel);
        for i in 0..inputs.len() {
            let gap = (ana[i] - fd[i]).abs();
            // Absolute floor covers f32 cancellation in the difference quotient;
            // relative term covers large gradients.
            let tol = 2e-3 + 2e-2 * fd[i].abs();
            if gap > worst {
                worst = gap;
                worst_at = i;
            }
            if gap > tol {
                failures.push(format!(
                    "  [{}] {} idx={} x={:.4} analytic={:.6} finite_diff={:.6} gap={:.6}",
                    label,
                    if parallel {
                        "backward_parallel"
                    } else {
                        "backward"
                    },
                    i,
                    inputs[i],
                    ana[i],
                    fd[i],
                    gap
                ));
            }
        }
    }

    println!(
        "[{}] max |analytic - finite_diff| = {:.6} at idx {} (x = {:.4})",
        label, worst, worst_at, inputs[worst_at]
    );
    assert!(
        failures.is_empty(),
        "grad_input disagrees with the forward pass ({} mismatches, worst {:.6}):\n{}",
        failures.len(),
        worst,
        failures.join("\n")
    );
    worst
}

// ===========================================================================
// CONTROLS - must pass whatever the clamp does. They prove the harness itself.
// ===========================================================================

#[test]
fn control_input_layer_inside_grid_range() {
    let cfg = config(4, 3, 5);
    let layer = KanLayer::new(4, 4, &cfg);
    // Knots sit at -1, -0.6, -0.2, 0.2, 0.6, 1.0 - stay off them.
    let inputs = vec![
        0.05, -0.37, 0.44, -0.13, //
        0.71, -0.82, 0.11, 0.33,
    ];
    let g_out = seeded_grad_output(2, layer.out_dim, 7);
    compare("control input layer, inside range", &layer, &inputs, &g_out);
}

#[test]
fn control_hidden_layer_inside_grid_range() {
    // in_dim != config.input_dim => identity normalization, exactly how
    // KanNetwork builds its hidden layers.
    let cfg = config(4, 3, 5);
    let layer = KanLayer::new(8, 4, &cfg);
    assert_eq!(
        layer.std,
        vec![1.0; 8],
        "hidden layer must be identity-normalized"
    );
    assert_eq!(layer.mean, vec![0.0; 8]);

    let inputs = vec![
        0.05, -0.37, 0.44, -0.13, 0.71, -0.82, 0.11, 0.33, //
        -0.05, 0.37, -0.44, 0.13, -0.71, 0.82, -0.11, -0.33,
    ];
    let g_out = seeded_grad_output(2, layer.out_dim, 11);
    compare(
        "control hidden layer, inside range",
        &layer,
        &inputs,
        &g_out,
    );
}

#[test]
fn control_input_layer_with_nonunit_std() {
    // Exercises the std_inv path itself: z = x / 0.4, all still inside range.
    let mut cfg = config(4, 3, 5);
    cfg.input_std = vec![0.4; 4];
    let layer = KanLayer::new(4, 4, &cfg);
    assert_eq!(layer.std, vec![0.4; 4]);

    let inputs = vec![0.02, -0.15, 0.18, -0.05, 0.28, -0.33, 0.04, 0.13];
    let g_out = seeded_grad_output(2, layer.out_dim, 13);
    compare(
        "control input layer, std=0.4, inside range",
        &layer,
        &inputs,
        &g_out,
    );
}

// ===========================================================================
// SATURATED - these are the bug.
// ===========================================================================

#[test]
fn hidden_layer_saturated_input_grad_must_be_zero() {
    for order in 2..=4 {
        let cfg = config(4, order, 5);
        let layer = KanLayer::new(8, 4, &cfg);

        // Every feature is far outside (-1, 1): the forward output is completely
        // insensitive to these inputs, so dL/dx == 0 exactly.
        let inputs = vec![
            1.7, -2.3, 3.1, -1.4, 2.05, -4.0, 1.25, -1.9, //
            -1.6, 2.4, -3.0, 1.45, -2.2, 5.0, -1.3, 1.85,
        ];
        let g_out = seeded_grad_output(2, layer.out_dim, 17 + order as u64);

        // Ground truth, independent of the FD probe: the forward pass is constant.
        let fd = fd_grad_input(&layer, &inputs, &g_out);
        assert!(
            fd.iter().all(|g| *g == 0.0),
            "saturated forward pass must be exactly flat, got {:?}",
            fd
        );

        compare(
            &format!("hidden layer, fully saturated, order={}", order),
            &layer,
            &inputs,
            &g_out,
        );
    }
}

#[test]
fn straddling_boundary_mixes_live_and_saturated_features() {
    let cfg = config(4, 3, 5);
    let layer = KanLayer::new(8, 4, &cfg);

    // Alternating: inside the range / outside the range, within the same sample.
    let inputs = vec![
        0.05, 1.4, -0.37, -2.6, 0.44, 3.3, -0.13, -1.15, //
        0.71, 1.05, -0.82, -1.9, 0.11, 2.2, 0.33, -1.6,
    ];
    let g_out = seeded_grad_output(2, layer.out_dim, 23);

    let fd = fd_grad_input(&layer, &inputs, &g_out);
    for (i, g) in fd.iter().enumerate() {
        if inputs[i].abs() > 1.0 + FD_STEP {
            assert_eq!(*g, 0.0, "saturated feature {} should be flat", i);
        } else {
            assert!(*g != 0.0, "live feature {} should have nonzero slope", i);
        }
    }

    compare("hidden layer, straddling boundary", &layer, &inputs, &g_out);
}

#[test]
fn input_layer_saturates_through_std_scaling() {
    // Raw inputs look harmless but std=0.4 pushes z outside the grid range.
    let mut cfg = config(4, 3, 5);
    cfg.input_std = vec![0.4; 4];
    let layer = KanLayer::new(4, 4, &cfg);

    let inputs = vec![0.9, -1.2, 2.0, -0.75, 1.5, -0.62, 0.83, -2.4];
    let g_out = seeded_grad_output(2, layer.out_dim, 29);

    let fd = fd_grad_input(&layer, &inputs, &g_out);
    assert!(
        fd.iter().all(|g| *g == 0.0),
        "all z = x/0.4 are outside (-1,1); forward must be flat, got {:?}",
        fd
    );

    compare("input layer, saturated via std", &layer, &inputs, &g_out);
}

#[test]
fn backward_and_backward_parallel_agree_on_saturated_inputs() {
    let cfg = config(4, 3, 5);
    let layer = KanLayer::new(8, 4, &cfg);
    let batch = 4;
    let inputs: Vec<f32> = (0..batch * 8)
        .map(|i| {
            let v = (i as f32) * 0.37 - 5.0;
            if i % 3 == 0 {
                v * 2.0
            } else {
                v * 0.1
            }
        })
        .collect();
    let g_out = seeded_grad_output(batch, layer.out_dim, 31);

    let seq = analytic_grad_input(&layer, &inputs, &g_out, false);
    let par = analytic_grad_input(&layer, &inputs, &g_out, true);
    for i in 0..seq.len() {
        assert!(
            (seq[i] - par[i]).abs() < 1e-5,
            "backward vs backward_parallel diverge at {}: {} vs {}",
            i,
            seq[i],
            par[i]
        );
    }
}

// ===========================================================================
// SEVERITY - does a real network actually saturate its hidden activations?
// ===========================================================================

/// Builds a 4 -> 8 -> 4 network whose layer-0 outputs leave the grid range, and
/// reports what fraction of the hidden layer's inputs are clamped.
fn saturating_network() -> (KanNetwork, Vec<f32>, Vec<f32>, usize) {
    let cfg = KanConfig {
        input_dim: 4,
        output_dim: 4,
        hidden_dims: vec![8],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(4242),
        ..Default::default()
    };
    let batch = 8;
    let mut network = KanNetwork::new(cfg.clone());

    // Freshly initialized weights are tiny; a trained layer's are not. Scale
    // layer 0 up so its outputs leave (-1, 1), which is what training does.
    for w in network.layers[0].weights.iter_mut() {
        *w *= 9.0;
    }

    let mut rng = SmallRng::seed_from_u64(555);
    let input: Vec<f32> = (0..batch * 4).map(|_| rng.gen_range(-0.9..0.9)).collect();
    let target: Vec<f32> = (0..batch * 4).map(|_| rng.gen_range(-1.0..1.0)).collect();
    (network, input, target, batch)
}

/// Raw (pre-clamp) outputs of layer 0 = the hidden layer's raw inputs.
fn layer0_outputs(network: &KanNetwork, input: &[f32], batch: usize) -> Vec<f32> {
    let mut ws = Workspace::default();
    let mut out = vec![0.0f32; batch * network.layers[0].out_dim];
    network.layers[0].forward_batch(input, &mut out, &mut ws);
    out
}

#[test]
fn hidden_activations_really_do_leave_the_grid_range() {
    let (network, input, _target, batch) = saturating_network();
    let hidden = layer0_outputs(&network, &input, batch);
    let clamped = hidden.iter().filter(|v| v.abs() > 1.0).count();
    let frac = clamped as f32 / hidden.len() as f32;
    println!(
        "hidden activations outside grid_range: {}/{} = {:.1}%",
        clamped,
        hidden.len(),
        frac * 100.0
    );
    assert!(
        clamped > 0,
        "this fixture is supposed to saturate; it did not, so the severity \
         argument below is untested"
    );
}

/// The one that matters for training: a wrong `grad_input` on the hidden layer
/// is what layer 0's weight gradients are built from.
#[test]
fn saturated_hidden_layer_corrupts_layer0_weight_gradients() {
    let (network, input, target, batch) = saturating_network();
    let opts = TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.0,
    };

    let mut net = network.clone();
    let mut ws = net.create_workspace(batch);
    // LR = 0: run the real training step purely to populate the gradient buffers.
    net.train_step_with_options(&input, &target, None, 0.0, &mut ws, &opts);
    let ana: Vec<f32> = ws.weight_grads[0].clone();

    let mse = |net: &KanNetwork, ws: &mut Workspace| -> f32 {
        let mut pred = vec![0.0f32; batch * 4];
        net.forward_batch(&input, &mut pred, ws);
        pred.iter()
            .zip(&target)
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / pred.len() as f32
    };

    // Which hidden activations are clamped is what decides whether the loss is
    // even differentiable at this weight; skip probes that move that pattern.
    let pattern = |net: &KanNetwork| -> Vec<bool> {
        layer0_outputs(net, &input, batch)
            .iter()
            .map(|v| v.abs() > 1.0)
            .collect()
    };
    let base_pattern = pattern(&network);

    let mut rng = SmallRng::seed_from_u64(909);
    let n_weights = network.layers[0].weights.len();
    let mut checked = 0;
    let mut worst = 0.0f32;
    let mut failures = Vec::new();

    for _ in 0..40 {
        let w_idx = rng.gen_range(0..n_weights);
        let h = 1e-3f32;

        let mut probe = network.clone();
        let orig = probe.layers[0].weights[w_idx];

        probe.layers[0].weights[w_idx] = orig + h;
        if pattern(&probe) != base_pattern {
            continue;
        }
        let plus = mse(&probe, &mut ws);

        probe.layers[0].weights[w_idx] = orig - h;
        if pattern(&probe) != base_pattern {
            continue;
        }
        let minus = mse(&probe, &mut ws);

        let fd = (plus - minus) / (2.0 * h);
        let gap = (ana[w_idx] - fd).abs();
        worst = worst.max(gap);
        checked += 1;

        if gap > 5e-3 + 5e-2 * fd.abs() {
            failures.push(format!(
                "  layer0 weight[{}]: analytic={:.6} finite_diff={:.6} gap={:.6}",
                w_idx, ana[w_idx], fd, gap
            ));
        }
    }

    println!(
        "layer0 weight gradients checked: {}, max |analytic - finite_diff| = {:.6}",
        checked, worst
    );
    assert!(checked >= 10, "not enough stable probes: {}", checked);
    assert!(
        failures.is_empty(),
        "layer 0 weight gradients are corrupted by the hidden layer's grad_input \
         ({} of {} mismatched, worst {:.6}):\n{}",
        failures.len(),
        checked,
        worst,
        failures.join("\n")
    );
}
