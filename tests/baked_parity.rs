//! Honest accuracy measurement for BakedModel (int8 quantized inference).
//!
//! Reports BOTH:
//!   (a) NRMSE = ||b - f||₂ / ||f||₂   (aggregate, over all outputs × test inputs)
//!   (b) Worst-case = max per-element |b - f| / |f| restricted to outputs whose
//!       |f| > tau, where tau = 0.1 × per-output std over the test set.
//!
//! Asserts NRMSE ≤ 5% (small config) / ≤ 10% (medium config).
//! Worst-case on significant outputs is printed but NOT gated — it is the
//! honest tail number that WS02's aggregate NRMSE concealed.

use arkan::{BakedModel, KanConfig, KanNetwork};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

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

/// Fast deterministic PRNG (xorshift64) for generating test data without pulling in rand.
fn random_inputs_in_range(n: usize, dim: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed ^ 0xdeadbeef_cafebabe;
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // map u64 → [lo, hi]
        let fv = (state as f64 / u64::MAX as f64) as f32 * (hi - lo) + lo;
        out.push(fv);
    }
    out
}

/// Count total f32 weights across all layers of a KanNetwork.
fn f32_weight_bytes(network: &KanNetwork) -> usize {
    network
        .layers
        .iter()
        .map(|l| (l.weights.len() + l.bias.len()) * 4)
        .sum()
}

// ---------------------------------------------------------------------------
// Accuracy metric helpers
// ---------------------------------------------------------------------------

/// Compute NRMSE and worst-case-on-significant-outputs for a given config.
///
/// Returns (nrmse, worst_case_significant).
fn measure_accuracy(
    network: &KanNetwork,
    baked: &BakedModel,
    test_inputs: &[f32],
    input_dim: usize,
    output_dim: usize,
) -> (f32, f32) {
    let n = test_inputs.len() / input_dim;

    // Collect all (f32_out, baked_out) pairs
    let mut f32_outs = vec![vec![0.0f32; output_dim]; n];
    let mut baked_outs = vec![vec![0.0f32; output_dim]; n];
    let mut workspace = network.create_workspace(1);

    for s in 0..n {
        let inp = &test_inputs[s * input_dim..(s + 1) * input_dim];
        network.forward_single(inp, &mut f32_outs[s], &mut workspace);
        baked.forward(inp, &mut baked_outs[s]);
    }

    // (a) NRMSE = sqrt(mean(|b - f|²)) / sqrt(mean(f²))
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_f = 0.0f64;
    for s in 0..n {
        for j in 0..output_dim {
            let e = (baked_outs[s][j] - f32_outs[s][j]) as f64;
            let fv = f32_outs[s][j] as f64;
            sum_sq_err += e * e;
            sum_sq_f += fv * fv;
        }
    }
    let count = (n * output_dim) as f64;
    let rmse = (sum_sq_err / count).sqrt();
    let rms_f = (sum_sq_f / count).sqrt().max(1e-9);
    let nrmse = (rmse / rms_f) as f32;

    // (b) worst-case on significant outputs, at several significance thresholds.
    // tau_j = factor * std_j. The 0.1σ threshold includes near-noise outputs;
    // 0.5σ/1.0σ show whether the tail affects decision-relevant outputs.
    let mut std_j = vec![0.0f32; output_dim];
    for j in 0..output_dim {
        let vals: Vec<f32> = (0..n).map(|s| f32_outs[s][j]).collect();
        let mean = vals.iter().copied().sum::<f32>() / n as f32;
        let var = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        std_j[j] = var.sqrt();
    }

    let worst_at = |factor: f32| -> f32 {
        let mut wc = 0.0f32;
        for s in 0..n {
            for j in 0..output_dim {
                let fv = f32_outs[s][j].abs();
                if fv > factor * std_j[j] {
                    let rel_err = (baked_outs[s][j] - f32_outs[s][j]).abs() / fv;
                    if rel_err > wc {
                        wc = rel_err;
                    }
                }
            }
        }
        wc
    };

    let worst_case = worst_at(0.1);
    println!(
        "    worst-case by significance: 0.1σ={:.1}%  0.5σ={:.1}%  1.0σ={:.1}%",
        worst_case * 100.0,
        worst_at(0.5) * 100.0,
        worst_at(1.0) * 100.0
    );

    (nrmse, worst_case)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Config 1 (small): 4 → [8] → 2, grid=5, order=3
#[test]
fn baked_parity_small() {
    let input_dim = 4;
    let hidden = vec![8];
    let output_dim = 2;
    let grid_size = 5;
    let order = 3;

    let network = make_network(input_dim, hidden, output_dim, grid_size, order, 42);

    // Calibration: 256 inputs in grid range [-1, 1]
    let cal = random_inputs_in_range(256, input_dim, 1234, -0.9, 0.9);
    // Test set: 2000 inputs
    let test = random_inputs_in_range(2000, input_dim, 5678, -0.9, 0.9);

    let baked = BakedModel::from_network(&network, Some(&cal));
    let f32_bytes = f32_weight_bytes(&network);
    let baked_bytes = baked.size_bytes();

    let (nrmse, worst_case) = measure_accuracy(&network, &baked, &test, input_dim, output_dim);

    println!(
        "\n[baked_parity_small] config=4→[8]→2, grid=5, order=3"
    );
    println!(
        "  NRMSE (aggregate L2 relative error):          {:.4} ({:.2}%)",
        nrmse,
        nrmse * 100.0
    );
    println!(
        "  Worst-case on significant outputs (tau=0.1σ): {:.4} ({:.2}%)",
        worst_case,
        worst_case * 100.0
    );
    println!(
        "  Size: baked={} B  f32_weights={} B  ratio={:.2}x",
        baked_bytes,
        f32_bytes,
        f32_bytes as f32 / baked_bytes as f32
    );

    assert!(
        nrmse <= 0.05,
        "GATE FAILED: NRMSE={:.4} ({:.2}%) exceeds 5% for small config",
        nrmse,
        nrmse * 100.0
    );
}

/// Config 2 (medium): 8 → [16, 8] → 4, grid=5, order=3
#[test]
fn baked_parity_medium() {
    let input_dim = 8;
    let hidden = vec![16, 8];
    let output_dim = 4;
    let grid_size = 5;
    let order = 3;

    let network = make_network(input_dim, hidden, output_dim, grid_size, order, 99);

    let cal = random_inputs_in_range(256, input_dim, 111, -0.9, 0.9);
    let test = random_inputs_in_range(2000, input_dim, 222, -0.9, 0.9);

    let baked = BakedModel::from_network(&network, Some(&cal));
    let f32_bytes = f32_weight_bytes(&network);
    let baked_bytes = baked.size_bytes();

    let (nrmse, worst_case) = measure_accuracy(&network, &baked, &test, input_dim, output_dim);

    println!(
        "\n[baked_parity_medium] config=8→[16,8]→4, grid=5, order=3"
    );
    println!(
        "  NRMSE (aggregate L2 relative error):          {:.4} ({:.2}%)",
        nrmse,
        nrmse * 100.0
    );
    println!(
        "  Worst-case on significant outputs (tau=0.1σ): {:.4} ({:.2}%)",
        worst_case,
        worst_case * 100.0
    );
    println!(
        "  Size: baked={} B  f32_weights={} B  ratio={:.2}x",
        baked_bytes,
        f32_bytes,
        f32_bytes as f32 / baked_bytes as f32
    );

    assert!(
        nrmse <= 0.10,
        "GATE FAILED: NRMSE={:.4} ({:.2}%) exceeds 10% for medium config",
        nrmse,
        nrmse * 100.0
    );
}

/// Config 3 (single-layer): 4 → 2, grid=5, order=3 (exercises no inter-layer path)
#[test]
fn baked_parity_single_layer() {
    let input_dim = 4;
    let hidden: Vec<usize> = vec![];
    let output_dim = 2;
    let grid_size = 5;
    let order = 3;

    let network = make_network(input_dim, hidden, output_dim, grid_size, order, 77);

    let cal = random_inputs_in_range(256, input_dim, 333, -0.9, 0.9);
    let test = random_inputs_in_range(2000, input_dim, 444, -0.9, 0.9);

    let baked = BakedModel::from_network(&network, Some(&cal));

    let (nrmse, worst_case) = measure_accuracy(&network, &baked, &test, input_dim, output_dim);

    println!(
        "\n[baked_parity_single_layer] config=4→2, grid=5, order=3"
    );
    println!(
        "  NRMSE (aggregate L2 relative error):          {:.4} ({:.2}%)",
        nrmse,
        nrmse * 100.0
    );
    println!(
        "  Worst-case on significant outputs (tau=0.1σ): {:.4} ({:.2}%)",
        worst_case,
        worst_case * 100.0
    );

    assert!(
        nrmse <= 0.05,
        "GATE FAILED: NRMSE={:.4} ({:.2}%) exceeds 5% for single-layer config",
        nrmse,
        nrmse * 100.0
    );
}

/// End-to-end baked accuracy across EVERY supported spline order (2..=5).
///
/// Why this exists: until now every baked test, bench and example hard-coded
/// `order = 3`, so the order-4 and order-5 code paths had zero end-to-end
/// coverage — and a Q-scale defect in their basis polynomials (max abs error
/// 0.208 and 0.775, end-to-end NRMSE 13.98% and 89.30%) survived undetected.
/// `make_network` already took `order` as a parameter; nothing used it.
///
/// This walks the whole supported range through the real pipeline: per-channel
/// weight quantization, the `local_basis_size` = 3..6 weight layout, the
/// inter-layer requant, and (under `serde`) the versioned round-trip.
#[test]
fn baked_parity_all_orders() {
    let input_dim = 8;
    let output_dim = 4;
    let grid_size = 5;

    let mut failures = Vec::new();
    for order in 2..=5usize {
        let network = make_network(input_dim, vec![16, 8], output_dim, grid_size, order, 4242);

        let cal = random_inputs_in_range(256, input_dim, 111, -0.9, 0.9);
        let test = random_inputs_in_range(2000, input_dim, 222, -0.9, 0.9);

        let baked = BakedModel::from_network(&network, Some(&cal));
        let (nrmse, worst) = measure_accuracy(&network, &baked, &test, input_dim, output_dim);

        println!(
            "[baked_parity_all_orders] order={order}: NRMSE={:.4} ({:.2}%), worst(0.1s)={:.4}",
            nrmse,
            nrmse * 100.0,
            worst
        );

        // Same 10% gate as the medium config, which shares this shape.
        if nrmse > 0.10 {
            failures.push(format!("order={order}: NRMSE={nrmse:.4} exceeds 10%"));
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("; "));
}
