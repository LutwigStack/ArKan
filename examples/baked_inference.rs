//! Baked (int8) Inference Example
//!
//! Demonstrates how to quantize a trained KAN network into a [`BakedModel`]
//! and run fixed-point inference.
//!
//! # What it shows
//!
//! 1. Building and (optionally training) a small [`KanNetwork`].
//! 2. Generating a calibration set and calling [`BakedModel::from_network`].
//! 3. Running [`BakedModel::forward`] on several inputs.
//! 4. Comparing baked output against f32 output.
//! 5. Reporting model size: baked bytes vs equivalent f32 weight bytes.
//!
//! # Run
//!
//! ```bash
//! cargo run --example baked_inference
//! ```
//!
//! Serialization round-trip is shown when the `serde` feature is enabled:
//!
//! ```bash
//! cargo run --example baked_inference --features serde
//! ```

use arkan::{BakedModel, KanConfig, KanNetwork};

/// Cheap pseudo-random number in [-range, +range] seeded by index.
fn pseudo_random(idx: u64, range: f32) -> f32 {
    // lcg-style hash — good enough for examples
    let h = idx
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let frac = (h >> 33) as f32 / (u32::MAX as f32); // [0, 1)
    frac * 2.0 * range - range
}

fn generate_inputs(n_samples: usize, input_dim: usize, seed: u64) -> Vec<f32> {
    (0..n_samples * input_dim)
        .map(|i| pseudo_random(seed ^ i as u64, 0.9))
        .collect()
}

fn main() {
    println!("=== ArKan Baked (int8) Inference Example ===\n");

    // ----------------------------------------------------------------
    // 1. Build a small KAN network
    // ----------------------------------------------------------------
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![8],
        grid_size: 5,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        multithreading_threshold: 128,
        simd_width: 8,
        init_seed: Some(42),
    };

    let mut network = KanNetwork::new(config.clone());

    println!("Network architecture: {}→{:?}→{}", config.input_dim, config.hidden_dims, config.output_dim);
    println!("Grid size: {}, Spline order: {}\n", config.grid_size, config.spline_order);

    // ----------------------------------------------------------------
    // 2. Brief training (a few steps so weights are non-trivial)
    //    This is optional — even random-init weights demonstrate baking.
    // ----------------------------------------------------------------
    let n_train = 200;
    let train_inputs = generate_inputs(n_train, config.input_dim, 1234);
    // Regression target: sin(x0 + x1) + cos(x2 * x3)
    let train_targets: Vec<f32> = (0..n_train)
        .flat_map(|s| {
            let x = &train_inputs[s * config.input_dim..(s + 1) * config.input_dim];
            let y0 = (x[0] + x[1]).sin();
            let y1 = (x[2] * x[3]).cos();
            [y0, y1]
        })
        .collect();

    let mut workspace = network.create_workspace(n_train);
    for _ in 0..10 {
        let _loss = network.train_step(&train_inputs, &train_targets, None, 5e-3, &mut workspace);
    }
    println!("Trained for 10 steps (LR=5e-3, MSE target).\n");

    // ----------------------------------------------------------------
    // 3. Generate calibration set and bake
    // ----------------------------------------------------------------
    let n_calib = 512;
    let calib_inputs = generate_inputs(n_calib, config.input_dim, 9999);

    let baked = BakedModel::from_network(&network, Some(&calib_inputs));
    println!("BakedModel created (calibrated on {} samples).", n_calib);

    // ----------------------------------------------------------------
    // 4. Size comparison
    // ----------------------------------------------------------------
    let baked_bytes = baked.size_bytes();

    // Count f32 parameters: weights + biases across all layers
    let f32_weight_bytes: usize = network
        .layers
        .iter()
        .map(|l| (l.weights.len() + l.bias.len()) * 4)
        .sum();

    let ratio = f32_weight_bytes as f64 / baked_bytes as f64;

    println!("\n--- Model size ---");
    println!("  f32 weights+biases : {:>7} bytes", f32_weight_bytes);
    println!("  BakedModel (int8)  : {:>7} bytes", baked_bytes);
    println!("  Compression ratio  : {:.2}x smaller", ratio);

    // ----------------------------------------------------------------
    // 5. Inference comparison on a few sample inputs
    // ----------------------------------------------------------------
    println!("\n--- Sample inference (baked vs f32) ---");
    println!("{:<6}  {:<22}  {:<22}  {}", "sample", "baked out", "f32 out", "abs_err");

    let n_test = 5;
    let test_inputs = generate_inputs(n_test, config.input_dim, 5678);
    let mut ws1 = network.create_workspace(1);

    let mut total_err: f64 = 0.0;

    for s in 0..n_test {
        let inp = &test_inputs[s * config.input_dim..(s + 1) * config.input_dim];

        let mut baked_out = vec![0.0f32; config.output_dim];
        baked.forward(inp, &mut baked_out);

        let mut f32_out = vec![0.0f32; config.output_dim];
        network.forward_single(inp, &mut f32_out, &mut ws1);

        let err: f32 = baked_out
            .iter()
            .zip(&f32_out)
            .map(|(b, f)| (b - f).abs())
            .sum::<f32>()
            / config.output_dim as f32;

        total_err += err as f64;

        println!(
            "  #{:<4}  [{:+.4}, {:+.4}]  [{:+.4}, {:+.4}]  {:.5}",
            s, baked_out[0], baked_out[1], f32_out[0], f32_out[1], err
        );
    }

    println!("\nMean abs error across {} samples: {:.5}", n_test, total_err / n_test as f64);
    println!("(NRMSE ~0.6–1.3% for calibrated nets; suitable for ranking/argmax)");

    // ----------------------------------------------------------------
    // 6. Optional: serialization round-trip (requires --features serde)
    // ----------------------------------------------------------------
    #[cfg(feature = "serde")]
    {
        println!("\n--- Serialization round-trip (serde feature) ---");
        let bytes = baked.to_bytes().expect("to_bytes failed");
        println!("  Serialized to {} bytes (header: 16 bytes magic+version, body: {} bytes)",
            bytes.len(), bytes.len() - 16);

        let baked2 = BakedModel::from_bytes(&bytes).expect("from_bytes failed");

        let inp = &test_inputs[..config.input_dim];
        let mut out_orig = vec![0.0f32; config.output_dim];
        let mut out_loaded = vec![0.0f32; config.output_dim];
        baked.forward(inp, &mut out_orig);
        baked2.forward(inp, &mut out_loaded);

        let identical = out_orig
            .iter()
            .zip(&out_loaded)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!("  Round-trip output identical: {}", identical);
    }

    println!("\nDone.");
}
