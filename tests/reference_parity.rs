//! Layered differential test: ArKan CPU training vs PyTorch reference.
//!
//! This test loads a reference JSON produced by `scripts/export_training_reference.py`
//! which contains:
//!   - Identical initial weights (copied from PyTorch random init)
//!   - A fixed toy dataset
//!   - PyTorch autograd outputs and gradients at step 0
//!   - Loss trajectory over N SGD steps
//!   - Final weights after N steps
//!
//! Three parity layers (each must pass before the next is meaningful):
//!   1. Forward parity: ArKan output matches PyTorch step0_forward (tol 1e-5)
//!   2. Gradient parity: ArKan weight/bias grads match step0_grads (tol 1e-4)
//!   3. Trajectory parity: per-step MSE loss matches loss_trajectory (rel tol 1e-3),
//!      and final weights match final_weights (tol 1e-3)
//!
//! HONESTY RULES:
//!   - If Layer 1 fails: STOP — weight bridge or forward bug. Do not proceed.
//!   - If Layer 2 fails: STOP — gradient bug. Do not proceed.
//!   - If Layer 3 diverges: REPORT divergence step and localization. Do not loosen tolerance.
//!
//! Run with: cargo test --test reference_parity -- --nocapture

use arkan::{KanConfig, KanNetwork};
use serde::Deserialize;
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// JSON schema matching export_training_reference.py output
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct RefConfig {
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    grid_size: usize,
    spline_order: usize,
    grid_range: (f32, f32),
    #[allow(dead_code)]
    global_basis_size: usize,
    #[allow(dead_code)]
    dims: Vec<usize>,
}

#[derive(Deserialize, Debug)]
struct LayerWeights {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct Dataset {
    inputs: Vec<f32>,
    targets: Vec<f32>,
    n_samples: usize,
}

#[derive(Deserialize, Debug)]
struct OptimizerConfig {
    #[serde(rename = "type")]
    opt_type: String,
    lr: f32,
    momentum: f32,
}

#[derive(Deserialize, Debug)]
struct Reference {
    config: RefConfig,
    optimizer: OptimizerConfig,
    dataset: Dataset,
    init_weights: Vec<LayerWeights>,
    step0_forward: Vec<f32>,
    step0_grads: Vec<LayerWeights>,
    loss_trajectory: Vec<f32>,
    final_weights: Vec<LayerWeights>,
}

// ---------------------------------------------------------------------------
// Helper: load reference JSON
// ---------------------------------------------------------------------------

fn load_reference() -> Reference {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest_dir)
        .join("tests")
        .join("reference_data")
        .join("training_ref_sgd_order3.json");

    let data = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Could not read reference file {:?}: {}", path, e));

    serde_json::from_str(&data).unwrap_or_else(|e| panic!("Could not parse reference JSON: {}", e))
}

// ---------------------------------------------------------------------------
// Helper: build ArKan network and inject reference initial weights
// ---------------------------------------------------------------------------

fn build_network_with_ref_weights(ref_data: &Reference) -> KanNetwork {
    let cfg = &ref_data.config;

    // mean=0, std=1 → normalization is identity for all inputs
    let config = KanConfig {
        input_dim: cfg.input_dim,
        output_dim: cfg.output_dim,
        hidden_dims: cfg.hidden_dims.clone(),
        grid_size: cfg.grid_size,
        spline_order: cfg.spline_order,
        grid_range: cfg.grid_range,
        input_mean: vec![0.0f32; cfg.input_dim],
        input_std: vec![1.0f32; cfg.input_dim],
        multithreading_threshold: 128, // small batch → sequential backward
        simd_width: 8,
        init_seed: None,
    };

    let mut network = KanNetwork::new(config);

    // Verify layer count and weight counts before injection
    assert_eq!(
        network.layers.len(),
        ref_data.init_weights.len(),
        "Layer count mismatch: ArKan={}, ref={}",
        network.layers.len(),
        ref_data.init_weights.len()
    );

    // Inject weights.
    //
    // Weight bridge mapping:
    //   PyTorch layout in export: flat weights[out_dim * in_dim * global_basis]
    //     stored as weights[out][in][k] for k in 0..global_basis (row-major)
    //   ArKan weight_index(j, i, k) = (j * in_dim + i) * global_basis_size + k
    //
    // These are byte-for-byte the same flat ordering — no permutation needed.
    // The Python export flattens weights[out][in][k] in C order, which is
    // exactly (out * in_dim + in) * global_basis + k = ArKan's weight_index.
    for (layer_idx, ref_layer) in ref_data.init_weights.iter().enumerate() {
        let layer = &mut network.layers[layer_idx];
        let expected_w = layer.weights.len();
        let expected_b = layer.bias.len();

        assert_eq!(
            ref_layer.weights.len(),
            expected_w,
            "Layer {}: weight count mismatch: ref={}, arkan={}",
            layer_idx,
            ref_layer.weights.len(),
            expected_w
        );
        assert_eq!(
            ref_layer.bias.len(),
            expected_b,
            "Layer {}: bias count mismatch: ref={}, arkan={}",
            layer_idx,
            ref_layer.bias.len(),
            expected_b
        );

        layer.weights.copy_from_slice(&ref_layer.weights);
        layer.bias.copy_from_slice(&ref_layer.bias);
    }

    network
}

// ---------------------------------------------------------------------------
// Helper: max absolute error between two slices
// ---------------------------------------------------------------------------

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Helper: run ArKan forward on entire dataset, return flat output
// ---------------------------------------------------------------------------

fn arkan_forward(network: &KanNetwork, dataset: &Dataset) -> Vec<f32> {
    let n = dataset.n_samples;
    let out_dim = network.config.output_dim;
    let mut workspace = network.create_workspace(n);

    let mut outputs = vec![0.0f32; n * out_dim];
    network.forward_batch(&dataset.inputs, &mut outputs, &mut workspace);
    outputs
}

// ---------------------------------------------------------------------------
// LAYER 1: Forward parity
// ---------------------------------------------------------------------------

#[test]
fn layer1_forward_parity() {
    let ref_data = load_reference();
    let network = build_network_with_ref_weights(&ref_data);

    let arkan_out = arkan_forward(&network, &ref_data.dataset);

    let max_err = max_abs_err(&arkan_out, &ref_data.step0_forward);
    println!(
        "[LAYER 1] Forward parity: max_abs_err = {:.2e}  (tol 1e-5)",
        max_err
    );

    // If this fails, the weight bridge or forward path has a bug.
    // Do NOT proceed to layer 2/3 until this passes.
    assert!(
        max_err < 1e-5,
        "[LAYER 1 FAIL] Forward parity failed: max_abs_err={:.4e} > 1e-5\n\
         This indicates a weight bridge mapping error or a forward-pass bug.\n\
         ArKan[:5]  = {:?}\n\
         PyTorch[:5] = {:?}",
        max_err,
        &arkan_out[..arkan_out.len().min(5)],
        &ref_data.step0_forward[..ref_data.step0_forward.len().min(5)],
    );
    println!("[LAYER 1] PASS");
}

// ---------------------------------------------------------------------------
// LAYER 2: Gradient parity
// ---------------------------------------------------------------------------

#[test]
fn layer2_gradient_parity() {
    // Layer 1 must have passed — check it inline
    let ref_data = load_reference();
    let network = build_network_with_ref_weights(&ref_data);

    let n = ref_data.dataset.n_samples;
    let out_dim = network.config.output_dim;

    // First verify forward parity (prerequisite)
    {
        let arkan_out = arkan_forward(&network, &ref_data.dataset);
        let fwd_err = max_abs_err(&arkan_out, &ref_data.step0_forward);
        assert!(
            fwd_err < 1e-5,
            "[LAYER 2 PREREQ FAIL] Forward parity not satisfied (err={:.4e}). \
             Fix LAYER 1 before testing gradients.",
            fwd_err
        );
    }

    // Exercise production backward with an independently computed MSE derivative.
    let mut workspace = network.create_workspace(n);

    // Forward with history
    let mut predictions = vec![0.0f32; n * out_dim];
    let pass = network
        .try_forward_for_backward(&ref_data.dataset.inputs, &mut predictions, &mut workspace)
        .unwrap();

    // Compute MSE gradient (same as ArKan's compute_masked_mse_loss_into)
    let mut grad_output = vec![0.0f32; n * out_dim];
    let count = (n * out_dim) as f32;
    for (idx, (pred, target)) in predictions
        .iter()
        .zip(ref_data.dataset.targets.iter())
        .enumerate()
    {
        let diff = pred - target;
        grad_output[idx] = 2.0 * diff / count;
    }

    let gradients = pass.backward(&grad_output).unwrap();
    let weight_grads = gradients.weights;
    let bias_grads = gradients.biases;

    // Compare with PyTorch reference gradients
    let tol = 1e-4;
    let mut all_pass = true;
    let mut max_w_err = 0.0f32;
    let mut max_b_err = 0.0f32;

    for (layer_idx, ref_grad) in ref_data.step0_grads.iter().enumerate() {
        let w_err = max_abs_err(&weight_grads[layer_idx], &ref_grad.weights);
        let b_err = max_abs_err(&bias_grads[layer_idx], &ref_grad.bias);
        max_w_err = max_w_err.max(w_err);
        max_b_err = max_b_err.max(b_err);

        println!(
            "[LAYER 2] Layer {}: weight_grad err={:.2e}, bias_grad err={:.2e}",
            layer_idx, w_err, b_err
        );

        if w_err >= tol {
            eprintln!(
                "[LAYER 2 FAIL] Layer {}: weight grad error {:.4e} >= tol {:.1e}",
                layer_idx, w_err, tol
            );
            eprintln!(
                "  ArKan[:5]   = {:?}",
                &weight_grads[layer_idx][..weight_grads[layer_idx].len().min(5)]
            );
            eprintln!(
                "  PyTorch[:5] = {:?}",
                &ref_grad.weights[..ref_grad.weights.len().min(5)]
            );
            all_pass = false;
        }
        if b_err >= tol {
            eprintln!(
                "[LAYER 2 FAIL] Layer {}: bias grad error {:.4e} >= tol {:.1e}",
                layer_idx, b_err, tol
            );
            all_pass = false;
        }
    }

    println!(
        "[LAYER 2] Gradient parity: max_weight_err={:.2e}, max_bias_err={:.2e}  (tol 1e-4)",
        max_w_err, max_b_err
    );

    assert!(
        all_pass,
        "[LAYER 2 FAIL] Gradient parity failed. This indicates a backward-pass bug in ArKan. \
         Forward parity was OK, so the error is in gradient computation or bias-correction."
    );
    println!("[LAYER 2] PASS");
}

// ---------------------------------------------------------------------------
// LAYER 3: Training trajectory parity
// ---------------------------------------------------------------------------

#[test]
fn layer3_trajectory_parity() {
    let ref_data = load_reference();

    // Verify prerequisites (layer 1 and 2 should have passed)
    {
        let network_check = build_network_with_ref_weights(&ref_data);
        let arkan_out = arkan_forward(&network_check, &ref_data.dataset);
        let fwd_err = max_abs_err(&arkan_out, &ref_data.step0_forward);
        assert!(
            fwd_err < 1e-5,
            "[LAYER 3 PREREQ FAIL] Forward parity not satisfied (err={:.4e}). \
             Fix LAYER 1 before testing trajectory.",
            fwd_err
        );
    }

    let mut network = build_network_with_ref_weights(&ref_data);
    let n = ref_data.dataset.n_samples;
    let n_steps = ref_data.loss_trajectory.len();
    let lr = ref_data.optimizer.lr;

    assert_eq!(
        ref_data.optimizer.opt_type, "sgd",
        "Only SGD supported in this test"
    );
    assert_eq!(
        ref_data.optimizer.momentum, 0.0,
        "Only momentum=0 (plain SGD) supported"
    );

    let mut workspace = network.create_workspace(n);

    let rel_tol = 1e-3f32; // relative tolerance for loss trajectory
    let mut diverged_at: Option<usize> = None;

    for step in 0..n_steps {
        let arkan_loss = network.train_step(
            &ref_data.dataset.inputs,
            &ref_data.dataset.targets,
            None,
            lr,
            &mut workspace,
        );
        let ref_loss = ref_data.loss_trajectory[step];

        // Relative error: |arkan - ref| / max(|ref|, 1e-8)
        let rel_err = (arkan_loss - ref_loss).abs() / (ref_loss.abs().max(1e-8));

        if step < 5 || step % 10 == 0 {
            println!(
                "[LAYER 3] step {:3}: arkan={:.6}, ref={:.6}, rel_err={:.2e}",
                step, arkan_loss, ref_loss, rel_err
            );
        }

        if rel_err >= rel_tol && diverged_at.is_none() {
            diverged_at = Some(step);
            eprintln!(
                "[LAYER 3] DIVERGENCE at step {}: arkan_loss={:.6}, ref_loss={:.6}, rel_err={:.4e}",
                step, arkan_loss, ref_loss, rel_err
            );
        }
    }

    // If diverged, localize and report — do NOT loosen tolerance
    if let Some(div_step) = diverged_at {
        // Compare final weights to see how far apart they are
        let mut max_w_err = 0.0f32;
        for (layer_idx, ref_final) in ref_data.final_weights.iter().enumerate() {
            let w_err = max_abs_err(&network.layers[layer_idx].weights, &ref_final.weights);
            let b_err = max_abs_err(&network.layers[layer_idx].bias, &ref_final.bias);
            max_w_err = max_w_err.max(w_err).max(b_err);
            println!(
                "[LAYER 3] Final weight diff layer {}: w_err={:.4e}, b_err={:.4e}",
                layer_idx, w_err, b_err
            );
        }

        // Localization report:
        // Layer 1 (forward) passed → spline forward is correct.
        // Layer 2 (gradient) passed → gradient computation is correct at step 0.
        // Divergence in trajectory → likely optimizer update or accumulated state error.
        panic!(
            "[LAYER 3 FAIL] Training trajectory diverged starting at step {}.\n\
             Layer 1 (forward) and Layer 2 (gradient at step 0) passed, so:\n\
             SUSPECT: optimizer weight update formula or per-step numerical drift.\n\
             Evidence: max final weight diff = {:.4e}\n\
             Divergence starts at step {}, check ArKan's SGD update in network.rs \
             (search 'w -= learning_rate * g') for correctness vs PyTorch's w -= lr * grad.",
            div_step, max_w_err, div_step
        );
    }

    // Check final weights match within tolerance
    let w_tol = 1e-3f32;
    let mut max_final_err = 0.0f32;
    for (layer_idx, ref_final) in ref_data.final_weights.iter().enumerate() {
        let w_err = max_abs_err(&network.layers[layer_idx].weights, &ref_final.weights);
        let b_err = max_abs_err(&network.layers[layer_idx].bias, &ref_final.bias);
        max_final_err = max_final_err.max(w_err).max(b_err);
        println!(
            "[LAYER 3] Final weight diff layer {}: w_err={:.4e}, b_err={:.4e}",
            layer_idx, w_err, b_err
        );
    }

    println!(
        "[LAYER 3] Trajectory parity: all {} steps within rel_tol={:.0e}, \
         max_final_weight_err={:.4e}  (tol {})",
        n_steps, rel_tol, max_final_err, w_tol
    );

    assert!(
        max_final_err < w_tol,
        "[LAYER 3 FAIL] Final weights diverged: max_err={:.4e} > tol={:.1e}",
        max_final_err,
        w_tol
    );

    println!("[LAYER 3] PASS — harness is green, training parity holds.");
}
