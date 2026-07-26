//! `grid_range` applies to EVERY layer, not just the input layer.
//!
//! Only layer 0 receives `input_mean`/`input_std` (see `KanNetwork::new` ->
//! `set_normalization`, src/network.rs). Every hidden layer is built with identity
//! normalization (mean=0, std=1) and there is no running-statistics update anywhere. So a
//! hidden layer's `z` IS the previous layer's raw activation, clamped to the shared
//! `grid_range`. Nothing bounds a KAN layer's output to its own grid range.
//!
//! The practical trap: picking `grid_range` from the range of the *inputs* silently kills
//! the hidden layers. A saturated input has zero derivative, so those features produce a
//! constant output and — since the clamp/gradient fix — a correctly zero gradient. They
//! stop learning.
//!
//! This bit the repo's own `examples/game2048`, which used `grid_range(0.0, 1.0)` reasoning
//! "one-hot values are 0 or 1". These tests pin the measured behaviour so the trap stays
//! visible and the fix stays fixed.

use arkan::{KanConfig, KanNetwork, SPAN_CLAMPED_FLAG};

/// The game2048 shape: 256 one-hot inputs -> [64, 32] -> 4 actions.
fn saturation_per_layer(grid_range: (f32, f32)) -> Vec<f32> {
    let config = KanConfig {
        input_dim: 256,
        output_dim: 4,
        hidden_dims: vec![64, 32],
        grid_size: 5,
        spline_order: 3,
        grid_range,
        input_mean: vec![0.0; 256],
        input_std: vec![1.0; 256],
        multithreading_threshold: 128,
        simd_width: 8,
        init_seed: Some(7),
    };
    let net = KanNetwork::new(config);
    let batch = 64;

    // 16 cells x 16 possible values, one-hot — exactly game2048's encoding.
    let mut inputs = vec![0.0f32; batch * 256];
    let mut s = 0x1234_5678u64;
    for b in 0..batch {
        for cell in 0..16 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            inputs[b * 256 + cell * 16 + (s % 16) as usize] = 1.0;
        }
    }

    let mut ws = net.create_workspace(batch);
    let mut out = vec![0.0f32; batch * 4];
    net.forward_batch_training(&inputs, &mut out, &mut ws);

    net.layers
        .iter()
        .enumerate()
        .map(|(li, layer)| {
            let n = batch * layer.in_dim;
            let clamped = ws.layers_grid_indices[li][..n]
                .iter()
                .filter(|s| *s & SPAN_CLAMPED_FLAG != 0)
                .count();
            100.0 * clamped as f32 / n as f32
        })
        .collect()
}

#[test]
fn grid_range_excluding_negatives_starves_hidden_layers() {
    let sat = saturation_per_layer((0.0, 1.0));
    println!("grid_range=(0,1) saturation per layer: {sat:?}");

    // Layer 0 is fine: the one-hot inputs really are within [0,1].
    assert!(
        sat[0] < 1.0,
        "layer 0 should not saturate on one-hot inputs, got {:.1}%",
        sat[0]
    );

    // The hidden layers are not, because their inputs are raw activations and every
    // negative one collapses onto the lower bound. Measured 43.6% and 48.9%.
    assert!(
        sat[1] > 30.0 && sat[2] > 30.0,
        "expected heavy hidden-layer saturation with a lower bound of 0.0, got {:.1}% and {:.1}%. \
         If this dropped, hidden-layer normalization may have changed — update this test and \
         the guidance in the docs to match.",
        sat[1],
        sat[2]
    );
}

#[test]
fn symmetric_grid_range_keeps_hidden_layers_alive() {
    for range in [(-1.0f32, 1.0f32), (-3.0, 3.0)] {
        let sat = saturation_per_layer(range);
        println!("grid_range={range:?} saturation per layer: {sat:?}");
        for (li, &s) in sat.iter().enumerate() {
            assert!(
                s < 1.0,
                "grid_range={range:?}: layer {li} saturated {s:.1}%, expected ~0%"
            );
        }
    }
}
