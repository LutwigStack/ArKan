//! Deployment benchmark: BakedModel (int8) vs f32 KanNetwork at batch=1.
//!
//! Measures:
//!   - Latency: `BakedModel::forward` vs `KanNetwork::forward_single` (Criterion, warmup + median)
//!   - Size:    `BakedModel::size_bytes()` vs f32 weight bytes (params × 4) — printed once
//!
//! Note: baked int8 is NOT expected to beat f32 at batch=1 on modern CPUs without
//! SIMD-optimized integer kernels. The win is in model size (deployment footprint),
//! not batch-1 latency. SIMD acceleration is a planned future epic.

use arkan::{BakedModel, KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ---------------------------------------------------------------------------
// Helpers
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

/// Fast deterministic PRNG (xorshift64) — no `rand` dep needed in benches.
fn random_inputs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0xdeadbeef_cafebabe;
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..(n * dim) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let fv = (state as f64 / u64::MAX as f64) as f32 * 1.8 - 0.9;
        out.push(fv);
    }
    out
}

/// Total bytes for f32 weights + biases across all KanNetwork layers.
fn f32_weight_bytes(network: &KanNetwork) -> usize {
    network
        .layers
        .iter()
        .map(|l| (l.weights.len() + l.bias.len()) * 4)
        .sum()
}

/// Print a one-shot size comparison (not inside the timed criterion loop).
fn print_size_comparison(label: &str, baked: &BakedModel, network: &KanNetwork) {
    let baked_bytes = baked.size_bytes();
    let f32_bytes = f32_weight_bytes(network);
    let ratio = f32_bytes as f32 / baked_bytes as f32;
    println!("[size] {label}: baked={baked_bytes} B  f32={f32_bytes} B  compression={ratio:.2}x");
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

/// Small config: 4 → [8] → 2, grid=5, order=3
fn bench_small(c: &mut Criterion) {
    let input_dim = 4;
    let output_dim = 2;
    let network = make_network(input_dim, vec![8], output_dim, 5, 3, 42);

    let cal = random_inputs(256, input_dim, 1234);
    let baked = BakedModel::from_network(&network, Some(&cal));

    // One-shot size comparison (printed once, before criterion loop)
    print_size_comparison("small 4→[8]→2 grid5 order3", &baked, &network);

    let input = random_inputs(1, input_dim, 9999);
    let mut f32_out = vec![0.0f32; output_dim];
    let mut baked_out = vec![0.0f32; output_dim];
    let mut workspace = network.create_workspace(1);

    let mut group = c.benchmark_group("baked_batch1_small");

    group.bench_function("f32_forward_single", |b| {
        b.iter(|| {
            network.forward_single(black_box(&input), black_box(&mut f32_out), &mut workspace);
        });
    });

    group.bench_function("baked_int8_forward", |b| {
        b.iter(|| {
            baked.forward(black_box(&input), black_box(&mut baked_out));
        });
    });

    group.finish();
}

/// Medium config: 8 → [16, 8] → 4, grid=5, order=3
fn bench_medium(c: &mut Criterion) {
    let input_dim = 8;
    let output_dim = 4;
    let network = make_network(input_dim, vec![16, 8], output_dim, 5, 3, 99);

    let cal = random_inputs(256, input_dim, 111);
    let baked = BakedModel::from_network(&network, Some(&cal));

    print_size_comparison("medium 8→[16,8]→4 grid5 order3", &baked, &network);

    let input = random_inputs(1, input_dim, 8888);
    let mut f32_out = vec![0.0f32; output_dim];
    let mut baked_out = vec![0.0f32; output_dim];
    let mut workspace = network.create_workspace(1);

    let mut group = c.benchmark_group("baked_batch1_medium");

    group.bench_function("f32_forward_single", |b| {
        b.iter(|| {
            network.forward_single(black_box(&input), black_box(&mut f32_out), &mut workspace);
        });
    });

    group.bench_function("baked_int8_forward", |b| {
        b.iter(|| {
            baked.forward(black_box(&input), black_box(&mut baked_out));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_small, bench_medium);
criterion_main!(benches);
