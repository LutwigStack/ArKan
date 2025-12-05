//! Single-sample latency benchmarks - critical for real-time poker.
//!
//! Tests:
//! - forward_single vs forward_batch(batch=1)
//! - Cold cache vs warm cache effects
//! - Latency percentiles (p50, p99, p999)

use arkan::{KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

fn make_input_single(dim: usize, grid_range: (f32, f32), seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

fn make_inputs_batch(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

/// Compare forward_single vs forward_batch(batch=1)
fn bench_single_vs_batch1(c: &mut Criterion) {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());

    let input_single = make_input_single(config.input_dim, config.grid_range, 42);
    let input_batch1 = make_inputs_batch(config.input_dim, config.grid_range, 1, 42);
    let mut output_single = vec![0.0f32; config.output_dim];
    let mut output_batch1 = vec![0.0f32; config.output_dim];
    let mut workspace = network.create_workspace(1);

    let mut group = c.benchmark_group("single_sample_latency");
    group.throughput(Throughput::Elements(config.input_dim as u64));

    // forward_single API
    group.bench_function("forward_single", |b| {
        b.iter(|| {
            network.forward_single(
                black_box(&input_single),
                black_box(&mut output_single),
                &mut workspace,
            );
        });
    });

    // forward_batch with batch=1
    group.bench_function("forward_batch_1", |b| {
        b.iter(|| {
            network.forward_batch(
                black_box(&input_batch1),
                black_box(&mut output_batch1),
                &mut workspace,
            );
        });
    });

    group.finish();
}

/// Latency distribution analysis (manual measurement for percentiles)
fn bench_latency_distribution(c: &mut Criterion) {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());

    let input = make_input_single(config.input_dim, config.grid_range, 42);
    let mut output = vec![0.0f32; config.output_dim];
    let mut workspace = network.create_workspace(1);

    let iterations = 10000;
    let mut latencies: Vec<f64> = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..1000 {
        network.forward_single(&input, &mut output, &mut workspace);
    }

    // Measure
    for _ in 0..iterations {
        let start = Instant::now();
        network.forward_single(&input, &mut output, &mut workspace);
        latencies.push(start.elapsed().as_nanos() as f64);
    }

    // Sort for percentiles
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = latencies[iterations / 2];
    let p90 = latencies[iterations * 90 / 100];
    let p99 = latencies[iterations * 99 / 100];
    let p999 = latencies[iterations * 999 / 1000];
    let min = latencies[0];
    let max = latencies[iterations - 1];
    let avg: f64 = latencies.iter().sum::<f64>() / iterations as f64;

    println!(
        "\n=== Latency Distribution (forward_single, {} samples) ===",
        iterations
    );
    println!("Min:  {:.2} µs", min / 1000.0);
    println!("Avg:  {:.2} µs", avg / 1000.0);
    println!("P50:  {:.2} µs", p50 / 1000.0);
    println!("P90:  {:.2} µs", p90 / 1000.0);
    println!("P99:  {:.2} µs", p99 / 1000.0);
    println!("P999: {:.2} µs", p999 / 1000.0);
    println!("Max:  {:.2} µs", max / 1000.0);
    println!(
        "Throughput: {:.0} inferences/sec (at P50)",
        1_000_000_000.0 / p50
    );

    // Criterion benchmark for comparison
    let mut group = c.benchmark_group("latency_distribution");
    group.bench_function("forward_single", |b| {
        b.iter(|| {
            network.forward_single(black_box(&input), black_box(&mut output), &mut workspace);
        });
    });
    group.finish();
}

/// Cold cache vs warm cache comparison
fn bench_cache_effects(c: &mut Criterion) {
    let config = KanConfig::preset();

    let mut group = c.benchmark_group("cache_effects");

    // Warm cache: same network, same input
    {
        let network = KanNetwork::new(config.clone());
        let input = make_input_single(config.input_dim, config.grid_range, 42);
        let mut output = vec![0.0f32; config.output_dim];
        let mut workspace = network.create_workspace(1);

        // Warmup
        for _ in 0..100 {
            network.forward_single(&input, &mut output, &mut workspace);
        }

        group.bench_function("warm_cache", |b| {
            b.iter(|| {
                network.forward_single(black_box(&input), black_box(&mut output), &mut workspace);
            });
        });
    }

    // Varying inputs (partial cache pollution)
    {
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);
        let inputs: Vec<Vec<f32>> = (0..100)
            .map(|i| make_input_single(config.input_dim, config.grid_range, i as u64))
            .collect();
        let mut output = vec![0.0f32; config.output_dim];
        let mut idx = 0_usize;

        group.bench_function("varying_inputs", |b| {
            b.iter(|| {
                let input = &inputs[idx % inputs.len()];
                network.forward_single(black_box(input), black_box(&mut output), &mut workspace);
                idx += 1;
            });
        });
    }

    group.finish();
}

/// Simulate poker solver workload: alternating inference and other work
fn bench_poker_workload(c: &mut Criterion) {
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());

    let input = make_input_single(config.input_dim, config.grid_range, 42);
    let mut output = vec![0.0f32; config.output_dim];
    let mut workspace = network.create_workspace(1);

    let mut group = c.benchmark_group("poker_workload");

    // Pure inference loop
    group.bench_function("pure_inference", |b| {
        b.iter(|| {
            network.forward_single(black_box(&input), black_box(&mut output), &mut workspace);
        });
    });

    // Inference + light processing (simulating game state update)
    group.bench_function("inference_with_processing", |b| {
        b.iter(|| {
            network.forward_single(black_box(&input), black_box(&mut output), &mut workspace);
            // Simulate light processing
            let sum: f32 = output.iter().sum();
            black_box(sum);
        });
    });

    // Inference + softmax (common post-processing)
    group.bench_function("inference_with_softmax", |b| {
        b.iter(|| {
            network.forward_single(black_box(&input), black_box(&mut output), &mut workspace);
            // Manual softmax on first 8 outputs (strategy)
            let max = output[..8]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = output[..8].iter().map(|x| (x - max).exp()).sum();
            for out in output.iter_mut().take(8) {
                *out = (*out - max).exp() / sum;
            }
            black_box(&output);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_vs_batch1,
    bench_latency_distribution,
    bench_cache_effects,
    bench_poker_workload,
);
criterion_main!(benches);
