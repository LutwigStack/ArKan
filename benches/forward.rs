//! Forward pass and training benchmarks.
//!
//! # Methodology
//!
//! Seed 42 fixes initial weights across cases. Each timed training step uses a
//! fresh clone; cloning and warming reusable scratch buffers are outside timing.
//! Inference reuses its workspace. Throughput counts batch * input_dim input
//! elements, not samples, FLOPS, or measured memory traffic.

use arkan::{KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

fn bench_forward(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("forward_batch");

    for &batch in &batch_sizes {
        // Fresh network per batch size: ensures identical starting weights.
        // Not warmup—prevents weight-dependent timing artifacts.
        let network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        // Workspace created once, reused across all iterations (zero-alloc after first).
        let mut workspace = network.create_workspace(batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &_batch| {
            b.iter(|| {
                network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
            });
        });
    }

    group.finish();
}

fn bench_train_step(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("train_step");

    for &batch in &batch_sizes {
        // Fresh network per batch size: identical initial weights.
        // Every training routine receives a fresh clone outside timing.
        let network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 123);
        let targets = make_inputs(config.output_dim, config.grid_range, batch, 321);
        // Warm scratch buffers on a discarded clone before the timed step.
        let mut workspace = network.create_workspace(batch);
        network
            .clone()
            .train_step(&inputs, &targets, None, 0.001, &mut workspace);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &_batch| {
            b.iter_batched_ref(
                || network.clone(),
                |network| {
                    network.train_step(
                        black_box(&inputs),
                        black_box(&targets),
                        None,
                        0.001,
                        &mut workspace,
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark comparing try_* methods overhead vs panicking versions.
///
/// This measures the cost of error checking in the try_* API.
/// Expected: near-zero overhead since checks use cheap comparisons.
fn bench_try_overhead(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let batch = 64;

    let network = KanNetwork::new(config.clone());
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let mut outputs = vec![0.0f32; batch * config.output_dim];
    let mut workspace = network.create_workspace(batch);

    let mut group = c.benchmark_group("try_overhead");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));

    // Panicking version
    group.bench_function("forward_batch", |b| {
        b.iter(|| {
            network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
        });
    });

    // Fallible version
    group.bench_function("try_forward_batch", |b| {
        b.iter(|| {
            let _ = network.try_forward_batch(
                black_box(&inputs),
                black_box(&mut outputs),
                &mut workspace,
            );
        });
    });

    group.finish();
}

/// Benchmark workspace creation: panicking vs fallible.
fn bench_workspace_creation(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let network = KanNetwork::new(config);

    let mut group = c.benchmark_group("workspace_creation");

    group.bench_function("create_workspace", |b| {
        b.iter(|| {
            black_box(network.create_workspace(64));
        });
    });

    group.bench_function("try_create_workspace", |b| {
        b.iter(|| {
            black_box(network.try_create_workspace(64).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_forward,
    bench_train_step,
    bench_try_overhead,
    bench_workspace_creation
);
criterion_main!(benches);
