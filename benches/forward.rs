//! Forward pass and training benchmarks.
//!
//! # Methodology
//!
//! **Network recreation per batch size**: A fresh `KanNetwork` is created for each
//! batch size to ensure identical initial weights. This is NOT warmup—it ensures
//! fair comparison across batch sizes since weight values affect computation time
//! (e.g., denormalized floats).
//!
//! **Workspace reuse**: The workspace is created once per batch size and reused
//! across all iterations. After the first iteration, all subsequent calls are
//! zero-allocation, measuring true steady-state performance.
//!
//! **Throughput metric**: `Elements` = `batch_size * input_dim`, representing
//! total floating-point inputs processed, not samples.

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
    let config = KanConfig::preset();

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
    let config = KanConfig::preset();

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("train_step");

    for &batch in &batch_sizes {
        // Fresh network per batch size: identical initial weights.
        // train_step modifies weights, but Criterion runs many iterations,
        // so steady-state behavior is measured (not first-iteration warmup).
        let mut network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 123);
        let targets = make_inputs(config.output_dim, config.grid_range, batch, 321);
        // Workspace reused across iterations for zero-alloc training.
        let mut workspace = network.create_workspace(batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &_batch| {
            b.iter(|| {
                network.train_step(
                    black_box(&inputs),
                    black_box(&targets),
                    None,
                    0.001,
                    &mut workspace,
                );
            });
        });
    }

    group.finish();
}

/// Benchmark comparing try_* methods overhead vs panicking versions.
///
/// This measures the cost of error checking in the try_* API.
/// Expected: near-zero overhead since checks use cheap comparisons.
fn bench_try_overhead(c: &mut Criterion) {
    let config = KanConfig::preset();
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
            let _ = network.try_forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
        });
    });

    group.finish();
}

/// Benchmark workspace creation: panicking vs fallible.
fn bench_workspace_creation(c: &mut Criterion) {
    let config = KanConfig::preset();
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

criterion_group!(benches, bench_forward, bench_train_step, bench_try_overhead, bench_workspace_creation);
criterion_main!(benches);
