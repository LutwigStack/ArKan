//! Backward pass benchmarks — isolate backpropagation from forward pass.
//!
//! This module helps identify training bottlenecks separate from inference.
//!
//! # Methodology
//!
//! **Comparison groups**:
//! - `forward_only`: inference-only forward pass (no history saved)
//! - `forward_training`: forward pass with intermediate value storage
//! - `full_train_step`: complete forward + backward + SGD update
//!
//! By comparing these, you can estimate:
//! - **History overhead**: `forward_training - forward_only`
//! - **Backward overhead**: `full_train_step - forward_training`
//!
//! **Single network instance**: Unlike forward.rs, these benchmarks share one
//! network across batch sizes. This is acceptable because we're comparing
//! relative costs, not absolute timings across batch sizes.
//!
//! **Workspace reuse**: Each batch size gets its own workspace, created once
//! and reused. After first iteration, all calls are zero-allocation.

use arkan::{KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

/// Benchmark forward pass only (for comparison with backward).
/// Measures pure inference without saving history for backward pass.
fn bench_forward_only(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    let batch_sizes = [1_usize, 16, 64, 256];
    let mut group = c.benchmark_group("forward_only");

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
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

/// Benchmark forward_batch_training (stores intermediate values for backward)
fn bench_forward_training(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    let batch_sizes = [1_usize, 16, 64, 256];
    let mut group = c.benchmark_group("forward_training");

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        let mut workspace = network.create_workspace(batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &_batch| {
            b.iter(|| {
                network.forward_batch_training(
                    black_box(&inputs),
                    black_box(&mut outputs),
                    &mut workspace,
                );
            });
        });
    }

    group.finish();
}

/// Benchmark full train_step (forward + backward + SGD update)
/// Compare with forward_only to understand backward overhead
fn bench_full_train_step(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let mut network = KanNetwork::new(config.clone());

    let batch_sizes = [1_usize, 16, 64, 256];
    let mut group = c.benchmark_group("full_train_step");

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_inputs(config.output_dim, config.grid_range, batch, 123);
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

/// Compute backward overhead by comparing train_step vs forward_training
/// backward_time ≈ train_step_time - forward_training_time
fn bench_backward_overhead(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let mut network = KanNetwork::new(config.clone());

    // Fixed batch for detailed analysis
    let batch = 64_usize;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let targets = make_inputs(config.output_dim, config.grid_range, batch, 123);
    let mut outputs = vec![0.0f32; batch * config.output_dim];
    let mut workspace = network.create_workspace(batch);

    let mut group = c.benchmark_group("backward_overhead_batch64");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));

    // Forward inference only
    group.bench_function("1_forward_inference", |b| {
        b.iter(|| {
            network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
        });
    });

    // Forward with training buffers
    group.bench_function("2_forward_training", |b| {
        b.iter(|| {
            network.forward_batch_training(
                black_box(&inputs),
                black_box(&mut outputs),
                &mut workspace,
            );
        });
    });

    // Full train step (forward + backward + update)
    group.bench_function("3_full_train_step", |b| {
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

    group.finish();
}

criterion_group!(
    benches,
    bench_forward_only,
    bench_forward_training,
    bench_full_train_step,
    bench_backward_overhead,
);
criterion_main!(benches);
