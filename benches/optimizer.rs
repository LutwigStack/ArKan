//! Optimizer benchmarks - Adam vs SGD overhead.
//!
//! Tests:
//! - Raw SGD (inline in train_step) vs raw train_step with options
//! - Optimizer initialization cost
//! - Memory overhead of optimizer state
//!
//! Note: Full Adam/SGD optimizer step benchmarks require manual gradient
//! extraction which adds overhead. These benchmarks focus on the train_step
//! variants available in the API.

use arkan::network::TrainOptions;
use arkan::{Adam, AdamConfig, KanConfig, KanNetwork, SGDConfig, SGD};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

/// Benchmark raw train_step (inline SGD, no momentum)
fn bench_raw_train_step(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let network = KanNetwork::new(config.clone());

    let batch_sizes = [1_usize, 16, 64, 256];
    let mut group = c.benchmark_group("raw_train_step");

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_inputs(config.output_dim, config.grid_range, batch, 123);
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

/// Benchmark train_step with gradient clipping
fn bench_train_step_clipping(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let network = KanNetwork::new(config.clone());
    let batch = 64;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let targets = make_inputs(config.output_dim, config.grid_range, batch, 123);
    let mut workspace = network.create_workspace(batch);
    network
        .clone()
        .train_step(&inputs, &targets, None, 0.001, &mut workspace);
    let norm = workspace
        .weight_grads
        .iter()
        .chain(&workspace.bias_grads)
        .flat_map(|g| g.as_slice())
        .map(|g| (*g as f64).powi(2))
        .sum::<f64>()
        .sqrt() as f32;
    assert!(
        norm.is_finite() && norm > 0.0,
        "clipping workload needs finite nonzero gradients"
    );
    println!("Seeded CPU clipping workload gradient norm: {norm}");
    let cases = [
        ("no_options", None, 0.0),
        ("clip_active_half_norm", Some(norm * 0.5), 0.0),
        ("clip_inactive_double_norm", Some(norm * 2.0), 0.0),
        ("weight_decay_0.01", None, 0.01),
        ("active_clip_and_decay", Some(norm * 0.5), 0.01),
    ];
    let mut group = c.benchmark_group("train_options_batch64");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
    for (name, max_grad_norm, weight_decay) in cases {
        let opts = TrainOptions {
            max_grad_norm,
            weight_decay,
        };
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || network.clone(),
                |network| {
                    black_box(network.train_step_with_options(
                        black_box(&inputs),
                        black_box(&targets),
                        None,
                        0.001,
                        &mut workspace,
                        &opts,
                    ));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Optimizer initialization cost
fn bench_optimizer_init(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let network = KanNetwork::new(config.clone());

    let mut group = c.benchmark_group("optimizer_init");

    group.bench_function("adam_new", |b| {
        b.iter(|| {
            black_box(Adam::new(&network, AdamConfig::default()));
        });
    });

    group.bench_function("sgd_new", |b| {
        b.iter(|| {
            black_box(SGD::new(&network, SGDConfig::with_momentum(0.001, 0.9)));
        });
    });

    group.finish();

    // Print memory analysis
    println!("\n=== Optimizer Memory Overhead ===");
    let params = network.param_count();
    let adam_mem = params * 4 * 2; // m and v buffers, f32
    let sgd_mom_mem = params * 4; // velocity buffer only

    println!("Network params: {}", params);
    println!(
        "Adam state memory: {} bytes ({:.1} KB)",
        adam_mem,
        adam_mem as f64 / 1024.0
    );
    println!(
        "SGD+momentum memory: {} bytes ({:.1} KB)",
        sgd_mom_mem,
        sgd_mom_mem as f64 / 1024.0
    );
    println!("Raw SGD memory: 0 bytes (no state)");
}

/// Compare different learning rates overhead (should be negligible)
fn bench_learning_rates(c: &mut Criterion) {
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let network = KanNetwork::new(config.clone());

    let batch = 64_usize;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let targets = make_inputs(config.output_dim, config.grid_range, batch, 123);
    let mut workspace = network.create_workspace(batch);
    network
        .clone()
        .train_step(&inputs, &targets, None, 0.001, &mut workspace);

    let lrs = [0.0001_f32, 0.001, 0.01, 0.1];
    let mut group = c.benchmark_group("learning_rates_batch64");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));

    for &lr in &lrs {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("lr_{}", lr)),
            &lr,
            |b, &lr| {
                b.iter_batched_ref(
                    || network.clone(),
                    |network| {
                        network.train_step(
                            black_box(&inputs),
                            black_box(&targets),
                            None,
                            lr,
                            &mut workspace,
                        );
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_raw_train_step,
    bench_train_step_clipping,
    bench_optimizer_init,
    bench_learning_rates,
);
criterion_main!(benches);
