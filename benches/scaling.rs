//! Architecture scaling benchmarks - different network depths and widths.
//!
//! Tests how ArKan scales with:
//! - Tiny networks: [3, 10, 1] (toy regression)
//! - Medium networks: [10, 64, 64, 10] (typical use)
//! - Large networks: [32, 128, 128, 128, 32] (heavy workload)
//! - Wide networks: [21, 256, 24] (single wide hidden layer)
//! - Deep networks: [21, 32, 32, 32, 32, 32, 24] (many narrow layers)

use arkan::{KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

fn make_config(input: usize, output: usize, hidden: Vec<usize>) -> KanConfig {
    KanConfig {
        input_dim: input,
        output_dim: output,
        hidden_dims: hidden,
        grid_size: 5,
        spline_order: 3,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; input],
        input_std: vec![1.0; input],
        multithreading_threshold: 128,
        simd_width: 8,
        init_seed: Some(42),
    }
}

/// Architecture configurations for benchmarking
struct ArchConfig {
    name: &'static str,
    input: usize,
    output: usize,
    hidden: Vec<usize>,
}

fn get_architectures() -> Vec<ArchConfig> {
    vec![
        ArchConfig {
            name: "tiny_3_10_1",
            input: 3,
            output: 1,
            hidden: vec![10],
        },
        ArchConfig {
            name: "medium_10_64_64_10",
            input: 10,
            output: 10,
            hidden: vec![64, 64],
        },
        ArchConfig {
            name: "poker_21_64_64_24",
            input: 21,
            output: 24,
            hidden: vec![64, 64],
        },
        ArchConfig {
            name: "large_32_128x3_32",
            input: 32,
            output: 32,
            hidden: vec![128, 128, 128],
        },
        ArchConfig {
            name: "wide_21_256_24",
            input: 21,
            output: 24,
            hidden: vec![256],
        },
        ArchConfig {
            name: "deep_21_32x5_24",
            input: 21,
            output: 24,
            hidden: vec![32, 32, 32, 32, 32],
        },
    ]
}

/// Forward pass scaling across architectures (batch=1 for latency)
fn bench_architecture_latency(c: &mut Criterion) {
    let archs = get_architectures();
    let mut group = c.benchmark_group("arch_latency_batch1");

    for arch in &archs {
        let config = make_config(arch.input, arch.output, arch.hidden.clone());
        let network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, 1, 42);
        let mut outputs = vec![0.0f32; config.output_dim];
        let mut workspace = network.create_workspace(1);

        group.throughput(Throughput::Elements(config.input_dim as u64));
        group.bench_with_input(BenchmarkId::new("forward", arch.name), arch, |b, _| {
            b.iter(|| {
                network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
            });
        });
    }

    group.finish();
}

/// Forward pass scaling across architectures (batch=64 for throughput)
fn bench_architecture_throughput(c: &mut Criterion) {
    let archs = get_architectures();
    let batch = 64_usize;
    let mut group = c.benchmark_group("arch_throughput_batch64");

    for arch in &archs {
        let config = make_config(arch.input, arch.output, arch.hidden.clone());
        let network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        let mut workspace = network.create_workspace(batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::new("forward", arch.name), arch, |b, _| {
            b.iter(|| {
                network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
            });
        });
    }

    group.finish();
}

/// Training step scaling across architectures
fn bench_architecture_training(c: &mut Criterion) {
    let archs = get_architectures();
    let batch = 64_usize;
    let mut group = c.benchmark_group("arch_training_batch64");

    for arch in &archs {
        let config = make_config(arch.input, arch.output, arch.hidden.clone());
        let mut network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_inputs(config.output_dim, config.grid_range, batch, 123);
        let mut workspace = network.create_workspace(batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::new("train_step", arch.name), arch, |b, _| {
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

/// Parameter count and memory footprint analysis
fn bench_param_count(c: &mut Criterion) {
    let archs = get_architectures();
    let mut group = c.benchmark_group("arch_param_count");

    for arch in &archs {
        let config = make_config(arch.input, arch.output, arch.hidden.clone());
        let network = KanNetwork::new(config.clone());

        // Just measure param_count() call overhead (should be negligible)
        group.bench_with_input(BenchmarkId::new("param_count", arch.name), arch, |b, _| {
            b.iter(|| black_box(network.param_count()));
        });
    }

    group.finish();

    // Print param counts for documentation
    println!("\n=== Architecture Parameter Counts ===");
    for arch in &archs {
        let config = make_config(arch.input, arch.output, arch.hidden.clone());
        let network = KanNetwork::new(config);
        let params = network.param_count();
        let memory_kb = (params * 4) as f64 / 1024.0; // f32 = 4 bytes
        println!("{}: {} params ({:.1} KB)", arch.name, params, memory_kb);
    }
}

criterion_group!(
    benches,
    bench_architecture_latency,
    bench_architecture_throughput,
    bench_architecture_training,
    bench_param_count,
);
criterion_main!(benches);
