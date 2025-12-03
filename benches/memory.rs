//! Memory throughput benchmarks - roofline analysis.
//!
//! Measures actual memory bandwidth utilization during forward pass.
//! Goal: understand how close we are to peak memory bandwidth.
//!
//! Key metrics:
//! - Bytes read/written per forward pass
//! - Achieved GB/s
//! - Arithmetic intensity (FLOPS/byte)

use arkan::{KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

/// Estimate bytes read/written for a forward pass
fn estimate_memory_bytes(config: &KanConfig, batch_size: usize) -> usize {
    let basis_size = config.basis_size();
    let layer_dims = config.layer_dims();

    let mut total_bytes = 0;

    // Input read
    total_bytes += batch_size * config.input_dim * 4;

    // Per layer
    for i in 0..layer_dims.len() - 1 {
        let in_dim = layer_dims[i];
        let out_dim = layer_dims[i + 1];

        // Read weights: [out_dim][in_dim][basis_size]
        total_bytes += out_dim * in_dim * basis_size * 4;

        // Read bias: [out_dim]
        total_bytes += out_dim * 4;

        // Read normalization (mean, std): [in_dim] * 2
        total_bytes += in_dim * 2 * 4;

        // Write intermediate output: [batch_size][out_dim]
        total_bytes += batch_size * out_dim * 4;

        // Basis values computed: [batch_size][in_dim][local_basis]
        // local_basis = order + 1 = 4 for cubic
        let local_basis = config.spline_order + 1;
        total_bytes += batch_size * in_dim * local_basis * 4;
    }

    // Output write
    total_bytes += batch_size * config.output_dim * 4;

    total_bytes
}

/// Benchmark with explicit memory throughput measurement
fn bench_memory_throughput(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    let batch_sizes = [1_usize, 16, 64, 256, 1024];
    let mut group = c.benchmark_group("memory_throughput");

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        let mut workspace = network.create_workspace(batch);

        let mem_bytes = estimate_memory_bytes(&config, batch);

        // Use bytes as throughput metric
        group.throughput(Throughput::Bytes(mem_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &_batch| {
            b.iter(|| {
                network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
            });
        });
    }

    group.finish();
}

/// Manual timing to compute GB/s
fn bench_manual_bandwidth(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    let batch = 256_usize;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let mut outputs = vec![0.0f32; batch * config.output_dim];
    let mut workspace = network.create_workspace(batch);

    let mem_bytes = estimate_memory_bytes(&config, batch);
    let iterations = 1000;

    let mut group = c.benchmark_group("bandwidth_analysis");

    group.bench_function(format!("batch{}_bytes{}", batch, mem_bytes), |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
            }
            start.elapsed()
        });
    });

    group.finish();

    // Print bandwidth analysis
    println!("\n=== Memory Bandwidth Analysis ===");
    println!("Config: {:?}", config.layer_dims());
    println!("Batch size: {}", batch);
    println!(
        "Estimated memory per forward: {} bytes ({:.2} KB)",
        mem_bytes,
        mem_bytes as f64 / 1024.0
    );

    // Warmup and measure
    for _ in 0..100 {
        network.forward_batch(&inputs, &mut outputs, &mut workspace);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        network.forward_batch(&inputs, &mut outputs, &mut workspace);
    }
    let elapsed = start.elapsed();

    let total_bytes = mem_bytes * iterations;
    let bandwidth_gbs = (total_bytes as f64 / 1e9) / elapsed.as_secs_f64();
    let time_per_forward_us = elapsed.as_micros() as f64 / iterations as f64;

    println!("Time per forward: {:.2} µs", time_per_forward_us);
    println!("Achieved bandwidth: {:.2} GB/s", bandwidth_gbs);
    println!("(Typical DDR4 peak: ~25-50 GB/s, DDR5: ~50-100 GB/s)");
}

/// Cache pressure analysis - vary batch size to see cache effects
fn bench_cache_pressure(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    // Batch sizes designed to stress different cache levels
    // L1 ~32KB, L2 ~256KB, L3 ~8MB (typical)
    let batch_sizes = [1_usize, 4, 16, 32, 64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("cache_pressure");

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        let mut workspace = network.create_workspace(batch);

        let mem_bytes = estimate_memory_bytes(&config, batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "forward",
                format!("batch{}_mem{}KB", batch, mem_bytes / 1024),
            ),
            &batch,
            |b, _| {
                b.iter(|| {
                    network.forward_batch(
                        black_box(&inputs),
                        black_box(&mut outputs),
                        &mut workspace,
                    );
                });
            },
        );
    }

    group.finish();
}

/// Workspace allocation vs reuse comparison
fn bench_workspace_reuse(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    let batch = 64_usize;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let mut outputs = vec![0.0f32; batch * config.output_dim];

    let mut group = c.benchmark_group("workspace_reuse");

    // With workspace reuse (normal usage)
    let mut workspace = network.create_workspace(batch);
    group.bench_function("with_reuse", |b| {
        b.iter(|| {
            network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut workspace);
        });
    });

    // Without workspace reuse (allocate each time)
    group.bench_function("without_reuse", |b| {
        b.iter(|| {
            let mut ws = network.create_workspace(batch);
            network.forward_batch(black_box(&inputs), black_box(&mut outputs), &mut ws);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_throughput,
    bench_manual_bandwidth,
    bench_cache_pressure,
    bench_workspace_reuse,
);
criterion_main!(benches);
