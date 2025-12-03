//! Comprehensive GPU Forward pass benchmarks.
//!
//! Run with: cargo bench --bench gpu_forward --features gpu -- --gpu
//!
//! # Benchmarks
//!
//! - `gpu_forward_batch`: Basic forward pass at different batch sizes
//! - `cpu_vs_gpu_batch256`: Direct CPU vs GPU comparison
//! - `gpu_arch_latency`: Architecture scaling (batch=1)
//! - `gpu_arch_throughput`: Architecture scaling (batch=64)
//! - `gpu_spline_order`: Spline order impact
//! - `gpu_grid_size`: Grid size impact
//! - `gpu_latency_distribution`: Latency percentiles
//! - `gpu_memory_throughput`: Memory bandwidth analysis
//!
//! # GPU Flag
//!
//! To enable GPU benchmarks, pass `--gpu` flag:
//! ```bash
//! cargo bench --bench gpu_forward --features gpu -- --gpu
//! ```
//!
//! Without `--gpu` flag, benchmarks are skipped (CI-safe default).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

#[cfg(feature = "gpu")]
use arkan::{KanConfig, KanNetwork};
#[cfg(feature = "gpu")]
use arkan::gpu::{GpuNetwork, WgpuBackend, WgpuOptions};

/// Check if --gpu flag was passed to criterion.
fn gpu_flag_enabled() -> bool {
    std::env::args().any(|arg| arg == "--gpu")
}

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

#[cfg(feature = "gpu")]
fn bench_gpu_forward(c: &mut Criterion) {
    // Check if --gpu flag is enabled (CI-safe: skip if not)
    if !gpu_flag_enabled() {
        eprintln!("GPU benchmarks skipped (--gpu flag not provided).");
        eprintln!("Run with: cargo bench --bench gpu_forward --features gpu -- --gpu");
        return;
    }

    // Initialize GPU backend once
    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => {
            println!("GPU: {} ({:?})", b.adapter_info().name, b.adapter_info().backend);
            b
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}. Skipping GPU benchmarks.", e);
            return;
        }
    };

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());
    
    // Create GPU network
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to create GPU network: {}. Skipping.", e);
            return;
        }
    };

    let batch_sizes = [1_usize, 8, 16, 64, 256, 1024];
    let mut group = c.benchmark_group("gpu_forward_batch");

    // Pre-create workspace for largest batch
    let max_batch = *batch_sizes.iter().max().unwrap();
    let mut workspace = match gpu_network.create_workspace(max_batch) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to create workspace: {}. Skipping.", e);
            return;
        }
    };

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &batch_size| {
            b.iter(|| {
                let output = gpu_network
                    .forward_batch(black_box(&inputs), batch_size, &mut workspace)
                    .expect("GPU forward failed");
                black_box(output);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_cpu_vs_gpu(c: &mut Criterion) {
    // Check if --gpu flag is enabled
    if !gpu_flag_enabled() {
        return;
    }

    // Initialize GPU backend
    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let batch = 256;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    
    // CPU workspace
    let mut cpu_workspace = cpu_network.create_workspace(batch);
    let mut cpu_outputs = vec![0.0f32; batch * config.output_dim];
    
    // GPU workspace
    let mut gpu_workspace = gpu_network.create_workspace(batch).expect("workspace");

    let mut group = c.benchmark_group("cpu_vs_gpu_batch256");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));

    // CPU benchmark
    group.bench_function("cpu", |b| {
        b.iter(|| {
            cpu_network.forward_batch(black_box(&inputs), black_box(&mut cpu_outputs), &mut cpu_workspace);
        });
    });

    // GPU benchmark
    group.bench_function("gpu", |b| {
        b.iter(|| {
            let output = gpu_network
                .forward_batch(black_box(&inputs), batch, &mut gpu_workspace)
                .expect("GPU forward failed");
            black_box(output);
        });
    });

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_forward(_c: &mut Criterion) {
    eprintln!("GPU benchmarks require the 'gpu' feature. Run with:");
    eprintln!("  cargo bench --bench gpu_forward --features gpu");
}

#[cfg(not(feature = "gpu"))]
fn bench_cpu_vs_gpu(_c: &mut Criterion) {}

// ============================================================================
// Architecture Scaling Benchmarks
// ============================================================================

#[cfg(feature = "gpu")]
struct ArchConfig {
    name: &'static str,
    input: usize,
    output: usize,
    hidden: Vec<usize>,
}

#[cfg(feature = "gpu")]
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

#[cfg(feature = "gpu")]
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

#[cfg(feature = "gpu")]
fn make_config_spline(input: usize, output: usize, hidden: Vec<usize>, grid: usize, order: usize) -> KanConfig {
    KanConfig {
        input_dim: input,
        output_dim: output,
        hidden_dims: hidden,
        grid_size: grid,
        spline_order: order,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; input],
        input_std: vec![1.0; input],
        multithreading_threshold: 128,
        simd_width: 8,
        init_seed: Some(42),
    }
}

#[cfg(feature = "gpu")]
fn bench_gpu_arch_latency(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let archs = get_architectures();
    let mut group = c.benchmark_group("gpu_arch_latency_batch1");

    for arch in &archs {
        let config = make_config(arch.input, arch.output, arch.hidden.clone());
        let cpu_network = KanNetwork::new(config.clone());
        
        let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let inputs = make_inputs(config.input_dim, config.grid_range, 1, 42);
        let mut workspace = match gpu_network.create_workspace(1) {
            Ok(w) => w,
            Err(_) => continue,
        };

        group.throughput(Throughput::Elements(config.input_dim as u64));
        group.bench_with_input(BenchmarkId::new("forward", arch.name), arch, |b, _| {
            b.iter(|| {
                let output = gpu_network
                    .forward_batch(black_box(&inputs), 1, &mut workspace)
                    .expect("GPU forward failed");
                black_box(output);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_gpu_arch_throughput(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let archs = get_architectures();
    let batch = 64_usize;
    let mut group = c.benchmark_group("gpu_arch_throughput_batch64");

    for arch in &archs {
        let config = make_config(arch.input, arch.output, arch.hidden.clone());
        let cpu_network = KanNetwork::new(config.clone());
        
        let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut workspace = match gpu_network.create_workspace(batch) {
            Ok(w) => w,
            Err(_) => continue,
        };

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::new("forward", arch.name), arch, |b, _| {
            b.iter(|| {
                let output = gpu_network
                    .forward_batch(black_box(&inputs), batch, &mut workspace)
                    .expect("GPU forward failed");
                black_box(output);
            });
        });
    }

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_arch_latency(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_arch_throughput(_c: &mut Criterion) {}

// ============================================================================
// Spline Configuration Benchmarks
// ============================================================================

#[cfg(feature = "gpu")]
fn bench_gpu_spline_order(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let batch = 64_usize;
    let orders = [(1, "linear"), (2, "quadratic"), (3, "cubic"), (4, "quartic"), (5, "quintic")];
    let mut group = c.benchmark_group("gpu_spline_order_grid5");

    for (order, name) in orders {
        let config = make_config_spline(21, 24, vec![64, 64], 5, order);
        let cpu_network = KanNetwork::new(config.clone());
        
        let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut workspace = match gpu_network.create_workspace(batch) {
            Ok(w) => w,
            Err(_) => continue,
        };

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::new("forward", name), &order, |b, _| {
            b.iter(|| {
                let output = gpu_network
                    .forward_batch(black_box(&inputs), batch, &mut workspace)
                    .expect("GPU forward failed");
                black_box(output);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_gpu_grid_size(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let batch = 64_usize;
    let grids = [3_usize, 5, 8, 12, 16];
    let mut group = c.benchmark_group("gpu_grid_size_order3");

    for grid in grids {
        let config = make_config_spline(21, 24, vec![64, 64], grid, 3);
        let cpu_network = KanNetwork::new(config.clone());
        
        let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut workspace = match gpu_network.create_workspace(batch) {
            Ok(w) => w,
            Err(_) => continue,
        };

        let basis_size = grid + 3;
        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("grid{}_basis{}", grid, basis_size)),
            &grid,
            |b, _| {
                b.iter(|| {
                    let output = gpu_network
                        .forward_batch(black_box(&inputs), batch, &mut workspace)
                        .expect("GPU forward failed");
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_spline_order(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_grid_size(_c: &mut Criterion) {}

// ============================================================================
// Latency Distribution (manual for percentiles)
// ============================================================================

#[cfg(feature = "gpu")]
fn bench_gpu_latency_distribution(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let inputs = make_inputs(config.input_dim, config.grid_range, 1, 42);
    let mut workspace = match gpu_network.create_workspace(1) {
        Ok(w) => w,
        Err(_) => return,
    };

    let iterations = 5000;
    let mut latencies: Vec<f64> = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..500 {
        let _ = gpu_network.forward_batch(&inputs, 1, &mut workspace);
    }

    // Measure
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = gpu_network.forward_batch(&inputs, 1, &mut workspace);
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
        "\n=== GPU Latency Distribution (forward batch=1, {} samples) ===",
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
    let mut group = c.benchmark_group("gpu_latency_distribution");
    group.bench_function("forward_batch1", |b| {
        b.iter(|| {
            let output = gpu_network
                .forward_batch(black_box(&inputs), 1, &mut workspace)
                .expect("GPU forward failed");
            black_box(output);
        });
    });
    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_latency_distribution(_c: &mut Criterion) {}

// ============================================================================
// Memory Throughput Analysis
// ============================================================================

#[cfg(feature = "gpu")]
fn estimate_gpu_memory_bytes(config: &KanConfig, batch_size: usize) -> usize {
    let basis_size = config.basis_size();
    let layer_dims = config.layer_dims();

    let mut total_bytes = 0;

    // Input upload
    total_bytes += batch_size * config.input_dim * 4;

    // Per layer
    for i in 0..layer_dims.len() - 1 {
        let in_dim = layer_dims[i];
        let out_dim = layer_dims[i + 1];

        // Weights: [out_dim][in_dim][basis_size] (padded to vec4)
        let padded_basis = ((basis_size + 3) / 4) * 4;
        total_bytes += out_dim * in_dim * padded_basis * 4;

        // Bias: [out_dim]
        total_bytes += out_dim * 4;

        // Intermediate output: [batch_size][out_dim]
        total_bytes += batch_size * out_dim * 4;
    }

    // Output download
    total_bytes += batch_size * config.output_dim * 4;

    total_bytes
}

#[cfg(feature = "gpu")]
fn bench_gpu_memory_throughput(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let batch_sizes = [1_usize, 16, 64, 256, 1024];
    let mut group = c.benchmark_group("gpu_memory_throughput");

    let max_batch = *batch_sizes.iter().max().unwrap();
    let mut workspace = match gpu_network.create_workspace(max_batch) {
        Ok(w) => w,
        Err(_) => return,
    };

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mem_bytes = estimate_gpu_memory_bytes(&config, batch);

        // Use bytes as throughput metric
        group.throughput(Throughput::Bytes(mem_bytes as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &batch_size| {
            b.iter(|| {
                let output = gpu_network
                    .forward_batch(black_box(&inputs), batch_size, &mut workspace)
                    .expect("GPU forward failed");
                black_box(output);
            });
        });
    }

    group.finish();

    // Print bandwidth analysis
    println!("\n=== GPU Memory Bandwidth Analysis ===");
    println!("Config: {:?}", config.layer_dims());
    
    let batch = 256;
    let mem_bytes = estimate_gpu_memory_bytes(&config, batch);
    println!(
        "Estimated memory per forward (batch={}): {} bytes ({:.2} KB)",
        batch,
        mem_bytes,
        mem_bytes as f64 / 1024.0
    );

    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let iterations = 500;

    // Warmup
    for _ in 0..50 {
        let _ = gpu_network.forward_batch(&inputs, batch, &mut workspace);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = gpu_network.forward_batch(&inputs, batch, &mut workspace);
    }
    let elapsed = start.elapsed();

    let total_bytes = mem_bytes * iterations;
    let bandwidth_gbs = (total_bytes as f64 / 1e9) / elapsed.as_secs_f64();
    let time_per_forward_us = elapsed.as_micros() as f64 / iterations as f64;

    println!("Time per forward: {:.2} µs", time_per_forward_us);
    println!("Achieved bandwidth: {:.2} GB/s", bandwidth_gbs);
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_memory_throughput(_c: &mut Criterion) {}

// ============================================================================
// CPU vs GPU Multi-Batch Comparison
// ============================================================================

#[cfg(feature = "gpu")]
fn bench_cpu_vs_gpu_scaling(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let batch_sizes = [1_usize, 8, 32, 64, 128, 256, 512];
    
    // Print comparison table
    println!("\n=== CPU vs GPU Forward Pass Scaling ===");
    println!("---------------------------------------------------------");
    println!("{:>8} {:>12} {:>12} {:>10}", "Batch", "CPU (µs)", "GPU (µs)", "Speedup");
    println!("---------------------------------------------------------");

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut cpu_outputs = vec![0.0f32; batch * config.output_dim];
        let mut cpu_workspace = cpu_network.create_workspace(batch);
        let mut gpu_workspace = match gpu_network.create_workspace(batch) {
            Ok(w) => w,
            Err(_) => continue,
        };

        // CPU timing
        let iterations = 100;
        
        // Warmup
        for _ in 0..10 {
            cpu_network.forward_batch(&inputs, &mut cpu_outputs, &mut cpu_workspace);
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            cpu_network.forward_batch(&inputs, &mut cpu_outputs, &mut cpu_workspace);
        }
        let cpu_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // GPU timing
        for _ in 0..10 {
            let _ = gpu_network.forward_batch(&inputs, batch, &mut gpu_workspace);
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = gpu_network.forward_batch(&inputs, batch, &mut gpu_workspace);
        }
        let gpu_time = start.elapsed().as_micros() as f64 / iterations as f64;

        let speedup = cpu_time / gpu_time;
        let speedup_str = if speedup >= 1.0 {
            format!("{:.2}x GPU", speedup)
        } else {
            format!("{:.2}x CPU", 1.0 / speedup)
        };
        
        println!("{:>8} {:>12.1} {:>12.1} {:>10}", batch, cpu_time, gpu_time, speedup_str);
    }
    println!("---------------------------------------------------------");

    // Criterion comparison at crossover point
    let crossover_batch = 64;
    let inputs = make_inputs(config.input_dim, config.grid_range, crossover_batch, 42);
    let mut cpu_outputs = vec![0.0f32; crossover_batch * config.output_dim];
    let mut cpu_workspace = cpu_network.create_workspace(crossover_batch);
    let mut gpu_workspace = gpu_network.create_workspace(crossover_batch).expect("workspace");

    let mut group = c.benchmark_group("cpu_vs_gpu_crossover_batch64");
    group.throughput(Throughput::Elements((crossover_batch * config.input_dim) as u64));

    group.bench_function("cpu", |b| {
        b.iter(|| {
            cpu_network.forward_batch(black_box(&inputs), black_box(&mut cpu_outputs), &mut cpu_workspace);
        });
    });

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let output = gpu_network
                .forward_batch(black_box(&inputs), crossover_batch, &mut gpu_workspace)
                .expect("GPU forward failed");
            black_box(output);
        });
    });

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_cpu_vs_gpu_scaling(_c: &mut Criterion) {}

// ============================================================================
// GPU Softmax Benchmarks
// ============================================================================

#[cfg(feature = "gpu")]
fn bench_gpu_softmax(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let batch_sizes = [1_usize, 16, 64, 256];
    let mut group = c.benchmark_group("gpu_softmax");

    let max_batch = *batch_sizes.iter().max().unwrap();
    let mut workspace = match gpu_network.create_workspace(max_batch) {
        Ok(w) => w,
        Err(_) => return,
    };

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);

        group.throughput(Throughput::Elements((batch * config.output_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &batch_size| {
            b.iter(|| {
                let output = gpu_network
                    .forward_batch_softmax(black_box(&inputs), batch_size, &mut workspace)
                    .expect("GPU forward+softmax failed");
                black_box(output);
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_forward_vs_forward_softmax(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let batch = 64_usize;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    
    let mut workspace = match gpu_network.create_workspace(batch) {
        Ok(w) => w,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("gpu_softmax_overhead_batch64");
    group.throughput(Throughput::Elements((batch * config.output_dim) as u64));

    // Forward only
    group.bench_function("forward_only", |b| {
        b.iter(|| {
            let output = gpu_network
                .forward_batch(black_box(&inputs), batch, &mut workspace)
                .expect("GPU forward failed");
            black_box(output);
        });
    });

    // Forward + softmax
    group.bench_function("forward_softmax", |b| {
        b.iter(|| {
            let output = gpu_network
                .forward_batch_softmax(black_box(&inputs), batch, &mut workspace)
                .expect("GPU forward+softmax failed");
            black_box(output);
        });
    });

    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_gpu_softmax(_c: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_forward_vs_forward_softmax(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_gpu_forward,
    bench_cpu_vs_gpu,
    bench_gpu_arch_latency,
    bench_gpu_arch_throughput,
    bench_gpu_spline_order,
    bench_gpu_grid_size,
    bench_gpu_latency_distribution,
    bench_gpu_memory_throughput,
    bench_cpu_vs_gpu_scaling,
    bench_gpu_softmax,
    bench_forward_vs_forward_softmax,
);
criterion_main!(benches);
