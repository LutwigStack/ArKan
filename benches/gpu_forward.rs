//! GPU Forward pass benchmarks.
//!
//! Run with: cargo bench --bench gpu_forward --features gpu -- --gpu
//!
//! # Notes
//!
//! These benchmarks measure GPU forward pass performance. For fair comparison
//! with CPU, both include data transfer time (upload/download).
//!
//! # GPU Flag
//!
//! To enable GPU benchmarks in CI-safe mode, pass `--gpu` flag:
//! ```bash
//! cargo bench --bench gpu_forward --features gpu -- --gpu
//! ```
//!
//! Without `--gpu` flag, benchmarks will be skipped gracefully (CI-safe default).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

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

criterion_group!(benches, bench_gpu_forward, bench_cpu_vs_gpu);
criterion_main!(benches);
