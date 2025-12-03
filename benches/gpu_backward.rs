//! GPU Backward pass and training step benchmarks.
//!
//! Run with: cargo bench --bench gpu_backward --features gpu -- --gpu
//!
//! # Notes
//!
//! These benchmarks measure GPU backward pass and training step performance.
//! For fair comparison with CPU, both include data transfer time.
//!
//! # GPU Flag
//!
//! To enable GPU benchmarks in CI-safe mode, pass `--gpu` flag:
//! ```bash
//! cargo bench --bench gpu_backward --features gpu -- --gpu
//! ```
//!
//! Without `--gpu` flag, benchmarks will be skipped gracefully (CI-safe default).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(feature = "gpu")]
use arkan::{KanConfig, KanNetwork, TrainOptions};
#[cfg(feature = "gpu")]
use arkan::gpu::{GpuNetwork, WgpuBackend, WgpuOptions};
#[cfg(feature = "gpu")]
use arkan::optimizer::{Adam, AdamConfig, SGD};

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

fn make_targets(dim: usize, batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed + 1000);
    (0..batch * dim)
        .map(|_| rng.gen_range(0.0..1.0))
        .collect()
}

#[cfg(feature = "gpu")]
fn bench_gpu_train_step_adam(c: &mut Criterion) {
    // Check if --gpu flag is enabled (CI-safe: skip if not)
    if !gpu_flag_enabled() {
        eprintln!("GPU benchmarks skipped (--gpu flag not provided).");
        eprintln!("Run with: cargo bench --bench gpu_backward --features gpu -- --gpu");
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
    let mut cpu_network = KanNetwork::new(config.clone());
    
    // Create GPU network
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to create GPU network: {}. Skipping.", e);
            return;
        }
    };

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("gpu_train_step_adam");

    // Pre-create workspace for largest batch
    let max_batch = *batch_sizes.iter().max().unwrap();
    let mut workspace = match gpu_network.create_workspace(max_batch) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to create workspace: {}. Skipping.", e);
            return;
        }
    };
    
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_targets(config.output_dim, batch, 42);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &batch_size| {
            b.iter(|| {
                let loss = gpu_network
                    .train_step_mse(
                        black_box(&inputs),
                        black_box(&targets),
                        batch_size,
                        &mut workspace,
                        &mut optimizer,
                        &mut cpu_network,
                    )
                    .expect("GPU train step failed");
                black_box(loss)
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_gpu_train_step_sgd(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    // Initialize GPU backend once
    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let mut cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("gpu_train_step_sgd");

    let max_batch = *batch_sizes.iter().max().unwrap();
    let mut workspace = match gpu_network.create_workspace(max_batch) {
        Ok(w) => w,
        Err(_) => return,
    };
    
    let mut optimizer = SGD::new(&cpu_network, 0.01, 0.9, 0.0);

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_targets(config.output_dim, batch, 42);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &batch_size| {
            b.iter(|| {
                let loss = gpu_network
                    .train_step_sgd(
                        black_box(&inputs),
                        black_box(&targets),
                        batch_size,
                        &mut workspace,
                        &mut optimizer,
                        &mut cpu_network,
                    )
                    .expect("GPU train step failed");
                black_box(loss)
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_gpu_train_step_with_options(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let mut cpu_network = KanNetwork::new(config.clone());
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("gpu_train_options");
    let batch = 64;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let targets = make_targets(config.output_dim, batch, 42);

    let mut workspace = match gpu_network.create_workspace(batch) {
        Ok(w) => w,
        Err(_) => return,
    };
    
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));

    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
    
    // No options
    group.bench_function("no_options", |b| {
        let opts = TrainOptions::default();
        b.iter(|| {
            let loss = gpu_network
                .train_step_with_options(
                    black_box(&inputs),
                    black_box(&targets),
                    None,
                    batch,
                    &mut workspace,
                    &mut optimizer,
                    &mut cpu_network,
                    &opts,
                )
                .expect("GPU train step failed");
            black_box(loss)
        });
    });
    
    // With gradient clipping
    group.bench_function("grad_clip", |b| {
        let opts = TrainOptions {
            max_grad_norm: Some(1.0),
            weight_decay: 0.0,
        };
        b.iter(|| {
            let loss = gpu_network
                .train_step_with_options(
                    black_box(&inputs),
                    black_box(&targets),
                    None,
                    batch,
                    &mut workspace,
                    &mut optimizer,
                    &mut cpu_network,
                    &opts,
                )
                .expect("GPU train step failed");
            black_box(loss)
        });
    });
    
    // With weight decay
    group.bench_function("weight_decay", |b| {
        let opts = TrainOptions {
            max_grad_norm: None,
            weight_decay: 0.01,
        };
        b.iter(|| {
            let loss = gpu_network
                .train_step_with_options(
                    black_box(&inputs),
                    black_box(&targets),
                    None,
                    batch,
                    &mut workspace,
                    &mut optimizer,
                    &mut cpu_network,
                    &opts,
                )
                .expect("GPU train step failed");
            black_box(loss)
        });
    });
    
    // Both options
    group.bench_function("both", |b| {
        let opts = TrainOptions {
            max_grad_norm: Some(1.0),
            weight_decay: 0.01,
        };
        b.iter(|| {
            let loss = gpu_network
                .train_step_with_options(
                    black_box(&inputs),
                    black_box(&targets),
                    None,
                    batch,
                    &mut workspace,
                    &mut optimizer,
                    &mut cpu_network,
                    &opts,
                )
                .expect("GPU train step failed");
            black_box(loss)
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_cpu_vs_gpu_train(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig::preset();
    let mut cpu_network = KanNetwork::new(config.clone());
    let mut gpu_cpu_network = cpu_network.clone();
    let mut cpu_workspace = cpu_network.create_workspace(64);
    
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &gpu_cpu_network) {
        Ok(n) => n,
        Err(_) => return,
    };

    let batch = 64;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let targets = make_targets(config.output_dim, batch, 42);

    let mut gpu_workspace = match gpu_network.create_workspace(batch) {
        Ok(w) => w,
        Err(_) => return,
    };
    
    let mut optimizer = Adam::new(&gpu_cpu_network, AdamConfig::with_lr(0.001));

    let mut group = c.benchmark_group("cpu_vs_gpu_train_batch64");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));

    // CPU benchmark
    group.bench_function("cpu", |b| {
        b.iter(|| {
            let loss = cpu_network.train_step(
                black_box(&inputs),
                black_box(&targets),
                None,
                0.001,
                &mut cpu_workspace,
            );
            black_box(loss)
        });
    });

    // GPU benchmark
    group.bench_function("gpu", |b| {
        b.iter(|| {
            let loss = gpu_network
                .train_step_mse(
                    black_box(&inputs),
                    black_box(&targets),
                    batch,
                    &mut gpu_workspace,
                    &mut optimizer,
                    &mut gpu_cpu_network,
                )
                .expect("GPU train step failed");
            black_box(loss)
        });
    });

    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_gpu_train_step_adam,
    bench_gpu_train_step_sgd,
    bench_gpu_train_step_with_options,
    bench_cpu_vs_gpu_train,
);

#[cfg(not(feature = "gpu"))]
fn dummy(_c: &mut Criterion) {
    eprintln!("GPU feature not enabled. Skipping GPU benchmarks.");
}

#[cfg(not(feature = "gpu"))]
criterion_group!(benches, dummy);

criterion_main!(benches);
