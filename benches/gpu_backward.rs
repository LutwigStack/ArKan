//! GPU Backward pass and training step benchmarks.
//!
//! Run with: ARKAN_GPU_BENCH=1 cargo bench --bench gpu_backward --features gpu
//!
//! # Notes
//!
//! These benchmarks measure GPU backward pass and training step performance.
//! For fair comparison with CPU, both include data transfer time.
//!
//! # Enabling GPU Benchmarks
//!
//! Set environment variable `ARKAN_GPU_BENCH=1`:
//! ```bash
//! # Linux/macOS
//! ARKAN_GPU_BENCH=1 cargo bench --bench gpu_backward --features gpu
//!
//! # Windows PowerShell
//! $env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_backward --features gpu
//! ```
//!
//! Without this variable, benchmarks are skipped (CI-safe default).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;

#[cfg(feature = "gpu")]
use arkan::gpu::{GpuAdam, GpuAdamConfig, GpuNetwork, WgpuBackend, WgpuOptions};
#[cfg(feature = "gpu")]
use arkan::optimizer::{Adam, AdamConfig, SGDConfig, SGD};
#[cfg(feature = "gpu")]
use arkan::{KanConfig, KanNetwork, TrainOptions};

/// Check if GPU benchmarks are enabled via environment variable.
///
/// Set `ARKAN_GPU_BENCH=1` to enable GPU benchmarks.
fn gpu_flag_enabled() -> bool {
    std::env::var("ARKAN_GPU_BENCH")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

fn make_targets(dim: usize, batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed + 1000);
    (0..batch * dim).map(|_| rng.gen_range(0.0..1.0)).collect()
}

#[cfg(feature = "gpu")]
fn bench_gpu_train_step_adam(c: &mut Criterion) {
    // Check if ARKAN_GPU_BENCH=1 is set (CI-safe: skip if not)
    if !gpu_flag_enabled() {
        eprintln!("GPU benchmarks skipped (ARKAN_GPU_BENCH not set).");
        eprintln!("Run with: ARKAN_GPU_BENCH=1 cargo bench --bench gpu_backward --features gpu");
        return;
    }

    // Initialize GPU backend once
    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => {
            println!(
                "GPU: {} ({:?})",
                b.adapter_info().name,
                b.adapter_info().backend
            );
            b
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}. Skipping GPU benchmarks.", e);
            return;
        }
    };

    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };

    // Create base CPU network to clone from (for stable measurements)
    let base_cpu_network = KanNetwork::new(config.clone());

    // Create GPU network wrapped in RefCell for interior mutability
    let gpu_network = match GpuNetwork::from_cpu(&backend, &base_cpu_network) {
        Ok(n) => RefCell::new(n),
        Err(e) => {
            eprintln!("Failed to create GPU network: {}. Skipping.", e);
            return;
        }
    };

    gpu_network
        .borrow_mut()
        .init_training()
        .expect("training buffer setup");
    backend.poll();

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("gpu_train_step_adam");

    // Pre-create workspace for largest batch
    let max_batch = *batch_sizes.iter().max().unwrap();
    let workspace = match gpu_network.borrow_mut().create_workspace(max_batch) {
        Ok(w) => RefCell::new(w),
        Err(e) => {
            eprintln!("Failed to create workspace: {}. Skipping.", e);
            return;
        }
    };

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_targets(config.output_dim, batch, 42);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &batch,
            |b, &batch_size| {
                // Use iter_batched to get fresh state each iteration
                b.iter_batched_ref(
                    || {
                        // Setup: create fresh CPU network and optimizer
                        let cpu_network = base_cpu_network.clone();
                        let optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));
                        // Sync weights to GPU (reset to initial state)
                        gpu_network
                            .borrow_mut()
                            .sync_weights_cpu_to_gpu(&cpu_network)
                            .unwrap();
                        backend.poll();
                        (cpu_network, optimizer)
                    },
                    |(cpu_network, optimizer)| {
                        let loss = gpu_network
                            .borrow_mut()
                            .train_step_mse(
                                black_box(&inputs),
                                black_box(&targets),
                                batch_size,
                                &mut workspace.borrow_mut(),
                                optimizer,
                                cpu_network,
                            )
                            .expect("GPU train step failed");
                        black_box(loss)
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_gpu_train_step_sgd(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => b,
        Err(_) => return,
    };

    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let base_cpu_network = KanNetwork::new(config.clone());

    let gpu_network = match GpuNetwork::from_cpu(&backend, &base_cpu_network) {
        Ok(n) => RefCell::new(n),
        Err(_) => return,
    };

    gpu_network
        .borrow_mut()
        .init_training()
        .expect("training buffer setup");
    backend.poll();

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("gpu_train_step_sgd");

    let max_batch = *batch_sizes.iter().max().unwrap();
    let workspace = match gpu_network.borrow_mut().create_workspace(max_batch) {
        Ok(w) => RefCell::new(w),
        Err(_) => return,
    };

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_targets(config.output_dim, batch, 42);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &batch,
            |b, &batch_size| {
                b.iter_batched_ref(
                    || {
                        let cpu_network = base_cpu_network.clone();
                        let optimizer = SGD::new(&cpu_network, SGDConfig::with_momentum(0.01, 0.9));
                        gpu_network
                            .borrow_mut()
                            .sync_weights_cpu_to_gpu(&cpu_network)
                            .unwrap();
                        backend.poll();
                        (cpu_network, optimizer)
                    },
                    |(cpu_network, optimizer)| {
                        let loss = gpu_network
                            .borrow_mut()
                            .train_step_sgd(
                                black_box(&inputs),
                                black_box(&targets),
                                batch_size,
                                &mut workspace.borrow_mut(),
                                optimizer,
                                cpu_network,
                            )
                            .expect("GPU train step failed");
                        black_box(loss)
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
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
    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let base_cpu_network = KanNetwork::new(config.clone());
    let gpu_network = match GpuNetwork::from_cpu(&backend, &base_cpu_network) {
        Ok(n) => RefCell::new(n),
        Err(_) => return,
    };
    gpu_network
        .borrow_mut()
        .init_training()
        .expect("training buffer setup");
    let batch = 64;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let targets = make_targets(config.output_dim, batch, 42);
    let mut workspace = match gpu_network.borrow_mut().create_workspace(batch) {
        Ok(w) => w,
        Err(_) => return,
    };
    // A CPU probe defines active/inactive clipping thresholds on the same seeded workload.
    let mut probe_workspace = base_cpu_network.create_workspace(batch);
    base_cpu_network
        .clone()
        .train_step(&inputs, &targets, None, 0.001, &mut probe_workspace);
    let norm = probe_workspace
        .weight_grads
        .iter()
        .chain(&probe_workspace.bias_grads)
        .flat_map(|g| g.as_slice())
        .map(|g| (*g as f64).powi(2))
        .sum::<f64>()
        .sqrt() as f32;
    assert!(
        norm.is_finite() && norm > 0.0,
        "clipping workload needs finite nonzero gradients"
    );
    println!("Seeded GPU clipping workload CPU-reference gradient norm: {norm}");
    backend.poll();
    let mut group = c.benchmark_group("gpu_train_options");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
    let cases = [
        ("no_options", None, 0.0),
        ("clip_active_half_norm", Some(norm * 0.5), 0.0),
        ("clip_inactive_double_norm", Some(norm * 2.0), 0.0),
        ("weight_decay_0.01", None, 0.01),
        ("active_clip_and_decay", Some(norm * 0.5), 0.01),
    ];
    for (name, max_grad_norm, weight_decay) in cases {
        let opts = TrainOptions {
            max_grad_norm,
            weight_decay,
        };
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || {
                    let cpu = base_cpu_network.clone();
                    let optimizer = Adam::new(&cpu, AdamConfig::with_lr(0.001));
                    gpu_network
                        .borrow_mut()
                        .sync_weights_cpu_to_gpu(&cpu)
                        .unwrap();
                    backend.poll();
                    (cpu, optimizer)
                },
                |(cpu, optimizer)| {
                    black_box(
                        gpu_network
                            .borrow_mut()
                            .train_step_with_options(
                                black_box(&inputs),
                                black_box(&targets),
                                None,
                                batch,
                                &mut workspace,
                                optimizer,
                                cpu,
                                &opts,
                            )
                            .expect("GPU train step failed"),
                    );
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }
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

    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let base_cpu_network = KanNetwork::new(config.clone());

    let gpu_network = match GpuNetwork::from_cpu(&backend, &base_cpu_network) {
        Ok(n) => RefCell::new(n),
        Err(_) => return,
    };

    let batch = 64;
    let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
    let targets = make_targets(config.output_dim, batch, 42);

    gpu_network
        .borrow_mut()
        .init_training()
        .expect("training buffer setup");
    backend.poll();
    let gpu_workspace = match gpu_network.borrow_mut().create_workspace(batch) {
        Ok(w) => RefCell::new(w),
        Err(_) => return,
    };

    let mut group = c.benchmark_group("cpu_vs_gpu_train_batch64");
    group.throughput(Throughput::Elements((batch * config.input_dim) as u64));

    // CPU benchmark - reset network each iteration for stable measurements
    group.bench_function("cpu", |b| {
        b.iter_batched_ref(
            || {
                let cpu_network = base_cpu_network.clone();
                let workspace = cpu_network.create_workspace(batch);
                let optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));
                (cpu_network, workspace, optimizer)
            },
            |(cpu_network, workspace, optimizer)| {
                let loss = cpu_network
                    .train_step_with_optimizer(
                        black_box(&inputs),
                        black_box(&targets),
                        None,
                        workspace,
                        optimizer,
                        &TrainOptions::default(),
                    )
                    .expect("CPU train step failed");
                black_box(loss)
            },
            criterion::BatchSize::PerIteration,
        );
    });

    // GPU hybrid benchmark - reset network and optimizer each iteration
    group.bench_function("gpu_hybrid", |b| {
        b.iter_batched_ref(
            || {
                let gpu_cpu_network = base_cpu_network.clone();
                let optimizer = Adam::new(&gpu_cpu_network, AdamConfig::with_lr(0.001));
                gpu_network
                    .borrow_mut()
                    .sync_weights_cpu_to_gpu(&gpu_cpu_network)
                    .unwrap();
                backend.poll();
                (gpu_cpu_network, optimizer)
            },
            |(gpu_cpu_network, optimizer)| {
                let loss = gpu_network
                    .borrow_mut()
                    .train_step_mse(
                        black_box(&inputs),
                        black_box(&targets),
                        batch,
                        &mut gpu_workspace.borrow_mut(),
                        optimizer,
                        gpu_cpu_network,
                    )
                    .expect("GPU train step failed");
                black_box(loss)
            },
            criterion::BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// Benchmark native GPU training (GPU optimizer state, synchronous loss readback).
///
/// This benchmark measures `train_step_gpu_native` performance, which keeps
/// all optimizer state and gradients on the GPU. Reading parameters back requires
/// explicit `sync_weights_gpu_to_cpu`; the returned loss is read back synchronously.
#[cfg(feature = "gpu")]
fn bench_gpu_native_training(c: &mut Criterion) {
    if !gpu_flag_enabled() {
        eprintln!("GPU benchmarks skipped (ARKAN_GPU_BENCH not set).");
        return;
    }

    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => {
            println!(
                "Native GPU training benchmark - GPU: {} ({:?})",
                b.adapter_info().name,
                b.adapter_info().backend
            );
            b
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}. Skipping.", e);
            return;
        }
    };

    let config = KanConfig {
        init_seed: Some(42),
        ..KanConfig::preset()
    };
    let cpu_network = KanNetwork::new(config.clone());

    let gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => RefCell::new(n),
        Err(e) => {
            eprintln!("Failed to create GPU network: {}. Skipping.", e);
            return;
        }
    };

    gpu_network
        .borrow_mut()
        .init_training()
        .expect("training buffer setup");
    backend.poll();

    let batch_sizes = [1_usize, 8, 16, 64, 256];
    let mut group = c.benchmark_group("gpu_native_training");

    let max_batch = *batch_sizes.iter().max().unwrap();
    let mut workspace = match gpu_network.borrow_mut().create_workspace(max_batch) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to create workspace: {}. Skipping.", e);
            return;
        }
    };

    // Create GpuAdam optimizer (native GPU optimizer)
    let layer_sizes = gpu_network.borrow().layer_param_sizes();
    let optimizer = RefCell::new(GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.001),
    ));

    for &batch in &batch_sizes {
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_targets(config.output_dim, batch, 42);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &batch,
            |b, &batch_size| {
                b.iter_batched_ref(
                    || {
                        gpu_network
                            .borrow_mut()
                            .sync_weights_cpu_to_gpu(&cpu_network)
                            .unwrap();
                        optimizer.borrow_mut().reset();
                        backend.poll();
                    },
                    |_| {
                        let loss = gpu_network
                            .borrow_mut()
                            .train_step_gpu_native(
                                black_box(&inputs),
                                black_box(&targets),
                                batch_size,
                                &mut workspace,
                                &mut optimizer.borrow_mut(),
                            )
                            .expect("Native GPU train step failed");
                        black_box(loss)
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }

    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_gpu_train_step_adam,
    bench_gpu_train_step_sgd,
    bench_gpu_train_step_with_options,
    bench_cpu_vs_gpu_train,
    bench_gpu_native_training,
);

#[cfg(not(feature = "gpu"))]
fn dummy(_c: &mut Criterion) {
    eprintln!("GPU feature not enabled. Skipping GPU benchmarks.");
}

#[cfg(not(feature = "gpu"))]
criterion_group!(benches, dummy);

criterion_main!(benches);
