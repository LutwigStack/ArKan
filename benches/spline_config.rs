//! Spline configuration benchmarks - find optimal grid_size and spline_order.
//!
//! Tests:
//! - grid_size: 3, 5, 8, 12, 16 (max supported)
//! - spline_order: 1 (linear), 2 (quadratic), 3 (cubic), 4, 5 (quintic)
//!
//! Trade-offs:
//! - Larger grid_size = more expressiveness, more parameters, more compute
//! - Higher spline_order = smoother functions, more local basis values

use arkan::{KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_inputs(dim: usize, grid_range: (f32, f32), batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * dim)
        .map(|_| rng.gen_range(grid_range.0..grid_range.1))
        .collect()
}

fn make_config_with_spline(grid_size: usize, spline_order: usize) -> KanConfig {
    KanConfig {
        input_dim: 21,
        output_dim: 24,
        hidden_dims: vec![64, 64],
        grid_size,
        spline_order,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 21],
        input_std: vec![1.0; 21],
        multithreading_threshold: 128,
        simd_width: 8,
        init_seed: Some(42),
    }
}

/// Benchmark different grid sizes with fixed spline_order=3 (cubic)
fn bench_grid_sizes(c: &mut Criterion) {
    let grid_sizes = [3_usize, 5, 8, 12, 16];
    let batch = 64_usize;
    let mut group = c.benchmark_group("grid_size_order3");

    for &grid_size in &grid_sizes {
        let config = make_config_with_spline(grid_size, 3);
        let network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        let mut workspace = network.create_workspace(batch);

        let basis_size = grid_size + 3; // grid_size + spline_order
        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("grid{}_basis{}", grid_size, basis_size)),
            &grid_size,
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

/// Benchmark different spline orders with fixed grid_size=5
fn bench_spline_orders(c: &mut Criterion) {
    let spline_orders = [1_usize, 2, 3, 4, 5];
    let batch = 64_usize;
    let mut group = c.benchmark_group("spline_order_grid5");

    for &order in &spline_orders {
        let config = make_config_with_spline(5, order);
        let network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        let mut workspace = network.create_workspace(batch);

        let basis_size = 5 + order;
        let order_name = match order {
            1 => "linear",
            2 => "quadratic",
            3 => "cubic",
            4 => "quartic",
            5 => "quintic",
            _ => "unknown",
        };
        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "forward",
                format!("order{}_{}_basis{}", order, order_name, basis_size),
            ),
            &order,
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

/// Full matrix: all combinations of grid_size and spline_order
fn bench_spline_matrix(c: &mut Criterion) {
    let grid_sizes = [3_usize, 5, 8];
    let spline_orders = [2_usize, 3, 4];
    let batch = 64_usize;
    let mut group = c.benchmark_group("spline_matrix");

    for &grid_size in &grid_sizes {
        for &order in &spline_orders {
            let config = make_config_with_spline(grid_size, order);
            let network = KanNetwork::new(config.clone());
            let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
            let mut outputs = vec![0.0f32; batch * config.output_dim];
            let mut workspace = network.create_workspace(batch);

            let basis_size = grid_size + order;
            group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    "forward",
                    format!("g{}_o{}_b{}", grid_size, order, basis_size),
                ),
                &(grid_size, order),
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
    }

    group.finish();
}

/// Training with different spline configs
fn bench_spline_training(c: &mut Criterion) {
    let configs = [
        (3_usize, 2_usize, "minimal"),
        (5, 3, "default"),
        (8, 3, "high_res"),
        (5, 5, "smooth"),
    ];
    let batch = 64_usize;
    let mut group = c.benchmark_group("spline_training");

    for &(grid_size, order, name) in &configs {
        let config = make_config_with_spline(grid_size, order);
        let mut network = KanNetwork::new(config.clone());
        let inputs = make_inputs(config.input_dim, config.grid_range, batch, 42);
        let targets = make_inputs(config.output_dim, config.grid_range, batch, 123);
        let mut workspace = network.create_workspace(batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("train_step", name),
            &(grid_size, order),
            |b, _| {
                b.iter(|| {
                    network.train_step(
                        black_box(&inputs),
                        black_box(&targets),
                        None,
                        0.001,
                        &mut workspace,
                    );
                });
            },
        );
    }

    group.finish();
}

/// Print parameter counts for different spline configs
fn bench_spline_params(c: &mut Criterion) {
    let mut group = c.benchmark_group("spline_params");

    // Minimal benchmark to trigger output
    let config = make_config_with_spline(5, 3);
    let network = KanNetwork::new(config);
    group.bench_function("param_count", |b| {
        b.iter(|| black_box(network.param_count()));
    });

    group.finish();

    // Print analysis
    println!("\n=== Spline Configuration Analysis ===");
    println!("Architecture: [21, 64, 64, 24]");
    println!(
        "{:<12} {:<12} {:<12} {:<12} {:<12}",
        "Grid", "Order", "Basis", "Params", "Memory"
    );
    println!("{}", "-".repeat(60));

    for grid_size in [3_usize, 5, 8, 12, 16] {
        for order in [2_usize, 3, 4] {
            let config = make_config_with_spline(grid_size, order);
            let network = KanNetwork::new(config);
            let params = network.param_count();
            let basis = grid_size + order;
            let memory_kb = (params * 4) as f64 / 1024.0;
            println!(
                "{:<12} {:<12} {:<12} {:<12} {:<12.1} KB",
                grid_size, order, basis, params, memory_kb
            );
        }
    }
}

criterion_group!(
    benches,
    bench_grid_sizes,
    bench_spline_orders,
    bench_spline_matrix,
    bench_spline_training,
    bench_spline_params,
);
criterion_main!(benches);
