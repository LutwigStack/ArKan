//! Optimizer benchmarks - Adam, SGD and warmed L-BFGS steps.
//!
//! Tests:
//! - Raw SGD (inline in train_step) vs raw train_step with options
//! - Optimizer initialization cost
//! - Memory overhead of optimizer state
//!
//! The AMP cases measure whole Adam/SGD steps with prebuilt gradients and warmed
//! optimizer state. Train-step cases include gradient extraction. L-BFGS cases
//! include their objective callback.

use arkan::network::TrainOptions;
use arkan::optimizer::{LineSearchMethod, SafetyConfig};
use arkan::{Adam, AdamConfig, KanConfig, KanNetwork, LBFGSConfig, SGDConfig, LBFGS, SGD};
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

/// Quadratic callback including the gradient allocation required by step_lbfgs.
fn lbfgs_objective(network: &KanNetwork, stationary: bool) -> arkan::ArkanResult<(f64, Vec<f32>)> {
    let mut gradient = vec![0.0; network.param_count()];
    if stationary {
        return Ok((7.0, gradient));
    }
    let x = network.layers[0].weights[0];
    gradient[0] = 2.0 * x;
    Ok((f64::from(x).powi(2), gradient))
}

/// Reconstruct and warm once: cloning an H1 optimizer would change history capacity.
fn warmed_lbfgs(width: usize, iterations: usize) -> (KanNetwork, LBFGS) {
    let mut network = KanNetwork::new(KanConfig {
        input_dim: width,
        output_dim: width,
        hidden_dims: vec![width],
        grid_size: 3,
        spline_order: 3,
        input_mean: vec![0.0; width],
        input_std: vec![1.0; width],
        init_seed: Some(42),
        ..KanConfig::default()
    });
    for layer in &mut network.layers {
        layer.weights.fill(0.0);
        layer.bias.fill(0.0);
    }
    network.layers[0].weights[0] = 1.0;
    assert_eq!(network.param_count(), 12 * width * width + 2 * width);
    let mut optimizer = LBFGS::new(
        &network,
        LBFGSConfig {
            lr: 0.25,
            max_iter: 1,
            max_eval: Some(2),
            tolerance_grad: 0.0,
            tolerance_change: 0.0,
            history_size: 3,
            line_search_fn: LineSearchMethod::NoLineSearch,
            safety: SafetyConfig::strict(),
        },
    );
    let loss = optimizer
        .step_lbfgs(&mut network, |net| lbfgs_objective(net, false))
        .expect("L-BFGS warm step failed");
    assert_eq!(loss.to_bits(), 0.25_f64.to_bits());
    assert_eq!(network.layers[0].weights[0].to_bits(), 0.5_f32.to_bits());
    assert_eq!(optimizer.num_evals(), 2);
    optimizer.config.max_iter = iterations;
    optimizer.config.max_eval = Some(iterations + 1);
    (network, optimizer)
}

/// Analytical checks run before Criterion's measured routine, not inside it.
fn verify_lbfgs_case(width: usize, iterations: usize, stationary: bool) {
    let (mut network, mut optimizer) = warmed_lbfgs(width, iterations);
    let mut calls = 0;
    let loss = optimizer
        .step_lbfgs(&mut network, |net| {
            calls += 1;
            lbfgs_objective(net, stationary)
        })
        .expect("L-BFGS verification failed");
    let expected_x: f32 = if stationary {
        0.5
    } else if iterations == 1 {
        0.375
    } else {
        81.0_f32 / 512.0
    };
    let expected_loss = if stationary {
        7.0
    } else {
        f64::from(expected_x).powi(2)
    };
    assert_eq!(loss.to_bits(), expected_loss.to_bits());
    assert_eq!(network.layers[0].weights[0].to_bits(), expected_x.to_bits());
    assert_eq!(calls, if stationary { 1 } else { iterations + 1 });
    assert_eq!(optimizer.num_evals(), 2 + calls);
    assert!(network
        .layers
        .iter()
        .flat_map(|layer| layer.weights.iter().chain(&layer.bias))
        .skip(1)
        .all(|value| value.to_bits() == 0.0_f32.to_bits()));
}

/// One warmed H1 step, including the ordinary allocating objective callback.
fn bench_lbfgs_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("lbfgs_step_with_callback");
    let cases = [
        ("p14_h1_fixed_k1", 1, 1, false),
        ("p784_h1_fixed_k4", 8, 4, false),
        ("p784_h1_stationary", 8, 1, true),
    ];
    for (name, width, iterations, stationary) in cases {
        verify_lbfgs_case(width, iterations, stationary);
        group.bench_function(name, |b| {
            b.iter_batched_ref(
                || warmed_lbfgs(width, iterations),
                |(network, optimizer)| {
                    let loss = optimizer
                        .step_lbfgs(network, |net| lbfgs_objective(net, stationary))
                        .expect("L-BFGS measured step failed");
                    black_box(loss);
                    black_box(&*network);
                    black_box(&*optimizer);
                },
                // Bound warmed tuples independently of Criterion's calibrated iteration count.
                criterion::BatchSize::NumIterations(16),
            );
        });
    }
    group.finish();
}

/// Whole public step on a cloned, already warmed model and optimizer.
fn bench_amp_case<O: arkan::optimizer::Optimizer + Clone>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    mut fixture: (KanNetwork, O),
    gradients: &(Vec<Vec<f32>>, Vec<Vec<f32>>),
    max_grad_norm: Option<f32>,
) {
    for _ in 0..5 {
        fixture
            .1
            .step(&mut fixture.0, &gradients.0, &gradients.1, max_grad_norm)
            .expect("AMP benchmark warmup failed");
    }
    group.bench_function(name, |b| {
        b.iter_batched_ref(
            || (fixture.0.clone(), fixture.1.clone()),
            |(network, optimizer)| {
                optimizer
                    .step(
                        black_box(&mut *network),
                        black_box(&gradients.0[..]),
                        black_box(&gradients.1[..]),
                        black_box(max_grad_norm),
                    )
                    .expect("AMP benchmark step failed");
                black_box(&*network);
                black_box(&*optimizer);
            },
            // Each clock covers one full step; cloning and all drops remain outside.
            criterion::BatchSize::NumIterations(1),
        );
    });
}

fn bench_amp_identity(c: &mut Criterion) {
    let network = KanNetwork::new(KanConfig {
        input_dim: 8,
        output_dim: 4,
        hidden_dims: vec![16, 16],
        grid_size: 5,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 8],
        input_std: vec![1.0; 8],
        multithreading_threshold: 1 << 20,
        simd_width: 8,
        init_seed: Some(11),
    });
    let gradients = (
        network
            .layers
            .iter()
            .map(|layer| vec![0.125; layer.weights.len()])
            .collect::<Vec<_>>(),
        network
            .layers
            .iter()
            .map(|layer| vec![-0.125; layer.bias.len()])
            .collect::<Vec<_>>(),
    );
    let norm = gradients
        .0
        .iter()
        .chain(&gradients.1)
        .flatten()
        .map(|&g| f64::from(g).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(norm > 0.25 && norm < 100.0, "AMP clipping fixture bounds");
    let cases = [
        ("checked_strict", true, false, 1.0, None),
        ("checked_skip", false, true, 1.0, None),
        ("unchecked_identity", false, false, 1.0, None),
        ("active_clip", true, false, 1.0, Some(0.25)),
        ("nonunit2", true, false, 2.0, None),
    ];
    let mut group = c.benchmark_group("amp_identity");
    for (name, fail_on_nan, skip_step_on_nan, factor, max_grad_norm) in cases {
        let safety = SafetyConfig {
            fail_on_nan,
            skip_step_on_nan,
            grad_scaling_factor: Some(factor),
            unscale_before_step: true,
        };
        let adam = Adam::new(
            &network,
            AdamConfig {
                lr: 0.125,
                weight_decay: 0.05,
                safety,
                ..Default::default()
            },
        );
        bench_amp_case(
            &mut group,
            &format!("adam_{name}"),
            (network.clone(), adam),
            &gradients,
            max_grad_norm,
        );
        let sgd = SGD::new(
            &network,
            SGDConfig {
                lr: 0.125,
                momentum: 0.5,
                weight_decay: 0.05,
                nesterov: true,
                safety,
            },
        );
        bench_amp_case(
            &mut group,
            &format!("sgd_{name}"),
            (network.clone(), sgd),
            &gradients,
            max_grad_norm,
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
    bench_lbfgs_step,
    bench_amp_identity,
);
criterion_main!(benches);
