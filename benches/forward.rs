use arkan::{KanConfig, KanNetwork};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_inputs(config: &KanConfig, batch: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..batch * config.input_dim)
        .map(|_| rng.gen_range(config.grid_range.0..config.grid_range.1))
        .collect()
}

fn bench_forward(c: &mut Criterion) {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    let batch_sizes = [1_usize, 16, 64, 256];
    let mut group = c.benchmark_group("forward_batch");

    for &batch in &batch_sizes {
        let inputs = make_inputs(&config, batch, 42);
        let mut outputs = vec![0.0f32; batch * config.output_dim];
        let mut workspace = network.create_workspace(batch);

        group.throughput(Throughput::Elements((batch * config.input_dim) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &_batch| {
            b.iter(|| {
                network.forward_batch(
                    black_box(&inputs),
                    black_box(&mut outputs),
                    &mut workspace,
                );
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);
