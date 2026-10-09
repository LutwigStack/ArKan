//! The README advertises zero-allocation inference and training. This asserts it.
//!
//! A counting `GlobalAlloc` wraps the system allocator and is armed only around the
//! measured region, after the workspace is warmed, so buffer growth on the first calls
//! is not counted. What remains is steady-state allocation — which for a "zero
//! allocation" claim must be exactly zero.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arkan::{Adam, AdamConfig, KanConfig, KanNetwork, SGDConfig, TrainOptions, SGD};

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static CLASSES: [AtomicUsize; 3] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

struct Counting;

fn record(class: usize, bytes: usize) {
    if ARMED.load(Ordering::Relaxed) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(bytes, Ordering::Relaxed);
        CLASSES[class].fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: Every operation forwards the original pointer/layout to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(0, layout.size());
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(1, layout.size());
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(2, new_size);
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Runs `f` with allocation counting armed; returns (allocations, bytes).
fn measure(f: impl FnOnce()) -> (usize, usize) {
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    for class in &CLASSES {
        class.store(0, Ordering::Relaxed);
    }
    ARMED.store(true, Ordering::Relaxed);
    f();
    ARMED.store(false, Ordering::Relaxed);
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

fn entropy_is_allocation_free() {
    let coefficients: [f32; 18] = std::array::from_fn(|i| (i as f32 - 7.0) * 0.13);
    let cases = [
        (8, 4),
        (16, 4),
        (18, 4),
        (8, 8),
        (8, 1),
        (0, 4),
        (8, 0),
        (8, 9),
    ];
    // Collect every row before assertions, so a baseline failure retains tail controls.
    let observations = cases.map(|(length, group_size)| {
        let control = measure(|| {});
        let result = measure(|| {
            black_box(arkan::loss::entropy_regularization(
                black_box(&coefficients[..length]),
                black_box(group_size),
            ));
        });
        let classes: [usize; 3] = std::array::from_fn(|i| CLASSES[i].load(Ordering::Relaxed));
        (control, result, classes)
    });
    for ((length, group_size), (control, (requests, bytes), classes)) in
        cases.into_iter().zip(observations)
    {
        println!(
            "entropy len={length} group={group_size} calls=1 control={control:?} \
             requests={requests} bytes={bytes} alloc={} alloc_zeroed={} realloc={}",
            classes[0], classes[1], classes[2]
        );
    }
    assert_eq!(
        observations[1].1, observations[2].1,
        "incomplete tail allocation cost"
    );
    for (control, result, _) in observations {
        assert_eq!(control, (0, 0), "empty entropy measurement window");
        assert_eq!(
            result,
            (0, 0),
            "entropy must not allocate temporary storage"
        );
    }
}

fn net(batch: usize) -> (KanNetwork, KanConfig) {
    let config = KanConfig {
        input_dim: 8,
        output_dim: 4,
        hidden_dims: vec![16, 16],
        grid_size: 5,
        spline_order: 3,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 8],
        input_std: vec![1.0; 8],
        multithreading_threshold: 1 << 20, // keep it single-threaded regardless of features
        simd_width: 8,
        init_seed: Some(11),
    };
    let _ = batch;
    (KanNetwork::new(config.clone()), config)
}

fn inference_is_allocation_free() {
    let batch = 32;
    let (network, config) = net(batch);
    let input = vec![0.25f32; batch * config.input_dim];
    let mut out = vec![0.0f32; batch * config.output_dim];
    let mut ws = network.create_workspace(batch);

    // Warm up: let every buffer reach its steady-state capacity.
    for _ in 0..5 {
        network.forward_batch(&input, &mut out, &mut ws);
    }

    // Control: measure an empty region first. Anything counted here is background
    // (lazy statics, test harness), not the library, and would otherwise be
    // misattributed to the first real measurement.
    let (ctrl_a, ctrl_b) = measure(|| {});
    println!("control (empty region): {ctrl_a} allocations, {ctrl_b} bytes");

    let (allocs, bytes) = measure(|| {
        for _ in 0..50 {
            network.forward_batch(&input, &mut out, &mut ws);
        }
    });
    println!("forward_batch x50: {allocs} allocations, {bytes} bytes");
    assert_eq!(
        allocs, 0,
        "forward_batch allocated {allocs} times ({bytes} B)"
    );

    let single = vec![0.25f32; config.input_dim];
    let mut single_out = vec![0.0f32; config.output_dim];
    for _ in 0..5 {
        network.forward_single(&single, &mut single_out, &mut ws);
    }
    let (allocs, bytes) = measure(|| {
        for _ in 0..50 {
            network.forward_single(&single, &mut single_out, &mut ws);
        }
    });
    println!("forward_single x50: {allocs} allocations, {bytes} bytes");
    assert_eq!(
        allocs, 0,
        "forward_single allocated {allocs} times ({bytes} B)"
    );
}

fn train_step_is_allocation_free() {
    let batch = 32;
    let (mut network, config) = net(batch);
    let input = vec![0.25f32; batch * config.input_dim];
    let target = vec![0.5f32; batch * config.output_dim];
    let mut ws = network.create_workspace(batch);

    // Warm up so every workspace buffer has reached steady-state capacity.
    for _ in 0..5 {
        network.train_step(&input, &target, None, 0.01, &mut ws);
    }

    let (allocs, bytes) = measure(|| {
        for _ in 0..50 {
            network.train_step(&input, &target, None, 0.01, &mut ws);
        }
    });
    println!("train_step x50: {allocs} allocations, {bytes} bytes");
    assert_eq!(
        allocs, 0,
        "train_step allocated {allocs} times ({bytes} B) over 50 warmed steps \
         -- the README advertises zero-allocation training"
    );
}

/// The optimizer step is the other half of a training iteration. A zero-allocation
/// `train_step` is worth little if `Adam::step` allocates the network's weights again
/// every iteration.
fn optimizer_step_is_allocation_free() {
    let batch = 32;
    let (mut network, config) = net(batch);
    let input = vec![0.25f32; batch * config.input_dim];
    let target = vec![0.5f32; batch * config.output_dim];
    let mut ws = network.create_workspace(batch);
    let opts = TrainOptions::default();

    let mut adam = Adam::new(&network, AdamConfig::default());
    for _ in 0..5 {
        network
            .train_step_with_optimizer(&input, &target, None, &mut ws, &mut adam, &opts)
            .unwrap();
    }
    let (allocs, bytes) = measure(|| {
        for _ in 0..50 {
            network
                .train_step_with_optimizer(&input, &target, None, &mut ws, &mut adam, &opts)
                .unwrap();
        }
    });
    println!("train_step_with_optimizer(Adam) x50: {allocs} allocations, {bytes} bytes");
    assert_eq!(allocs, 0, "Adam path allocated {allocs} times ({bytes} B)");

    let mut sgd = SGD::new(&network, SGDConfig::default());
    for _ in 0..5 {
        network
            .train_step_with_optimizer(&input, &target, None, &mut ws, &mut sgd, &opts)
            .unwrap();
    }
    let (allocs, bytes) = measure(|| {
        for _ in 0..50 {
            network
                .train_step_with_optimizer(&input, &target, None, &mut ws, &mut sgd, &opts)
                .unwrap();
        }
    });
    println!("train_step_with_optimizer(SGD) x50: {allocs} allocations, {bytes} bytes");
    assert_eq!(allocs, 0, "SGD path allocated {allocs} times ({bytes} B)");
}

/// Single entry point on purpose.
///
/// The counter behind `measure` is a process-global `GlobalAlloc`, but libtest runs
/// `#[test]` functions on parallel threads by default. As separate tests these three
/// sections each observed the *sum* of all threads' allocations and reported identical
/// bogus totals (228 allocations / 550240 bytes each). Sequencing them inside one test
/// is the fix that does not depend on remembering `--test-threads=1`.
/// `create_workspace(n)` must allocate for `n`, not for `multithreading_threshold`.
///
/// `Workspace::new` used to pre-reserve `config.multithreading_threshold` rows -
/// a field documented as consulted only with the `parallel` feature - so a caller
/// asking for a batch-1 workspace on the latency path got the default 128 rows of
/// buffers, tunable only through a knob named after multithreading. Measured in
/// bytes rather than asserted structurally, because the point is the memory.
fn workspace_size_ignores_the_multithreading_threshold() {
    let mut per_threshold = Vec::new();
    for threshold in [1usize, 8, 128, 1 << 20] {
        let config = KanConfig {
            input_dim: 784,
            output_dim: 10,
            hidden_dims: vec![64, 32],
            grid_size: 12,
            spline_order: 3,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.0; 784],
            input_std: vec![1.0; 784],
            multithreading_threshold: threshold,
            simd_width: 8,
            init_seed: Some(1),
        };
        let network = KanNetwork::new(config);

        let mut ws = None;
        let (_, bytes) = measure(|| ws = Some(network.create_workspace(1)));
        assert_eq!(ws.expect("workspace").batch_capacity(), 1);
        println!("threshold={threshold}: create_workspace(1) allocated {bytes} bytes");
        per_threshold.push(bytes);
    }

    assert!(
        per_threshold.windows(2).all(|w| w[0] == w[1]),
        "a batch-1 workspace must cost the same at every multithreading_threshold, \
         got {per_threshold:?} for thresholds [1, 8, 128, 1<<20]"
    );
}

/// The `# Errors` contract of `try_create_workspace` has to hold for a config that
/// `validate` accepts. It used to panic out of the `Result` instead, because the
/// infallible `Workspace::new` reserved `multithreading_threshold` rows *before*
/// the `try_reserve` that would have reported the overflow.
fn try_create_workspace_does_not_panic_on_a_valid_config() {
    let config = KanConfig {
        input_dim: 784,
        output_dim: 10,
        hidden_dims: vec![64, 32],
        grid_size: 12,
        spline_order: 3,
        grid_range: (-3.0, 3.0),
        input_mean: vec![0.0; 784],
        input_std: vec![1.0; 784],
        // The natural way to write "never take the parallel branch".
        multithreading_threshold: 1 << 30,
        simd_width: 8,
        init_seed: Some(1),
    };
    assert!(config.validate().is_ok(), "config must be accepted");

    let network = KanNetwork::new(config);
    let ws = network
        .try_create_workspace(32)
        .expect("a batch-32 workspace fits regardless of multithreading_threshold");
    assert_eq!(ws.batch_capacity(), 32);
}

#[test]
fn allocation_budget() {
    entropy_is_allocation_free();
    inference_is_allocation_free();
    train_step_is_allocation_free();
    optimizer_step_is_allocation_free();
    workspace_size_ignores_the_multithreading_threshold();
    try_create_workspace_does_not_panic_on_a_valid_config();
    #[cfg(feature = "serde")]
    serialization::exports_have_one_output_allocation();
}

#[cfg(feature = "serde")]
mod serialization {
    use super::*;
    use arkan::baked::BakedModel;
    fn serialization_fixture_network() -> KanNetwork {
        let mut network = KanNetwork::new(KanConfig {
            input_dim: 2,
            output_dim: 1,
            hidden_dims: vec![2],
            grid_size: 3,
            spline_order: 2,
            grid_range: (-2.0, 2.0),
            input_mean: vec![0.25, -0.5],
            input_std: vec![0.5, 2.0],
            init_seed: Some(7),
            ..KanConfig::default()
        });
        for (i, layer) in network.layers.iter_mut().enumerate() {
            for (j, weight) in layer.weights.iter_mut().enumerate() {
                *weight = (j as f32 - 3.0) * (i + 1) as f32 / 32.0;
            }
            for (j, bias) in layer.bias.iter_mut().enumerate() {
                *bias = (j + 1) as f32 / 16.0;
            }
        }
        network.layers[1].set_normalization(&[0.125, -0.25], &[1.5, 0.75]);
        network
    }

    fn large_network(width: usize) -> KanNetwork {
        KanNetwork::new(KanConfig {
            input_dim: 21,
            output_dim: 24,
            hidden_dims: vec![width, width],
            grid_size: 5,
            spline_order: 3,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.0; 21],
            input_std: vec![1.0; 21],
            multithreading_threshold: 128,
            simd_width: 8,
            init_seed: Some(42),
        })
    }

    fn networks() -> [KanNetwork; 3] {
        [
            serialization_fixture_network(),
            large_network(64),
            large_network(128),
        ]
    }

    fn baked_models(networks: &[KanNetwork; 3]) -> [BakedModel; 3] {
        [
            BakedModel::try_from_network(&networks[0], Some(&[-0.5, 0.25, 0.25, -0.5, 1.0, 0.75]))
                .unwrap(),
            BakedModel::try_from_network(&networks[1], None).unwrap(),
            BakedModel::try_from_network(&networks[2], None).unwrap(),
        ]
    }

    pub(super) fn exports_have_one_output_allocation() {
        let networks = networks();
        let baked = baked_models(&networks);
        let ids = ["n_literal", "b_literal", "n_64", "b_64", "n_128", "b_128"];
        let observations = std::array::from_fn::<_, 6, _>(|row| {
            let i = row / 2;
            let body = if row % 2 == 0 {
                bincode::serialize(&networks[i]).unwrap()
            } else {
                bincode::serialize(&baked[i]).unwrap()
            };
            let header = if row % 2 == 0 { 9 } else { 16 };
            let mut output = None;
            let counts = measure(|| {
                output = Some(if row % 2 == 0 {
                    networks[i].to_bytes()
                } else {
                    baked[i].to_bytes()
                });
            });
            let classes: [usize; 3] = std::array::from_fn(|i| CLASSES[i].load(Ordering::Relaxed));
            let output = output.unwrap().unwrap();
            let length = output.len();
            let capacity = output.capacity();
            drop(output);
            (header + body.len(), counts, classes, length, capacity)
        });
        // Print all original rows before the intentional one-request RED assertion.
        for (id, (expected, counts, classes, length, capacity)) in ids.into_iter().zip(observations)
        {
            println!("serialization id={id} expected={expected} requests={} bytes={} classes={classes:?} len={length} cap={capacity}", counts.0, counts.1);
        }
        for (expected, counts, classes, length, capacity) in observations {
            assert_eq!(
                counts,
                (1, expected),
                "export must allocate only its returned output"
            );
            assert_eq!(classes, [1, 0, 0], "export must not reallocate");
            assert_eq!((length, capacity), (expected, expected));
        }
    }
}
