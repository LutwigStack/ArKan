//! The README advertises zero-allocation inference and training. This asserts it.
//!
//! A counting `GlobalAlloc` wraps the system allocator and is armed only around the
//! measured region, after the workspace is warmed, so buffer growth on the first calls
//! is not counted. What remains is steady-state allocation — which for a "zero
//! allocation" claim must be exactly zero.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arkan::{Adam, AdamConfig, KanConfig, KanNetwork, SGDConfig, TrainOptions, SGD};

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        System.alloc(layout)
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
    ARMED.store(true, Ordering::Relaxed);
    f();
    ARMED.store(false, Ordering::Relaxed);
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
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
#[test]
fn allocation_budget() {
    inference_is_allocation_free();
    train_step_is_allocation_free();
    optimizer_step_is_allocation_free();
}
