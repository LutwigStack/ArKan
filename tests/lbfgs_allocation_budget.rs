//! A stationary L-BFGS step needs at most its original and working snapshots.
//! Count requested bytes, not peak live memory; the objective supplies its gradient.

use arkan::{KanConfig, KanNetwork, LBFGSConfig, LBFGS};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static ARMED: AtomicBool = AtomicBool::new(false);
static REQUESTS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

fn record(bytes: usize) {
    if ARMED.load(Ordering::Relaxed) {
        REQUESTS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

// SAFETY: Every operation forwards the original pointer/layout to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    REQUESTS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    let result = f();
    ARMED.store(false, Ordering::Relaxed);
    (
        result,
        REQUESTS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

// One test avoids parallel test functions contaminating this process-wide counter.
#[test]
fn stationary_step_uses_at_most_two_parameter_buffers() {
    let ((), requests, bytes) = measure(|| {});
    assert_eq!((requests, bytes), (0, 0), "empty measurement window");

    for width in [1, 8] {
        let mut network = KanNetwork::new(KanConfig {
            input_dim: width,
            output_dim: width,
            hidden_dims: vec![width],
            grid_size: 3,
            spline_order: 3,
            input_mean: vec![0.0; width],
            input_std: vec![1.0; width],
            init_seed: Some(42),
            ..Default::default()
        });
        let before = LBFGS::flatten_params(&network);
        let parameter_bytes = before.len() * std::mem::size_of::<f32>();
        let mut optimizer = LBFGS::new(&network, LBFGSConfig::default());
        let mut gradient = Some(vec![0.0; before.len()]);
        let mut calls = 0;
        let step = || {
            optimizer.step_lbfgs(&mut network, |_| {
                calls += 1;
                Ok((7.0, gradient.take().unwrap()))
            })
        };
        let (result, requests, bytes) = measure(step);
        println!(
            "width={width} parameter_bytes={parameter_bytes} requests={requests} bytes={bytes}"
        );
        assert_eq!(result.unwrap(), 7.0);
        assert_eq!(calls, 1);
        assert_eq!(optimizer.num_evals(), 1);
        assert_eq!(LBFGS::flatten_params(&network), before);
        assert!(requests <= 2, "stationary step allocated {requests} times");
        assert!(
            bytes <= 2 * parameter_bytes,
            "stationary step requested {bytes} bytes"
        );
    }
}
