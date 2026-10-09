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

    // All setup and numerical checks are unarmed; reuse this sole global-counter test.
    for (width, history, n, h) in [
        (1, "H1", 14, 1),
        (1, "H3", 14, 3),
        (8, "H1", 784, 1),
        (8, "H3", 784, 3),
        (1, "E1", 14, 1),
        (1, "H0", 14, 0),
        (8, "H0", 784, 0),
        (1, "H0", 0, 0),
        (1, "H1", 0, 1),
        (1, "H3", 0, 3),
    ] {
        let (network, optimizer) = l3_allocation_fixture(width, history);
        let before: Vec<u32> = LBFGS::flatten_params(&network)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let evaluations = optimizer.num_evals();
        let probe = vec![2.0; n];
        for repeat in 0..3 {
            let (output, requests, bytes) = measure(|| optimizer.two_loop_recursion(&probe));
            println!("L3 width={width} history={history} n={n} h={h} repeat={repeat} requests={requests} bytes={bytes}");
            assert_eq!(output.len(), n);
            let expected = if h == 0 { -2.0f32 } else { -1.0f32 };
            assert!(
                output.iter().all(|v| v.to_bits() == expected.to_bits()),
                "analytic direction"
            );
            assert_eq!(optimizer.num_evals(), evaluations);
            assert_eq!(
                LBFGS::flatten_params(&network)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                before
            );
            let expected_requests = if h == 0 {
                usize::from(n > 0)
            } else if n == 0 {
                1
            } else {
                3
            };
            let expected_bytes = if h == 0 { 4 * n } else { 12 * n + 8 * h };
            assert_eq!(requests, expected_requests, "L3 removed local r buffer");
            assert_eq!(bytes, expected_bytes, "L3 exact requested bytes");
            drop(output); // Unarmed; do not retain outputs across repetitions.
        }
    }
}

fn l3_allocation_fixture(width: usize, history: &str) -> (KanNetwork, LBFGS) {
    use arkan::optimizer::{LineSearchMethod, SafetyConfig};
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
    for layer in &mut network.layers {
        layer.weights.fill(0.0);
        layer.bias.fill(0.0);
    }
    network.layers[0].weights[0] = 1.0;
    let mut optimizer = LBFGS::new(
        &network,
        LBFGSConfig {
            lr: 0.25,
            max_iter: 1,
            max_eval: Some(2),
            tolerance_grad: 0.0,
            tolerance_change: 0.0,
            history_size: if history == "E1" { 1 } else { 3 },
            line_search_fn: LineSearchMethod::NoLineSearch,
            safety: SafetyConfig::strict(),
        },
    );
    let steps = match history {
        "H0" => 0,
        "H1" => 1,
        "H3" | "E1" => 3,
        _ => unreachable!(),
    };
    let n = 12 * width * width + 2 * width;
    for step in 0..steps {
        if step > 0 {
            optimizer.config.lr = 0.5;
        }
        let loss = optimizer
            .step_lbfgs(&mut network, |net| {
                let x = net.layers[0].weights[0];
                let mut gradient = vec![0.0; n];
                gradient[0] = 2.0 * x;
                Ok((f64::from(x).powi(2), gradient))
            })
            .unwrap();
        let expected = [0.5f32, 0.25, 0.125][step];
        assert_eq!(network.layers[0].weights[0].to_bits(), expected.to_bits());
        assert_eq!(loss.to_bits(), f64::from(expected).powi(2).to_bits());
    }
    let parameters = LBFGS::flatten_params(&network);
    assert_eq!(parameters.len(), n);
    assert!(parameters[1..].iter().all(|v| v.to_bits() == 0));
    assert_eq!(optimizer.num_evals(), 2 * steps);
    (network, optimizer)
}
