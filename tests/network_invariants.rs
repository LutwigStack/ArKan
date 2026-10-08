use arkan::{KanConfig, KanNetwork, Workspace};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNTING.with(|armed| {
            if armed.get() {
                ALLOCATIONS.with(|n| n.set(n.get() + 1));
            }
        });
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNTING.with(|armed| {
            if armed.get() {
                ALLOCATIONS.with(|n| n.set(n.get() + 1));
            }
        });
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        COUNTING.with(|armed| {
            if armed.get() {
                ALLOCATIONS.with(|n| n.set(n.get() + 1));
            }
        });
        System.realloc(ptr, layout, size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn config(hidden_dims: Vec<usize>) -> KanConfig {
    KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims,
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        init_seed: Some(7),
        ..KanConfig::default()
    }
}

#[test]
fn single_layer_batch_does_not_allocate_after_warmup() {
    let net = KanNetwork::new(config(vec![]));
    let mut ws = net.create_workspace(4);
    let input = [0.3; 8];
    let mut output = [0.0; 4];
    net.try_forward_batch(&input, &mut output, &mut ws).unwrap();
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|armed| armed.set(true));
    for _ in 0..20 {
        net.try_forward_batch(&input, &mut output, &mut ws).unwrap();
    }
    COUNTING.with(|armed| armed.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}

#[test]
fn empty_batch_policy_is_independent_of_topology() {
    for hidden in [vec![], vec![3]] {
        let mut net = KanNetwork::new(config(hidden));
        let mut ws = net.create_workspace(1);
        assert!(net.try_forward_batch(&[], &mut [], &mut ws).is_ok());
        assert!(net
            .try_forward_batch_training(&[], &mut [], &mut ws)
            .is_ok());
        assert!(net.try_forward_batch(&[], &mut [0.0], &mut ws).is_err());
        assert!(net
            .try_forward_batch_training(&[], &mut [0.0], &mut ws)
            .is_err());
        assert_eq!(
            net.try_train_step(&[], &[], None, 0.01, &mut ws).unwrap(),
            0.0
        );
    }
}

#[test]
fn zero_input_dimension_returns_errors_before_division() {
    let mut net = KanNetwork::new(config(vec![]));
    let mut ws = net.create_workspace(1);
    net.config.input_dim = 0;
    assert!(net.try_forward_batch(&[], &mut [], &mut ws).is_err());
    assert!(net
        .try_forward_batch_training(&[], &mut [], &mut ws)
        .is_err());
    assert!(net.try_train_step(&[], &[], None, 0.01, &mut ws).is_err());
    assert!(net.try_create_workspace(1).is_err());
}

#[test]
fn cleared_layers_cannot_leave_output_untouched_successfully() {
    let mut net = KanNetwork::new(config(vec![]));
    let mut ws = net.create_workspace(1);
    net.layers.clear();
    let mut output = [123.0];
    assert!(net
        .try_forward_single(&[0.0; 2], &mut output, &mut ws)
        .is_err());
    assert_eq!(output, [123.0]);
    assert!(net
        .try_forward_batch(&[0.0; 2], &mut output, &mut ws)
        .is_err());
}

#[test]
fn public_layout_mutations_return_errors() {
    for mutation in 0..5 {
        let mut net = KanNetwork::new(config(vec![3]));
        let mut ws = net.create_workspace(1);
        match mutation {
            0 => net.config.hidden_dims[0] = 4,
            1 => net.layers[0].weights.clear(),
            2 => net.layers[0].out_dim = 0,
            3 => net.config.simd_width = 0,
            _ => net.config.grid_size = usize::MAX,
        }
        assert!(net
            .try_forward_single(&[0.0; 2], &mut [0.0], &mut ws)
            .is_err());
    }
}

#[test]
fn workspace_reserve_grows_for_a_different_model_at_same_batch_size() {
    let first = config(vec![]);
    let wider = config(vec![16]);
    let mut ws = Workspace::new(&first);
    ws.try_reserve(4, &first).unwrap();
    ws.try_reserve(4, &wider).unwrap();
    assert!(ws.z_buffer.capacity() >= 64);
    assert!(ws.basis_values.capacity() >= 64 * wider.basis_size_aligned());
    let net = KanNetwork::new(wider);
    let mut reused = [0.0; 4];
    let mut fresh = [0.0; 4];
    net.try_forward_batch(&[0.3; 8], &mut reused, &mut ws)
        .unwrap();
    net.try_forward_batch(&[0.3; 8], &mut fresh, &mut net.create_workspace(4))
        .unwrap();
    assert_eq!(reused, fresh);
}

#[test]
fn gradient_buffers_expose_exact_active_shapes_on_reuse() {
    let mut ws = Workspace::new(&config(vec![]));
    ws.try_prepare_grad_buffers(&[(10, 3), (7, 2)]).unwrap();
    ws.try_prepare_grad_buffers(&[(4, 1)]).unwrap();
    assert_eq!(ws.weight_grads.len(), 1);
    assert_eq!(ws.bias_grads.len(), 1);
    assert_eq!(ws.weight_grads[0].len(), 4);
    assert_eq!(ws.bias_grads[0].len(), 1);
}

#[cfg(feature = "serde")]
#[test]
fn malformed_models_are_rejected_at_all_import_boundaries() {
    for mutation in 0..6 {
        let mut net = KanNetwork::new(config(vec![]));
        match mutation {
            0 => net.config.input_dim = 0,
            1 => net.layers.clear(),
            2 => net.layers[0].weights.clear(),
            3 => net.layers[0].weights[0] = f32::NAN,
            4 => net.layers[0].mean[0] = f32::INFINITY,
            _ => net.layers[0].std[0] = 0.0,
        }
        let raw = bincode::serialize(&net).unwrap();
        assert!(
            KanNetwork::from_bytes_legacy(&raw).is_err(),
            "mutation {mutation}"
        );
        assert!(
            bincode::deserialize::<KanNetwork>(&raw).is_err(),
            "mutation {mutation}"
        );
        let mut bytes = b"ARKAN".to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(raw);
        assert!(
            KanNetwork::from_bytes(&bytes).is_err(),
            "mutation {mutation}"
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn import_rebuilds_derived_network_caches() {
    let net = KanNetwork::new(config(vec![3]));
    let mut data = serde_json::to_value(&net).unwrap();
    data["layer_dims"] = serde_json::json!([999]);
    data["layer_param_sizes"] = serde_json::json!([[0, 0]]);
    let mut loaded: KanNetwork = serde_json::from_value(data).unwrap();
    let mut ws = loaded.create_workspace(1);
    let mut expected = [0.0];
    let mut actual = [0.0];
    net.try_forward_single(&[0.2; 2], &mut expected, &mut net.create_workspace(1))
        .unwrap();
    loaded
        .try_forward_single(&[0.2; 2], &mut actual, &mut ws)
        .unwrap();
    assert_eq!(actual, expected);
    assert!(loaded
        .try_train_step(&[0.2; 2], &[0.1], None, 0.01, &mut ws)
        .unwrap()
        .is_finite());
    let canonical = serde_json::to_value(loaded).unwrap();
    assert_eq!(canonical["layer_dims"], serde_json::json!([2, 3, 1]));
    assert_ne!(canonical["layer_param_sizes"], serde_json::json!([[0, 0]]));
}

#[test]
fn network_clipping_preserves_large_finite_gradient_direction() {
    let mut net = KanNetwork::new(config(vec![]));
    let mut ws = net.create_workspace(1);
    let original_bias = net.layers[0].bias[0];
    let opts = arkan::TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.0,
    };
    net.try_train_step_with_options(&[0.2; 2], &[1e20], None, 0.1, &mut ws, &opts)
        .unwrap();
    let norm = ws
        .weight_grads
        .iter()
        .chain(&ws.bias_grads)
        .flatten()
        .map(|&g| (g as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "clipped norm {norm}");
    assert!(net.layers[0].bias[0] > original_bias);
}

#[cfg(feature = "serde")]
#[test]
fn canonical_model_bytes_survive_import_unchanged() {
    let net = KanNetwork::new(config(vec![3]));
    let bytes = net.to_bytes().unwrap();
    let loaded = KanNetwork::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.to_bytes().unwrap(), bytes);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_training_reuses_scratch_across_layer_shapes() {
    let mut cfg = config(vec![3, 4]);
    cfg.multithreading_threshold = 1;
    let mut net = KanNetwork::new(cfg);
    let mut ws = net.create_workspace(8);
    let input = [0.2; 16];
    let target = [0.1; 8];
    // Warm Rayon and scratch capacities across all layer shapes.
    for _ in 0..2 {
        net.try_train_step(&input, &target, None, 0.0, &mut ws)
            .unwrap();
    }
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|armed| armed.set(true));
    for _ in 0..10 {
        net.try_train_step(&input, &target, None, 0.0, &mut ws)
            .unwrap();
    }
    COUNTING.with(|armed| armed.set(false));
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}

#[test]
fn invalid_training_options_are_rejected_before_parameters_change() {
    for case in 0..8 {
        let mut net = KanNetwork::new(config(vec![]));
        let mut ws = net.create_workspace(1);
        let original_weights = net.layers[0].weights.clone();
        let original_bias = net.layers[0].bias.clone();
        let mut opts = arkan::TrainOptions::default();
        let mut lr = 0.01;
        match case {
            0 => opts.max_grad_norm = Some(f32::NAN),
            1 => opts.max_grad_norm = Some(0.0),
            2 => opts.max_grad_norm = Some(-1.0),
            3 => opts.weight_decay = f32::INFINITY,
            4 => opts.weight_decay = -1.0,
            5 => lr = f32::NAN,
            6 => lr = f32::INFINITY,
            _ => lr = -0.01,
        }
        assert!(
            net.try_train_step_with_options(&[0.2; 2], &[0.1], None, lr, &mut ws, &opts)
                .is_err(),
            "case {case}"
        );
        assert_eq!(net.layers[0].weights, original_weights);
        assert_eq!(net.layers[0].bias, original_bias);
        assert_eq!(ws.history_batch_size(), 0);
    }
}

#[test]
fn empty_training_does_not_decay_parameters() {
    let mut net = KanNetwork::new(config(vec![]));
    let mut ws = net.create_workspace(1);
    let original_weights = net.layers[0].weights.clone();
    let opts = arkan::TrainOptions {
        max_grad_norm: None,
        weight_decay: 0.2,
    };
    assert_eq!(
        net.try_train_step_with_options(&[], &[], None, 0.1, &mut ws, &opts)
            .unwrap(),
        0.0
    );
    assert_eq!(net.layers[0].weights, original_weights);
}
