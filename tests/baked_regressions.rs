use arkan::{BakedModel, KanConfig, KanNetwork};

fn network(dim: usize, range: (f32, f32)) -> KanNetwork {
    KanNetwork::new(KanConfig {
        input_dim: dim,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: 5,
        spline_order: 3,
        grid_range: range,
        input_mean: vec![0.0; dim],
        input_std: vec![1.0; dim],
        init_seed: Some(42),
        ..KanConfig::default()
    })
}

#[test]
fn wide_q16_range_keeps_the_upper_span() {
    let mut net = network(1, (-30000.0, 30000.0));
    net.layers[0].weights = (0..8).map(|i| i as f32).collect();
    net.layers[0].bias.fill(0.0);
    let baked = BakedModel::from_network(&net, None);
    let mut output = [0.0];
    baked.forward(&[30000.0], &mut output);
    // At the top, cubic weights [4,5,6,7] interpolate to 6.
    assert!((output[0] - 6.0).abs() < 0.05, "{output:?}");
}

#[test]
fn out_of_q16_range_fails_clearly_at_bake_time() {
    let net = network(1, (-40000.0, -39000.0));
    let panic = std::panic::catch_unwind(|| BakedModel::from_network(&net, None));
    let message = panic.unwrap_err();
    let message = message
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| message.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(message.contains("Q15.16"), "{message}");
}

#[test]
fn empty_calibration_uses_the_uncalibrated_fallback() {
    let mut net = network(1, (-1.0, 1.0));
    net.layers[0].weights.fill(0.0);
    net.layers[0].bias.fill(0.25);
    let baked = BakedModel::from_network(&net, Some(&[]));
    assert!(baked.uncalibrated);
    let mut output = [0.0];
    baked.forward(&[0.0], &mut output);
    assert_eq!(output, [0.25]);
}

#[test]
fn partial_calibration_is_rejected_instead_of_discarded() {
    let net = network(2, (-1.0, 1.0));
    for values in [&[0.0][..], &[0.0, 0.0, 0.5][..]] {
        assert!(std::panic::catch_unwind(|| BakedModel::from_network(&net, Some(values))).is_err());
    }
}

#[cfg(feature = "serde")]
#[test]
fn import_rejects_invalid_executable_metadata() {
    let net = network(1, (-1.0, 1.0));
    let baked = BakedModel::from_network(&net, None);
    let mutations: &[fn(&mut BakedModel)] = &[
        |b| b.layers[0].weights_i8.clear(),
        |b| b.layers.clear(),
        |b| b.layers[0].in_dim = usize::MAX,
        |b| b.layers[0].q_bias.clear(),
        |b| b.layers[0].requant_shift[0] = 128,
        |b| b.layers[0].norm_shift = 63,
        |b| b.layers[0].h_q16 = 0,
        |b| b.layers[0].order = 6,
        |b| b.layers[0].global_basis_size = 7,
        |b| b.layers[0].q_rmin = b.layers[0].q_rmax + 1,
        |b| b.layers[0].std[0] = 0.0,
        |b| b.layers[0].mean[0] = f32::NAN,
        |b| b.layers[0].s_act_out = f32::INFINITY,
        |b| b.layers[0].norm_a_fixed[0] = i32::MAX,
        |b| b.layers[0].requant_m0[0] = 0,
        |b| b.layers[0].q_bias[0] = i64::MAX,
    ];
    for (i, mutate) in mutations.iter().enumerate() {
        let mut invalid = baked.clone();
        mutate(&mut invalid);
        let bytes = invalid.to_bytes().unwrap();
        assert!(
            BakedModel::from_bytes(&bytes).is_err(),
            "accepted malformed fixture {i}"
        );
    }
    let bytes = baked.to_bytes().unwrap();
    let loaded = BakedModel::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.to_bytes().unwrap(), bytes);
}

thread_local! {
    static ALLOCATIONS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
struct Meter;
unsafe impl std::alloc::GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        std::alloc::System.alloc(layout)
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: std::alloc::Layout) {
        std::alloc::System.dealloc(pointer, layout);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        std::alloc::System.realloc(pointer, layout, size)
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;

#[test]
fn warmed_inference_does_not_allocate() {
    for order in 2..=5 {
        let config = KanConfig {
            hidden_dims: vec![4, 3],
            spline_order: order,
            ..network(2, (-1.0, 1.0)).config
        };
        let net = KanNetwork::new(config);
        let baked = BakedModel::from_network(&net, None);
        let mut output = [0.0];
        let mut workspace = baked.create_workspace();
        baked.forward_with_workspace(&[0.1, -0.1], &mut output, &mut workspace);
        ALLOCATIONS.with(|count| count.set(Some(0)));
        for _ in 0..10 {
            baked.forward_with_workspace(&[0.1, -0.1], &mut output, &mut workspace);
        }
        let count = ALLOCATIONS.with(|count| count.replace(None).unwrap());
        assert_eq!(
            count, 0,
            "allocations during warmed inference, order={order}"
        );
    }
}

#[test]
fn cached_bases_follow_each_input_across_channels_orders_and_calls() {
    for order in 2..=5 {
        let mut config = network(2, (-1.0, 1.0)).config;
        config.spline_order = order;
        config.output_dim = 3;
        let mut net = KanNetwork::new(config);
        let layer = &mut net.layers[0];
        for (j, factor) in [1.0, 2.0, -1.0].iter().enumerate() {
            for i in 0..2 {
                for k in 0..layer.global_basis_size {
                    layer.weights[(j * 2 + i) * layer.global_basis_size + k] = *factor * k as f32;
                }
            }
        }
        layer.bias.fill(0.0);
        let baked = BakedModel::from_network(&net, None);
        let mut workspace = baked.create_workspace();
        for (input, base_sum) in [
            ([-1.0, -1.0], 0.0),
            ([0.0, 0.0], 5.0),
            ([1.0, -1.0], 5.0),
            ([1.0, 1.0], 10.0),
        ] {
            let mut output = [0.0; 3];
            baked.forward_with_workspace(&input, &mut output, &mut workspace);
            // Uniform cardinal splines reproduce linear coefficients: position + (degree-1)/2.
            let sum = base_sum + (order - 1) as f32;
            for (actual, expected) in output.into_iter().zip([sum, 2.0 * sum, -sum]) {
                assert!(
                    (actual - expected).abs() < 0.12,
                    "order={order}, input={input:?}: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn fallible_bake_rejects_invalid_calibration_and_backend_ranges() {
    let net = network(2, (-1.0, 1.0));
    for values in [
        &[0.0][..],
        &[0.0, 0.0, 0.5][..],
        &[0.0, f32::NAN][..],
        &[f32::INFINITY, 0.0][..],
    ] {
        assert!(BakedModel::try_from_network(&net, Some(values)).is_err());
    }
    for range in [
        (-40000.0, -39000.0),
        (32768.0, 32769.0),
        (-30000.0, 30000.0),
    ] {
        let mut net = network(1, range);
        if range == (-30000.0, 30000.0) {
            // Endpoints fit, but one interval does not when G=1.
            net = KanNetwork::new(KanConfig {
                grid_size: 1,
                ..net.config
            });
        }
        let err = BakedModel::try_from_network(&net, None).unwrap_err();
        assert!(err.to_string().contains("Q15.16"), "{err}");
    }
}

#[test]
fn insufficient_workspace_fails_before_inference() {
    let small = BakedModel::from_network(&network(1, (-1.0, 1.0)), None);
    let large = BakedModel::from_network(&network(2, (-1.0, 1.0)), None);
    let mut workspace = small.create_workspace();
    let mut output = [0.0];
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        large.forward_with_workspace(&[0.0, 0.0], &mut output, &mut workspace);
    }))
    .is_err());
}

#[test]
fn requant_saturates_before_narrowing_to_i64() {
    let mut baked = BakedModel::from_network(&network(1, (-1.0, 1.0)), None);
    let layer = &mut baked.layers[0];
    layer.weights_i8.fill(0);
    layer.requant_m0[0] = (1 << 30) - 1;
    layer.requant_shift[0] = 0;
    layer.s_act_out = 1.0;
    for (bias, expected) in [
        (1i64 << 40, i32::MAX as f32),
        (-(1i64 << 40), i32::MIN as f32),
    ] {
        baked.layers[0].q_bias[0] = bias;
        baked.validate().unwrap();
        let mut output = [0.0];
        baked.forward(&[0.0], &mut output);
        assert_eq!(output, [expected]);
    }
}

#[test]
fn fallible_bake_rejects_stale_range_knots_with_and_without_calibration() {
    let mut net = network(1, (-1.0, 1.0));
    net.config.grid_range = (-2.0, 2.0);
    net.layers[0].grid_range = (-2.0, 2.0);
    let results: Vec<bool> = [None, Some(&[1.0][..])]
        .into_iter()
        .map(|calibration| {
            matches!(
                std::panic::catch_unwind(|| BakedModel::try_from_network(&net, calibration)),
                Ok(Err(_))
            )
        })
        .collect();
    assert_eq!(results, [true, true], "stale range cache must return Err");
}

#[test]
fn fallible_bake_rejects_stale_order_knots_with_and_without_calibration() {
    let config = KanConfig {
        spline_order: 2,
        grid_size: 1,
        ..network(1, (-1.0, 1.0)).config
    };
    let mut net = KanNetwork::new(config);
    net.config.spline_order = 5;
    let layer = &mut net.layers[0];
    layer.order = 5;
    layer.global_basis_size = 6;
    layer.local_basis_size = 6;
    layer.basis_aligned = 8;
    layer.weights.resize(6, 1.0);
    let results: Vec<bool> = [None, Some(&[0.0][..])]
        .into_iter()
        .map(|calibration| {
            matches!(
                std::panic::catch_unwind(|| BakedModel::try_from_network(&net, calibration)),
                Ok(Err(_))
            )
        })
        .collect();
    assert_eq!(
        results,
        [true, true],
        "stale order cache must return Err without panicking"
    );
}
