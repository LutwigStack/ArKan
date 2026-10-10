use arkan::{BakedModel, KanConfig, KanNetwork};

fn network(hidden: bool, range: (f32, f32), grid: usize, order: usize) -> KanNetwork {
    KanNetwork::new(KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: if hidden { vec![1] } else { vec![] },
        grid_size: grid,
        spline_order: order,
        grid_range: range,
        input_mean: vec![0.0],
        input_std: vec![1.0],
        init_seed: Some(42),
        ..KanConfig::default()
    })
}

fn check_tiny_channel(bias_only: bool) {
    for order in 2..=5 {
        let mut net = network(true, (-3.0, 3.0), 5, order);
        let tiny = 2.0f32.powi(-21);
        net.layers[0]
            .weights
            .fill(if bias_only { 0.0 } else { tiny });
        net.layers[0].bias.fill(if bias_only { tiny } else { 0.0 });
        net.layers[1].weights = (0..5 + order).map(|k| k as f32).collect();
        net.layers[1].bias.fill(0.0);
        net.layers[1].std[0] = tiny;
        for calibration in [None, Some(&[0.0][..])] {
            let baked = BakedModel::try_from_network(&net, calibration).unwrap();
            let mut workspace = baked.create_workspace();
            for x in [-2.0, 0.0, 2.0] {
                let mut cpu = [0.0];
                let mut actual = [0.0];
                net.try_forward_single(&[x], &mut cpu, &mut net.create_workspace(1))
                    .unwrap();
                baked.forward_with_workspace(&[x], &mut actual, &mut workspace);
                assert!(
                    (actual[0] - cpu[0]).abs() < 0.035,
                    "order={order}, bias_only={bias_only}, x={x}: {actual:?} vs {cpu:?}"
                );
            }
        }
    }
}

#[test]
fn tiny_weights_preserve_a_normalized_hidden_feature() {
    check_tiny_channel(false);
}

#[test]
fn tiny_bias_only_preserves_a_normalized_hidden_feature() {
    check_tiny_channel(true);
}

#[test]
fn narrow_grid_uses_the_whole_extent_without_interval_rounding_drift() {
    for order in 2..=5 {
        let mut net = network(false, (0.0, 95.0 / 65536.0), 64, order);
        net.layers[0].weights = (0..64 + order).map(|k| k as f32).collect();
        net.layers[0].bias.fill(0.0);
        // Calibrate at the maximum so output clipping cannot explain geometry error.
        let baked = BakedModel::try_from_network(&net, Some(&[95.0 / 65536.0])).unwrap();
        let mut workspace = baked.create_workspace();
        for tick in [0.0, 1.0, 16.0, 32.0, 64.0, 80.0, 94.0] {
            let input = [tick / 65536.0];
            let mut cpu = [0.0];
            let mut actual = [0.0];
            net.try_forward_single(&input, &mut cpu, &mut net.create_workspace(1))
                .unwrap();
            baked.forward_with_workspace(&input, &mut actual, &mut workspace);
            assert!(
                (actual[0] - cpu[0]).abs() < 0.3,
                "order={order}, tick={tick}: {actual:?} vs {cpu:?}"
            );
        }
    }
}

#[test]
fn unrepresentable_nonzero_channel_scales_are_rejected() {
    for bias_only in [false, true] {
        let mut net = network(false, (-3.0, 3.0), 5, 3);
        net.layers[0]
            .weights
            .fill(if bias_only { 0.0 } else { f32::MIN_POSITIVE });
        net.layers[0]
            .bias
            .fill(if bias_only { f32::MIN_POSITIVE } else { 0.0 });
        assert!(BakedModel::try_from_network(&net, Some(&[0.0])).is_err());
    }
}

#[test]
fn finite_channel_scales_with_unrepresentable_requantization_are_rejected() {
    // Each coefficient scale fits f32, but the requant multiplier would be
    // rounded to zero or exceed its fixed-point maximum with heuristic calibration.
    for weight in [1e-25, 1e30] {
        let mut net = network(false, (-3.0, 3.0), 5, 3);
        net.layers[0].weights.fill(weight);
        net.layers[0].bias.fill(0.0);
        assert!(BakedModel::try_from_network(&net, None).is_err());
    }
}

#[test]
fn whole_extent_preserves_wide_grid_endpoints_and_clamps() {
    for range in [(-32768.0, 32768.0), (0.0, 95.0 / 65536.0)] {
        for order in 2..=5 {
            let mut net = network(false, range, 64, order);
            net.layers[0].weights = (0..64 + order).map(|k| k as f32).collect();
            net.layers[0].bias.fill(0.0);
            let baked = BakedModel::try_from_network(&net, Some(&[range.1])).unwrap();
            #[cfg(feature = "serde")]
            let baked = BakedModel::from_bytes(&baked.to_bytes().unwrap()).unwrap();
            let mut workspace = baked.create_workspace();
            let mut bottom = [0.0];
            let mut top = [0.0];
            baked.forward_with_workspace(&[range.0], &mut bottom, &mut workspace);
            baked.forward_with_workspace(&[range.1], &mut top, &mut workspace);
            // Uniform B-splines reproduce a coefficient ramp with offset (order-1)/2.
            let offset = (order - 1) as f32 / 2.0;
            assert!((bottom[0] - offset).abs() < 0.3, "{bottom:?}");
            // The narrow upper clamp loses less than one tick (64/95 of a span).
            assert!((top[0] - (64.0 + offset)).abs() < 0.95, "{top:?}");
            for (input, expected) in [(range.0 - 1.0, bottom), (range.1 + 1.0, top)] {
                let mut actual = [0.0];
                baked.forward_with_workspace(&[input], &mut actual, &mut workspace);
                assert_eq!(actual, expected);
            }
        }
    }
}
