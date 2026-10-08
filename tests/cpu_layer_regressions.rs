use arkan::{KanConfig, KanLayer, KanNetwork, Workspace};

fn config() -> KanConfig {
    KanConfig::builder()
        .input_dim(2)
        .output_dim(1)
        .hidden_dims(vec![2, 2])
        .normalization(vec![1.0, -0.5], vec![0.2, 0.5])
        .seed(42)
        .build()
        .unwrap()
}

#[test]
fn hidden_layers_of_input_width_use_identity_normalization() {
    let cfg = config();
    let mut net = KanNetwork::new(cfg.clone());
    assert_eq!(net.layers[0].mean, cfg.input_mean);
    for layer in &net.layers[1..] {
        assert_eq!(layer.mean, vec![0.0; 2]);
        assert_eq!(layer.std, vec![1.0; 2]);
    }
    assert_eq!(KanLayer::new(2, 1, &cfg).mean, cfg.input_mean);
    assert_eq!(KanLayer::new(3, 1, &cfg).mean, vec![0.0; 3]);
    let mut workspace = net.create_workspace(2);
    let loss = net.train_step(
        &[1.0, -0.5, 1.1, -0.4],
        &[1.0, 0.0],
        None,
        0.0,
        &mut workspace,
    );
    assert!(loss.is_finite());
    assert!(workspace.weight_grads[0].iter().all(|x| x.is_finite()));
    assert!(workspace.weight_grads[0].iter().any(|&x| x != 0.0));
}

#[test]
fn constructors_reject_invalid_layer_configuration_without_panicking() {
    for case in 0..7 {
        let mut cfg = config();
        match case {
            0 => cfg.simd_width = 0,
            1 => cfg.spline_order = 8,
            2 => cfg.grid_size = 0,
            3 => cfg.grid_range = (f32::NEG_INFINITY, 1.0),
            4 => cfg.input_std.clear(),
            5 => cfg.input_mean.clear(),
            _ => cfg.input_std[0] = -1.0,
        }
        let result = std::panic::catch_unwind(|| KanLayer::try_new(2, 1, &cfg));
        assert!(result.is_ok(), "case {case} panicked");
        assert!(
            result.unwrap().is_err(),
            "case {case} accepted invalid config"
        );
    }
}

#[test]
fn normalization_rejects_nonfinite_stats_without_mutation() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for is_mean in [true, false] {
            let mut cfg = config();
            if is_mean {
                cfg.input_mean[0] = bad;
            } else {
                cfg.input_std[0] = bad;
            }
            assert!(cfg.validate().is_err());
            assert!(KanLayer::try_new(2, 1, &cfg).is_err());
            let mut original = config();
            assert!(original
                .set_normalization(cfg.input_mean, cfg.input_std)
                .is_err());
            assert_eq!(original.input_mean, config().input_mean);
            assert_eq!(original.input_std, config().input_std);
        }
    }
}

#[test]
fn empty_batch_still_validates_output_shape() {
    let cfg = config();
    let layer = KanLayer::new(2, 1, &cfg);
    let mut workspace = Workspace::new(&cfg);
    assert!(layer
        .try_forward_batch(&[], &mut [123.0], &mut workspace)
        .is_err());
    assert!(layer
        .try_forward_batch(&[], &mut [], &mut workspace)
        .is_ok());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_overwrites_reused_input_gradients_and_is_repeatable() {
    use arkan::spline::{compute_knots, find_span};
    let cfg = config();
    let layer = KanLayer::new(2, 1, &cfg);
    let knots = compute_knots(cfg.grid_size, cfg.spline_order, cfg.grid_range);
    let mut workspace = Workspace::new(&cfg);
    for batch in [1, 7, 128, 513] {
        let input: Vec<f32> = (0..batch * 2).map(|i| (i as f32 * 0.37).sin()).collect();
        let spans: Vec<u32> = input
            .iter()
            .map(|&x| find_span(x, &knots, cfg.spline_order, cfg.grid_size) as u32)
            .collect();
        let go: Vec<f32> = (0..batch).map(|i| (i as f32 * 0.73).cos()).collect();
        let mut expected = vec![9.0; input.len()];
        let mut ew = vec![0.0; layer.weights.len()];
        let mut eb = vec![0.0; layer.bias.len()];
        layer.backward(
            &input,
            &spans,
            &go,
            Some(&mut expected),
            &mut ew,
            &mut eb,
            &mut workspace,
        );
        let mut gi = vec![9.0; input.len()];
        let mut first = None;
        for _ in 0..12 {
            let mut gw = vec![0.0; ew.len()];
            let mut gb = vec![0.0; eb.len()];
            layer.backward_parallel(&input, &spans, &go, Some(&mut gi), &mut gw, &mut gb);
            assert_eq!(gi, expected);
            for (&a, &b) in gw.iter().chain(&gb).zip(ew.iter().chain(&eb)) {
                assert!((a - b).abs() < 1e-4, "{a} != {b}");
            }
            if let Some((ref fw, ref fb)) = first {
                assert_eq!(&gw, fw);
                assert_eq!(&gb, fb);
            } else {
                first = Some((gw, gb));
            }
        }
        layer.backward_parallel(
            &input,
            &spans,
            &vec![0.0; batch],
            Some(&mut gi),
            &mut ew,
            &mut eb,
        );
        assert!(gi.iter().all(|&x| x == 0.0));
    }
}

#[test]
fn layer_normalization_update_rejects_nonfinite_before_clamping() {
    let mut layer = KanLayer::new(2, 1, &config());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(layer.try_set_normalization(&[bad, 0.0], &[1.0; 2]).is_err());
        assert!(layer.try_set_normalization(&[0.0; 2], &[bad, 1.0]).is_err());
        assert_eq!(layer.mean, config().input_mean);
        assert_eq!(layer.std, config().input_std);
    }
    layer
        .try_set_normalization(&[0.0; 2], &[0.0, -1.0])
        .unwrap();
    assert_eq!(layer.std, vec![arkan::config::EPSILON; 2]);
}

#[cfg(feature = "serde")]
#[test]
fn deserialization_rejects_invalid_layout_and_preserves_valid_models() {
    let layer = KanLayer::new(2, 1, &config());
    let valid = serde_json::to_value(&layer).unwrap();
    let restored: KanLayer = serde_json::from_value(valid.clone()).unwrap();
    assert_eq!(restored.weights, layer.weights);
    for (field, value) in [
        ("in_dim", serde_json::json!(0)),
        ("out_dim", serde_json::json!(usize::MAX)),
        ("order", serde_json::json!(usize::MAX)),
        ("grid_size", serde_json::json!(usize::MAX)),
        ("global_basis_size", serde_json::json!(1)),
        ("local_basis_size", serde_json::json!(1)),
        ("basis_aligned", serde_json::json!(1)),
        ("simd_width", serde_json::json!(0)),
        ("mean", serde_json::json!([])),
        ("std", serde_json::json!([0.0, 1.0])),
        ("weights", serde_json::json!([])),
        ("bias", serde_json::json!([])),
        ("grid_range", serde_json::json!([1.0, -1.0])),
    ] {
        let mut data = valid.clone();
        data[field] = value;
        let result = std::panic::catch_unwind(|| serde_json::from_value::<KanLayer>(data));
        assert!(result.is_ok(), "{field} panicked");
        assert!(result.unwrap().is_err(), "invalid {field} accepted");
    }
}

#[cfg(feature = "serde")]
#[test]
fn binary_import_rejects_nonfinite_parameters() {
    let mut layer = KanLayer::new(2, 1, &config());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        layer.weights[0] = value;
        assert!(bincode::deserialize::<KanLayer>(&bincode::serialize(&layer).unwrap()).is_err());
        layer.weights[0] = 0.0;
        layer.bias[0] = value;
        assert!(bincode::deserialize::<KanLayer>(&bincode::serialize(&layer).unwrap()).is_err());
        layer.bias[0] = 0.0;
    }
}
