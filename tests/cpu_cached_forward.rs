use arkan::{Adam, AdamConfig, KanConfig, KanNetwork, TrainOptions, Workspace};

fn model(width: usize, order: usize, simd: usize, threshold: usize) -> KanNetwork {
    KanNetwork::new(KanConfig {
        input_dim: width,
        output_dim: 3,
        hidden_dims: vec![width],
        spline_order: order,
        grid_size: 5,
        grid_range: (-3.0, 3.0),
        input_mean: (0..width).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect(),
        input_std: (0..width).map(|i| [0.5, 1.0, 2.0][i % 3]).collect(),
        simd_width: simd,
        multithreading_threshold: threshold,
        init_seed: Some(42),
    })
}

fn inputs(net: &KanNetwork, batch: usize) -> Vec<f32> {
    let points = [
        -4.0,
        -3.0,
        f32::from_bits((-3.0f32).to_bits() - 1),
        -1.8,
        0.0,
        1.8,
        f32::from_bits(3.0f32.to_bits() - 1),
        3.0,
        4.0,
    ];
    (0..batch * net.config.input_dim)
        .map(|n| {
            let i = n % net.config.input_dim;
            net.config.input_mean[i]
                + net.config.input_std[i] * points[(n / net.config.input_dim + i) % points.len()]
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn history(net: &KanNetwork, ws: &Workspace, batch: usize) -> Vec<u32> {
    let mut result = Vec::new();
    for (li, layer) in net.layers.iter().enumerate() {
        let size = batch * layer.in_dim;
        assert_eq!(ws.layers_inputs[li].len(), size);
        assert_eq!(ws.layers_grid_indices[li].len(), size);
        result.extend(bits(ws.layers_inputs[li].as_slice()));
        result.extend(&ws.layers_grid_indices[li]);
    }
    if batch > 0 {
        let last = net.layers.last().unwrap();
        result.extend(bits(
            &ws.basis_values.as_slice()[..batch * last.in_dim * last.basis_aligned()],
        ));
    }
    result
}

fn in_pool<T: Send>(workers: usize, f: impl FnOnce() -> T + Send) -> T {
    #[cfg(feature = "parallel")]
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = workers;
        f()
    }
}

fn compare_forward(net: &KanNetwork, ws: &mut Workspace, batch: usize) {
    let input = inputs(net, batch);
    let mut actual = vec![0.0; batch * net.config.output_dim];
    net.try_forward_batch_training(&input, &mut actual, ws)
        .unwrap();
    let mut replay = net.create_workspace(batch);
    let mut current = input;
    let mut expected_history = Vec::new();
    for layer in &net.layers {
        let mut output = vec![0.0; batch * layer.out_dim];
        layer
            .try_forward_batch(&current, &mut output, &mut replay)
            .unwrap();
        expected_history.extend(bits(replay.z_buffer.as_slice()));
        expected_history.extend(&replay.grid_indices[..batch * layer.in_dim]);
        current = output;
    }
    if batch > 0 {
        let last = net.layers.last().unwrap();
        expected_history.extend(bits(
            &replay.basis_values.as_slice()[..batch * last.in_dim * last.basis_aligned()],
        ));
    }
    assert_eq!(bits(&actual), bits(&current));
    assert_eq!(history(net, ws, batch), expected_history);
}

#[test]
fn cached_rows_match_serial_replay_at_threshold_and_chunk_boundaries() {
    // Wrong chunk offsets, partial-row handling, or changed arithmetic breaks this comparison.
    for workers in [1, 2, 4] {
        in_pool(workers, || {
            for threshold in [1, 128, usize::MAX] {
                let net = model(9, 3, 8, threshold);
                let mut ws = net.create_workspace(1);
                for batch in [0, 1, 2, 3, 7, 127, 128, 129, 3, 0] {
                    compare_forward(&net, &mut ws, batch);
                }
            }
        });
    }
}

#[test]
fn cached_history_preserves_orders_simd_tails_and_cross_model_reuse() {
    in_pool(4, || {
        let mut ws = model(1, 1, 8, 1).create_workspace(1);
        for order in 1..=7 {
            for width in [1, 3, 4, 7, 8, 9, 16, 21, 64] {
                for simd in [4, 8, 16] {
                    let net = model(width, order, simd, 1);
                    for batch in [7, 2, 0, 3] {
                        compare_forward(&net, &mut ws, batch);
                    }
                }
            }
        }
    });
}

fn parameter_bits(net: &KanNetwork) -> Vec<u32> {
    net.layers
        .iter()
        .flat_map(|l| l.weights.iter().chain(&l.bias))
        .map(|v| v.to_bits())
        .collect()
}

fn gradient_bits(ws: &Workspace) -> Vec<u32> {
    ws.weight_grads
        .iter()
        .chain(&ws.bias_grads)
        .flatten()
        .map(|v| v.to_bits())
        .collect()
}

// The external frozen-control oracle also uses these real consumer snapshots.
fn consumer_snapshot(width: usize, batch: usize, threshold: usize) -> Vec<u32> {
    let net = model(width, 3, 8, threshold);
    let input = inputs(&net, batch);
    let target: Vec<f32> = (0..batch * 3)
        .map(|i| (i % 7) as f32 * 0.125 - 0.3)
        .collect();
    let mut snapshot = Vec::new();
    for custom in [false, true] {
        let mut ws = net.create_workspace(1);
        let mut output = vec![0.0; target.len()];
        let pass = net
            .try_forward_for_backward(&input, &mut output, &mut ws)
            .unwrap();
        let mut derivative = vec![0.0; target.len()];
        let loss = if custom {
            let (loss, gradient) = arkan::loss::masked_huber(&output, &target, 0.4, None);
            derivative.copy_from_slice(&gradient);
            loss
        } else {
            arkan::loss::masked_mse_into(&output, &target, None, &mut derivative).unwrap()
        };
        pass.backward(&derivative).unwrap();
        snapshot.push(loss.to_bits());
        snapshot.extend(bits(&output));
        // Backward reuses the basis cache; forward's exact basis is checked by replay above.
        snapshot.extend(history(&net, &ws, batch));
        snapshot.extend(gradient_bits(&ws));
        if !custom {
            let mut basic = net.clone();
            let mut basic_ws = basic.create_workspace(1);
            let basic_loss = basic
                .try_train_step(&input, &target, None, 0.0, &mut basic_ws)
                .unwrap();
            assert_eq!(basic_loss.to_bits(), loss.to_bits());
            assert_eq!(gradient_bits(&basic_ws), gradient_bits(&ws));
            assert_eq!(parameter_bits(&basic), parameter_bits(&net));
        }
    }
    for lr in [0.0, 0.003] {
        let mut basic = net.clone();
        let mut ws = basic.create_workspace(1);
        snapshot.push(
            basic
                .try_train_step(&input, &target, None, lr, &mut ws)
                .unwrap()
                .to_bits(),
        );
        snapshot.extend(gradient_bits(&ws));
        snapshot.extend(parameter_bits(&basic));
        let mut optimized = net.clone();
        let mut optimizer = Adam::new(
            &optimized,
            AdamConfig {
                lr,
                ..AdamConfig::default()
            },
        );
        let mut ws = optimized.create_workspace(1);
        snapshot.push(
            optimized
                .train_step_with_optimizer(
                    &input,
                    &target,
                    None,
                    &mut ws,
                    &mut optimizer,
                    &TrainOptions::default(),
                )
                .unwrap()
                .to_bits(),
        );
        snapshot.extend(gradient_bits(&ws));
        snapshot.extend(parameter_bits(&optimized));
    }
    snapshot
}

#[test]
fn real_consumers_keep_forward_and_parallel_gradient_bits_across_workers() {
    for width in [9, 64] {
        for batch in [2, 3, 7, 129] {
            // Threshold 1 selects the same parallel backward reduction in every pool.
            let expected = in_pool(1, || consumer_snapshot(width, batch, 1));
            for workers in [2, 4] {
                assert_eq!(
                    in_pool(workers, || consumer_snapshot(width, batch, 1)),
                    expected
                );
            }
        }
    }
}

#[test]
fn dropping_completed_pass_preserves_gradients_and_allows_smaller_reuse() {
    // A forward that clears gradients or leaves workers borrowing the workspace breaks reuse.
    in_pool(4, || {
        let net = model(9, 3, 8, 1);
        let mut ws = net.create_workspace(129);
        let input = inputs(&net, 129);
        let mut output = vec![0.0; 129 * 3];
        net.try_forward_for_backward(&input, &mut output, &mut ws)
            .unwrap()
            .backward(&vec![0.25; output.len()])
            .unwrap();
        assert!(ws.weight_grads.iter().flatten().any(|&g| g != 0.0));
        assert!(ws.bias_grads.iter().flatten().any(|&g| g != 0.0));
        let parameters = parameter_bits(&net);
        let gradients = gradient_bits(&ws);
        let pass = net
            .try_forward_for_backward(&input, &mut output, &mut ws)
            .unwrap();
        drop(pass);
        assert_eq!(parameter_bits(&net), parameters);
        assert_eq!(gradient_bits(&ws), gradients);

        let mut smaller = vec![0.0; 7 * 3];
        net.try_forward_for_backward(&inputs(&net, 7), &mut smaller, &mut ws)
            .unwrap()
            .backward(&vec![0.25; smaller.len()])
            .unwrap();
        let mut fresh = net.create_workspace(7);
        let mut expected = vec![0.0; 7 * 3];
        net.try_forward_for_backward(&inputs(&net, 7), &mut expected, &mut fresh)
            .unwrap()
            .backward(&vec![0.25; expected.len()])
            .unwrap();
        assert_eq!(bits(&smaller), bits(&expected));
        assert_eq!(history(&net, &ws, 7), history(&net, &fresh, 7));
        assert_eq!(gradient_bits(&ws), gradient_bits(&fresh));
        assert_eq!(parameter_bits(&net), parameters);
    });
}

fn shared_model_snapshot(net: &KanNetwork, batch: usize) -> Vec<u32> {
    let mut ws = net.create_workspace(1);
    let input = inputs(net, batch);
    let mut output = vec![0.0; batch * 3];
    net.try_forward_batch_training(&input, &mut output, &mut ws)
        .unwrap();
    let mut snapshot = bits(&output);
    snapshot.extend(history(net, &ws, batch));
    net.try_forward_for_backward(&input, &mut output, &mut ws)
        .unwrap()
        .backward(&vec![0.25; output.len()])
        .unwrap();
    snapshot.extend(gradient_bits(&ws));
    snapshot
}

#[test]
fn concurrent_shared_model_consumers_match_sequential_independent_workspaces() {
    // Sharing mutable prepared caches or returning before joined workers finish changes snapshots.
    in_pool(4, || {
        let net = model(9, 3, 8, 1);
        let parameters = parameter_bits(&net);
        let expected = [
            shared_model_snapshot(&net, 129),
            shared_model_snapshot(&net, 7),
        ];
        let actual = std::thread::scope(|scope| {
            let large = scope.spawn(|| shared_model_snapshot(&net, 129));
            let small = scope.spawn(|| shared_model_snapshot(&net, 7));
            [large.join().unwrap(), small.join().unwrap()]
        });
        assert_eq!(actual, expected);
        assert_eq!(parameter_bits(&net), parameters);
    });
}
