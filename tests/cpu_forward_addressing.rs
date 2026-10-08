use arkan::{KanConfig, KanLayer, Workspace};

#[test]
fn normalized_boundaries_keep_masked_spans_and_active_coefficients() {
    // Literal frozen-control output, independent of the coefficient-addressing loop.
    let expected = [
        0x4a49dc70, 0x4a49dc6d, 0x4a49dc73, 0xcb22ca2c, 0xcb22ca2c, 0xcb22ca2b, 0x4a258cbf,
        0x4a258cbc, 0x4a258cc2, 0xcae92eb8, 0xcae92eb9, 0xcae92eb6, 0xcb1ede26, 0xcb1ede26,
        0xcb1ede25, 0x4b1c3022, 0x4b1c3022, 0x4b1c3023, 0x4a7302e3, 0x4a7302e0, 0x4a7302e6,
        0xc03df63f, 0xc06df63f, 0xc00df63f, 0x48c24922, 0x48c2490a, 0x48c2493a, 0x493ec7d5,
        0x493ec7c9, 0x493ec7e1, 0xc0a5c23c, 0xc0bdc23c, 0xc08dc23c, 0xcaccb936, 0xcaccb938,
        0xcaccb935, 0x4b547f87, 0x4b547f86, 0x4b547f88,
    ];
    let points = [
        -4.0f32,
        -3.0,
        f32::from_bits((-3.0f32).to_bits() - 1),
        f32::from_bits((-1.8f32).to_bits() + 1),
        -1.8,
        f32::from_bits((-1.8f32).to_bits() - 1),
        0.0,
        f32::from_bits(1.8f32.to_bits() - 1),
        1.8,
        f32::from_bits(1.8f32.to_bits() + 1),
        f32::from_bits(3.0f32.to_bits() - 1),
        3.0,
        4.0,
    ];
    let mean: Vec<f32> = (0..9).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
    let std: Vec<f32> = (0..9).map(|i| [0.5, 1.0, 2.0][i % 3]).collect();
    let mut cfg = KanConfig::builder()
        .input_dim(9)
        .output_dim(3)
        .hidden_dims(vec![])
        .normalization(mean.clone(), std.clone())
        .seed(42)
        .build()
        .unwrap();
    cfg.spline_order = 7;
    cfg.grid_size = 5;
    cfg.grid_range = (-3.0, 3.0);
    cfg.simd_width = 8;
    let mut layer = KanLayer::new(9, 3, &cfg);
    for (i, weight) in layer.weights.iter_mut().enumerate() {
        *weight = [
            16777216.0,
            0.125,
            -16777216.0,
            3.0,
            -7.0,
            1.0,
            65536.0,
            -65536.0,
            0.03125,
        ][i % 9];
    }
    layer.bias.copy_from_slice(&[0.25, -0.5, 1.0]);
    let inputs: Vec<f32> = (0..points.len() * 9)
        .map(|n| mean[n % 9] + std[n % 9] * points[(n / 9 + n % 9) % points.len()])
        .collect();
    let mut output = [0.0; 39];
    layer.forward_batch(&inputs, &mut output, &mut Workspace::new(&cfg));
    assert_eq!(output.map(f32::to_bits), expected);
}

#[test]
fn cancellation_keeps_simd_lane_order_and_scalar_tail() {
    // Frozen BASE 28a0d9a outputs: reassociating the lane sums changes these bits.
    for (width, expected) in [
        (
            16,
            [
                1109458944, 1109262336, 1110507520, 1110310912, 1109458944, 1109262336,
            ],
        ),
        (
            21,
            [
                1266679830, 1113718785, 1266679833, 1113366528, 1266679829, 1113849857,
            ],
        ),
    ] {
        let mut cfg = KanConfig::builder()
            .input_dim(width)
            .output_dim(2)
            .hidden_dims(vec![])
            .seed(42)
            .build()
            .unwrap();
        let terms = [
            16777216.0,
            1.0,
            -16777216.0,
            3.0,
            16777216.0,
            -16777216.0,
            7.0,
            9.0,
        ];
        let inputs: Vec<f32> = [-3.0, 0.0, 3.0]
            .iter()
            .flat_map(|&x| vec![x; width])
            .collect();
        let mut scalar_bits = [0; 6];
        for simd in [16, 8] {
            cfg.simd_width = simd;
            let mut layer = KanLayer::new(width, 2, &cfg);
            for (i, coefficients) in layer
                .weights
                .chunks_mut(layer.global_basis_size)
                .enumerate()
            {
                coefficients.fill(terms[i % terms.len()]);
            }
            layer.bias.copy_from_slice(&[0.25, -0.5]);
            let mut output = [0.0; 6];
            layer.forward_batch(&inputs, &mut output, &mut Workspace::new(&cfg));
            if simd == 16 {
                scalar_bits = output.map(f32::to_bits);
            } else {
                assert_eq!(output.map(f32::to_bits), expected, "width={width}");
                assert_ne!(
                    output.map(f32::to_bits),
                    scalar_bits,
                    "fixture must expose reassociation"
                );
            }
        }
    }
}
