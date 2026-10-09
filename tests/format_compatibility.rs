#![cfg(feature = "serde")]

use arkan::baked::BakedModel;
use arkan::config::KanConfig;
use arkan::network::KanNetwork;

fn fixture_network() -> KanNetwork {
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

#[test]
fn network_v1_and_raw_legacy_bytes_survive_runtime_changes() {
    let versioned = include_bytes!("fixtures/formats/network-v1.bin");
    let legacy = include_bytes!("fixtures/formats/network-legacy.bin");
    let original = fixture_network();
    assert_eq!(original.to_bytes().unwrap(), versioned);
    assert_eq!(bincode::serialize(&original).unwrap(), legacy);
    for restored in [
        KanNetwork::from_bytes(versioned).unwrap(),
        KanNetwork::from_bytes_legacy(legacy).unwrap(),
        bincode::deserialize::<KanNetwork>(legacy).unwrap(),
    ] {
        assert_eq!(restored.to_bytes().unwrap(), versioned);
        assert_eq!(restored.layers[1].mean, [0.125, -0.25]);
        assert_eq!(restored.layers[1].std, [1.5, 0.75]);
        // Compatibility module imports must still refer to the root types.
        let layer: &arkan::layer::KanLayer = &restored.layers[0];
        let root_layer: &arkan::KanLayer = layer;
        assert_eq!(root_layer.in_dim, 2);
        let mut workspace: arkan::buffer::Workspace = restored.create_workspace(1);
        let mut actual = [0.0];
        restored.forward_single(&[0.25, -0.5], &mut actual, &mut workspace);
        let mut expected = [0.0];
        original.forward_single(&[0.25, -0.5], &mut expected, &mut workspace);
        assert_eq!(actual, expected);
    }
}

#[test]
fn direct_serde_keeps_network_and_layer_field_names() {
    let json = include_str!("fixtures/formats/network.json");
    let network = fixture_network();
    assert_eq!(serde_json::to_string_pretty(&network).unwrap(), json);
    let restored: KanNetwork = serde_json::from_str(json).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), network.to_bytes().unwrap());
}

#[test]
fn baked_v2_bytes_and_json_survive_runtime_changes() {
    let versioned = include_bytes!("fixtures/formats/baked-v2.bin");
    let json = include_str!("fixtures/formats/baked.json");
    let original = BakedModel::try_from_network(
        &fixture_network(),
        Some(&[-0.5, 0.25, 0.25, -0.5, 1.0, 0.75]),
    )
    .unwrap();
    assert_eq!(original.to_bytes().unwrap(), versioned);
    assert_eq!(serde_json::to_string_pretty(&original).unwrap(), json);
    for restored in [
        BakedModel::from_bytes(versioned).unwrap(),
        serde_json::from_str::<BakedModel>(json).unwrap(),
    ] {
        assert_eq!(restored.to_bytes().unwrap(), versioned);
        let mut expected = [0.0];
        let mut actual = [0.0];
        original.forward(&[0.25, -0.5], &mut expected);
        restored.forward(&[0.25, -0.5], &mut actual);
        assert_eq!(actual, expected);
    }
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
    [fixture_network(), large_network(64), large_network(128)]
}

fn baked_models(networks: &[KanNetwork; 3]) -> [BakedModel; 3] {
    [
        BakedModel::try_from_network(&networks[0], Some(&[-0.5, 0.25, 0.25, -0.5, 1.0, 0.75]))
            .unwrap(),
        BakedModel::try_from_network(&networks[1], None).unwrap(),
        BakedModel::try_from_network(&networks[2], None).unwrap(),
    ]
}

fn network_wire_oracle(network: &KanNetwork) -> Vec<u8> {
    let mut expected = b"ARKAN\x01\x00\x00\x00".to_vec();
    expected.extend(bincode::serialize(network).unwrap());
    expected
}

fn baked_wire_oracle(model: &BakedModel) -> Vec<u8> {
    let mut expected = b"KAN_BAKED_v1\x02\x00\x00\x00".to_vec();
    expected.extend(bincode::serialize(model).unwrap());
    expected
}

#[test]
fn public_exports_match_direct_bincode_with_literal_headers() {
    let networks = networks();
    let baked = baked_models(&networks);
    for i in 0..3 {
        assert_eq!(
            networks[i].to_bytes().unwrap(),
            network_wire_oracle(&networks[i])
        );
        assert_eq!(baked[i].to_bytes().unwrap(), baked_wire_oracle(&baked[i]));
    }
}

#[test]
fn cpu_export_preserves_signed_zero_infinity_and_nan_payload_bits() {
    let patterns = [
        0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321,
    ];
    for bits in patterns {
        let mut network = fixture_network();
        network.layers[0].weights[0] = f32::from_bits(bits);
        network.layers[0].bias[0] = f32::from_bits(bits);
        assert_eq!(network.to_bytes().unwrap(), network_wire_oracle(&network));
    }
    let mut mixed = fixture_network();
    for (i, weight) in mixed.layers[0].weights.iter_mut().enumerate() {
        *weight = f32::from_bits(patterns[i % patterns.len()]);
    }
    mixed.layers[0].bias[0] = f32::from_bits(0x00000000);
    mixed.layers[0].bias[1] = f32::from_bits(0x80000000);
    assert_eq!(mixed.to_bytes().unwrap(), network_wire_oracle(&mixed));
}

#[test]
fn mutable_invalid_baked_models_still_export_but_import_rejects_them() {
    let valid = BakedModel::try_from_network(
        &fixture_network(),
        Some(&[-0.5, 0.25, 0.25, -0.5, 1.0, 0.75]),
    )
    .unwrap();
    for case in 0..4 {
        let mut model = valid.clone();
        match case {
            0 => {
                model.layers[0].weights_i8.pop();
            }
            1 => model.layers[0].requant_shift[0] = 63,
            2 => model.layers[0].std[0] = 0.0,
            3 => model.layers[0].s_act_out = f32::from_bits(0x7fc12345),
            _ => unreachable!(),
        }
        let bytes = model.to_bytes().unwrap();
        assert_eq!(bytes, baked_wire_oracle(&model));
        assert!(matches!(
            *BakedModel::from_bytes(&bytes).unwrap_err(),
            bincode::ErrorKind::Custom(_)
        ));
        let body = bincode::serialize(&model).unwrap();
        assert!(matches!(
            *bincode::deserialize::<BakedModel>(&body).unwrap_err(),
            bincode::ErrorKind::Custom(_)
        ));
    }
    let bytes = valid.to_bytes().unwrap();
    assert_eq!(
        BakedModel::from_bytes(&bytes).unwrap().to_bytes().unwrap(),
        bytes
    );
}
