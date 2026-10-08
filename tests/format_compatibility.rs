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
