use arkan::config::KanConfig;
use arkan::network::KanNetwork;
use arkan::optimizer::{Optimizer, SGDConfig, SGD};

fn network(order: usize, grid: usize) -> KanNetwork {
    KanNetwork::new(KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![],
        grid_size: grid,
        spline_order: order,
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        init_seed: Some(7),
        ..KanConfig::default()
    })
}

#[test]
fn equal_parameter_counts_do_not_allow_replacing_the_checked_geometry() {
    let mut original = network(2, 3);
    let replacement = network(3, 2);
    assert_eq!(
        original.layers[0].weights.len(),
        replacement.layers[0].weights.len()
    );
    let mut workspace = original.create_workspace(1);
    original.config = replacement.config;
    original.layers = replacement.layers;
    let mut output = [123.0];
    assert!(original
        .try_forward_single(&[0.0, 0.0], &mut output, &mut workspace)
        .is_err());
    assert_eq!(output, [123.0]);
}

#[test]
fn optimizer_rejects_invalid_model_before_updating_parameters() {
    let mut network = network(2, 3);
    let mut optimizer = SGD::new(&network, SGDConfig::with_lr(0.1));
    let weights = network.layers[0].weights.clone();
    let bias = network.layers[0].bias.clone();
    network.config.input_dim = 0;
    assert!(optimizer
        .step(
            &mut network,
            &[vec![1.0; weights.len()]],
            &[vec![1.0]],
            None
        )
        .is_err());
    assert_eq!(network.layers[0].weights, weights);
    assert_eq!(network.layers[0].bias, bias);
}

#[test]
fn fixed_parameter_views_update_the_real_model_without_changing_normalization() {
    let mut network = network(2, 3);
    network.layers[0].set_normalization(&[0.25, -0.5], &[0.5, 2.0]);
    {
        let mut parameters = network.try_parameters_mut().unwrap();
        assert_eq!(parameters.len(), 1);
        for layer in parameters.iter_mut() {
            layer.weights.fill(0.0);
            layer.bias.fill(0.375);
        }
    }
    let mut workspace = network.create_workspace(1);
    let mut output = [0.0];
    network.forward_single(&[0.25, -0.5], &mut output, &mut workspace);
    assert_eq!(output, [0.375]);
    assert_eq!(network.layers[0].mean, [0.25, -0.5]);
    assert_eq!(network.layers[0].std, [0.5, 2.0]);
}
