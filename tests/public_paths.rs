use arkan::config::{KanConfig, KanConfigBuilder};
use arkan::network::KanNetwork;

#[test]
fn legacy_imports_and_responsibility_modules_share_types_and_execution() {
    let config: KanConfig = KanConfigBuilder::new()
        .input_dim(2)
        .output_dim(1)
        .hidden_dims(vec![2])
        .normalization(vec![0.0; 2], vec![1.0; 2])
        .build()
        .unwrap();
    let config: arkan::model::KanConfig = config;
    let mut network: KanNetwork = KanNetwork::new(config);
    let model: &mut arkan::model::KanNetwork = &mut network;
    let _: arkan::ParametersMut<'_> = model.try_parameters_mut().unwrap();
    let layer: &arkan::layer::KanLayer = &model.layers[0];
    let _: &arkan::cpu::KanLayer = layer;
    let mut workspace: arkan::buffer::Workspace = model.create_workspace(1);
    let scratch: &mut arkan::cpu::Workspace = &mut workspace;
    let memory: &arkan::memory::AlignedBuffer = &scratch.layer_input;
    let _: &arkan::buffer::Tensor = memory;
    for parameters in model.try_parameters_mut().unwrap().iter_mut() {
        parameters.weights.fill(0.0);
        parameters.bias.fill(0.25);
    }
    let mut output = [0.0];
    model.forward_single(&[0.1, -0.2], &mut output, scratch);
    assert_eq!(output, [0.25]);
    let pass: arkan::ForwardPass<'_, '_> = model
        .try_forward_for_backward(&[0.1, -0.2], &mut output, scratch)
        .unwrap();
    let _: arkan::Gradients<'_> = pass.backward(&[1.0]).unwrap();
    let baked = arkan::BakedModel::try_from_network(model, None).unwrap();
    let mut baked_scratch: arkan::BakedWorkspace = baked.create_workspace();
    baked.forward_with_workspace(&[0.1, -0.2], &mut output, &mut baked_scratch);
    assert_eq!(output, [0.25]);
    let (_loss, logits_gradient) =
        arkan::masked_categorical_cross_entropy_with_logits(&[0.0, 0.0], &[1.0, 0.0], 2, None);
    assert_eq!(logits_gradient, [-0.5, 0.5]);
    let (_loss, probability_gradient) =
        arkan::masked_categorical_cross_entropy_probabilities(&[0.5, 0.5], &[1.0, 0.0], 2, None);
    assert_eq!(probability_gradient, [-2.0, 0.0]);
    let (_, _, _, poker_gradient) =
        arkan::poker_combined_loss_probabilities(&[0.5; 24], &[0.5; 24], 0.0);
    assert_eq!(poker_gradient.len(), 24);
    let options: arkan::network::TrainOptions = arkan::training::TrainOptions::default();
    let _: arkan::TrainOptions = options;
    let knots = arkan::spline::compute_knots(3, 2, (-1.0, 1.0));
    let span = arkan::math::find_span(0.0, &knots, 2, 3);
    let mut basis = [0.0; 4];
    arkan::math::compute_basis(0.0, span, &knots, 2, &mut basis);
    assert!((basis.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}
