//! Exact initialized-capacity regression previews for single-sample inference.

use arkan::cpu::Workspace;
use arkan::memory::AlignedBuffer;
use arkan::{KanConfig, KanNetwork};

fn capacity_bits(buffer: &AlignedBuffer) -> Vec<u32> {
    // SAFETY: AlignedBuffer documents that all [0, capacity) elements are
    // initialized. The shared borrow keeps the allocation alive and immutable.
    // Read before clone or resize: both can erase the spare-capacity evidence.
    let initialized = unsafe { std::slice::from_raw_parts(buffer.as_ptr(), buffer.capacity()) };
    initialized.iter().map(|value| value.to_bits()).collect()
}

fn bias_only(hidden: Vec<usize>, biases: &[&[f32]]) -> KanNetwork {
    let mut network = KanNetwork::new(KanConfig {
        input_dim: 1,
        input_mean: vec![0.0],
        input_std: vec![1.0],
        hidden_dims: hidden,
        output_dim: 1,
        init_seed: Some(42),
        multithreading_threshold: usize::MAX,
        ..KanConfig::default()
    });
    assert_eq!(network.layers.len(), biases.len());
    for (layer, values) in network.layers.iter_mut().zip(biases) {
        layer.weights.fill(0.0);
        layer.bias.copy_from_slice(values);
    }
    network
}

fn prefill(workspace: &mut Workspace) {
    workspace.z_buffer.resize(16);
    for (index, value) in workspace.z_buffer.as_mut_slice().iter_mut().enumerate() {
        *value = 20.0 + index as f32;
    }
    workspace.z_buffer.resize(0);
}

fn check_tail(network: &KanNetwork, workspace: &mut Workspace, prefix: &[f32], output: f32) {
    let identities = (
        workspace.z_buffer.as_ptr(),
        workspace.layer_input.as_ptr(),
        workspace.layer_output.as_ptr(),
        workspace.basis_values.as_ptr(),
    );
    let mut prediction = [f32::from_bits(0x7fc0_1234)];
    network
        .try_forward_single(&[0.0], &mut prediction, workspace)
        .unwrap();
    assert_eq!(prediction[0].to_bits(), output.to_bits());
    assert_eq!(
        workspace.z_buffer.len(),
        network.layers.last().unwrap().in_dim
    );
    assert_eq!(workspace.z_buffer.capacity(), 16);
    let mut expected = (0..16)
        .map(|index| (20.0 + index as f32).to_bits())
        .collect::<Vec<_>>();
    for (destination, value) in expected.iter_mut().zip(prefix) {
        *destination = value.to_bits();
    }
    assert_eq!(capacity_bits(&workspace.z_buffer), expected);
    assert_eq!(
        identities,
        (
            workspace.z_buffer.as_ptr(),
            workspace.layer_input.as_ptr(),
            workspace.layer_output.as_ptr(),
            workspace.basis_values.as_ptr(),
        )
    );
}

#[test]
fn single_inference_retains_the_hidden_initialized_tail() {
    let network = bias_only(vec![4, 1], &[&[0.5, 2.0, 3.0, 4.0], &[0.5], &[0.25]]);
    let mut workspace = network.create_workspace(1);
    prefill(&mut workspace);
    for _ in 0..2 {
        check_tail(&network, &mut workspace, &[0.5, 2.0, 3.0, 4.0], 0.25);
    }
}

#[test]
fn single_inference_retains_mixed_suffixes_across_workspace_reuse() {
    let narrow = bias_only(vec![4, 1], &[&[0.5, 2.0, 3.0, 4.0], &[0.5], &[0.25]]);
    let mixed = bias_only(
        vec![4, 2, 3],
        &[
            &[0.5, 2.0, 3.0, 4.0],
            &[0.25, 0.75],
            &[0.5, 0.875, -0.25],
            &[-0.75],
        ],
    );
    let mut workspace = mixed.create_workspace(1);
    prefill(&mut workspace);
    check_tail(&narrow, &mut workspace, &[0.5, 2.0, 3.0, 4.0], 0.25);
    for _ in 0..2 {
        check_tail(&mixed, &mut workspace, &[0.5, 0.875, -0.25, 4.0], -0.75);
    }
    check_tail(&narrow, &mut workspace, &[0.5, 2.0, 3.0, 4.0], 0.25);
}

#[test]
fn rejected_single_shape_retains_output_and_initialized_tail() {
    let network = bias_only(vec![4, 1], &[&[0.5, 2.0, 3.0, 4.0], &[0.5], &[0.25]]);
    let mut workspace = network.create_workspace(1);
    prefill(&mut workspace);
    let before = capacity_bits(&workspace.z_buffer);
    let identity = workspace.z_buffer.as_ptr();
    let mut output = [f32::from_bits(0x7fc0_1234)];
    let error = network
        .try_forward_single(&[], &mut output, &mut workspace)
        .unwrap_err();
    assert!(matches!(
        error,
        arkan::error::ArkanError::ShapeMismatch { .. }
    ));
    assert_eq!(output[0].to_bits(), 0x7fc0_1234);
    assert_eq!(workspace.z_buffer.as_ptr(), identity);
    assert_eq!(workspace.z_buffer.len(), 0);
    assert_eq!(workspace.z_buffer.capacity(), 16);
    assert_eq!(capacity_bits(&workspace.z_buffer), before);
}
