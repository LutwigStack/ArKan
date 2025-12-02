use arkan::{KanConfig, KanNetwork};

fn main() {
    // Build a default config (poker preset) and network.
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());

    // Preallocate workspace for the maximum batch we expect (here: 8).
    let mut workspace = network.create_workspace(8);

    // Single-sample forward: inputs length must be input_dim.
    let mut inputs = vec![0.0f32; config.input_dim];
    inputs[0] = 0.5;

    let mut outputs = vec![0.0f32; config.output_dim];
    network.forward_single(&inputs, &mut outputs, &mut workspace);
    println!("single forward, out[0] = {}", outputs[0]);

    // Batch forward (batch = 8).
    let batch = 8;
    let mut batch_inputs = vec![0.0f32; batch * config.input_dim];
    let mut batch_outputs = vec![0.0f32; batch * config.output_dim];
    network.forward_batch(&batch_inputs, &mut batch_outputs, &mut workspace);
    println!("batch forward, first sample out[0] = {}", batch_outputs[0]);
}
