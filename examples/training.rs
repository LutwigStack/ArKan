//! Training Example for ArKan
//!
//! This example demonstrates how to train a KAN network using ArKan's
//! zero-allocation training pipeline.
//!
//! # Features Demonstrated
//!
//! - Network configuration and initialization
//! - Training loop with Adam optimizer
//! - Gradient clipping and weight decay
//! - Loss tracking and early stopping
//!
//! # Run
//!
//! ```bash
//! cargo run --example training
//! ```

use arkan::optimizer::{Adam, AdamConfig};
use arkan::{KanConfig, KanNetwork, TrainOptions};

fn main() {
    println!("=== ArKan Training Example ===\n");

    // 1. Configure network
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![16, 16],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.5; 4],
        input_std: vec![0.3; 4],
        ..Default::default()
    };
    config.validate().expect("Invalid config");

    println!("Network configuration:");
    println!("  Input:  {} neurons", config.input_dim);
    println!("  Hidden: {:?}", config.hidden_dims);
    println!("  Output: {} neurons", config.output_dim);

    // 2. Create network and workspace
    let mut network = KanNetwork::new(config.clone());
    let batch_size = 32;
    let mut workspace = network.create_workspace(batch_size);

    println!("\nNetwork statistics:");
    println!("  Layers:     {}", network.num_layers());
    println!("  Parameters: {}", network.param_count());
    println!("  Batch size: {}", batch_size);

    // 3. Create Adam optimizer
    let optimizer_config = AdamConfig {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0,
    };
    let optimizer = Adam::new(&network, optimizer_config);

    println!("\nOptimizer: Adam");
    println!("  Learning rate: {}", optimizer.learning_rate());

    // 4. Set training options (gradient clipping + weight decay)
    let train_opts = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.01,
    };
    network.set_default_train_options(train_opts);

    println!("\nTraining options:");
    println!("  Gradient clipping: max norm = 1.0");
    println!("  Weight decay:      0.01");

    // 5. Generate synthetic dataset (XOR-like pattern)
    let num_samples = 256;
    let (inputs, targets) = generate_xor_dataset(num_samples, config.input_dim, config.output_dim);

    println!("\nDataset: {} samples", num_samples);

    // 6. Training loop
    println!("\n--- Training ---\n");

    let epochs = 100;
    let mut best_loss = f32::MAX;
    let mut no_improvement_count = 0;
    let patience = 10;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let num_batches = num_samples / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end_input = start * config.input_dim + batch_size * config.input_dim;
            let end_target = start * config.output_dim + batch_size * config.output_dim;

            let batch_inputs = &inputs[start * config.input_dim..end_input];
            let batch_targets = &targets[start * config.output_dim..end_target];

            // Train step with SGD (or use train_step_with_options for advanced control)
            let loss = network.train_step(
                batch_inputs,
                batch_targets,
                None, // MSE loss, no mask
                optimizer.learning_rate(),
                &mut workspace,
            );

            epoch_loss += loss;
        }

        epoch_loss /= num_batches as f32;

        // Print progress every 10 epochs
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("Epoch {:3}: loss = {:.6}", epoch + 1, epoch_loss);
        }

        // Early stopping check
        if epoch_loss < best_loss * 0.999 {
            best_loss = epoch_loss;
            no_improvement_count = 0;
        } else {
            no_improvement_count += 1;
            if no_improvement_count >= patience {
                println!(
                    "\nEarly stopping at epoch {} (no improvement for {} epochs)",
                    epoch + 1,
                    patience
                );
                break;
            }
        }
    }

    // 7. Final evaluation
    println!("\n--- Evaluation ---\n");

    let mut total_loss = 0.0;
    let mut outputs = vec![0.0f32; batch_size * config.output_dim];

    let num_eval_batches = num_samples / batch_size;
    for batch_idx in 0..num_eval_batches {
        let start = batch_idx * batch_size;
        let end_input = start * config.input_dim + batch_size * config.input_dim;
        let end_target = start * config.output_dim + batch_size * config.output_dim;

        let batch_inputs = &inputs[start * config.input_dim..end_input];
        let batch_targets = &targets[start * config.output_dim..end_target];

        network.forward_batch(batch_inputs, &mut outputs, &mut workspace);

        // Compute MSE
        for (pred, target) in outputs.iter().zip(batch_targets.iter()) {
            let diff = pred - target;
            total_loss += diff * diff;
        }
    }

    let final_mse = total_loss / (num_samples * config.output_dim) as f32;
    println!("Final MSE: {:.6}", final_mse);
    println!("Best training loss: {:.6}", best_loss);

    // 8. Show sample predictions
    println!("\n--- Sample Predictions ---\n");

    let mut single_output = vec![0.0f32; config.output_dim];
    for i in 0..4 {
        let sample_input = &inputs[i * config.input_dim..(i + 1) * config.input_dim];
        let sample_target = &targets[i * config.output_dim..(i + 1) * config.output_dim];

        network.forward_single(sample_input, &mut single_output, &mut workspace);

        println!("Sample {}:", i + 1);
        println!("  Input:  {:?}", &sample_input[..sample_input.len().min(4)]);
        println!("  Target: {:?}", sample_target);
        println!("  Output: {:?}", single_output);
        println!();
    }

    println!("Training complete!");
}

/// Generate synthetic XOR-like dataset for demonstration.
fn generate_xor_dataset(
    num_samples: usize,
    input_dim: usize,
    output_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    use std::f32::consts::PI;

    let mut inputs = Vec::with_capacity(num_samples * input_dim);
    let mut targets = Vec::with_capacity(num_samples * output_dim);

    for i in 0..num_samples {
        // Generate input pattern
        let phase = (i as f32 / num_samples as f32) * 2.0 * PI;

        for j in 0..input_dim {
            let val = ((phase + j as f32 * 0.5).sin() + 1.0) / 2.0;
            inputs.push(val);
        }

        // Generate XOR-like target: high when inputs are "different"
        let sum: f32 = inputs[i * input_dim..(i + 1) * input_dim]
            .iter()
            .sum::<f32>()
            / input_dim as f32;

        for j in 0..output_dim {
            // Target depends on input pattern
            let target = if j % 2 == 0 {
                if sum > 0.3 && sum < 0.7 {
                    1.0
                } else {
                    0.0
                }
            } else if sum <= 0.3 || sum >= 0.7 {
                1.0
            } else {
                0.0
            };
            targets.push(target);
        }
    }

    (inputs, targets)
}
