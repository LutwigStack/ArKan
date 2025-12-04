//! MNIST Classification using ArKan KAN Network
//!
//! This example demonstrates training a KAN network for image classification.
//! MNIST is a dataset of 28x28 grayscale handwritten digits (0-9).
//!
//! # Usage
//!
//! ```bash
//! # Download MNIST data first (will be downloaded automatically)
//! cargo run --release
//! ```

use arkan::{KanConfigBuilder, KanNetwork, Workspace, TrainOptions};
use indicatif::{ProgressBar, ProgressStyle};
use mnist::{Mnist, MnistBuilder};
use rand::seq::SliceRandom;
use std::time::Instant;

/// Normalize pixel values to [-1, 1] range (matching KAN grid range)
fn normalize_image(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| (p as f32 / 127.5) - 1.0).collect()
}

/// Convert label to one-hot encoding
fn one_hot(label: u8, num_classes: usize) -> Vec<f32> {
    let mut oh = vec![0.0f32; num_classes];
    oh[label as usize] = 1.0;
    oh
}

/// Softmax function for output probabilities
fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Get predicted class from output
fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Cross-entropy loss
fn cross_entropy_loss(predicted: &[f32], target: &[f32]) -> f32 {
    let probs = softmax(predicted);
    -target
        .iter()
        .zip(probs.iter())
        .map(|(&t, &p)| t * (p + 1e-10).ln())
        .sum::<f32>()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║         MNIST Classification with ArKan KAN               ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Load MNIST dataset
    println!("📥 Loading MNIST dataset...");
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    let train_size = trn_lbl.len();
    let test_size = tst_lbl.len();
    println!("  Training samples: {}", train_size);
    println!("  Test samples:     {}", test_size);
    println!();

    // Prepare training data
    println!("🔧 Preparing data...");
    let train_images: Vec<Vec<f32>> = (0..train_size)
        .map(|i| normalize_image(&trn_img[i * 784..(i + 1) * 784]))
        .collect();
    let train_labels: Vec<Vec<f32>> = trn_lbl.iter().map(|&l| one_hot(l, 10)).collect();

    let test_images: Vec<Vec<f32>> = (0..test_size)
        .map(|i| normalize_image(&tst_img[i * 784..(i + 1) * 784]))
        .collect();
    let test_labels: Vec<u8> = tst_lbl.clone();

    // Create KAN network
    // Architecture: 784 (28x28) -> 32 -> 10 (digits)
    // Simpler network for faster convergence
    println!("🧠 Creating KAN network...");
    let config = KanConfigBuilder::new()
        .input_dim(784)           // 28x28 pixels
        .hidden_dims(vec![32])    // Single hidden layer
        .output_dim(10)           // 10 digit classes
        .spline_order(3)          // Cubic splines
        .grid_size(5)             // 5 grid points
        .grid_range(-1.0, 1.0)    // Match normalized input range
        .build()?;

    println!("  Architecture: 784 -> 32 -> 10");
    println!();

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let train_opts = TrainOptions {
        max_grad_norm: Some(5.0),  // Increased clip threshold
        weight_decay: 0.0,        // No weight decay for now
    };

    // Training parameters
    let epochs = 10;
    let batch_size = 128;  // Larger batch
    let learning_rate = 0.01f32;  // Much higher learning rate
    let num_batches = train_size / batch_size;

    println!("🎯 Training configuration:");
    println!("  Epochs:      {}", epochs);
    println!("  Batch size:  {}", batch_size);
    println!("  Batches:     {}", num_batches);
    println!();

    // Training loop
    println!("🚀 Starting training...");
    println!("─────────────────────────────────────────────────────────────");

    let mut indices: Vec<usize> = (0..train_size).collect();
    let start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();

        // Shuffle training data
        indices.shuffle(&mut rand::thread_rng());

        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} Epoch {msg} [{bar:30.cyan/blue}] {pos}/{len}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.set_message(format!("{}/{}", epoch, epochs));

        let mut epoch_loss = 0.0f32;
        let mut correct = 0usize;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_indices = &indices[batch_start..batch_start + batch_size];

            // Prepare batch data
            let batch_inputs: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| train_images[i].iter().cloned())
                .collect();
            let batch_targets: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| train_labels[i].iter().cloned())
                .collect();

            // Forward pass to compute predictions (for accuracy)
            let mut batch_outputs = vec![0.0f32; batch_size * 10];
            for (i, &idx) in batch_indices.iter().enumerate() {
                let input = &train_images[idx];
                let output = &mut batch_outputs[i * 10..(i + 1) * 10];
                network.forward_single(input, output, &mut workspace);

                // Check if prediction is correct
                let predicted = argmax(output);
                let actual = trn_lbl[idx] as usize;
                if predicted == actual {
                    correct += 1;
                }

                // Accumulate loss
                epoch_loss += cross_entropy_loss(output, &train_labels[idx]);
            }

            // Training step with SGD
            network.train_step_with_options(
                &batch_inputs,
                &batch_targets,
                None,  // No mask
                learning_rate,
                &mut workspace,
                &train_opts,
            );

            pb.inc(1);
        }

        pb.finish_and_clear();

        let epoch_time = epoch_start.elapsed().as_secs_f64();
        let train_acc = 100.0 * correct as f64 / (num_batches * batch_size) as f64;
        let avg_loss = epoch_loss / (num_batches * batch_size) as f32;

        // Evaluate on test set
        let mut test_correct = 0usize;
        let mut test_output = vec![0.0f32; 10];
        for i in 0..test_size {
            network.forward_single(&test_images[i], &mut test_output, &mut workspace);
            if argmax(&test_output) == test_labels[i] as usize {
                test_correct += 1;
            }
        }
        let test_acc = 100.0 * test_correct as f64 / test_size as f64;

        println!(
            "Epoch {:2}/{} | Loss: {:.4} | Train Acc: {:5.2}% | Test Acc: {:5.2}% | Time: {:.1}s",
            epoch, epochs, avg_loss, train_acc, test_acc, epoch_time
        );
    }

    let total_time = start.elapsed().as_secs_f64();
    println!("─────────────────────────────────────────────────────────────");
    println!();

    // Final evaluation
    println!("📊 Final Evaluation:");
    println!("─────────────────────────────────────────────────────────────");

    let mut final_correct = 0usize;
    let mut output = vec![0.0f32; 10];
    let mut confusion = vec![vec![0usize; 10]; 10]; // confusion[actual][predicted]

    for i in 0..test_size {
        network.forward_single(&test_images[i], &mut output, &mut workspace);
        let predicted = argmax(&output);
        let actual = test_labels[i] as usize;

        if predicted == actual {
            final_correct += 1;
        }
        confusion[actual][predicted] += 1;
    }

    let final_acc = 100.0 * final_correct as f64 / test_size as f64;

    println!("Test Accuracy: {:.2}% ({}/{})", final_acc, final_correct, test_size);
    println!("Total Time:    {:.1}s", total_time);
    println!();

    // Print per-digit accuracy
    println!("Per-digit accuracy:");
    for digit in 0..10 {
        let total: usize = confusion[digit].iter().sum();
        let correct = confusion[digit][digit];
        let acc = 100.0 * correct as f64 / total as f64;
        println!("  Digit {}: {:5.2}% ({}/{})", digit, acc, correct, total);
    }

    println!();
    println!("✅ Training complete!");

    Ok(())
}
