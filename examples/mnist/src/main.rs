//! MNIST Classification using ArKan KAN Network
//!
//! This example demonstrates training a KAN network for image classification.
//! MNIST is a dataset of 28x28 grayscale handwritten digits (0-9).
//!
//! # Usage
//!
//! ```bash
//! # CPU training
//! cargo run --release
//!
//! # GPU training (requires 'gpu' feature)
//! cargo run --release --features gpu -- --gpu
//! ```

use arkan::{KanConfigBuilder, KanNetwork, Workspace, TrainOptions};
use mnist::{Mnist, MnistBuilder};
use rand::seq::SliceRandom;
use std::time::Instant;

#[cfg(feature = "gpu")]
use arkan::gpu::{GpuNetwork, GpuAdam, GpuAdamConfig, WgpuBackend, WgpuOptions};

/// Normalize pixel values using z-score standardization
fn normalize_image(pixels: &[u8], mean: f32, std: f32) -> Vec<f32> {
    pixels.iter().map(|&p| ((p as f32) - mean) / std).collect()
}

/// Compute mean and std of the entire dataset
fn compute_stats(images: &[u8]) -> (f32, f32) {
    let total_pixels = images.len();
    let mean = images.iter().map(|&p| p as f64).sum::<f64>() / total_pixels as f64;
    let variance = images.iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>() / total_pixels as f64;
    (mean as f32, variance.sqrt() as f32)
}

/// Convert label to one-hot encoding
fn one_hot(label: u8, num_classes: usize) -> Vec<f32> {
    let mut oh = vec![0.0f32; num_classes];
    oh[label as usize] = 1.0;
    oh
}

/// Get predicted class from output
fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Parse command line arguments
fn parse_args() -> bool {
    std::env::args().any(|arg| arg == "--gpu")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let use_gpu = parse_args();
    
    println!("╔═══════════════════════════════════════════════════════════╗");
    if use_gpu {
        println!("║     MNIST Classification with ArKan KAN (GPU Mode)        ║");
    } else {
        println!("║     MNIST Classification with ArKan KAN (CPU Mode)        ║");
    }
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

    // Compute dataset statistics for z-score normalization
    println!("🔧 Preparing data...");
    let (mean, std) = compute_stats(&trn_img);
    
    let train_images: Vec<Vec<f32>> = (0..train_size)
        .map(|i| normalize_image(&trn_img[i * 784..(i + 1) * 784], mean, std))
        .collect();
    let train_labels: Vec<Vec<f32>> = trn_lbl.iter().map(|&l| one_hot(l, 10)).collect();

    let test_images: Vec<Vec<f32>> = (0..test_size)
        .map(|i| normalize_image(&tst_img[i * 784..(i + 1) * 784], mean, std))
        .collect();
    let test_labels: Vec<u8> = tst_lbl.clone();

    // Create KAN network - optimized architecture
    println!("🧠 Creating KAN network...");
    let config = KanConfigBuilder::new()
        .input_dim(784)                    // 28x28 pixels
        .hidden_dims(vec![64, 32])         // Two hidden layers (best so far)
        .output_dim(10)                    // 10 digit classes
        .spline_order(3)                   // Cubic splines (optimal)
        .grid_size(12)                     // Grid 12 (best result: 92.76%)
        .grid_range(-3.0, 3.0)             // Z-score range
        .build()?;

    println!("  Architecture: 784 -> 64 -> 32 -> 10");
    println!("  Grid size:    12, Spline order: 3");
    println!();

    let mut network = KanNetwork::new(config.clone());
    let mut workspace = Workspace::new(&config);

    let train_opts = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.0,                 // No weight decay - try without
    };

    // Training parameters
    let epochs = if use_gpu { 50 } else { 5 };
    let batch_size = if use_gpu { 256 } else { 512 };
    let initial_lr = if use_gpu { 0.02f32 } else { 0.03f32 };
    let num_batches = train_size / batch_size;

    println!("🎯 Training configuration:");
    println!("  Epochs:      {}", epochs);
    println!("  Batch size:  {}", batch_size);
    println!("  Batches/epoch: {}", num_batches);
    println!("  Initial LR:  {} (cosine decay)", initial_lr);
    println!();

    // GPU setup if enabled
    #[cfg(feature = "gpu")]
    let (_gpu_backend, mut gpu_network, mut gpu_workspace, mut gpu_optimizer) = if use_gpu {
        println!("🖥️  Initializing GPU backend...");
        let backend = WgpuBackend::init(WgpuOptions::default())?;
        println!("  Device: {}", backend.adapter_info().name);
        
        let mut gpu_net = GpuNetwork::from_cpu(&backend, &network)?;
        let gpu_ws = gpu_net.create_workspace(batch_size)?;
        
        let layer_sizes = gpu_net.layer_param_sizes();
        let optimizer = GpuAdam::new(
            backend.device_arc(),
            backend.queue_arc(),
            &layer_sizes,
            GpuAdamConfig::with_lr(initial_lr),
        );
        
        let mem_stats = gpu_net.memory_stats();
        println!("  GPU Memory: {:.2} MB", mem_stats.total_mb());
        println!();
        
        (Some(backend), Some(gpu_net), Some(gpu_ws), Some(optimizer))
    } else {
        (None, None, None, None)
    };

    #[cfg(not(feature = "gpu"))]
    let _use_gpu = false;

    // Training loop
    println!("🚀 Starting training...");
    println!("─────────────────────────────────────────────────────────────");

    let mut indices: Vec<usize> = (0..train_size).collect();
    let start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        
        // Cosine annealing learning rate
        let lr = initial_lr * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / epochs as f32).cos());
        
        // Update LR for GPU optimizer
        #[cfg(feature = "gpu")]
        if use_gpu {
            if let Some(ref mut gpu_opt) = gpu_optimizer {
                gpu_opt.set_lr(lr);
            }
        }

        // Shuffle training data
        indices.shuffle(&mut rand::thread_rng());

        // Train all batches - NO progress bar, NO accuracy check during training
        // Just pure training for maximum speed
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

            #[cfg(feature = "gpu")]
            if use_gpu {
                let gpu_net = gpu_network.as_mut().unwrap();
                let gpu_ws = gpu_workspace.as_mut().unwrap();
                let gpu_opt = gpu_optimizer.as_mut().unwrap();
                let _ = gpu_net.train_step_gpu_native(
                    &batch_inputs, &batch_targets, batch_size, gpu_ws, gpu_opt,
                )?;
            }

            #[cfg(feature = "gpu")]
            if !use_gpu {
                network.train_step_with_options(
                    &batch_inputs, &batch_targets, None, lr, &mut workspace, &train_opts,
                );
            }
            
            #[cfg(not(feature = "gpu"))]
            {
                network.train_step_with_options(
                    &batch_inputs, &batch_targets, None, lr, &mut workspace, &train_opts,
                );
            }
        }

        let epoch_time = epoch_start.elapsed().as_secs_f64();
        
        // Quick test accuracy using batch forward
        let mut test_correct = 0usize;
        let eval_batch = 1000;  // Large eval batch for speed
        
        #[cfg(feature = "gpu")]
        if use_gpu {
            let gpu_net = gpu_network.as_mut().unwrap();
            let gpu_ws = gpu_workspace.as_mut().unwrap();
            
            for chunk_start in (0..test_size).step_by(eval_batch) {
                let chunk_end = (chunk_start + eval_batch).min(test_size);
                let chunk_size = chunk_end - chunk_start;
                let chunk_inputs: Vec<f32> = (chunk_start..chunk_end)
                    .flat_map(|i| test_images[i].iter().cloned())
                    .collect();
                let output = gpu_net.forward_batch(&chunk_inputs, chunk_size, gpu_ws)?;
                for (i, idx) in (chunk_start..chunk_end).enumerate() {
                    if argmax(&output[i * 10..(i + 1) * 10]) == test_labels[idx] as usize {
                        test_correct += 1;
                    }
                }
            }
        }
        
        #[cfg(feature = "gpu")]
        if !use_gpu {
            for chunk_start in (0..test_size).step_by(eval_batch) {
                let chunk_end = (chunk_start + eval_batch).min(test_size);
                let chunk_size = chunk_end - chunk_start;
                let chunk_inputs: Vec<f32> = (chunk_start..chunk_end)
                    .flat_map(|i| test_images[i].iter().cloned())
                    .collect();
                let mut chunk_outputs = vec![0.0f32; chunk_size * 10];
                network.forward_batch(&chunk_inputs, &mut chunk_outputs, &mut workspace);
                for (i, idx) in (chunk_start..chunk_end).enumerate() {
                    if argmax(&chunk_outputs[i * 10..(i + 1) * 10]) == test_labels[idx] as usize {
                        test_correct += 1;
                    }
                }
            }
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            for chunk_start in (0..test_size).step_by(eval_batch) {
                let chunk_end = (chunk_start + eval_batch).min(test_size);
                let chunk_size = chunk_end - chunk_start;
                let chunk_inputs: Vec<f32> = (chunk_start..chunk_end)
                    .flat_map(|i| test_images[i].iter().cloned())
                    .collect();
                let mut chunk_outputs = vec![0.0f32; chunk_size * 10];
                network.forward_batch(&chunk_inputs, &mut chunk_outputs, &mut workspace);
                for (i, idx) in (chunk_start..chunk_end).enumerate() {
                    if argmax(&chunk_outputs[i * 10..(i + 1) * 10]) == test_labels[idx] as usize {
                        test_correct += 1;
                    }
                }
            }
        }
        
        let test_acc = 100.0 * test_correct as f64 / test_size as f64;

        println!(
            "Epoch {:2}/{} | Test Acc: {:5.2}% | LR: {:.5} | Time: {:.1}s",
            epoch, epochs, test_acc, lr, epoch_time
        );
    }

    // Sync GPU weights back to CPU if needed
    #[cfg(feature = "gpu")]
    if use_gpu {
        if let Some(ref gpu_net) = gpu_network {
            gpu_net.sync_weights_to_cpu(&mut network)?;
        }
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
