//! Test training on sinusoid - achieving perfect sin(x) approximation

use arkan::{KanConfig, KanNetwork, TrainOptions};

fn train_sinusoid(seed: u64, epochs: usize, lr: f32, hidden: usize, verbose: bool) -> (f32, f32, f32) {
    let config = KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![hidden],  // Configurable hidden neurons
        grid_size: 16,              // Max grid size for smooth approximation
        spline_order: 3,
        grid_range: (-1.5, 1.5),
        input_mean: vec![0.0],
        input_std: vec![1.0],
        multithreading_threshold: 16,
        simd_width: 8,
        init_seed: Some(seed),
    };

    let mut network = KanNetwork::new(config.clone());
    let n_samples = 100;
    let mut workspace = network.create_workspace(n_samples);

    // Generate training data: sin(x) for x in [-pi, pi]
    let mut inputs = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * (i as f32) / (n_samples as f32 - 1.0);
        let y = x.sin();
        inputs.push(x);
        targets.push(y);
    }

    // Normalize inputs to [-1, 1] for better grid coverage
    let input_min = inputs.iter().cloned().fold(f32::INFINITY, f32::min);
    let input_max = inputs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let input_range = input_max - input_min;
    let inputs_norm: Vec<f32> = inputs.iter().map(|x| 2.0 * (x - input_min) / input_range - 1.0).collect();

    let options = TrainOptions {
        max_grad_norm: Some(5.0),
        weight_decay: 0.0,
    };

    // Training loop with cosine annealing
    for epoch in 0..epochs {
        let current_lr = lr * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / epochs as f32).cos());
        
        let _ = network.try_train_step_with_options(
            &inputs_norm,
            &targets,
            None,
            current_lr,
            &mut workspace,
            &options
        );
        
        if verbose && (epoch % 2000 == 0 || epoch == epochs - 1) {
            let mut preds = vec![0.0f32; n_samples];
            network.forward_batch(&inputs_norm, &mut preds, &mut workspace);
            let mse: f32 = preds.iter().zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>() / n_samples as f32;
            let max_err: f32 = preds.iter().zip(targets.iter())
                .map(|(p, t)| (p - t).abs())
                .fold(0.0f32, f32::max);
            println!("  Epoch {:5}: lr={:.5}, mse={:.8}, max_err={:.6}", epoch, current_lr, mse, max_err);
        }
    }

    // Final evaluation
    let mut final_preds = vec![0.0f32; n_samples];
    network.forward_batch(&inputs_norm, &mut final_preds, &mut workspace);
    
    let mse: f32 = final_preds.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>() / n_samples as f32;
    let mae: f32 = final_preds.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .sum::<f32>() / n_samples as f32;
    let max_err: f32 = final_preds.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).abs())
        .fold(0.0f32, f32::max);

    (mse, mae, max_err)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Training Test: sin(x) - Perfect Approximation ===\n");

    // Phase 1: Find best seed
    println!("Phase 1: Finding best initialization seed...");
    let seeds = [42, 123, 456, 789, 1337, 2023, 3141, 9999];
    let mut best_mse = f32::MAX;
    let mut best_seed = 42;

    for &seed in &seeds {
        let (mse, _, _) = train_sinusoid(seed, 1000, 0.1, 8, false);
        if mse < best_mse {
            best_mse = mse;
            best_seed = seed;
        }
        println!("  Seed {:5}: MSE = {:.8}", seed, mse);
    }
    println!("Best seed: {} with MSE = {:.8}\n", best_seed, best_mse);

    // Phase 2: Long training with best seed
    println!("Phase 2: Long training with best seed (8 hidden neurons)...");
    let (mse, mae, max_err) = train_sinusoid(best_seed, 10000, 0.1, 8, true);

    println!("\n=== Final Results ===");
    println!("MSE:       {:.10}", mse);
    println!("MAE:       {:.8}", mae);
    println!("Max Error: {:.8}", max_err);

    if mse < 1e-6 {
        println!("\n🎯 PERFECT FIT! MSE < 1e-6");
    } else if mse < 1e-5 {
        println!("\n🎯 NEAR-PERFECT FIT! MSE < 1e-5");
    } else if mse < 1e-4 {
        println!("\n✅ EXCELLENT FIT! MSE < 1e-4");
    } else if mse < 0.001 {
        println!("\n✅ VERY GOOD FIT! MSE < 0.001");
    } else if mse < 0.01 {
        println!("\n✅ GOOD FIT! MSE < 0.01");
    } else {
        println!("\n⚠️ Could be better...");
    }

    // Phase 3: Extra long training to see limit
    println!("\n\nPhase 3: Extended training (50000 epochs, 16 neurons)...");
    // Find best seed for 16 neurons
    let mut best_mse_16 = f32::MAX;
    let mut best_seed_16 = 42;
    for &seed in &seeds {
        let (mse, _, _) = train_sinusoid(seed, 1000, 0.1, 16, false);
        if mse < best_mse_16 {
            best_mse_16 = mse;
            best_seed_16 = seed;
        }
    }
    println!("Best seed for 16 neurons: {} with MSE = {:.8}", best_seed_16, best_mse_16);
    let (mse, mae, max_err) = train_sinusoid(best_seed_16, 50000, 0.2, 16, true);
    
    println!("\n=== Extended Results ===");
    println!("MSE:       {:.10}", mse);
    println!("MAE:       {:.8}", mae);
    println!("Max Error: {:.8}", max_err);
    
    // Show sample predictions
    println!("\n=== Sample Predictions ===");
    println!("{:>10} {:>12} {:>12} {:>12}", "x", "sin(x)", "predicted", "error");
    println!("{}", "-".repeat(50));
    
    let config = KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![16],
        grid_size: 16,
        spline_order: 3,
        grid_range: (-1.5, 1.5),
        input_mean: vec![0.0],
        input_std: vec![1.0],
        multithreading_threshold: 16,
        simd_width: 8,
        init_seed: Some(best_seed_16),
    };
    
    let mut network = KanNetwork::new(config);
    let mut workspace = network.create_workspace(100);
    
    // Quick training to get final model
    let options = TrainOptions {
        max_grad_norm: Some(5.0),
        weight_decay: 0.0,
    };
    
    let n_samples = 100;
    let mut inputs_norm = Vec::with_capacity(n_samples);
    let mut targets = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * (i as f32) / (n_samples as f32 - 1.0);
        inputs_norm.push(x / std::f32::consts::PI); // Normalize to [-1, 1]
        targets.push(x.sin());
    }
    
    for epoch in 0..50000 {
        let lr = 0.2 * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / 50000.0).cos());
        let _ = network.try_train_step_with_options(&inputs_norm, &targets, None, lr, &mut workspace, &options);
    }
    
    let mut preds = vec![0.0f32; n_samples];
    network.forward_batch(&inputs_norm, &mut preds, &mut workspace);
    
    for i in (0..n_samples).step_by(10) {
        let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * (i as f32) / (n_samples as f32 - 1.0);
        let target = targets[i];
        let pred = preds[i];
        let err = (target - pred).abs();
        println!("{:10.4} {:12.6} {:12.6} {:12.6}", x, target, pred, err);
    }

    Ok(())
}
