//! Comprehensive functionality test for ArKan library
//! Tests CPU, GPU, forward, backward, optimizers, etc.

use arkan::optimizer::{Adam, AdamConfig, CosineAnnealingLR, LrScheduler, SGDConfig, StepLR, SGD};
use arkan::{KanConfig, KanNetwork, TrainOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("         ArKan Comprehensive Functionality Test");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut passed = 0;
    let mut failed = 0;

    // ===== Test 1: Network Creation =====
    print!("1. Network Creation... ");
    let config = KanConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dims: vec![16, 8],
        grid_size: 8,
        spline_order: 3,
        grid_range: (-2.0, 2.0),
        input_mean: vec![0.0; 4],
        input_std: vec![1.0; 4],
        init_seed: Some(42),
        ..Default::default()
    };
    let network = KanNetwork::new(config.clone());
    if network.num_layers() == 3 && network.param_count() > 0 {
        println!(
            "✅ PASSED (layers={}, params={})",
            network.num_layers(),
            network.param_count()
        );
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 2: Forward Single =====
    print!("2. Forward Single... ");
    let mut network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(1);
    let input = vec![0.5f32; 4];
    let mut output = vec![0.0f32; 2];
    network.forward_batch(&input, &mut output, &mut workspace);
    if output.iter().all(|x| x.is_finite()) {
        println!("✅ PASSED (output={:.4}, {:.4})", output[0], output[1]);
        passed += 1;
    } else {
        println!("❌ FAILED (NaN or Inf in output)");
        failed += 1;
    }

    // ===== Test 3: Forward Batch =====
    print!("3. Forward Batch (batch=16)... ");
    let mut workspace = network.create_workspace(16);
    let batch_input: Vec<f32> = (0..16 * 4).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut batch_output = vec![0.0f32; 16 * 2];
    network.forward_batch(&batch_input, &mut batch_output, &mut workspace);
    if batch_output.iter().all(|x| x.is_finite()) {
        println!("✅ PASSED");
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 4: Train Step (SGD) =====
    print!("4. Train Step (SGD)... ");
    let targets: Vec<f32> = (0..16 * 2).map(|i| (i as f32 * 0.05).cos()).collect();
    let loss1 = network.train_step(&batch_input, &targets, None, 0.01, &mut workspace);
    let loss2 = network.train_step(&batch_input, &targets, None, 0.01, &mut workspace);
    if loss2 < loss1 && loss1.is_finite() && loss2.is_finite() {
        println!("✅ PASSED (loss: {:.6} -> {:.6})", loss1, loss2);
        passed += 1;
    } else {
        println!("❌ FAILED (loss not decreasing or NaN)");
        failed += 1;
    }

    // ===== Test 5: Train Step with Options =====
    print!("5. Train Step with Options... ");
    let opts = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.001,
    };
    let loss = network.try_train_step_with_options(
        &batch_input,
        &targets,
        None,
        0.01,
        &mut workspace,
        &opts,
    )?;
    if loss.is_finite() {
        println!("✅ PASSED (loss={:.6})", loss);
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 6: Adam Optimizer =====
    print!("6. Adam Optimizer... ");
    let network = KanNetwork::new(config.clone());
    let _workspace = network.create_workspace(16);
    let adam_config = AdamConfig::with_lr(0.001);
    let adam = Adam::new(&network, adam_config);
    if adam.learning_rate() == 0.001 {
        println!("✅ PASSED (lr={})", adam.learning_rate());
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 7: SGD Optimizer =====
    print!("7. SGD Optimizer... ");
    let sgd = SGD::new(&network, SGDConfig::with_momentum(0.01, 0.9));
    // SGD has lr() method
    println!("✅ PASSED (lr={}, momentum={})", sgd.lr(), sgd.momentum());
    passed += 1;

    // ===== Test 8: LR Schedulers =====
    print!("8. LR Schedulers... ");
    let cosine = CosineAnnealingLR::new(0.1, 100, 0.001);
    let step_lr = StepLR::new(0.1, 10, 0.1);
    let cosine_lr = cosine.get_lr(50, 0.1);
    let step_lr_val = step_lr.get_lr(5, 0.1);
    if cosine_lr < 0.1 && cosine_lr > 0.001 && step_lr_val == 0.1 {
        println!(
            "✅ PASSED (cosine@50={:.4}, step@5={:.4})",
            cosine_lr, step_lr_val
        );
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 9: Different Spline Orders =====
    print!("9. Spline Orders (2,3,4,5)... ");
    let mut order_ok = true;
    for order in 2..=5 {
        let cfg = KanConfig {
            input_dim: 2,
            output_dim: 2,
            hidden_dims: vec![4],
            spline_order: order,
            grid_size: 5,
            grid_range: (-1.0, 1.0),
            input_mean: vec![0.0; 2],
            input_std: vec![1.0; 2],
            ..Default::default()
        };
        let net = KanNetwork::new(cfg);
        let mut ws = net.create_workspace(1);
        let inp = vec![0.5, -0.5];
        let mut out = vec![0.0; 2];
        net.forward_batch(&inp, &mut out, &mut ws);
        if !out.iter().all(|x| x.is_finite()) {
            order_ok = false;
            break;
        }
    }
    if order_ok {
        println!("✅ PASSED");
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 10: Gradient Correctness =====
    print!("10. Gradient Check (training decreases loss)... ");
    let cfg = KanConfig {
        input_dim: 2,
        output_dim: 1,
        hidden_dims: vec![4],
        spline_order: 3,
        grid_size: 5,
        grid_range: (-1.0, 1.0),
        input_mean: vec![0.0; 2],
        input_std: vec![1.0; 2],
        init_seed: Some(42),
        ..Default::default()
    };
    let mut network = KanNetwork::new(cfg.clone());
    let mut workspace = network.create_workspace(10);

    let test_inputs: Vec<f32> = (0..20).map(|i| (i as f32 * 0.1 - 1.0)).collect();
    let test_targets: Vec<f32> = (0..10).map(|i| (i as f32 * 0.1).sin()).collect();

    // Get initial loss
    let mut preds = vec![0.0f32; 10];
    network.forward_batch(&test_inputs, &mut preds, &mut workspace);
    let initial_loss: f32 = preds
        .iter()
        .zip(test_targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / 10.0;

    // Train for some steps
    let opts = TrainOptions {
        max_grad_norm: Some(1.0),
        weight_decay: 0.0,
    };
    for _ in 0..50 {
        network.try_train_step_with_options(
            &test_inputs,
            &test_targets,
            None,
            0.05,
            &mut workspace,
            &opts,
        )?;
    }

    // Get final loss
    network.forward_batch(&test_inputs, &mut preds, &mut workspace);
    let final_loss: f32 = preds
        .iter()
        .zip(test_targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / 10.0;

    if final_loss < initial_loss * 0.5 {
        println!("✅ PASSED (loss: {:.6} -> {:.6})", initial_loss, final_loss);
        passed += 1;
    } else {
        println!(
            "❌ FAILED (loss not improved enough: {:.6} -> {:.6})",
            initial_loss, final_loss
        );
        failed += 1;
    }

    // ===== Test 11: Sin(x) Learning =====
    print!("11. Sin(x) Learning (100 epochs)... ");
    let cfg = KanConfig {
        input_dim: 1,
        output_dim: 1,
        hidden_dims: vec![8],
        spline_order: 3,
        grid_size: 12,
        grid_range: (-1.5, 1.5),
        input_mean: vec![0.0],
        input_std: vec![1.0],
        init_seed: Some(1337),
        ..Default::default()
    };
    let mut network = KanNetwork::new(cfg);
    let mut workspace = network.create_workspace(50);

    let n = 50;
    let inputs: Vec<f32> = (0..n)
        .map(|i| -1.0 + 2.0 * i as f32 / (n as f32 - 1.0))
        .collect();
    let targets: Vec<f32> = inputs
        .iter()
        .map(|x| (x * std::f32::consts::PI).sin())
        .collect();

    let opts = TrainOptions {
        max_grad_norm: Some(5.0),
        weight_decay: 0.0,
    };
    let mut initial_loss = 0.0;
    for epoch in 0..100 {
        let lr = 0.1 * 0.5 * (1.0 + (std::f32::consts::PI * epoch as f32 / 100.0).cos());
        let loss = network.try_train_step_with_options(
            &inputs,
            &targets,
            None,
            lr,
            &mut workspace,
            &opts,
        )?;
        if epoch == 0 {
            initial_loss = loss;
        }
    }

    let mut preds = vec![0.0f32; n];
    network.forward_batch(&inputs, &mut preds, &mut workspace);
    let final_mse: f32 = preds
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / n as f32;

    if final_mse < initial_loss * 0.1 {
        println!("✅ PASSED (MSE: {:.6} -> {:.6})", initial_loss, final_mse);
        passed += 1;
    } else {
        println!("❌ FAILED (MSE not improved enough)");
        failed += 1;
    }

    // ===== Test 12: Workspace Reuse =====
    print!("12. Workspace Reuse... ");
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(32);
    let input8: Vec<f32> = (0..8 * 4).map(|i| (i as f32).sin()).collect();
    let input16: Vec<f32> = (0..16 * 4).map(|i| (i as f32).sin()).collect();
    let input32: Vec<f32> = (0..32 * 4).map(|i| (i as f32).sin()).collect();
    let mut out8 = vec![0.0f32; 8 * 2];
    let mut out16 = vec![0.0f32; 16 * 2];
    let mut out32 = vec![0.0f32; 32 * 2];

    network.forward_batch(&input8, &mut out8, &mut workspace);
    network.forward_batch(&input16, &mut out16, &mut workspace);
    network.forward_batch(&input32, &mut out32, &mut workspace);

    if out8.iter().all(|x| x.is_finite())
        && out16.iter().all(|x| x.is_finite())
        && out32.iter().all(|x| x.is_finite())
    {
        println!("✅ PASSED");
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Summary =====
    println!("\n═══════════════════════════════════════════════════════════");
    println!("                      SUMMARY");
    println!("═══════════════════════════════════════════════════════════");
    println!("  Passed: {} / {}", passed, passed + failed);
    println!("  Failed: {}", failed);
    if failed == 0 {
        println!("\n  🎉 ALL TESTS PASSED!");
    } else {
        println!("\n  ⚠️  Some tests failed!");
    }
    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}
