//! GPU Comprehensive Test for ArKan
//! Tests GPU forward, backward, training, optimizers
//!
//! Run with: cargo run --example gpu_comprehensive_test --features gpu --release

use arkan::gpu::{GpuNetwork, WgpuBackend, WgpuOptions};
use arkan::optimizer::{Adam, AdamConfig};
use arkan::{KanConfig, KanNetwork};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("          ArKan GPU Comprehensive Test");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut passed = 0;
    let mut failed = 0;

    // ===== Test 1: GPU Backend Init =====
    print!("1. GPU Backend Init... ");
    let backend = WgpuBackend::init(WgpuOptions::default())?;
    println!("✅ PASSED ({})", backend.adapter_info().name);
    passed += 1;

    // ===== Test 2: GPU Network Creation =====
    print!("2. GPU Network Creation... ");
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
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)?;
    println!("✅ PASSED (GPU params={})", gpu_network.param_count());
    passed += 1;

    // ===== Test 3: GPU Forward Pass =====
    print!("3. GPU Forward Batch (batch=16)... ");
    let mut workspace = gpu_network.create_workspace(16)?;
    let batch_input: Vec<f32> = (0..16 * 4).map(|i| (i as f32 * 0.1).sin()).collect();
    let output = gpu_network.forward_batch(&batch_input, 16, &mut workspace)?;
    if output.iter().all(|x| x.is_finite()) {
        println!("✅ PASSED");
        passed += 1;
    } else {
        println!("❌ FAILED (NaN in output)");
        failed += 1;
    }

    // ===== Test 4: CPU/GPU Parity =====
    print!("4. CPU/GPU Forward Parity... ");
    let mut cpu_workspace = cpu_network.create_workspace(16);
    let mut cpu_output = vec![0.0f32; 16 * 2];
    cpu_network.forward_batch(&batch_input, &mut cpu_output, &mut cpu_workspace);

    let max_diff: f32 = output
        .iter()
        .zip(cpu_output.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0, f32::max);

    if max_diff < 1e-4 {
        println!("✅ PASSED (max diff = {:.6})", max_diff);
        passed += 1;
    } else {
        println!("❌ FAILED (max diff = {:.6})", max_diff);
        failed += 1;
    }

    // ===== Test 5: GPU Training =====
    print!("5. GPU Training Step... ");
    let targets: Vec<f32> = (0..16 * 2).map(|i| (i as f32 * 0.05).cos()).collect();
    let mut cpu_net = cpu_network.clone();
    let mut adam = Adam::new(&cpu_net, AdamConfig::with_lr(0.01));

    let loss1 = gpu_network.train_step_mse(
        &batch_input,
        &targets,
        16,
        &mut workspace,
        &mut adam,
        &mut cpu_net,
    )?;
    let loss2 = gpu_network.train_step_mse(
        &batch_input,
        &targets,
        16,
        &mut workspace,
        &mut adam,
        &mut cpu_net,
    )?;

    if loss2 < loss1 && loss1.is_finite() {
        println!("✅ PASSED (loss: {:.6} -> {:.6})", loss1, loss2);
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 6: GPU Softmax =====
    print!("6. GPU Softmax... ");
    // Use forward_batch_softmax convenience method
    let mut ws_softmax = gpu_network.create_workspace(16)?;
    let softmax_out = gpu_network.forward_batch_softmax(&batch_input, 16, &mut ws_softmax)?;

    // Check softmax properties: sum = 1, all > 0
    let mut softmax_ok = true;
    for b in 0..16 {
        let sum: f32 = softmax_out[b * 2..(b + 1) * 2].iter().sum();
        if (sum - 1.0).abs() > 1e-3 || softmax_out[b * 2..].iter().take(2).any(|x| *x < 0.0) {
            softmax_ok = false;
            break;
        }
    }
    if softmax_ok {
        println!("✅ PASSED (sum≈1.0 for all samples)");
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 7: Weight Sync Roundtrip =====
    print!("7. Weight Sync Roundtrip... ");
    let mut cpu_net2 = cpu_network.clone();

    // Modify CPU weights
    cpu_net2.layers[0].weights[0] = 999.0;

    // Sync to GPU
    gpu_network.sync_weights(&cpu_net2)?;

    // Sync back
    gpu_network.sync_weights_to_cpu(&mut cpu_net2)?;

    if (cpu_net2.layers[0].weights[0] - 999.0).abs() < 1e-5 {
        println!("✅ PASSED");
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 8: Different Batch Sizes =====
    print!("8. Different Batch Sizes (1,8,32,64)... ");
    let mut batch_ok = true;
    for batch_size in [1, 8, 32, 64] {
        let mut ws = gpu_network.create_workspace(batch_size)?;
        let input: Vec<f32> = (0..batch_size * 4).map(|i| (i as f32).sin()).collect();
        let out = gpu_network.forward_batch(&input, batch_size, &mut ws)?;
        if !out.iter().all(|x| x.is_finite()) {
            batch_ok = false;
            break;
        }
    }
    if batch_ok {
        println!("✅ PASSED");
        passed += 1;
    } else {
        println!("❌ FAILED");
        failed += 1;
    }

    // ===== Test 9: GPU Memory Stats =====
    print!("9. GPU Memory Stats... ");
    let stats = gpu_network.memory_stats();
    if stats.total_bytes > 0 {
        println!(
            "✅ PASSED (total={:.2} KB)",
            stats.total_bytes as f64 / 1024.0
        );
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
        println!("\n  🎉 ALL GPU TESTS PASSED!");
    } else {
        println!("\n  ⚠️  Some tests failed!");
    }
    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}
