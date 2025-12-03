//! GPU Forward Pass Example
//!
//! This example demonstrates how to use the GPU backend for KAN network inference.
//!
//! Run with: cargo run --example gpu_forward --features gpu

#[cfg(feature = "gpu")]
use arkan::{KanConfig, KanNetwork};

#[cfg(feature = "gpu")]
use arkan::gpu::{GpuNetwork, WgpuBackend, WgpuOptions};

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("This example requires the 'gpu' feature. Run with:");
        eprintln!("  cargo run --example gpu_forward --features gpu");
        std::process::exit(1);
    }

    #[cfg(feature = "gpu")]
    run_gpu_example();
}

#[cfg(feature = "gpu")]
fn run_gpu_example() {
    println!("=== ArKan GPU Forward Pass Example ===\n");

    // 1. Create a CPU network
    println!("1. Creating CPU network...");
    let config = KanConfig {
        input_dim: 21,
        output_dim: 24,
        hidden_dims: vec![64, 64],
        grid_size: 5,
        spline_order: 3,
        input_mean: vec![0.0; 21],
        input_std: vec![1.0; 21],
        ..Default::default()
    };
    config.validate().expect("Invalid config");

    let cpu_network = KanNetwork::new(config.clone());
    println!("   Layers: {:?}", cpu_network.layers.len());
    println!("   Parameters: {}", cpu_network.param_count());

    // 2. Initialize GPU backend
    println!("\n2. Initializing GPU backend...");
    let backend = match WgpuBackend::init(WgpuOptions::default()) {
        Ok(b) => {
            println!("   Adapter: {}", b.adapter_info().name);
            println!("   Backend: {:?}", b.adapter_info().backend);
            b
        }
        Err(e) => {
            eprintln!("   Failed to initialize GPU: {}", e);
            eprintln!("   Make sure you have a compatible GPU and drivers installed.");
            std::process::exit(1);
        }
    };

    // 3. Create GPU network
    println!("\n3. Creating GPU network...");
    let mut gpu_network = match GpuNetwork::from_cpu(&backend, &cpu_network) {
        Ok(n) => {
            println!("   GPU network created successfully");
            println!("   GPU parameters: {}", n.param_count());
            n
        }
        Err(e) => {
            eprintln!("   Failed to create GPU network: {}", e);
            std::process::exit(1);
        }
    };

    // 4. Create workspace
    println!("\n4. Creating GPU workspace...");
    let batch_size = 64;
    let mut workspace = gpu_network.create_workspace(batch_size).expect("Failed to create workspace");
    println!("   Max batch size: {}", workspace.max_batch);

    // 5. Prepare input data
    println!("\n5. Preparing input data...");
    let input: Vec<f32> = (0..batch_size * config.input_dim)
        .map(|i| (i as f32 * 0.01) % 1.0)
        .collect();
    println!("   Input shape: [{}, {}]", batch_size, config.input_dim);

    // 6. Run forward pass on GPU
    println!("\n6. Running GPU forward pass...");
    let start = std::time::Instant::now();
    let output = gpu_network
        .forward_batch(&input, batch_size, &mut workspace)
        .expect("GPU forward failed");
    let gpu_time = start.elapsed();
    println!("   GPU time: {:?}", gpu_time);
    println!("   Output shape: [{}, {}]", batch_size, config.output_dim);

    // 7. Compare with CPU
    println!("\n7. Running CPU forward pass for comparison...");
    let mut cpu_workspace = cpu_network.create_workspace(batch_size);
    let mut cpu_output = vec![0.0f32; batch_size * config.output_dim];
    
    let start = std::time::Instant::now();
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);
    let cpu_time = start.elapsed();
    println!("   CPU time: {:?}", cpu_time);

    // 8. Check results
    println!("\n8. Comparing results...");
    let max_diff: f32 = output
        .iter()
        .zip(&cpu_output)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("   Max difference: {:.6}", max_diff);
    
    if max_diff < 0.1 {
        println!("   ✓ Results match within tolerance");
    } else {
        println!("   ⚠ Results differ significantly (this may be due to different spline implementations)");
    }

    // 9. Performance summary
    println!("\n=== Performance Summary ===");
    println!("Batch size: {}", batch_size);
    println!("GPU time: {:?}", gpu_time);
    println!("CPU time: {:?}", cpu_time);
    if cpu_time > gpu_time {
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    } else {
        println!("Note: GPU may be slower for small batches due to overhead");
    }

    println!("\n✓ Example completed successfully!");
}
