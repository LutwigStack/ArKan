//! GPU Memory Safety Tests
//!
//! Tests for GPU memory limits, bounds checking, and precision.
//! These tests verify that the GPU implementation handles edge cases safely.
//!
//! Run with: cargo test --features gpu --test gpu_memory_safety -- --ignored

#![cfg(feature = "gpu")]

use arkan::gpu::{
    GpuNetwork, GpuTensor, GpuWorkspace, WgpuBackend, WgpuOptions, MAX_VRAM_ALLOC,
};
use arkan::{ArkanError, KanConfig, KanNetwork};

/// Helper to create a valid config with proper normalization
fn make_config(input_dim: usize, output_dim: usize, hidden_dims: Vec<usize>) -> KanConfig {
    KanConfig {
        input_dim,
        output_dim,
        hidden_dims,
        spline_order: 3,
        grid_size: 5,
        grid_range: (-2.0, 2.0),
        input_mean: vec![0.0; input_dim],
        input_std: vec![1.0; input_dim],
        ..Default::default()
    }
}

// =============================================================================
// GPU Memory Exhaustion Tests
// =============================================================================

#[test]
#[ignore = "Requires GPU"]
fn test_tensor_upload_exceeds_vram_limit() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Try to create a tensor larger than MAX_VRAM_ALLOC (2GB)
    // MAX_VRAM_ALLOC = 2GB = 2 * 1024^3 bytes
    // f32 = 4 bytes, so max elements = 2GB / 4 = 512M elements
    // We try to allocate slightly more
    let too_large_elements = (MAX_VRAM_ALLOC / 4) as usize + 1000;

    // This should fail with BatchTooLarge, not panic
    let result = GpuTensor::uninit(
        &backend.device,
        vec![too_large_elements],
        wgpu::BufferUsages::STORAGE,
    );

    match result {
        Err(ArkanError::BatchTooLarge(requested, max)) => {
            assert!(requested > max, "requested={} should be > max={}", requested, max);
            println!("✅ Correctly rejected tensor allocation: requested={}, max={}", requested, max);
        }
        Ok(_) => panic!("Should have failed with BatchTooLarge"),
        Err(other) => panic!("Wrong error type: {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_workspace_exceeds_vram_limit() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Try to create workspace with huge batch that exceeds VRAM
    // batch * dim * sizeof(f32) > MAX_VRAM_ALLOC
    let huge_batch = (MAX_VRAM_ALLOC / 4 / 1024) as usize + 1; // Approx 512K batches with dim=1024

    let result = GpuWorkspace::new(&backend.device, huge_batch, 1024, 1024);

    match result {
        Err(ArkanError::BatchTooLarge(_, _)) => {
            println!("✅ Workspace correctly rejected huge batch allocation");
        }
        Ok(_) => panic!("Should have failed with BatchTooLarge"),
        Err(other) => panic!("Wrong error type: {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_workspace_ensure_capacity_rejects_huge_batch() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Start with small workspace
    let mut workspace = GpuWorkspace::new(&backend.device, 32, 64, 64)
        .expect("Failed to create small workspace");

    // Try to resize to huge batch
    let huge_batch = (MAX_VRAM_ALLOC / 4 / 64) as usize + 1;
    let result = workspace.ensure_capacity(&backend.device, huge_batch);

    match result {
        Err(ArkanError::BatchTooLarge(_, _)) => {
            println!("✅ ensure_capacity correctly rejected huge batch");
        }
        Ok(_) => panic!("Should have failed with BatchTooLarge"),
        Err(other) => panic!("Wrong error type: {:?}", other),
    }
}

#[test]
#[ignore = "Requires GPU"]
fn test_forward_batch_shape_mismatch_returns_error() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let config = make_config(8, 4, vec![]);
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut workspace = gpu_network.create_workspace(32).expect("workspace");

    // Wrong input size: claim batch=32 but provide data for batch=16
    let input = vec![0.5f32; 16 * config.input_dim]; // Only 16 samples
    let result = gpu_network.forward_batch(&input, 32, &mut workspace);

    match result {
        Err(ArkanError::ShapeMismatch { .. }) => {
            println!("✅ forward_batch correctly rejected mismatched input");
        }
        Ok(_) => panic!("Should have failed with ShapeMismatch"),
        Err(other) => panic!("Wrong error type: {:?}", other),
    }
}

// =============================================================================
// Shader Bounds Checking Tests
// =============================================================================

#[test]
#[ignore = "Requires GPU"]
fn test_shader_bounds_with_non_power_of_two_batch() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let config = make_config(7, 3, vec![]); // Not power of 2
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(17); // Not power of 2

    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(17).expect("workspace");

    // Non-power-of-2 batch: 17 samples
    let input: Vec<f32> = (0..17 * config.input_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();

    // CPU reference
    let mut cpu_output = vec![0.0f32; 17 * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

    // GPU forward - must not crash or produce garbage
    let gpu_output = gpu_network
        .forward_batch(&input, 17, &mut gpu_workspace)
        .expect("GPU forward with non-power-of-2");

    // Compare
    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "Non-power-of-2 batch mismatch: max_diff={}",
        max_diff
    );

    println!("✅ Shader bounds OK with non-power-of-2 batch (max_diff={})", max_diff);
}

#[test]
#[ignore = "Requires GPU"]
fn test_shader_bounds_with_batch_size_one() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let config = make_config(4, 2, vec![8]);
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(1);

    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(1).expect("workspace");

    let input = vec![0.5f32; config.input_dim];

    // CPU reference
    let mut cpu_output = vec![0.0f32; config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

    // GPU forward with batch=1
    let gpu_output = gpu_network
        .forward_batch(&input, 1, &mut gpu_workspace)
        .expect("GPU forward batch=1");

    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff < 1e-5, "Batch=1 mismatch: max_diff={}", max_diff);

    println!("✅ Shader bounds OK with batch=1 (max_diff={})", max_diff);
}

#[test]
#[ignore = "Requires GPU"]
fn test_shader_bounds_large_output_dim() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Large output dimension to stress workgroup indexing
    let config = make_config(16, 513, vec![]); // 513 not divisible by 64 (workgroup size)
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(8);

    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(8).expect("workspace");

    let input: Vec<f32> = (0..8 * config.input_dim)
        .map(|i| (i as f32 * 0.05).cos())
        .collect();

    let mut cpu_output = vec![0.0f32; 8 * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

    let gpu_output = gpu_network
        .forward_batch(&input, 8, &mut gpu_workspace)
        .expect("GPU forward large out_dim");

    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "Large out_dim mismatch: max_diff={}",
        max_diff
    );

    println!(
        "✅ Shader bounds OK with out_dim=513 (max_diff={})",
        max_diff
    );
}

#[test]
#[ignore = "Requires GPU"]
fn test_shader_bounds_extreme_input_values() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let config = make_config(4, 2, vec![]);
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(4);

    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(4).expect("workspace");

    // Extreme input values: way outside grid range, infinities, etc.
    let input = vec![
        // Sample 1: normal
        0.0, 0.5, -0.5, 1.0,
        // Sample 2: outside grid range
        -100.0, 100.0, -1000.0, 1000.0,
        // Sample 3: boundary values
        -2.0, 2.0, -2.0, 2.0, // Exact grid boundaries
        // Sample 4: very small values
        1e-30, -1e-30, 1e-20, -1e-20,
    ];

    // CPU reference
    let mut cpu_output = vec![0.0f32; 4 * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

    // GPU forward
    let gpu_output = gpu_network
        .forward_batch(&input, 4, &mut gpu_workspace)
        .expect("GPU forward extreme inputs");

    // Should not produce NaN/Inf
    for (i, &v) in gpu_output.iter().enumerate() {
        assert!(v.is_finite(), "GPU output[{}] is not finite: {}", i, v);
    }

    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "Extreme inputs mismatch: max_diff={}",
        max_diff
    );

    println!(
        "✅ Shader handles extreme inputs correctly (max_diff={})",
        max_diff
    );
}

// =============================================================================
// GPU Precision Tests (f32 only - document limitation)
// =============================================================================

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_precision_f32_accumulation() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Large network to test f32 accumulation precision
    let config = KanConfig {
        input_dim: 128, // Many inputs to accumulate
        output_dim: 64,
        hidden_dims: vec![],
        spline_order: 3,
        grid_size: 8,
        grid_range: (-2.0, 2.0),
        input_mean: vec![0.0; 128],
        input_std: vec![1.0; 128],
        ..Default::default()
    };
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(16);

    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(16).expect("workspace");

    // Input that causes many small additions
    let input: Vec<f32> = (0..16 * config.input_dim)
        .map(|i| ((i as f32 * 0.01).sin() * 0.1))
        .collect();

    let mut cpu_output = vec![0.0f32; 16 * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

    let gpu_output = gpu_network
        .forward_batch(&input, 16, &mut gpu_workspace)
        .expect("GPU forward");

    // f32 accumulation may differ due to parallel summation order
    // Tolerance should be reasonable for f32
    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // f32 precision: ~7 significant digits, so relative error up to ~1e-5 is expected
    // With large accumulations, absolute error can be larger
    assert!(
        max_diff < 1e-3,
        "f32 accumulation precision exceeded: max_diff={} (expected < 1e-3)",
        max_diff
    );

    println!(
        "✅ f32 precision test passed: max_diff={:.2e} (in_dim=128)",
        max_diff
    );
}

#[test]
#[ignore = "Requires GPU"]
fn test_gpu_precision_deterministic() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    let config = make_config(32, 16, vec![64]);
    let cpu_network = KanNetwork::new(config.clone());
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut workspace = gpu_network.create_workspace(32).expect("workspace");

    let input: Vec<f32> = (0..32 * config.input_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();

    // Run same input multiple times
    let mut outputs = Vec::new();
    for _ in 0..5 {
        let output = gpu_network
            .forward_batch(&input, 32, &mut workspace)
            .expect("GPU forward");
        outputs.push(output);
    }

    // All outputs must be exactly equal (deterministic)
    for (i, out) in outputs.iter().enumerate().skip(1) {
        for (j, (a, b)) in outputs[0].iter().zip(out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "GPU non-deterministic at run {} element {}: {} != {}",
                i,
                j,
                a,
                b
            );
        }
    }

    println!("✅ GPU forward is deterministic (5 identical runs)");
}

// =============================================================================
// Multi-Layer Bounds Tests
// =============================================================================

#[test]
#[ignore = "Requires GPU"]
fn test_multi_layer_intermediate_buffer_bounds() {
    let backend = WgpuBackend::init(WgpuOptions::default()).expect("GPU init");

    // Network with varying layer sizes to stress intermediate buffer indexing
    // All primes, not divisible by workgroup
    let config = KanConfig {
        input_dim: 13,   // Prime number
        output_dim: 7,   // Prime number
        hidden_dims: vec![31, 17, 11], // All primes
        spline_order: 3,
        grid_size: 5,
        grid_range: (-2.0, 2.0),
        input_mean: vec![0.0; 13],
        input_std: vec![1.0; 13],
        ..Default::default()
    };
    let cpu_network = KanNetwork::new(config.clone());
    let mut cpu_workspace = cpu_network.create_workspace(19); // Prime batch size

    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network).expect("GPU network");
    let mut gpu_workspace = gpu_network.create_workspace(19).expect("workspace");

    let input: Vec<f32> = (0..19 * config.input_dim)
        .map(|i| (i as f32 * 0.07).sin())
        .collect();

    let mut cpu_output = vec![0.0f32; 19 * config.output_dim];
    cpu_network.forward_batch(&input, &mut cpu_output, &mut cpu_workspace);

    let gpu_output = gpu_network
        .forward_batch(&input, 19, &mut gpu_workspace)
        .expect("GPU forward");

    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "Multi-layer prime dimensions mismatch: max_diff={}",
        max_diff
    );

    println!(
        "✅ Multi-layer with prime dimensions OK (max_diff={})",
        max_diff
    );
}

// =============================================================================
// Documentation Tests - Known Limitations
// =============================================================================

/// Documents that f16 (half precision) is NOT supported.
///
/// ArKan GPU backend uses f32 exclusively for:
/// - Weight storage
/// - Input/output buffers
/// - All intermediate computations
///
/// This is a deliberate design choice for:
/// 1. Maximum precision in numerical computations
/// 2. Consistent behavior with CPU implementation
/// 3. Avoiding mixed-precision complexity
///
/// If f16 support is needed in the future, it would require:
/// - New shader variants with `f16` types
/// - Half extension support check via wgpu capabilities
/// - Conversion utilities between f32/f16
#[test]
fn test_f16_not_supported_documented() {
    // This test exists purely for documentation purposes
    // f16 is not supported - this is a known limitation
    println!("ℹ️  f16 (half precision) is NOT supported - f32 only");
    println!("   This is documented in FUNCTIONALITY_AUDIT.md");
}

/// Documents that Multi-GPU is NOT supported.
///
/// ArKan uses a single GPU via wgpu with:
/// - `WgpuBackend::init()` selects one adapter
/// - All operations use that single device
///
/// For multi-GPU support, users should:
/// 1. Create multiple WgpuBackend instances (one per GPU)
/// 2. Manually partition work across GPUs
/// 3. Handle data transfer between GPUs
#[test]
fn test_multi_gpu_not_supported_documented() {
    // This test exists purely for documentation purposes
    println!("ℹ️  Multi-GPU is NOT supported");
    println!("   Users can create multiple WgpuBackend instances for manual multi-GPU");
}
