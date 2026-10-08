//! Every code snippet in `README.md` and `docs/BENCHMARKS.md`, copied verbatim.
//!
//! The README is the first thing a stranger reads and the last thing anyone
//! re-runs. `backend.adapter_name()` sat in both language halves of the README
//! for two releases; the method is `adapter_info()`. Nothing caught it, because
//! the README is not a doctest target and the GPU snippets are marked `ignore`.
//!
//! So the snippets live here too. The `#[cfg(feature = "gpu")]` ones are never
//! called — type-checking them is the whole point, since running them needs an
//! adapter no CI runner has. `cargo clippy --all-targets --features gpu` in CI
//! is what compiles them.
//!
//! ponytail: copies, not includes. Extracting fenced blocks out of a bilingual
//! README at test time would cost a parser and a dependency to catch the same
//! class of bug; a stale copy here fails just as loudly, because a copy that
//! stops matching the README is exactly what a reviewer is looking at.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// README: "Baked (int8) Inference" -> Usage
// ---------------------------------------------------------------------------

#[test]
fn readme_baked_usage() {
    use arkan::{BakedModel, KanConfig, KanNetwork};

    // 1. Train a KanNetwork as usual.
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    // ... train ...

    // 2. Collect a calibration set (flat: n_samples * input_dim f32 values).
    //    Use REAL representative inputs: the 99.9th percentile of the activations
    //    they produce sets the baked activation scale.
    //    256-1024 samples is typical. Empty calibration uses the uncalibrated heuristic.
    let n_samples = 256;
    let calibration: Vec<f32> = (0..n_samples * config.input_dim)
        .map(|i| (i % 17) as f32 / 17.0 - 0.5) // stand-in for your real data
        .collect();

    // 3. Bake.
    let baked = BakedModel::from_network(&network, Some(&calibration));

    // 4. Run fixed-point inference.
    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];
    baked.forward(&input, &mut output);

    // 5. Check size.
    println!("Baked model: {} bytes", baked.size_bytes());

    assert!(baked.size_bytes() > 0);
}

/// Empty calibration must report the fallback rather than silently collapse output.
#[test]
fn readme_empty_calibration_uses_uncalibrated_fallback() {
    use arkan::{BakedModel, KanConfig, KanNetwork};

    let mut config = KanConfig::preset();
    config.init_seed = Some(20260726);
    let mut network = KanNetwork::new(config.clone());
    // A constant output isolates activation-scale handling from quantized weights.
    let last = network.layers.last_mut().unwrap();
    last.weights.fill(0.0);
    last.bias.fill(0.25);
    let baked = BakedModel::from_network(&network, Some(&[]));
    assert!(baked.uncalibrated);

    let input = vec![0.5f32; config.input_dim];
    let mut baked_out = vec![0.0f32; config.output_dim];
    baked.forward(&input, &mut baked_out);
    let mut f32_out = vec![0.0f32; config.output_dim];
    let mut ws = network.create_workspace(1);
    network.forward_single(&input, &mut f32_out, &mut ws);
    assert_eq!(f32_out, vec![0.25; config.output_dim]);
    assert_eq!(baked_out, f32_out);
}

// ---------------------------------------------------------------------------
// README: "Baked (int8) Inference" -> Serialization
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
#[test]
fn readme_baked_serialization() -> Result<(), Box<dyn std::error::Error>> {
    use arkan::{BakedModel, KanConfig, KanNetwork};

    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    let calibration = vec![0.25f32; 256 * config.input_dim];
    let baked = BakedModel::from_network(&network, Some(&calibration));

    // Serialize - prepends 12-byte magic + 4-byte version, then bincode body.
    let bytes: Vec<u8> = baked.to_bytes()?;

    // Deserialize - validates magic and version before parsing; returns Err on mismatch.
    let baked2 = BakedModel::from_bytes(&bytes)?;

    assert_eq!(&bytes[..12], arkan::MAGIC_BAKED);
    assert_eq!(baked2.size_bytes(), baked.size_bytes());
    Ok(())
}

// ---------------------------------------------------------------------------
// README: "Quick Start" / "Быстрый старт"
// ---------------------------------------------------------------------------

#[test]
fn readme_quick_start() {
    use arkan::{KanConfig, KanNetwork};

    // 1. Configuration
    let config = KanConfig::preset();

    // 2. Network initialization
    let network = KanNetwork::new(config.clone());

    // 3. Create Workspace (memory allocated once)
    let mut workspace = network.create_workspace(64); // Max batch size = 64

    // 4. Data preparation
    let inputs = vec![0.0f32; 64 * config.input_dim];
    let mut outputs = vec![0.0f32; 64 * config.output_dim];

    // 5. Inference (Zero allocations here!)
    network.forward_batch(&inputs, &mut outputs, &mut workspace);

    println!("Inference done. Output[0]: {}", outputs[0]);

    assert!(outputs.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// README: "GPU Backend" -> Usage. Compile-checked only; needs a real adapter.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn readme_gpu_usage() -> Result<(), Box<dyn std::error::Error>> {
    use arkan::gpu::{GpuNetwork, WgpuBackend, WgpuOptions};
    use arkan::optimizer::{Adam, AdamConfig};
    use arkan::{KanConfig, KanNetwork};

    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())?;
    println!("GPU: {}", backend.adapter_info().name);

    // Create CPU network
    let config = KanConfig::preset();
    let mut cpu_network = KanNetwork::new(config.clone());

    // Create GPU network from CPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)?;
    let mut workspace = gpu_network.create_workspace(64)?;

    // Forward inference
    let input = vec![0.5f32; config.input_dim];
    let output = gpu_network.forward_single(&input, &mut workspace)?;
    let _ = output;

    // Training with Adam optimizer
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));
    let target = vec![1.0f32; config.output_dim];

    let loss = gpu_network.train_step_mse(
        &input,
        &target,
        1,
        &mut workspace,
        &mut optimizer,
        &mut cpu_network,
    )?;

    println!("Loss: {}", loss);
    Ok(())
}

// ---------------------------------------------------------------------------
// README: "Weight Synchronization"
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn readme_weight_sync(
    gpu_network: &mut arkan::gpu::GpuNetwork,
    cpu_network: &mut arkan::KanNetwork,
) -> Result<(), Box<dyn std::error::Error>> {
    // Sync weights from CPU to GPU (after loading a model)
    gpu_network.sync_weights_cpu_to_gpu(cpu_network)?;

    // Sync weights from GPU to CPU (for saving/export)
    gpu_network.sync_weights_gpu_to_cpu(cpu_network)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// README: "Training with Options"
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn readme_train_with_options(
    gpu_network: &mut arkan::gpu::GpuNetwork,
    cpu_network: &mut arkan::KanNetwork,
    workspace: &mut arkan::gpu::GpuWorkspace,
    optimizer: &mut arkan::Adam,
    input: &[f32],
    target: &[f32],
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use arkan::TrainOptions;

    let opts = TrainOptions {
        max_grad_norm: Some(1.0), // Gradient clipping
        weight_decay: 0.01,       // AdamW-style weight decay
    };

    let loss = gpu_network.train_step_with_options(
        input,
        target,
        None,
        batch_size,
        workspace,
        optimizer,
        cpu_network,
        &opts,
    )?;
    let _ = loss;
    Ok(())
}

// ---------------------------------------------------------------------------
// README: "Choosing Backend"
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
// The README shows these as alternatives, one per line; only the last binding
// is read here. Shadowing is the point.
#[allow(unused_variables)]
fn readme_choosing_backend() -> Result<(), Box<dyn std::error::Error>> {
    use arkan::gpu::{WgpuBackend, WgpuOptions};

    // High-performance GPU (default, 2GB limit)
    let backend = WgpuBackend::init(WgpuOptions::default())?;

    // Compute-optimized (unlimited VRAM)
    let backend = WgpuBackend::init(WgpuOptions::compute())?;

    // Custom VRAM limit in GB (recommended for known hardware)
    let backend = WgpuBackend::init(WgpuOptions::with_max_vram(3))?; // 3GB

    // Percentage of device max (works on AMD/Intel, not useful for NVIDIA)
    let backend = WgpuBackend::init(WgpuOptions::with_max_vram_percent(30))?;

    // No VRAM limit (use device max)
    let backend = WgpuBackend::init(WgpuOptions::unlimited_vram())?;

    // Low-memory/integrated GPU
    let backend = WgpuBackend::init(WgpuOptions::low_memory())?;

    // Force specific adapter
    let opts = WgpuOptions {
        force_adapter_name: Some("NVIDIA".to_string()),
        ..Default::default()
    };
    let backend = WgpuBackend::init(opts)?;
    let _ = backend;
    Ok(())
}

// ---------------------------------------------------------------------------
// docs/BENCHMARKS.md: "Native GPU Training" -> API Usage
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
fn benchmarks_native_gpu_training(
    backend: &arkan::gpu::WgpuBackend,
    cpu_network: &arkan::KanNetwork,
    workspace: &mut arkan::gpu::GpuWorkspace,
    input: &[f32],
    target: &[f32],
    batch_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use arkan::gpu::{GpuAdam, GpuAdamConfig, GpuNetwork};

    // Create network and optimizer
    let mut gpu_network = GpuNetwork::from_cpu(backend, cpu_network)?;
    let layer_sizes = gpu_network.layer_param_sizes();
    let mut optimizer = GpuAdam::new(
        backend.device_arc(),
        backend.queue_arc(),
        &layer_sizes,
        GpuAdamConfig::with_lr(0.001),
    );

    // Native GPU training - no CPU transfers!
    let loss =
        gpu_network.train_step_gpu_native(input, target, batch_size, workspace, &mut optimizer)?;
    let _ = loss;
    Ok(())
}
