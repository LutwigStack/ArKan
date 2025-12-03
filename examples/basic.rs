//! Basic ArKan Inference Example
//!
//! This example demonstrates the fundamental usage of ArKan for
//! Kolmogorov-Arnold Network inference with zero allocations.
//!
//! # Key Concepts
//!
//! 1. **KanConfig**: Configuration for network architecture
//! 2. **KanNetwork**: The neural network instance
//! 3. **Workspace**: Pre-allocated buffer for zero-allocation inference
//!
//! # Performance
//!
//! ArKan achieves ~30µs latency for single-sample inference by:
//! - Pre-allocating all memory via Workspace
//! - Using SIMD-optimized B-spline evaluation
//! - Cache-friendly memory layout
//!
//! # Run
//!
//! ```bash
//! cargo run --example basic
//! ```

use arkan::{KanConfig, KanNetwork};

fn main() {
    println!("=== ArKan Basic Example ===\n");

    // Build a default config (poker preset) and network.
    let config = KanConfig::preset();
    println!("Network config:");
    println!("  Input:  {} neurons", config.input_dim);
    println!("  Hidden: {:?}", config.hidden_dims);
    println!("  Output: {} neurons", config.output_dim);
    println!("  Grid:   {}, Order: {}", config.grid_size, config.spline_order);

    let network = KanNetwork::new(config.clone());
    println!("\nNetwork created:");
    println!("  Layers:     {}", network.num_layers());
    println!("  Parameters: {}", network.param_count());

    // Preallocate workspace for the maximum batch we expect (here: 8).
    // This allocation happens ONCE. All subsequent calls are zero-alloc.
    let mut workspace = network.create_workspace(8);
    println!("\nWorkspace allocated for max batch size: 8");

    // Single-sample forward: inputs length must be input_dim.
    let mut inputs = vec![0.0f32; config.input_dim];
    inputs[0] = 0.5;

    let mut outputs = vec![0.0f32; config.output_dim];

    println!("\n--- Single Forward Pass ---");
    let start = std::time::Instant::now();
    network.forward_single(&inputs, &mut outputs, &mut workspace);
    let elapsed = start.elapsed();
    
    println!("Time:   {:?}", elapsed);
    println!("Output: [0]={:.6}, [1]={:.6}, ...", outputs[0], outputs[1]);

    // Batch forward (batch = 8).
    let batch = 8;
    let batch_inputs = vec![0.0f32; batch * config.input_dim];
    let mut batch_outputs = vec![0.0f32; batch * config.output_dim];

    println!("\n--- Batch Forward Pass (batch={}) ---", batch);
    let start = std::time::Instant::now();
    network.forward_batch(&batch_inputs, &mut batch_outputs, &mut workspace);
    let elapsed = start.elapsed();
    
    println!("Time:   {:?}", elapsed);
    println!("Output sample [0]: {:.6}", batch_outputs[0]);

    println!("\n✓ Zero-allocation inference complete!");
}
