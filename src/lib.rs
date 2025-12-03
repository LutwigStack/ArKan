//! # ArKan — High-Performance Kolmogorov-Arnold Networks
//!
//! [![Crates.io](https://img.shields.io/crates/v/arkan.svg)](https://crates.io/crates/arkan)
//! [![Documentation](https://docs.rs/arkan/badge.svg)](https://docs.rs/arkan)
//! [![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/your-username/arkan/blob/main/LICENSE)
//!
//! **ArKan** is a zero-allocation, SIMD-optimized implementation of
//! [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (KAN)
//! designed for latency-critical applications like poker solvers and game AI.
//!
//! ## Why ArKan?
//!
//! | Feature | ArKan | PyTorch KAN |
//! |---------|-------|-------------|
//! | Single inference | **30 µs** | 990 µs |
//! | Memory allocation | Zero (hot path) | Dynamic |
//! | Dependencies | Minimal | Heavy |
//!
//! ## Quick Start
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork};
//!
//! // Create a network with poker-optimized architecture
//! let config = KanConfig::default_poker(); // [21, 64, 64, 24]
//! let network = KanNetwork::new(config.clone());
//!
//! // Preallocate workspace (reuse across calls for zero-alloc)
//! let mut workspace = network.create_workspace(64);
//!
//! // Single inference (~30 µs)
//! let input = vec![0.5f32; config.input_dim];
//! let mut output = vec![0.0f32; config.output_dim];
//! network.forward_single(&input, &mut output, &mut workspace);
//!
//! // Batch inference (better throughput)
//! let batch_size = 64;
//! let batch_input = vec![0.5f32; batch_size * config.input_dim];
//! let mut batch_output = vec![0.0f32; batch_size * config.output_dim];
//! network.forward_batch(&batch_input, &mut batch_output, &mut workspace);
//! ```
//!
//! ## Training
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork};
//!
//! let config = KanConfig::default_poker();
//! let mut network = KanNetwork::new(config.clone());
//! let mut workspace = network.create_workspace(64);
//!
//! // Generate dummy data
//! let inputs = vec![0.5f32; 64 * config.input_dim];
//! let targets = vec![0.1f32; 64 * config.output_dim];
//!
//! // Single training step (zero-allocation after warmup)
//! let loss = network.train_step(&inputs, &targets, None, 0.001, &mut workspace);
//! println!("Loss: {:.4}", loss);
//! ```
//!
//! ## Architecture
//!
//! ArKan implements the KAN equation:
//!
//! ```text
//! y[j] = Σᵢ Σₖ c[j,i,k] · Bₖ(x[i]) + bias[j]
//! ```
//!
//! where `Bₖ` are B-spline basis functions computed via the Cox-de Boor algorithm.
//!
//! ### Memory Layout
//!
//! - **Weights**: `[Output, Input, Basis]` — row-major for cache efficiency
//! - **Buffers**: 64-byte aligned for AVX-512 compatibility
//! - **Workspace**: Preallocated buffers eliminate hot-path allocations
//!
//! ## Feature Flags
//!
//! | Flag | Description | Default |
//! |------|-------------|---------|
//! | `serde` | Serialization via `serde` + `bincode` | Off |
//! | `quantization` | Half-precision (f16) support | Off |
//! | `parallel` | Rayon parallelization | Off |
//! | `simd` | Explicit SIMD intrinsics | Off |
//!
//! Enable features in `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! arkan = { version = "0.1", features = ["serde"] }
//! ```
//!
//! ## Modules
//!
//! - [`config`] — Network configuration and validation
//! - [`network`] — Main [`KanNetwork`] struct with forward/backward passes
//! - [`layer`] — Individual [`KanLayer`] with B-spline computation
//! - [`buffer`] — [`AlignedBuffer`] and [`Workspace`] for zero-allocation
//! - [`spline`] — SIMD-optimized B-spline basis functions
//! - [`optimizer`] — [`Adam`] and [`SGD`] optimizers
//! - [`loss`] — Loss functions with masking support
//! - [`baked`] — Quantized models for deployment (WIP)
//!
//! ## Performance Tips
//!
//! 1. **Reuse [`Workspace`]**: Create once, use for all forward/backward calls
//! 2. **Use [`KanNetwork::forward_single`]** for real-time play (2x faster than batch=1)
//! 3. **Batch training**: Group samples for better cache utilization
//! 4. **Grid size 5, order 3**: Best speed/accuracy tradeoff for most tasks
//!
//! ## Example: Poker Solver Integration
//!
//! See [`examples/basic.rs`](https://github.com/LutwigStack/ArKan/blob/main/examples/basic.rs)
//! for a complete example.
//!
//! ## License
//!
//! Licensed under either of Apache License, Version 2.0 or MIT license at your option.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![doc(html_root_url = "https://docs.rs/arkan/0.1.0")]

pub mod baked;
pub mod buffer;
pub mod config;
pub mod error;
pub mod layer;
pub mod loss;
pub mod network;
pub mod optimizer;
pub mod spline;

// GPU backend (only available with "gpu" feature)
#[cfg(feature = "gpu")]
pub mod gpu;

// Re-exports for convenience
pub use baked::BakedModel;
pub use buffer::{AlignedBuffer, Workspace, CACHE_LINE};
pub use config::{ConfigError, KanConfig, LayerConfig, DEFAULT_GRID_SIZE, EPSILON};
pub use error::{ArkanError, ArkanResult};
pub use layer::KanLayer;
pub use loss::{masked_cross_entropy, masked_mse, masked_softmax, poker_combined_loss, softmax};
pub use network::{KanNetwork, TrainOptions};
pub use optimizer::{Adam, AdamConfig, AdamState, CosineAnnealingLR, LrScheduler, StepLR, SGD};
pub use spline::{
    compute_basis, compute_basis_and_deriv, compute_knots, find_span, normalize_batch,
};

// GPU re-exports (only available with "gpu" feature)
#[cfg(feature = "gpu")]
pub use gpu::{GpuLayer, GpuTensor, GpuWorkspace, LayerUniforms, WgpuBackend, WgpuOptions};

/// Library version from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic bytes for serialized spline models.
///
/// Used to identify ArKan model files during deserialization.
pub const MAGIC_SPLINE: &[u8; 12] = b"KAN_SPLINE_1";

/// Magic bytes for baked/quantized models.
///
/// Used to identify quantized ArKan model files.
pub const MAGIC_BAKED: &[u8; 12] = b"KAN_BAKED_v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_quick_start_example() {
        // This test ensures the Quick Start example in docs compiles
        let config = KanConfig::default_poker();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input = vec![0.5f32; config.input_dim];
        let mut output = vec![0.0f32; config.output_dim];
        network.forward_single(&input, &mut output, &mut workspace);

        // Output should be non-trivial (network initialized with random weights)
        assert_eq!(output.len(), config.output_dim);
    }
}
