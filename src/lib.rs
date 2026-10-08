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
//! | Single inference, `[21,64,64,24]` | **~15 µs** (`forward_single`) | ~1.5 ms (estimated) |
//! | Memory allocation | Zero (hot path) | Dynamic |
//! | Dependencies | `wide`, `rand`, `thiserror` | Heavy |
//!
//! The PyTorch figure is extrapolated, not measured at this shape, and ArKan
//! *loses* to PyTorch's BLAS kernels at batch 256+. See `docs/BENCHMARKS.md`
//! for the full picture, including what this library is bad at.
//!
//! ## Quick Start
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork};
//!
//! // Create a network with poker-optimized architecture
//! let config = KanConfig::preset(); // [21, 64, 64, 24]
//! let network = KanNetwork::new(config.clone());
//!
//! // Preallocate workspace (reuse across calls for zero-alloc)
//! let mut workspace = network.create_workspace(64);
//!

// Allow manual divisibility checks (x % y == 0) - is_multiple_of is nightly-only
// This lint may not exist in older clippy versions, so we allow unknown lints
#![allow(unknown_lints)]
#![allow(clippy::manual_is_multiple_of)]
//! // Single inference (~15 µs on the preset config)
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
//! let config = KanConfig::preset();
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
//! - **Buffers**: 64-byte (cache-line) aligned
//! - **Workspace**: Preallocated buffers eliminate hot-path allocations
//!
//! ## Feature Flags
//!
//! | Flag | Description | Default |
//! |------|-------------|---------|
//! | `parallel` | Rayon multi-core paths (see below) | Off |
//! | `serde` | Serialization via `serde` + `bincode` | Off |
//! | `gpu` | GPU backend via wgpu (Vulkan/DX12/Metal) | Off |
//!
//! A default build pulls only `wide`, `rand` and `thiserror` — no `rayon`, no
//! `wgpu`. SIMD is **not** a feature flag: B-spline evaluation is vectorized
//! unconditionally through the `wide` crate.
//!
//! `parallel` adds three things and nothing else:
//! `KanLayer::backward_parallel`, `KanNetwork::forward_batch_parallel`, and the
//! automatic parallel branch of the backward pass inside
//! [`KanNetwork::train_step`] for batches at or above
//! [`KanConfig::multithreading_threshold`]. Without it those two methods do not
//! exist and the backward pass is always single-threaded — identical gradients,
//! just one core.
//!
//! Enable features in `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! arkan = { version = "0.4", features = ["serde"] }
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
//! - [`baked`] — [`BakedModel`]: int8 quantized inference path (per-channel
//!   weights, int16 basis). Smaller than f32, **not** faster; see
//!   `docs/BENCHMARKS.md` for the latency and the per-output tail before using it
//!
//! ## Performance Tips
//!
//! 1. **Reuse [`Workspace`]**: Create once, use for all forward/backward calls
//! 2. **Use [`KanNetwork::forward_single`]** for real-time play (~1.8x faster than `forward_batch(1)`)
//! 3. **Batch training**: Group samples for better cache utilization
//! 4. **Grid size 5, order 3**: Best speed/accuracy tradeoff for most tasks
//! 5. **Size `grid_range` for the activations, not the inputs**: it is shared by
//!    every layer, but only layer 0 receives `input_mean`/`input_std`. A range
//!    picked from the input distribution can saturate ~45% of every hidden layer,
//!    which zeroes their gradients. See `docs/ARCHITECTURE.md`.
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
#![doc(html_root_url = "https://docs.rs/arkan/0.4.0")]

pub mod baked;
pub mod buffer;
pub mod config;
pub mod error;
pub mod layer;
pub mod loss;
pub mod model;
pub mod network;
pub mod optimizer;
pub mod spline;
pub mod training;

#[cfg(feature = "serde")]
mod format;

// GPU backend (only available with "gpu" feature)
#[cfg(feature = "gpu")]
pub mod gpu;

// Re-exports for convenience
pub use baked::BakedModel;
pub use buffer::{
    checked_buffer_size, checked_buffer_size3, AlignedBuffer, Tensor, TensorView, Workspace,
    WorkspaceGuard, CACHE_LINE, MAX_BUFFER_ELEMENTS,
};
pub use config::{
    ConfigError, KanConfig, KanConfigBuilder, LayerConfig, DEFAULT_GRID_SIZE, EPSILON,
    MAX_GPU_SPLINE_ORDER, MAX_GRID_SIZE, MAX_SPLINE_ORDER, MIN_GPU_SPLINE_ORDER,
};
pub use error::{ArkanError, ArkanResult};
pub use layer::KanLayer;
pub use loss::{
    entropy_regularization, kan_combined_loss, kan_regularization_gradient, l1_sparsity_gradient,
    l1_sparsity_loss, masked_bce_with_logits, masked_categorical_cross_entropy,
    masked_cross_entropy, masked_huber, masked_mae, masked_mse, masked_rmse, masked_softmax,
    pde_residual_loss, poker_combined_loss, r_squared, smoothness_gradient, smoothness_penalty,
    softmax, KanLossConfig,
};
pub use network::{KanNetwork, TrainOptions};
pub use optimizer::{
    Adam, AdamConfig, AdamState, CosineAnnealingLR, LBFGSConfig, LineSearchMethod, LrScheduler,
    Optimizer, ParamGroup, SGDConfig, SafetyConfig, StepLR, LBFGS, SGD,
};
pub use spline::{
    compute_basis, compute_basis_and_deriv, compute_knots, find_span, normalize_batch,
    SPAN_CLAMPED_FLAG, SPAN_INDEX_MASK,
};

// GPU re-exports (only available with "gpu" feature)
#[cfg(feature = "gpu")]
pub use gpu::{
    GpuLayer, GpuNetwork, GpuTensor, GpuTensorView, GpuWorkspace, LayerUniforms, PipelineCache,
    PowerPreference, WgpuBackend, WgpuOptions,
};

/// Library version from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic bytes for baked/quantized models.
///
/// Prepended by `BakedModel::to_bytes` to identify the file type, and checked
/// by `BakedModel::from_bytes` before any deserialization. Both methods require
/// the `serde` feature, so these are plain names rather than links — an intra-doc
/// link would break `cargo doc` on the default build.
pub const MAGIC_BAKED: &[u8; 12] = b"KAN_BAKED_v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        // VERSION is a static string, so we just verify it exists and is not empty
        let v = VERSION;
        assert!(!v.is_empty());
    }

    #[test]
    fn test_quick_start_example() {
        // This test ensures the Quick Start example in docs compiles
        let config = KanConfig::preset();
        let network = KanNetwork::new(config.clone());
        let mut workspace = network.create_workspace(1);

        let input = vec![0.5f32; config.input_dim];
        let mut output = vec![0.0f32; config.output_dim];
        network.forward_single(&input, &mut output, &mut workspace);

        // Output should be non-trivial (network initialized with random weights)
        assert_eq!(output.len(), config.output_dim);
    }
}

/// Compiles every ```rust fence in README.md as a doctest.
///
/// Without this the README is checked by nothing. `tests/readme_snippets.rs` is a
/// hand-typed copy, so it can only catch rot in the copy, never divergence between the
/// copy and the README itself — a reviewer appended a fence calling a fictional API and
/// `clippy --all-targets --all-features`, `test`, `test --doc`, `doc` and `package` all
/// exited 0. README.md ships inside the published package (`readme = "README.md"`), so
/// including it here is package-safe.
///
/// Fences that cannot run unaided are annotated in the README itself (`no_run` for GPU
/// paths, `text` for shell blocks) rather than being exempted here.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
