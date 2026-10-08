//! KAN network with multi-layer support and zero-allocation inference.
//!
//! This module provides [`KanNetwork`], the main entry point for using ArKan.
//! It supports both inference and training with configurable options.
//!
//! # Example: Inference
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork};
//!
//! let config = KanConfig::preset();
//! let network = KanNetwork::new(config.clone());
//! let mut workspace = network.create_workspace(1);
//!
//! let input = vec![0.5f32; config.input_dim];
//! let mut output = vec![0.0f32; config.output_dim];
//!
//! // ~30 µs latency, zero allocations
//! network.forward_single(&input, &mut output, &mut workspace);
//! ```
//!
//! # Example: Training
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork, TrainOptions};
//!
//! let config = KanConfig::preset();
//! let mut network = KanNetwork::new(config.clone());
//! let mut workspace = network.create_workspace(64);
//!
//! let inputs = vec![0.5f32; 64 * config.input_dim];
//! let targets = vec![0.1f32; 64 * config.output_dim];
//!
//! // Training with gradient clipping
//! let opts = TrainOptions {
//!     max_grad_norm: Some(1.0),
//!     weight_decay: 0.01,
//! };
//! network.set_default_train_options(opts);
//!
//! let loss = network.train_step(&inputs, &targets, None, 0.001, &mut workspace);
//! ```
//!
//! # Performance
//!
//! | Method | Batch Size | Time | Use Case |
//! |--------|-----------|------|----------|
//! | `forward_single` | 1 | ~15 µs | Real-time play |
//! | `forward_batch` | 1 | ~30 µs | General inference |
//! | `forward_batch` | 64 | ~2 ms | Batch inference |
//! | `train_step` | 64 | ~5 ms | Training |

pub use crate::model::KanNetwork;
pub use crate::training::TrainOptions;
