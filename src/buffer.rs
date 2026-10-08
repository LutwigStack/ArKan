//! Aligned buffers and workspace for zero-allocation inference.
//!
//! This module provides two key types:
//!
//! - [`AlignedBuffer`] — 64-byte aligned buffer for SIMD operations
//! - [`Workspace`] — Preallocated buffers for forward/backward passes
//! - [`WorkspaceGuard`] — RAII guard for exception-safe buffer management
//!
//! # Zero-Allocation Pattern
//!
//! ArKan achieves zero-allocation inference by preallocating all buffers
//! in the [`Workspace`]. Create it once and reuse:
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork, Workspace};
//!
//! let config = KanConfig::preset();
//! let network = KanNetwork::new(config.clone());
//!
//! // Allocate workspace once for max batch size
//! let mut workspace = network.create_workspace(64);
//!
//! // All subsequent calls are zero-allocation
//! let input = vec![0.5f32; config.input_dim];
//! let mut output = vec![0.0f32; config.output_dim];
//!
//! for _ in 0..1000 {
//!     network.forward_single(&input, &mut output, &mut workspace);
//! }
//! ```
//!
//! # Memory Alignment
//!
//! [`AlignedBuffer`] uses 64-byte alignment ([`CACHE_LINE`]) to ensure
//! optimal performance with AVX-512 SIMD instructions.
//!
//! # Exception Safety
//!
//! The [`WorkspaceGuard`] provides basic exception safety guarantee:
//! buffers are returned to workspace even if a panic occurs during
//! forward/backward passes.

pub use crate::cpu::{Workspace, WorkspaceGuard};
pub use crate::memory::*;
