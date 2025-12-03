#![forbid(unsafe_op_in_unsafe_fn)]

//! # ArKan - High-Performance Kolmogorov-Arnold Network
//!
//! Zero-allocation inference, SIMD-optimized B-spline evaluation,
//! and optional quantized (f16) baked models for production.
//!
//! ## Architecture
//! - Row-Major weight layout: `[Output, Input, Basis]`
//! - Aligned buffers (64-byte) for AVX-512
//! - Preallocated workspace to eliminate hot-path allocations
//!
//! ## Usage
//! ```rust,ignore
//! use arkan::{KanConfig, KanNetwork, Workspace};
//!
//! let config = KanConfig::default_poker();
//! let network = KanNetwork::new(config);
//! let mut workspace = Workspace::new(&network.config);
//!
//! // Zero-allocation forward pass
//! network.forward_batch(&inputs, &mut outputs, &mut workspace);
//! ```

pub mod baked;
pub mod buffer;
pub mod config;
pub mod layer;
pub mod loss;
pub mod network;
pub mod optimizer;
pub mod spline;

// Re-exports
pub use baked::BakedModel;
pub use buffer::{AlignedBuffer, Workspace, CACHE_LINE};
pub use config::{KanConfig, LayerConfig, DEFAULT_GRID_SIZE, EPSILON};
pub use layer::KanLayer;
pub use loss::{masked_cross_entropy, masked_mse, masked_softmax, poker_combined_loss, softmax};
pub use network::KanNetwork;
pub use optimizer::{Adam, AdamConfig, AdamState, CosineAnnealingLR, LrScheduler, StepLR, SGD};
pub use spline::{
    compute_basis, compute_basis_and_deriv, compute_knots, find_span, normalize_batch,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic bytes for serialized spline models
pub const MAGIC_SPLINE: &[u8; 12] = b"KAN_SPLINE_1";

/// Magic bytes for baked/quantized models
pub const MAGIC_BAKED: &[u8; 12] = b"KAN_BAKED_v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
