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

pub mod buffer;
pub mod config;
pub mod layer;
pub mod loss;
pub mod network;
pub mod optimizer;
pub mod spline;
pub mod baked;

// Re-exports
pub use buffer::{AlignedBuffer, Workspace, CACHE_LINE};
pub use config::{KanConfig, LayerConfig, EPSILON, DEFAULT_GRID_SIZE};
pub use layer::KanLayer;
pub use loss::{masked_mse, masked_cross_entropy, poker_combined_loss, softmax, masked_softmax};
pub use network::KanNetwork;
pub use optimizer::{Adam, AdamConfig, AdamState, SGD, LrScheduler, StepLR, CosineAnnealingLR};
pub use spline::{compute_knots, find_span, compute_basis, compute_basis_and_deriv, normalize_batch};
pub use baked::BakedModel;

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
