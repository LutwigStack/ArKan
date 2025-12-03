//! Configuration for KAN network.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Minimum value for standard deviation to avoid division by zero.
pub const EPSILON: f32 = 1e-6;

/// Default grid size (number of intervals).
pub const DEFAULT_GRID_SIZE: usize = 5;

/// Default spline order (cubic).
pub const DEFAULT_SPLINE_ORDER: usize = 3;

/// Number of basis functions = grid_size + spline_order.
#[inline]
pub const fn basis_size(grid_size: usize, spline_order: usize) -> usize {
    grid_size + spline_order
}

/// KAN network configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KanConfig {
    /// Input dimension.
    pub input_dim: usize,

    /// Output dimension.
    pub output_dim: usize,

    /// Hidden layer dimensions (empty = no hidden layers).
    pub hidden_dims: Vec<usize>,

    /// Grid size (number of intervals). Max 16 for SIMD index optimization.
    pub grid_size: usize,

    /// Spline order (3 = cubic, recommended).
    pub spline_order: usize,

    /// Grid range (min, max) for normalized inputs.
    pub grid_range: (f32, f32),

    /// Input feature means for normalization.
    pub input_mean: Vec<f32>,

    /// Input feature standard deviations for normalization.
    pub input_std: Vec<f32>,

    /// Batch size threshold for multithreading.
    /// Batches smaller than this are processed single-threaded.
    pub multithreading_threshold: usize,

    /// SIMD vector width for basis alignment (8 for AVX2, 16 for AVX-512).
    pub simd_width: usize,

    /// Optional seed for deterministic initialization (None => random).
    pub init_seed: Option<u64>,
}

impl Default for KanConfig {
    fn default() -> Self {
        Self {
            input_dim: 21,
            output_dim: 16, // 8 probs + 8 q-values
            hidden_dims: vec![10],
            grid_size: DEFAULT_GRID_SIZE,
            spline_order: DEFAULT_SPLINE_ORDER,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.0; 21],
            input_std: vec![1.0; 21],
            multithreading_threshold: 128,
            simd_width: 8, // AVX2
            init_seed: None,
        }
    }
}

impl KanConfig {
    /// Creates a configuration optimized for poker solver.
    ///
    /// - Input: 21 features (from nn crate)
    /// - Output: 24 (8 strategy probs + 8 Q-values + 8 mask)
    /// - Hidden: [64, 64]
    pub fn default_poker() -> Self {
        Self {
            input_dim: 21,
            output_dim: 24,
            hidden_dims: vec![64, 64],
            grid_size: 5,
            spline_order: 3,
            grid_range: (-3.0, 3.0),
            input_mean: vec![0.5; 21], // Will be computed from data
            input_std: vec![0.3; 21],  // Will be computed from data
            multithreading_threshold: 128,
            simd_width: 8,
            init_seed: None,
        }
    }

    /// Number of basis functions per connection.
    #[inline]
    pub fn basis_size(&self) -> usize {
        basis_size(self.grid_size, self.spline_order)
    }

    /// Basis size padded to SIMD width.
    #[inline]
    pub fn basis_size_aligned(&self) -> usize {
        let bs = self.basis_size();
        bs.div_ceil(self.simd_width) * self.simd_width
    }

    /// Total number of layers (hidden + output).
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.hidden_dims.len() + 1
    }

    /// Layer dimensions: [input, hidden..., output].
    pub fn layer_dims(&self) -> Vec<usize> {
        let mut dims = Vec::with_capacity(self.num_layers() + 1);
        dims.push(self.input_dim);
        dims.extend_from_slice(&self.hidden_dims);
        dims.push(self.output_dim);
        dims
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.input_dim == 0 {
            return Err(ConfigError::InvalidDimension("input_dim must be > 0"));
        }
        if self.output_dim == 0 {
            return Err(ConfigError::InvalidDimension("output_dim must be > 0"));
        }
        if self.grid_size == 0 || self.grid_size > 16 {
            return Err(ConfigError::InvalidGridSize(self.grid_size));
        }
        if self.spline_order == 0 || self.spline_order > 5 {
            return Err(ConfigError::InvalidSplineOrder(self.spline_order));
        }
        if self.grid_range.0 >= self.grid_range.1 {
            return Err(ConfigError::InvalidGridRange);
        }
        if self.input_mean.len() != self.input_dim {
            return Err(ConfigError::MismatchedNormalization("input_mean"));
        }
        if self.input_std.len() != self.input_dim {
            return Err(ConfigError::MismatchedNormalization("input_std"));
        }
        // init_seed: any value is acceptable, None => random
        // SIMD width must be a power of 2 (4, 8, 16)
        if !matches!(self.simd_width, 4 | 8 | 16) {
            return Err(ConfigError::InvalidSimdWidth(self.simd_width));
        }
        // Ensure order+1 <= global_basis_size (always true, but verify)
        let global_basis = self.basis_size();
        if self.spline_order + 1 > global_basis {
            return Err(ConfigError::InvalidDimension(
                "spline_order + 1 must be <= grid_size + spline_order",
            ));
        }
        Ok(())
    }

    /// Updates normalization parameters from data statistics.
    pub fn set_normalization(&mut self, mean: Vec<f32>, std: Vec<f32>) {
        debug_assert_eq!(mean.len(), self.input_dim);
        debug_assert_eq!(std.len(), self.input_dim);
        self.input_mean = mean;
        self.input_std = std.into_iter().map(|s| s.max(EPSILON)).collect();
    }
}

/// Per-layer configuration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerConfig {
    pub in_dim: usize,
    pub out_dim: usize,
    pub grid_size: usize,
    pub spline_order: usize,
    pub grid_range: (f32, f32),
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            in_dim: 21,
            out_dim: 64,
            grid_size: DEFAULT_GRID_SIZE,
            spline_order: 3,
            grid_range: (-3.0, 3.0),
        }
    }
}

/// Configuration errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid dimension: {0}")]
    InvalidDimension(&'static str),

    #[error("Grid size must be 1-16, got {0}")]
    InvalidGridSize(usize),

    #[error("Spline order must be 1-5, got {0}")]
    InvalidSplineOrder(usize),

    #[error("Invalid grid range")]
    InvalidGridRange,

    #[error("Mismatched normalization array: {0}")]
    MismatchedNormalization(&'static str),

    #[error("SIMD width must be 4, 8, or 16, got {0}")]
    InvalidSimdWidth(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KanConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_poker_config() {
        let config = KanConfig::default_poker();
        assert!(config.validate().is_ok());
        assert_eq!(config.input_dim, 21);
        assert_eq!(config.output_dim, 24);
    }

    #[test]
    fn test_basis_size() {
        let config = KanConfig::default();
        // G=5, k=3 → basis = 8
        assert_eq!(config.basis_size(), 8);
        // Already aligned to 8
        assert_eq!(config.basis_size_aligned(), 8);
    }

    #[test]
    fn test_layer_dims() {
        let config = KanConfig::default_poker();
        let dims = config.layer_dims();
        assert_eq!(dims, vec![21, 64, 64, 24]);
    }

    #[test]
    fn test_invalid_grid_size() {
        let mut config = KanConfig::default();
        config.grid_size = 20;
        assert!(config.validate().is_err());
    }
}
