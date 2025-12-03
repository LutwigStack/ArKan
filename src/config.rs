//! Network configuration and hyperparameters.
//!
//! This module provides [`KanConfig`] for configuring KAN networks,
//! including architecture, spline parameters, and normalization.
//!
//! # Example
//!
//! ```rust
//! use arkan::KanConfig;
//!
//! // Use preset for poker solver
//! let config = KanConfig::default_poker();
//! assert_eq!(config.layer_dims(), vec![21, 64, 64, 24]);
//!
//! // Or customize
//! let config = KanConfig {
//!     input_dim: 10,
//!     output_dim: 5,
//!     hidden_dims: vec![32, 32],
//!     grid_size: 8,
//!     spline_order: 3,
//!     ..Default::default()
//! };
//! ```
//!
//! # Spline Parameters
//!
//! The spline configuration determines expressiveness and speed:
//!
//! | Parameter | Typical Values | Effect |
//! |-----------|---------------|--------|
//! | `grid_size` | 3-16 | More intervals = finer control, more params |
//! | `spline_order` | 2-5 | Higher = smoother functions, more compute |
//!
//! **Recommended**: `grid_size=5`, `spline_order=3` (cubic) for balanced performance.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Minimum value for standard deviation to avoid division by zero.
///
/// Used in normalization calculations to prevent numerical instability.
pub const EPSILON: f32 = 1e-6;

/// Default grid size (number of intervals).
///
/// A grid size of 5 provides a good balance between expressiveness
/// and computational cost for most applications.
pub const DEFAULT_GRID_SIZE: usize = 5;

/// Default spline order (cubic).
///
/// Cubic splines (order 3) offer smooth second derivatives,
/// making them ideal for function approximation.
pub const DEFAULT_SPLINE_ORDER: usize = 3;

/// Computes the number of basis functions for given spline parameters.
///
/// For a B-spline with `grid_size` intervals and polynomial `order`,
/// the number of basis functions is `grid_size + order`.
///
/// # Example
///
/// ```rust
/// use arkan::config::basis_size;
///
/// // Cubic spline with 5 intervals: 5 + 3 = 8 basis functions
/// assert_eq!(basis_size(5, 3), 8);
/// ```
#[inline]
pub const fn basis_size(grid_size: usize, spline_order: usize) -> usize {
    grid_size + spline_order
}

/// KAN network configuration.
///
/// This struct defines all hyperparameters for a Kolmogorov-Arnold Network,
/// including architecture, spline parameters, and input normalization.
///
/// # Creating a Configuration
///
/// ```rust
/// use arkan::KanConfig;
///
/// // Preset for poker solver (recommended starting point)
/// let config = KanConfig::default_poker();
///
/// // Custom configuration
/// let config = KanConfig {
///     input_dim: 10,
///     output_dim: 5,
///     hidden_dims: vec![64, 64],
///     grid_size: 5,
///     spline_order: 3,
///     input_mean: vec![0.0; 10], // Must match input_dim
///     input_std: vec![1.0; 10],  // Must match input_dim
///     ..Default::default()
/// };
///
/// // Always validate before use
/// config.validate().expect("Invalid configuration");
/// ```
///
/// # Architecture
///
/// The network has layers: `input_dim → hidden_dims[0] → ... → output_dim`.
/// Use [`layer_dims`](Self::layer_dims) to get the full dimension list.
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
    /// Architecture: `[21, 64, 64, 24]` with cubic splines (order 3, grid 5).
    ///
    /// - **Input**: 21 features (game state encoding)
    /// - **Output**: 24 values (8 strategy probs + 8 Q-values + 8 mask)
    /// - **Hidden**: Two layers of 64 neurons each
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::KanConfig;
    ///
    /// let config = KanConfig::default_poker();
    /// assert_eq!(config.layer_dims(), vec![21, 64, 64, 24]);
    /// ```
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

    /// Returns the number of basis functions per connection.
    ///
    /// This equals `grid_size + spline_order`. For the default poker config
    /// (grid=5, order=3), this returns 8.
    #[inline]
    pub fn basis_size(&self) -> usize {
        basis_size(self.grid_size, self.spline_order)
    }

    /// Returns basis size padded to SIMD width for aligned memory access.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::KanConfig;
    ///
    /// let config = KanConfig::default_poker();
    /// // basis_size=8 is already aligned to simd_width=8
    /// assert_eq!(config.basis_size_aligned(), 8);
    /// ```
    #[inline]
    pub fn basis_size_aligned(&self) -> usize {
        let bs = self.basis_size();
        bs.div_ceil(self.simd_width) * self.simd_width
    }

    /// Returns total number of layers (hidden + output).
    ///
    /// Note: This does NOT include the input "layer" (which is just data).
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.hidden_dims.len() + 1
    }

    /// Returns all layer dimensions: `[input, hidden..., output]`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::KanConfig;
    ///
    /// let config = KanConfig::default_poker();
    /// assert_eq!(config.layer_dims(), vec![21, 64, 64, 24]);
    /// ```
    pub fn layer_dims(&self) -> Vec<usize> {
        let mut dims = Vec::with_capacity(self.num_layers() + 1);
        dims.push(self.input_dim);
        dims.extend_from_slice(&self.hidden_dims);
        dims.push(self.output_dim);
        dims
    }

    /// Validates the configuration and returns any errors.
    ///
    /// Should be called before creating a [`KanNetwork`](crate::KanNetwork).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - Dimensions are zero
    /// - `grid_size` not in 1..=16
    /// - `spline_order` not in 1..=5
    /// - `grid_range.0 >= grid_range.1`
    /// - Normalization arrays don't match `input_dim`
    /// - `simd_width` is not 4, 8, or 16
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::KanConfig;
    ///
    /// let config = KanConfig::default_poker();
    /// assert!(config.validate().is_ok());
    ///
    /// let bad_config = KanConfig {
    ///     grid_size: 100, // Invalid!
    ///     ..Default::default()
    /// };
    /// assert!(bad_config.validate().is_err());
    /// ```
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
    ///
    /// # Arguments
    ///
    /// * `mean` - Per-feature mean values, length must equal `input_dim`
    /// * `std` - Per-feature standard deviations, clamped to [`EPSILON`] minimum
    ///
    /// # Panics
    ///
    /// Debug-asserts if `mean.len() != input_dim` or `std.len() != input_dim`.
    pub fn set_normalization(&mut self, mean: Vec<f32>, std: Vec<f32>) {
        debug_assert_eq!(mean.len(), self.input_dim);
        debug_assert_eq!(std.len(), self.input_dim);
        self.input_mean = mean;
        self.input_std = std.into_iter().map(|s| s.max(EPSILON)).collect();
    }
}

/// Per-layer configuration (for advanced use cases).
///
/// Most users should use [`KanConfig`] instead. This struct is useful
/// for fine-grained control over individual layers.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerConfig {
    /// Input dimension for this layer.
    pub in_dim: usize,
    /// Output dimension for this layer.
    pub out_dim: usize,
    /// Number of grid intervals for B-splines.
    pub grid_size: usize,
    /// B-spline polynomial order (3 = cubic).
    pub spline_order: usize,
    /// Input normalization range `(min, max)`.
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

/// Errors returned by [`KanConfig::validate`].
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// A dimension parameter is invalid (zero or mismatched).
    #[error("Invalid dimension: {0}")]
    InvalidDimension(&'static str),

    /// Grid size is out of valid range (1-16).
    #[error("Grid size must be 1-16, got {0}")]
    InvalidGridSize(usize),

    /// Spline order is out of valid range (1-5).
    #[error("Spline order must be 1-5, got {0}")]
    InvalidSplineOrder(usize),

    /// Grid range is invalid (min >= max).
    #[error("Invalid grid range")]
    InvalidGridRange,

    /// Normalization arrays don't match input dimension.
    #[error("Mismatched normalization array: {0}")]
    MismatchedNormalization(&'static str),

    /// SIMD width is not a valid value (4, 8, or 16).
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
