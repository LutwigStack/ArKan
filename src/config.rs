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
//! let config = KanConfig::preset();
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
//! | `grid_size` | 3-64 | More intervals = finer control, more params |
//! | `spline_order` | 2-5 | Higher = smoother functions, more compute |
//!
//! **Recommended**: `grid_size=5`, `spline_order=3` (cubic) for balanced performance.
//!
//! # Constants
//!
//! - [`MAX_SPLINE_ORDER`]: Maximum supported spline order (7 for CPU, 5 for GPU)
//! - [`MAX_GPU_SPLINE_ORDER`]: Maximum spline order supported by GPU shaders

use std::borrow::Cow;

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

/// Maximum supported grid size.
///
/// Higher grid sizes allow finer control of the spline functions,
/// but increase memory usage and computation time quadratically.
/// For most applications, grid_size <= 16 is sufficient.
pub const MAX_GRID_SIZE: usize = 64;

/// Default spline order (cubic).
///
/// Cubic splines (order 3) offer smooth second derivatives,
/// making them ideal for function approximation.
pub const DEFAULT_SPLINE_ORDER: usize = 3;

/// Maximum supported spline order for CPU computation.
///
/// This limit exists because the Cox-de Boor algorithm uses stack-allocated
/// arrays of size `MAX_SPLINE_ORDER + 1`. Higher orders would require
/// larger stack allocations.
///
/// # Note
///
/// GPU shaders support a smaller range (2-5). See [`MAX_GPU_SPLINE_ORDER`].
pub const MAX_SPLINE_ORDER: usize = 7;

/// Maximum spline order supported by GPU shaders.
///
/// GPU shaders use optimized, order-specific basis functions.
/// Currently supported: orders 2, 3, 4, 5.
pub const MAX_GPU_SPLINE_ORDER: usize = 5;

/// Minimum spline order supported by GPU shaders.
pub const MIN_GPU_SPLINE_ORDER: usize = 2;

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
#[must_use]
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
/// let config = KanConfig::preset();
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
#[must_use]
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
    /// let config = KanConfig::preset();
    /// assert_eq!(config.layer_dims(), vec![21, 64, 64, 24]);
    /// ```
    #[must_use = "this creates a new config without modifying anything"]
    pub fn preset() -> Self {
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

    /// Creates a new builder for constructing a `KanConfig`.
    ///
    /// The builder pattern provides a fluent API for configuration
    /// with validation on build.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::KanConfig;
    ///
    /// let config = KanConfig::builder()
    ///     .input_dim(10)
    ///     .output_dim(5)
    ///     .hidden_dims(vec![32, 32])
    ///     .grid_size(8)
    ///     .spline_order(3)
    ///     .build()
    ///     .expect("valid config");
    ///
    /// assert_eq!(config.layer_dims(), vec![10, 32, 32, 5]);
    /// ```
    #[must_use]
    pub fn builder() -> KanConfigBuilder {
        KanConfigBuilder::new()
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
    /// let config = KanConfig::preset();
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
    /// let config = KanConfig::preset();
    /// assert_eq!(config.layer_dims(), vec![21, 64, 64, 24]);
    /// ```
    #[must_use]
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
    /// - `grid_size` not in 1..=[`MAX_GRID_SIZE`] (64)
    /// - `spline_order` not in 1..=[`MAX_SPLINE_ORDER`]
    /// - `grid_range.0 >= grid_range.1`
    /// - Normalization arrays don't match `input_dim`
    /// - `simd_width` is not 4, 8, or 16
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::KanConfig;
    ///
    /// let config = KanConfig::preset();
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
            return Err(ConfigError::InvalidDimension(Cow::Borrowed(
                "input_dim must be > 0",
            )));
        }
        if self.output_dim == 0 {
            return Err(ConfigError::InvalidDimension(Cow::Borrowed(
                "output_dim must be > 0",
            )));
        }
        if self.grid_size == 0 || self.grid_size > MAX_GRID_SIZE {
            return Err(ConfigError::InvalidGridSize(self.grid_size));
        }
        if self.spline_order == 0 || self.spline_order > MAX_SPLINE_ORDER {
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
        // Warn about zero/negative values in input_std (will be clamped to EPSILON)
        if self.input_std.iter().any(|&s| s <= 0.0) {
            #[cfg(feature = "gpu")]
            log::warn!(
                "input_std contains zero or negative values; will be clamped to EPSILON ({})",
                EPSILON
            );
            #[cfg(not(feature = "gpu"))]
            eprintln!(
                "Warning: input_std contains zero or negative values; will be clamped to EPSILON ({})",
                EPSILON
            );
        }
        // init_seed: any value is acceptable, None => random
        // SIMD width must be a power of 2 (4, 8, 16)
        if !matches!(self.simd_width, 4 | 8 | 16) {
            return Err(ConfigError::InvalidSimdWidth(self.simd_width));
        }
        // Ensure order+1 <= global_basis_size (always true, but verify)
        let global_basis = self.basis_size();
        if self.spline_order + 1 > global_basis {
            return Err(ConfigError::InvalidDimension(Cow::Borrowed(
                "spline_order + 1 must be <= grid_size + spline_order",
            )));
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
    InvalidDimension(Cow<'static, str>),

    /// Grid size is out of valid range (1-16).
    #[error("Grid size must be 1-16, got {0}")]
    InvalidGridSize(usize),

    /// Spline order is out of valid range (1-7 for CPU, 2-5 for GPU).
    #[error("Spline order must be 1-7 for CPU, 2-5 for GPU, got {0}")]
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

/// Builder for creating [`KanConfig`] with a fluent API.
///
/// # Example
///
/// ```rust
/// use arkan::KanConfig;
///
/// let config = KanConfig::builder()
///     .input_dim(10)
///     .output_dim(5)
///     .hidden_dims(vec![32, 32])
///     .build()
///     .expect("valid config");
///
/// assert_eq!(config.input_dim, 10);
/// assert_eq!(config.output_dim, 5);
/// ```
#[derive(Debug, Clone)]
pub struct KanConfigBuilder {
    input_dim: Option<usize>,
    output_dim: Option<usize>,
    hidden_dims: Vec<usize>,
    grid_size: usize,
    spline_order: usize,
    grid_range: (f32, f32),
    input_mean: Option<Vec<f32>>,
    input_std: Option<Vec<f32>>,
    multithreading_threshold: usize,
    simd_width: usize,
    init_seed: Option<u64>,
}

impl Default for KanConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl KanConfigBuilder {
    /// Creates a new builder with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            input_dim: None,
            output_dim: None,
            hidden_dims: vec![],
            grid_size: DEFAULT_GRID_SIZE,
            spline_order: DEFAULT_SPLINE_ORDER,
            grid_range: (-3.0, 3.0),
            input_mean: None,
            input_std: None,
            multithreading_threshold: 128,
            simd_width: 8,
            init_seed: None,
        }
    }

    /// Sets the input dimension (required).
    #[must_use]
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = Some(dim);
        self
    }

    /// Sets the output dimension (required).
    #[must_use]
    pub fn output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }

    /// Sets the hidden layer dimensions.
    ///
    /// Empty vector means no hidden layers (direct input→output).
    #[must_use]
    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Sets the grid size (number of spline intervals).
    ///
    /// Default: 5. Valid range: 1-16.
    #[must_use]
    pub fn grid_size(mut self, size: usize) -> Self {
        self.grid_size = size;
        self
    }

    /// Sets the spline order.
    ///
    /// Default: 3 (cubic). Valid range: 1-7 (CPU), 2-5 (GPU).
    #[must_use]
    pub fn spline_order(mut self, order: usize) -> Self {
        self.spline_order = order;
        self
    }

    /// Sets the grid range for input normalization.
    ///
    /// Default: (-3.0, 3.0).
    #[must_use]
    pub fn grid_range(mut self, min: f32, max: f32) -> Self {
        self.grid_range = (min, max);
        self
    }

    /// Sets the input normalization parameters.
    ///
    /// If not set, defaults to mean=0, std=1 for all inputs.
    #[must_use]
    pub fn normalization(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.input_mean = Some(mean);
        self.input_std = Some(std);
        self
    }

    /// Sets the multithreading threshold.
    ///
    /// Batches smaller than this are processed single-threaded.
    /// Default: 128.
    #[must_use]
    pub fn multithreading_threshold(mut self, threshold: usize) -> Self {
        self.multithreading_threshold = threshold;
        self
    }

    /// Sets the SIMD width for memory alignment.
    ///
    /// Default: 8 (AVX2). Valid values: 4, 8, 16.
    #[must_use]
    pub fn simd_width(mut self, width: usize) -> Self {
        self.simd_width = width;
        self
    }

    /// Sets the initialization seed for deterministic weights.
    ///
    /// Default: None (random initialization).
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.init_seed = Some(seed);
        self
    }

    /// Builds the configuration, validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if:
    /// - `input_dim` or `output_dim` not set
    /// - Any validation check fails (see [`KanConfig::validate`])
    pub fn build(self) -> Result<KanConfig, ConfigError> {
        let input_dim = self
            .input_dim
            .ok_or(ConfigError::InvalidDimension(Cow::Borrowed(
                "input_dim not set",
            )))?;
        let output_dim = self
            .output_dim
            .ok_or(ConfigError::InvalidDimension(Cow::Borrowed(
                "output_dim not set",
            )))?;

        let input_mean = self.input_mean.unwrap_or_else(|| vec![0.0; input_dim]);
        let input_std = self
            .input_std
            .map(|s| s.into_iter().map(|v| v.max(EPSILON)).collect())
            .unwrap_or_else(|| vec![1.0; input_dim]);

        let config = KanConfig {
            input_dim,
            output_dim,
            hidden_dims: self.hidden_dims,
            grid_size: self.grid_size,
            spline_order: self.spline_order,
            grid_range: self.grid_range,
            input_mean,
            input_std,
            multithreading_threshold: self.multithreading_threshold,
            simd_width: self.simd_width,
            init_seed: self.init_seed,
        };

        config.validate()?;
        Ok(config)
    }
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
        let config = KanConfig::preset();
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
        let config = KanConfig::preset();
        let dims = config.layer_dims();
        assert_eq!(dims, vec![21, 64, 64, 24]);
    }

    #[test]
    fn test_invalid_grid_size() {
        // grid_size = 65 exceeds MAX_GRID_SIZE (64)
        let config = KanConfig {
            grid_size: 65,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_valid_large_grid_size() {
        // grid_size = 64 is now valid (MAX_GRID_SIZE)
        let config = KanConfig {
            grid_size: 64,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // grid_size = 32 also valid
        let config32 = KanConfig {
            grid_size: 32,
            ..Default::default()
        };
        assert!(config32.validate().is_ok());
    }

    #[test]
    fn test_builder_basic() {
        let config = KanConfig::builder()
            .input_dim(10)
            .output_dim(5)
            .hidden_dims(vec![32, 32])
            .build()
            .expect("valid config");

        assert_eq!(config.input_dim, 10);
        assert_eq!(config.output_dim, 5);
        assert_eq!(config.hidden_dims, vec![32, 32]);
        assert_eq!(config.layer_dims(), vec![10, 32, 32, 5]);
    }

    #[test]
    fn test_builder_all_options() {
        let config = KanConfig::builder()
            .input_dim(4)
            .output_dim(2)
            .hidden_dims(vec![8])
            .grid_size(8)
            .spline_order(2)
            .grid_range(-1.0, 1.0)
            .normalization(vec![0.5; 4], vec![0.2; 4])
            .multithreading_threshold(64)
            .simd_width(4)
            .seed(42)
            .build()
            .expect("valid config");

        assert_eq!(config.grid_size, 8);
        assert_eq!(config.spline_order, 2);
        assert_eq!(config.grid_range, (-1.0, 1.0));
        assert_eq!(config.input_mean, vec![0.5; 4]);
        assert_eq!(config.multithreading_threshold, 64);
        assert_eq!(config.simd_width, 4);
        assert_eq!(config.init_seed, Some(42));
    }

    #[test]
    fn test_builder_missing_input_dim() {
        let result = KanConfig::builder().output_dim(5).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_output_dim() {
        let result = KanConfig::builder().input_dim(10).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_grid_size() {
        let result = KanConfig::builder()
            .input_dim(10)
            .output_dim(5)
            .grid_size(100) // Invalid
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_no_hidden_layers() {
        let config = KanConfig::builder()
            .input_dim(10)
            .output_dim(5)
            .build()
            .expect("valid config");

        assert!(config.hidden_dims.is_empty());
        assert_eq!(config.layer_dims(), vec![10, 5]);
    }

    #[test]
    fn test_builder_default_normalization() {
        let config = KanConfig::builder()
            .input_dim(4)
            .output_dim(2)
            .build()
            .expect("valid config");

        // Should default to mean=0, std=1
        assert_eq!(config.input_mean, vec![0.0; 4]);
        assert_eq!(config.input_std, vec![1.0; 4]);
    }
}
