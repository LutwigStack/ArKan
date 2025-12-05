//! Unified error types for ArKan.
//!
//! This module provides [`ArkanError`], a unified error type that covers both
//! CPU and GPU backend errors. It uses the `thiserror` crate for ergonomic
//! error handling.
//!
//! # Example
//!
//! ```rust
//! use arkan::ArkanError;
//!
//! fn validate_shape(expected: &[usize], got: &[usize]) -> Result<(), ArkanError> {
//!     if expected != got {
//!         return Err(ArkanError::ShapeMismatch {
//!             expected: expected.to_vec(),
//!             got: got.to_vec(),
//!         });
//!     }
//!     Ok(())
//! }
//! ```

use std::borrow::Cow;

use thiserror::Error;

use crate::config::ConfigError;

/// Unified error type for ArKan operations.
///
/// This enum covers all possible errors that can occur during ArKan operations,
/// including both CPU and GPU backends. GPU-specific variants are only available
/// when the `gpu` feature is enabled.
#[derive(Error, Debug)]
pub enum ArkanError {
    /// CPU backend error with descriptive message.
    #[error("CPU backend error: {0}")]
    Cpu(String),

    /// GPU device lost or disconnected.
    ///
    /// This typically occurs when the GPU driver crashes or the device is removed.
    #[cfg(feature = "gpu")]
    #[error("GPU device error: {0}")]
    DeviceError(#[from] wgpu::Error),

    /// GPU device request failed.
    ///
    /// Could not create a device with the requested features/limits.
    #[cfg(feature = "gpu")]
    #[error("Failed to create GPU device: {0}")]
    DeviceRequestFailed(#[from] wgpu::RequestDeviceError),

    /// GPU buffer or resource validation error.
    ///
    /// Occurs when buffer sizes don't match expected values or when
    /// bind group validation fails.
    #[cfg(feature = "gpu")]
    #[error("Buffer validation error: {0}")]
    Validation(String),

    /// GPU adapter request failed.
    ///
    /// No suitable GPU adapter was found matching the requested criteria.
    #[cfg(feature = "gpu")]
    #[error("Failed to find suitable GPU adapter: {0}")]
    AdapterNotFound(String),

    /// GPU buffer async operation failed.
    #[cfg(feature = "gpu")]
    #[error("Buffer async error: {0}")]
    BufferAsync(#[from] wgpu::BufferAsyncError),

    /// Shape mismatch between expected and actual tensor shapes.
    ///
    /// This is a common error when passing incorrectly sized inputs
    /// to network forward/backward passes.
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected tensor shape.
        expected: Vec<usize>,
        /// Actual tensor shape received.
        got: Vec<usize>,
    },

    /// Batch size exceeds the configured maximum limit.
    ///
    /// This can occur when the requested batch size is larger than
    /// what the workspace or GPU memory can handle.
    #[error("Batch size {0} exceeds limit {1}")]
    BatchTooLarge(usize, usize),

    /// GPU hardware doesn't support required limits.
    ///
    /// This occurs when the GPU doesn't have enough resources
    /// (e.g., max buffer size, max bindings) for the requested operation.
    #[cfg(feature = "gpu")]
    #[error("Unsupported GPU limits: {0}")]
    UnsupportedLimits(String),

    /// Buffer operation failed.
    ///
    /// Generic buffer error for upload/download operations.
    #[cfg(feature = "gpu")]
    #[error("Buffer operation failed: {0}")]
    BufferError(String),

    /// Shader compilation or execution error.
    #[cfg(feature = "gpu")]
    #[error("Shader error: {0}")]
    ShaderError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    /// Integer overflow in size calculations.
    ///
    /// This occurs when buffer size calculations overflow usize,
    /// typically with very large batch sizes or dimensions.
    #[error("Integer overflow: {0}")]
    Overflow(String),

    /// Workspace is in an invalid state.
    ///
    /// This can occur if a previous operation panicked and left
    /// the workspace in an inconsistent state.
    #[error("Invalid workspace state: {0}")]
    InvalidWorkspace(String),

    /// Unsupported spline order for GPU.
    ///
    /// GPU shaders only support spline orders 2-5.
    #[cfg(feature = "gpu")]
    #[error("Unsupported spline order {0}: GPU supports orders 2-5")]
    UnsupportedOrder(usize),

    /// I/O error during model save/load operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Incompatible model version.
    ///
    /// The serialized model was created with an incompatible version.
    #[error("Incompatible model version: expected {expected}, got {got}")]
    IncompatibleVersion {
        /// Expected version.
        expected: u32,
        /// Actual version found.
        got: u32,
    },

    // =========================================================================
    // Optimizer-specific errors (v2.1)
    // =========================================================================

    /// Parameter group index out of bounds.
    ///
    /// Occurs when trying to access or modify a parameter group that doesn't exist.
    #[error("Group index {index} out of bounds (total groups: {total_groups})")]
    GroupIndexOutOfBounds {
        /// Requested index.
        index: usize,
        /// Total number of parameter groups.
        total_groups: usize,
    },

    /// Tensor shape mismatch between parameter and gradient.
    ///
    /// Parameter and gradient tensors must have identical shapes for update.
    #[error("Tensor shape mismatch: param shape {param_shape:?}, grad shape {grad_shape:?}")]
    TensorShapeMismatch {
        /// Shape of the parameter tensor.
        param_shape: Vec<usize>,
        /// Shape of the gradient tensor.
        grad_shape: Vec<usize>,
    },

    /// NaN value encountered during optimization.
    ///
    /// This can occur in gradients or loss values, indicating numerical instability.
    #[error("NaN encountered at param index {param_index} in {context}")]
    NaNEncountered {
        /// Index of the parameter where NaN was detected.
        param_index: usize,
        /// Context: "gradient", "loss", "param", etc.
        context: String,
    },

    /// Line search failed in L-BFGS optimizer.
    ///
    /// The line search algorithm could not find a satisfactory step size.
    #[error("Line search failed: {reason}")]
    LineSearchFailed {
        /// Description of why line search failed.
        reason: String,
    },

    /// Optimizer state version mismatch.
    ///
    /// Occurs when optimizer state is out of sync with model topology
    /// (e.g., after Grid Extension without calling bump_version).
    #[error("State version mismatch: optimizer version {optimizer_version}, expected {expected_version}")]
    StateVersionMismatch {
        /// Current optimizer state version.
        optimizer_version: u64,
        /// Expected version (from model).
        expected_version: u64,
    },

    /// Optimizer closure returned an error.
    ///
    /// Used by L-BFGS when the loss closure fails.
    #[error("Closure error: {0}")]
    ClosureError(String),

    /// General optimizer error.
    #[error("Optimizer error: {0}")]
    Optimizer(String),
}

/// Result type alias for ArKan operations.
pub type ArkanResult<T> = Result<T, ArkanError>;

impl ArkanError {
    /// Creates a CPU error with the given message.
    pub fn cpu<S: Into<String>>(msg: S) -> Self {
        ArkanError::Cpu(msg.into())
    }

    /// Creates a shape mismatch error.
    pub fn shape_mismatch(expected: &[usize], got: &[usize]) -> Self {
        ArkanError::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        }
    }

    /// Creates a batch too large error.
    pub fn batch_too_large(requested: usize, limit: usize) -> Self {
        ArkanError::BatchTooLarge(requested, limit)
    }

    /// Creates a configuration error from a ConfigError.
    pub fn config(err: ConfigError) -> Self {
        ArkanError::Config(err)
    }

    /// Creates a configuration error with a message.
    pub fn config_msg<S: AsRef<str>>(msg: S) -> Self {
        ArkanError::Config(ConfigError::InvalidDimension(Cow::Owned(
            msg.as_ref().to_string(),
        )))
    }

    /// Creates an overflow error.
    pub fn overflow<S: Into<String>>(msg: S) -> Self {
        ArkanError::Overflow(msg.into())
    }

    /// Creates an invalid workspace error.
    pub fn invalid_workspace<S: Into<String>>(msg: S) -> Self {
        ArkanError::InvalidWorkspace(msg.into())
    }

    /// Creates an incompatible version error.
    pub fn incompatible_version(expected: u32, got: u32) -> Self {
        ArkanError::IncompatibleVersion { expected, got }
    }

    // =========================================================================
    // Optimizer error constructors
    // =========================================================================

    /// Creates a group index out of bounds error.
    pub fn group_index_out_of_bounds(index: usize, total_groups: usize) -> Self {
        ArkanError::GroupIndexOutOfBounds { index, total_groups }
    }

    /// Creates a tensor shape mismatch error.
    pub fn tensor_shape_mismatch(param_shape: &[usize], grad_shape: &[usize]) -> Self {
        ArkanError::TensorShapeMismatch {
            param_shape: param_shape.to_vec(),
            grad_shape: grad_shape.to_vec(),
        }
    }

    /// Creates a NaN encountered error.
    pub fn nan_encountered<S: Into<String>>(param_index: usize, context: S) -> Self {
        ArkanError::NaNEncountered {
            param_index,
            context: context.into(),
        }
    }

    /// Creates a line search failed error.
    pub fn line_search_failed<S: Into<String>>(reason: S) -> Self {
        ArkanError::LineSearchFailed {
            reason: reason.into(),
        }
    }

    /// Creates a state version mismatch error.
    pub fn state_version_mismatch(optimizer_version: u64, expected_version: u64) -> Self {
        ArkanError::StateVersionMismatch {
            optimizer_version,
            expected_version,
        }
    }

    /// Creates a closure error.
    pub fn closure_error<S: Into<String>>(msg: S) -> Self {
        ArkanError::ClosureError(msg.into())
    }

    /// Creates a general optimizer error.
    pub fn optimizer<S: Into<String>>(msg: S) -> Self {
        ArkanError::Optimizer(msg.into())
    }

    /// Creates an unsupported order error (GPU only).
    #[cfg(feature = "gpu")]
    pub fn unsupported_order(order: usize) -> Self {
        ArkanError::UnsupportedOrder(order)
    }

    /// Creates a GPU validation error.
    #[cfg(feature = "gpu")]
    pub fn validation<S: Into<String>>(msg: S) -> Self {
        ArkanError::Validation(msg.into())
    }

    /// Creates a GPU buffer error.
    #[cfg(feature = "gpu")]
    pub fn buffer<S: Into<String>>(msg: S) -> Self {
        ArkanError::BufferError(msg.into())
    }

    /// Creates a GPU unsupported limits error.
    #[cfg(feature = "gpu")]
    pub fn unsupported_limits<S: Into<String>>(msg: S) -> Self {
        ArkanError::UnsupportedLimits(msg.into())
    }

    /// Creates a shader error.
    #[cfg(feature = "gpu")]
    pub fn shader<S: Into<String>>(msg: S) -> Self {
        ArkanError::ShaderError(msg.into())
    }

    /// Creates an adapter not found error.
    #[cfg(feature = "gpu")]
    pub fn adapter_not_found<S: Into<String>>(msg: S) -> Self {
        ArkanError::AdapterNotFound(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_error() {
        let err = ArkanError::cpu("test error");
        assert!(err.to_string().contains("CPU backend error"));
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_shape_mismatch() {
        let err = ArkanError::shape_mismatch(&[1, 2, 3], &[1, 2, 4]);
        let msg = err.to_string();
        assert!(msg.contains("Shape mismatch"));
        assert!(msg.contains("[1, 2, 3]"));
        assert!(msg.contains("[1, 2, 4]"));
    }

    #[test]
    fn test_batch_too_large() {
        let err = ArkanError::batch_too_large(1000, 512);
        let msg = err.to_string();
        assert!(msg.contains("1000"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_config_error() {
        let err = ArkanError::config(ConfigError::InvalidDimension(Cow::Borrowed(
            "test dimension",
        )));
        assert!(err.to_string().contains("Configuration error"));
    }
}
