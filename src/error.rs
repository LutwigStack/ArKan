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

use thiserror::Error;

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
    Config(String),

    /// I/O error during model save/load operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
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

    /// Creates a configuration error.
    pub fn config<S: Into<String>>(msg: S) -> Self {
        ArkanError::Config(msg.into())
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
        let err = ArkanError::config("invalid config");
        assert!(err.to_string().contains("Configuration error"));
    }
}
