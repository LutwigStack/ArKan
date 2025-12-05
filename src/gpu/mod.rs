//! GPU Backend for ArKan using wgpu.
//!
//! This module provides GPU-accelerated KAN network operations using the wgpu
//! graphics API. It is only available when the `gpu` feature is enabled.
//!
//! # Architecture
//!
//! The GPU backend uses a split bind group strategy for efficient resource management:
//!
//! - **Group 0 (Static - Layer)**: Weights, Bias, Uniforms - created once per layer
//! - **Group 1 (Dynamic - Workspace)**: Input, Output - resized as needed
//!
//! # Example
//!
//! ```rust,no_run
//! use arkan::gpu::{WgpuBackend, WgpuOptions, GpuTensor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the backend
//! let backend = WgpuBackend::init(WgpuOptions::default())?;
//!
//! // Upload data to GPU
//! let data = vec![1.0f32; 1024];
//! let tensor = GpuTensor::upload(&backend.device, &backend.queue, &data, vec![1024])?;
//!
//! // Download data from GPU
//! let result = tensor.download(&backend.device, &backend.queue)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Memory Management
//!
//! The GPU backend uses a resize policy for workspace buffers:
//!
//! - If `requested_size > current_capacity`: Reallocate buffer
//! - If `requested_size > MAX_VRAM_ALLOC`: Return [`ArkanError::BatchTooLarge`](crate::ArkanError::BatchTooLarge)
//!
//! # Public API
//!
//! - [`WgpuBackend`] — GPU device initialization and management
//! - [`GpuTensor`] — GPU tensor abstraction with upload/download
//! - [`LayerUniforms`] — Uniform buffer structures (std140 layout)
//! - [`GpuWorkspace`] — GPU workspace with resize policy
//! - [`GpuLayer`] — GPU layer implementation with bind groups
//! - [`GpuNetwork`] — GPU network with forward/backward passes
//! - [`shaders`] — Dynamic shader generation for variable spline orders
//! - [`optimizer`] — GPU-accelerated optimizers (Adam, SGD)

mod backend;
mod layer;
mod network;
pub mod optimizer;
mod pipeline;
pub mod shaders;
mod tensor;
mod uniforms;
mod workspace;

pub use backend::{PowerPreference, WgpuBackend, WgpuOptions};
pub use layer::GpuLayer;
pub use network::{GpuForwardHandle, GpuMemoryStats, GpuNetwork};
pub use optimizer::{
    AdamUniforms, GpuAdam, GpuAdamConfig, GpuAdamLayerState, GpuSgd, GpuSgdConfig,
    GpuSgdLayerState, SgdUniforms,
};
pub use pipeline::{workgroup_count, PipelineCache, WORKGROUP_SIZE};
pub use shaders::{
    generate_backward_input_shader, generate_backward_weights_shader, generate_forward_shader,
    generate_forward_training_shader, ADAM_SHADER, GRAD_CLIP_SHADER, SGD_SHADER,
};
pub use tensor::{GpuTensor, GpuTensorView};
pub use uniforms::LayerUniforms;
pub use workspace::GpuWorkspace;

/// Maximum VRAM allocation per buffer (2GB).
///
/// This limit prevents excessive memory allocation on GPUs and ensures
/// compatibility with most hardware configurations.
pub const MAX_VRAM_ALLOC: u64 = 2 * 1024 * 1024 * 1024;

/// Default alignment for GPU buffers (256 bytes).
///
/// This alignment ensures compatibility with most GPU architectures and
/// meets the requirements for uniform buffer offsets.
pub const GPU_BUFFER_ALIGNMENT: u64 = 256;

/// Checks if a size in bytes exceeds the maximum VRAM allocation limit.
#[inline]
pub fn exceeds_vram_limit(size_bytes: u64) -> bool {
    size_bytes > MAX_VRAM_ALLOC
}

/// Aligns a size to the specified alignment.
#[inline]
pub const fn align_to(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

/// Pads a dimension to be a multiple of 4 (for vec4 access in shaders).
#[inline]
pub const fn pad_to_vec4(dim: usize) -> usize {
    (dim + 3) & !3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_limit() {
        assert!(!exceeds_vram_limit(1024));
        assert!(!exceeds_vram_limit(MAX_VRAM_ALLOC));
        assert!(exceeds_vram_limit(MAX_VRAM_ALLOC + 1));
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 256), 0);
        assert_eq!(align_to(1, 256), 256);
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(257, 256), 512);
    }

    #[test]
    fn test_pad_to_vec4() {
        assert_eq!(pad_to_vec4(0), 0);
        assert_eq!(pad_to_vec4(1), 4);
        assert_eq!(pad_to_vec4(4), 4);
        assert_eq!(pad_to_vec4(5), 8);
        assert_eq!(pad_to_vec4(7), 8);
        assert_eq!(pad_to_vec4(8), 8);
    }
}
