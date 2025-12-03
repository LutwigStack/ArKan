//! Uniform buffer structures for GPU shaders.
//!
//! This module defines `#[repr(C)]` structures that match the std140 layout
//! requirements for WGSL uniform buffers.
//!
//! # Memory Layout (std140)
//!
//! The std140 layout has strict alignment requirements:
//! - `f32`: 4-byte aligned
//! - `u32`: 4-byte aligned  
//! - `vec2<f32>`: 8-byte aligned
//! - `vec3<f32>`: 16-byte aligned (!)
//! - `vec4<f32>`: 16-byte aligned
//! - Structures: 16-byte aligned (rounded up)
//!
//! All structures in this module are designed to meet these requirements.

use bytemuck::{Pod, Zeroable};

/// Uniform buffer for layer configuration.
///
/// This structure is uploaded to GPU uniform buffer and used by the
/// forward kernel to configure spline computation.
///
/// # Layout
///
/// Total size: 32 bytes (2 × vec4), aligned to 16 bytes.
///
/// ```text
/// Offset  Size  Field
/// 0       4     grid_min
/// 4       4     grid_max
/// 8       4     grid_size
/// 12      4     order
/// 16      4     in_dim
/// 20      4     out_dim
/// 24      4     basis_padded
/// 28      4     batch_size
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LayerUniforms {
    /// Minimum value of the spline grid (typically 0.0).
    pub grid_min: f32,
    /// Maximum value of the spline grid (typically 1.0).
    pub grid_max: f32,
    /// Number of grid intervals.
    pub grid_size: u32,
    /// Spline order (degree).
    pub order: u32,

    /// Input dimension.
    pub in_dim: u32,
    /// Output dimension.
    pub out_dim: u32,
    /// Padded basis size (aligned to 4 for vec4 access).
    pub basis_padded: u32,
    /// Current batch size (dynamic parameter).
    pub batch_size: u32,
}

impl LayerUniforms {
    /// Creates a new LayerUniforms.
    pub fn new(
        grid_min: f32,
        grid_max: f32,
        grid_size: u32,
        order: u32,
        in_dim: u32,
        out_dim: u32,
        basis_padded: u32,
        batch_size: u32,
    ) -> Self {
        Self {
            grid_min,
            grid_max,
            grid_size,
            order,
            in_dim,
            out_dim,
            basis_padded,
            batch_size,
        }
    }

    /// Creates uniforms from layer configuration.
    pub fn from_layer_config(
        grid_min: f32,
        grid_max: f32,
        grid_size: usize,
        order: usize,
        in_dim: usize,
        out_dim: usize,
        batch_size: usize,
    ) -> Self {
        // Global basis size = grid_size + order
        let global_basis = grid_size + order;
        // Pad to multiple of 4 for vec4 access
        let basis_padded = (global_basis + 3) & !3;

        Self {
            grid_min,
            grid_max,
            grid_size: grid_size as u32,
            order: order as u32,
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            basis_padded: basis_padded as u32,
            batch_size: batch_size as u32,
        }
    }

    /// Updates the batch size (for dynamic batching).
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size as u32;
        self
    }

    /// Returns the size in bytes.
    pub const fn size_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Extended uniform buffer with additional training parameters.
///
/// Used during backward pass for gradient computation.
///
/// # Layout
///
/// Total size: 48 bytes (3 × vec4), aligned to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TrainingUniforms {
    // First vec4 (16 bytes)
    /// Minimum value of the spline grid.
    pub grid_min: f32,
    /// Maximum value of the spline grid.
    pub grid_max: f32,
    /// Number of grid intervals.
    pub grid_size: u32,
    /// Spline order (degree).
    pub order: u32,

    // Second vec4 (16 bytes)
    /// Input dimension.
    pub in_dim: u32,
    /// Output dimension.
    pub out_dim: u32,
    /// Padded basis size.
    pub basis_padded: u32,
    /// Current batch size.
    pub batch_size: u32,

    // Third vec4 (16 bytes) - training specific
    /// Learning rate.
    pub learning_rate: f32,
    /// Weight decay coefficient.
    pub weight_decay: f32,
    /// Gradient clipping threshold (0 = disabled).
    pub grad_clip: f32,
    /// Padding for alignment.
    pub _padding: u32,
}

impl TrainingUniforms {
    /// Creates training uniforms from layer config and training parameters.
    pub fn new(
        base: LayerUniforms,
        learning_rate: f32,
        weight_decay: f32,
        grad_clip: f32,
    ) -> Self {
        Self {
            grid_min: base.grid_min,
            grid_max: base.grid_max,
            grid_size: base.grid_size,
            order: base.order,
            in_dim: base.in_dim,
            out_dim: base.out_dim,
            basis_padded: base.basis_padded,
            batch_size: base.batch_size,
            learning_rate,
            weight_decay,
            grad_clip,
            _padding: 0,
        }
    }

    /// Returns the size in bytes.
    pub const fn size_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

/// Compact uniform for simple operations (e.g., softmax).
///
/// # Layout
///
/// Total size: 16 bytes (1 × vec4), aligned to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimpleUniforms {
    /// Total number of elements.
    pub num_elements: u32,
    /// Dimension for reduction operations.
    pub dim: u32,
    /// Batch size.
    pub batch_size: u32,
    /// Padding.
    pub _padding: u32,
}

impl SimpleUniforms {
    /// Creates new simple uniforms.
    pub fn new(num_elements: u32, dim: u32, batch_size: u32) -> Self {
        Self {
            num_elements,
            dim,
            batch_size,
            _padding: 0,
        }
    }

    /// Returns the size in bytes.
    pub const fn size_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_uniforms_size() {
        // Must be 32 bytes (2 vec4)
        assert_eq!(LayerUniforms::size_bytes(), 32);
    }

    #[test]
    fn test_training_uniforms_size() {
        // Must be 48 bytes (3 vec4)
        assert_eq!(TrainingUniforms::size_bytes(), 48);
    }

    #[test]
    fn test_simple_uniforms_size() {
        // Must be 16 bytes (1 vec4)
        assert_eq!(SimpleUniforms::size_bytes(), 16);
    }

    #[test]
    fn test_layer_uniforms_from_config() {
        let uniforms = LayerUniforms::from_layer_config(
            0.0, 1.0, // grid range
            5,        // grid_size
            3,        // order
            21,       // in_dim
            64,       // out_dim
            32,       // batch_size
        );

        assert_eq!(uniforms.grid_min, 0.0);
        assert_eq!(uniforms.grid_max, 1.0);
        assert_eq!(uniforms.grid_size, 5);
        assert_eq!(uniforms.order, 3);
        assert_eq!(uniforms.in_dim, 21);
        assert_eq!(uniforms.out_dim, 64);
        // basis = 5 + 3 = 8, already multiple of 4
        assert_eq!(uniforms.basis_padded, 8);
        assert_eq!(uniforms.batch_size, 32);
    }

    #[test]
    fn test_basis_padding() {
        // grid_size=5, order=2 -> basis=7 -> padded=8
        let uniforms = LayerUniforms::from_layer_config(0.0, 1.0, 5, 2, 10, 10, 1);
        assert_eq!(uniforms.basis_padded, 8);

        // grid_size=4, order=3 -> basis=7 -> padded=8
        let uniforms = LayerUniforms::from_layer_config(0.0, 1.0, 4, 3, 10, 10, 1);
        assert_eq!(uniforms.basis_padded, 8);

        // grid_size=8, order=3 -> basis=11 -> padded=12
        let uniforms = LayerUniforms::from_layer_config(0.0, 1.0, 8, 3, 10, 10, 1);
        assert_eq!(uniforms.basis_padded, 12);
    }

    #[test]
    fn test_pod_zeroable() {
        // Verify Pod and Zeroable traits work
        let zeros: LayerUniforms = Zeroable::zeroed();
        assert_eq!(zeros.grid_min, 0.0);
        assert_eq!(zeros.batch_size, 0);

        // Verify bytemuck cast works
        let uniforms = LayerUniforms::new(0.0, 1.0, 5, 3, 21, 64, 8, 32);
        let bytes: &[u8] = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 32);
    }
}
