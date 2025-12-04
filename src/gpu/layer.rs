//! GPU Layer implementation with bind group management.
//!
//! This module provides [`GpuLayer`] which holds GPU resources for a single
//! KAN layer, including weights, bias, and bind groups.

use crate::error::ArkanResult;
use crate::gpu::uniforms::LayerUniforms;
use crate::gpu::{pad_to_vec4, GpuTensor};
use crate::layer::KanLayer;
use wgpu::util::DeviceExt;

/// GPU representation of a KAN layer.
///
/// Holds all GPU resources needed for forward pass computation:
/// - Weight buffer (packed as vec4 for efficient access)
/// - Bias buffer
/// - Uniform buffer
/// - Static bind group (Group 0)
///
/// # Bind Group Layout
///
/// Group 0 (Static - per layer):
/// - Binding 0: Weights (storage, read) - `array<vec4<f32>>`
/// - Binding 1: Bias (storage, read)
/// - Binding 2: Uniforms (uniform)
///
/// # Weight Layout
///
/// Weights are stored as `array<vec4<f32>>` where:
/// - Total vec4 count = out_dim * in_dim * basis_vec4s
/// - basis_vec4s = ceil(basis_padded / 4)
/// - Each vec4 contains 4 consecutive basis weights
///
/// # Training Support
///
/// When `init_training()` is called, gradient buffers are allocated:
/// - `grad_weights`: accumulated gradients for weights
/// - `grad_bias`: accumulated gradients for bias
///
/// These buffers are used by GPU-native optimizers ([`GpuAdam`], [`GpuSgd`])
/// to update weights entirely on GPU without CPU transfers.
pub struct GpuLayer {
    /// Weight tensor [out_dim, in_dim, basis_vec4s] stored as vec4.
    pub weights: GpuTensor,
    /// Bias tensor `(out_dim,)`.
    pub bias: GpuTensor,
    /// Uniform buffer.
    pub uniforms_buffer: wgpu::Buffer,
    /// Cached uniforms for CPU-side updates.
    pub uniforms: LayerUniforms,

    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
    /// Grid size.
    pub grid_size: usize,
    /// Spline order.
    pub order: usize,
    /// Global basis size (grid_size + order).
    pub global_basis_size: usize,
    /// Padded basis size (aligned to 4).
    pub basis_padded: usize,
    /// Number of vec4s per (out, in) pair: ceil(basis_padded / 4).
    pub basis_vec4s: usize,

    /// Static bind group (weights, bias, uniforms).
    pub bind_group: wgpu::BindGroup,
    /// Bind group layout for static resources.
    pub bind_group_layout: wgpu::BindGroupLayout,

    // ==================== Training Buffers ====================
    // These are allocated on-demand when init_training() is called.
    /// Gradient buffer for weights (same layout as weights).
    /// `None` until `init_training()` is called.
    pub grad_weights: Option<GpuTensor>,
    /// Gradient buffer for bias.
    /// `None` until `init_training()` is called.
    pub grad_bias: Option<GpuTensor>,
}

impl GpuLayer {
    /// Creates a GPU layer from a CPU KAN layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `cpu_layer` - The source CPU layer to upload.
    ///
    /// # Returns
    ///
    /// A new `GpuLayer` with uploaded weights and bias.
    ///
    /// # Weight Packing
    ///
    /// Weights are packed into vec4 format for efficient GPU access:
    /// - Original: [out_dim, in_dim, global_basis_size]
    /// - Padded: [out_dim, in_dim, basis_padded] (basis_padded = align4(global_basis_size))
    /// - GPU: [out_dim * in_dim * basis_vec4s] vec4s (basis_vec4s = ceil(basis_padded/4))
    pub fn from_cpu_layer(device: &wgpu::Device, cpu_layer: &KanLayer) -> ArkanResult<Self> {
        let in_dim = cpu_layer.in_dim;
        let out_dim = cpu_layer.out_dim;
        let grid_size = cpu_layer.grid_size;
        let order = cpu_layer.order;
        let global_basis_size = cpu_layer.global_basis_size;
        let basis_padded = pad_to_vec4(global_basis_size);
        let basis_vec4s = (basis_padded + 3) / 4; // ceil division

        // Pack weights into vec4 format
        // Each vec4 contains 4 consecutive basis weights
        let packed_weights = Self::pack_weights_vec4(
            cpu_layer.weights.as_slice(),
            out_dim,
            in_dim,
            global_basis_size,
            basis_padded,
            basis_vec4s,
        );

        let weights = GpuTensor::storage_read(
            device,
            &packed_weights,
            vec![out_dim * in_dim * basis_vec4s * 4], // Total floats in vec4 array
        )?;

        let bias = GpuTensor::storage_read(device, cpu_layer.bias.as_slice(), vec![out_dim])?;

        // Create uniforms
        let uniforms = LayerUniforms::from_layer_config(
            cpu_layer.grid_range.0, // grid_min
            cpu_layer.grid_range.1, // grid_max
            grid_size,
            order,
            in_dim,
            out_dim,
            1, // batch_size will be updated per forward call
        );

        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuLayer Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = Self::create_bind_group_layout(device);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuLayer BindGroup (Static)"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            weights,
            bias,
            uniforms_buffer,
            uniforms,
            in_dim,
            out_dim,
            grid_size,
            order,
            global_basis_size,
            basis_padded,
            basis_vec4s,
            bind_group,
            bind_group_layout,
            grad_weights: None,
            grad_bias: None,
        })
    }

    /// Creates the bind group layout for static layer resources.
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuLayer BindGroupLayout (Static)"),
            entries: &[
                // Weights (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bias (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Creates the bind group layout for dynamic workspace resources.
    pub fn create_workspace_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuWorkspace BindGroupLayout (Dynamic)"),
            entries: &[
                // Input (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output (storage, read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Packs weights into vec4 format for efficient GPU access.
    ///
    /// Input layout: [out_dim, in_dim, global_basis]
    /// Output layout: [out_dim * in_dim * basis_vec4s] vec4s (flattened as f32 array)
    ///
    /// Each vec4 contains 4 consecutive basis weights, padded with zeros if needed.
    fn pack_weights_vec4(
        weights: &[f32],
        out_dim: usize,
        in_dim: usize,
        global_basis: usize,
        _basis_padded: usize, // Used for documentation, actual padding uses basis_vec4s * 4
        basis_vec4s: usize,
    ) -> Vec<f32> {
        let total_vec4s = out_dim * in_dim * basis_vec4s;
        let mut packed = vec![0.0f32; total_vec4s * 4];

        for o in 0..out_dim {
            for i in 0..in_dim {
                let src_offset = (o * in_dim + i) * global_basis;
                let dst_vec4_base = (o * in_dim + i) * basis_vec4s;

                // Copy weights into vec4 positions
                for k in 0..global_basis {
                    let dst_offset = dst_vec4_base * 4 + k;
                    packed[dst_offset] = weights[src_offset + k];
                }
                // Remaining positions (from global_basis to basis_padded) are already zero
            }
        }

        packed
    }

    /// Updates the batch size in uniforms.
    pub fn update_batch_size(&mut self, queue: &wgpu::Queue, batch_size: usize) {
        self.uniforms.batch_size = batch_size as u32;
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    /// Updates weights from CPU layer.
    pub fn update_weights(&mut self, queue: &wgpu::Queue, cpu_layer: &KanLayer) {
        let packed_weights = Self::pack_weights_vec4(
            cpu_layer.weights.as_slice(),
            self.out_dim,
            self.in_dim,
            self.global_basis_size,
            self.basis_padded,
            self.basis_vec4s,
        );
        self.weights.update(queue, &packed_weights);
    }

    /// Updates bias from CPU layer.
    pub fn update_bias(&mut self, queue: &wgpu::Queue, cpu_layer: &KanLayer) {
        self.bias.update(queue, cpu_layer.bias.as_slice());
    }

    /// Returns the total weight count (vec4 aligned).
    pub fn weight_count(&self) -> usize {
        self.out_dim * self.in_dim * self.basis_vec4s * 4
    }

    /// Returns the bias count.
    pub fn bias_count(&self) -> usize {
        self.out_dim
    }

    /// Returns the total parameter count.
    pub fn param_count(&self) -> usize {
        self.weight_count() + self.bias_count()
    }

    /// Returns the GPU memory used by weights in bytes.
    pub fn weights_bytes(&self) -> usize {
        self.weight_count() * std::mem::size_of::<f32>()
    }

    /// Returns the GPU memory used by bias in bytes.
    pub fn bias_bytes(&self) -> usize {
        self.bias_count() * std::mem::size_of::<f32>()
    }

    // ==================== Training Support ====================

    /// Initializes gradient buffers for training.
    ///
    /// Call this once before using GPU-native training. Allocates:
    /// - `grad_weights`: same size as weights buffer
    /// - `grad_bias`: same size as bias buffer
    ///
    /// These buffers are used by [`GpuAdam`] and [`GpuSgd`] optimizers.
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::BufferError`] if buffer creation fails.
    pub fn init_training(&mut self, device: &wgpu::Device) -> crate::ArkanResult<()> {
        use crate::ArkanError;

        if self.grad_weights.is_some() {
            return Ok(()); // Already initialized
        }

        // Gradient for weights (same layout as weights: vec4 packed)
        let grad_weights_zeros = vec![0.0f32; self.weight_count()];
        self.grad_weights = Some(
            GpuTensor::storage_rw(
                device,
                &grad_weights_zeros,
                vec![self.out_dim * self.in_dim * self.basis_vec4s * 4],
            )
            .map_err(|e| {
                ArkanError::buffer(format!("Failed to create grad_weights buffer: {}", e))
            })?,
        );

        // Gradient for bias
        let grad_bias_zeros = vec![0.0f32; self.out_dim];
        self.grad_bias = Some(
            GpuTensor::storage_rw(device, &grad_bias_zeros, vec![self.out_dim]).map_err(|e| {
                ArkanError::buffer(format!("Failed to create grad_bias buffer: {}", e))
            })?,
        );

        Ok(())
    }

    /// Returns true if training buffers are initialized.
    #[inline]
    pub fn is_training_initialized(&self) -> bool {
        self.grad_weights.is_some()
    }

    /// Zeros the gradient buffers.
    ///
    /// Call this at the start of each training step before backward pass.
    pub fn zero_grads(&self, queue: &wgpu::Queue) {
        if let Some(ref gw) = self.grad_weights {
            let zeros = vec![0.0f32; self.weight_count()];
            gw.update(queue, &zeros);
        }
        if let Some(ref gb) = self.grad_bias {
            let zeros = vec![0.0f32; self.out_dim];
            gb.update(queue, &zeros);
        }
    }

    /// Returns the weight buffer reference (for optimizer).
    #[inline]
    pub fn weights_buffer(&self) -> &wgpu::Buffer {
        &self.weights.buffer
    }

    /// Returns the bias buffer reference (for optimizer).
    #[inline]
    pub fn bias_buffer(&self) -> &wgpu::Buffer {
        &self.bias.buffer
    }

    /// Returns the grad_weights buffer reference (for optimizer).
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::InvalidWorkspace`] if training is not initialized.
    #[inline]
    pub fn grad_weights_buffer(&self) -> crate::ArkanResult<&wgpu::Buffer> {
        use crate::ArkanError;
        Ok(&self
            .grad_weights
            .as_ref()
            .ok_or_else(|| {
                ArkanError::invalid_workspace(
                    "Training not initialized. Call init_training() first.",
                )
            })?
            .buffer)
    }

    /// Returns the grad_bias buffer reference (for optimizer).
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::InvalidWorkspace`] if training is not initialized.
    #[inline]
    pub fn grad_bias_buffer(&self) -> crate::ArkanResult<&wgpu::Buffer> {
        use crate::ArkanError;
        Ok(&self
            .grad_bias
            .as_ref()
            .ok_or_else(|| {
                ArkanError::invalid_workspace(
                    "Training not initialized. Call init_training() first.",
                )
            })?
            .buffer)
    }

    // ==================== Backward Pass Support ====================

    /// Creates the bind group layout for backward pass.
    ///
    /// Group 0 (backward):
    /// - Binding 0: Weights (storage, read)
    /// - Binding 1: BackwardUniforms (uniform)
    pub fn create_backward_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuLayer Backward BindGroupLayout"),
            entries: &[
                // Weights (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // BackwardUniforms (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Creates a backward bind group for this layer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `layout` - The backward bind group layout.
    /// * `backward_uniforms_buffer` - Buffer containing BackwardUniforms.
    pub fn create_backward_bind_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        backward_uniforms_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuLayer Backward BindGroup"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: backward_uniforms_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

impl std::fmt::Debug for GpuLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLayer")
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("grid_size", &self.grid_size)
            .field("order", &self.order)
            .field("global_basis_size", &self.global_basis_size)
            .field("basis_padded", &self.basis_padded)
            .field("basis_vec4s", &self.basis_vec4s)
            .field("weight_count", &self.weight_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_weights_vec4() {
        // 2x2x3 weights -> packed into vec4 format
        // basis_padded = 4, basis_vec4s = 1
        let weights = vec![
            1.0, 2.0, 3.0, // [0,0,:]
            4.0, 5.0, 6.0, // [0,1,:]
            7.0, 8.0, 9.0, // [1,0,:]
            10.0, 11.0, 12.0, // [1,1,:]
        ];

        let packed = GpuLayer::pack_weights_vec4(&weights, 2, 2, 3, 4, 1);

        // 2*2*1 = 4 vec4s = 16 floats
        assert_eq!(packed.len(), 16);
        // First vec4: weights[0,0,:] = [1,2,3,0]
        assert_eq!(packed[0..4], [1.0, 2.0, 3.0, 0.0]);
        // Second vec4: weights[0,1,:] = [4,5,6,0]
        assert_eq!(packed[4..8], [4.0, 5.0, 6.0, 0.0]);
        // Third vec4: weights[1,0,:] = [7,8,9,0]
        assert_eq!(packed[8..12], [7.0, 8.0, 9.0, 0.0]);
        // Fourth vec4: weights[1,1,:] = [10,11,12,0]
        assert_eq!(packed[12..16], [10.0, 11.0, 12.0, 0.0]);
    }

    #[test]
    fn test_pack_weights_vec4_multiple_vec4s() {
        // 1x1x5 weights -> basis_padded = 8, basis_vec4s = 2
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let packed = GpuLayer::pack_weights_vec4(&weights, 1, 1, 5, 8, 2);

        // 1*1*2 = 2 vec4s = 8 floats
        assert_eq!(packed.len(), 8);
        // First vec4: [1,2,3,4]
        assert_eq!(packed[0..4], [1.0, 2.0, 3.0, 4.0]);
        // Second vec4: [5,0,0,0]
        assert_eq!(packed[4..8], [5.0, 0.0, 0.0, 0.0]);
    }
}
