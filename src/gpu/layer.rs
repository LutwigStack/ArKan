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
/// - Weight buffer (padded for vec4 access)
/// - Bias buffer
/// - Uniform buffer
/// - Static bind group (Group 0)
///
/// # Bind Group Layout
///
/// Group 0 (Static - per layer):
/// - Binding 0: Weights (storage, read)
/// - Binding 1: Bias (storage, read)
/// - Binding 2: Uniforms (uniform)
pub struct GpuLayer {
    /// Weight tensor [out_dim, in_dim, basis_padded].
    pub weights: GpuTensor,
    /// Bias tensor [out_dim].
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

    /// Static bind group (weights, bias, uniforms).
    pub bind_group: wgpu::BindGroup,
    /// Bind group layout for static resources.
    pub bind_group_layout: wgpu::BindGroupLayout,
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
    pub fn from_cpu_layer(device: &wgpu::Device, cpu_layer: &KanLayer) -> ArkanResult<Self> {
        let in_dim = cpu_layer.in_dim;
        let out_dim = cpu_layer.out_dim;
        let grid_size = cpu_layer.grid_size;
        let order = cpu_layer.order;
        let global_basis_size = cpu_layer.global_basis_size;
        let basis_padded = pad_to_vec4(global_basis_size);

        // Pad weights for vec4 access
        // Original layout: [out_dim, in_dim, global_basis_size]
        // Padded layout: [out_dim, in_dim, basis_padded]
        let padded_weights = Self::pad_weights(
            cpu_layer.weights.as_slice(),
            out_dim,
            in_dim,
            global_basis_size,
            basis_padded,
        );

        let weights = GpuTensor::storage_read(
            device,
            &padded_weights,
            vec![out_dim, in_dim, basis_padded],
        );

        let bias = GpuTensor::storage_read(device, cpu_layer.bias.as_slice(), vec![out_dim]);

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
            bind_group,
            bind_group_layout,
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

    /// Pads weights for vec4 access in shaders.
    fn pad_weights(
        weights: &[f32],
        out_dim: usize,
        in_dim: usize,
        global_basis: usize,
        basis_padded: usize,
    ) -> Vec<f32> {
        if global_basis == basis_padded {
            return weights.to_vec();
        }

        let mut padded = vec![0.0f32; out_dim * in_dim * basis_padded];

        for o in 0..out_dim {
            for i in 0..in_dim {
                let src_offset = (o * in_dim + i) * global_basis;
                let dst_offset = (o * in_dim + i) * basis_padded;
                padded[dst_offset..dst_offset + global_basis]
                    .copy_from_slice(&weights[src_offset..src_offset + global_basis]);
            }
        }

        padded
    }

    /// Updates the batch size in uniforms.
    pub fn update_batch_size(&mut self, queue: &wgpu::Queue, batch_size: usize) {
        self.uniforms.batch_size = batch_size as u32;
        queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }

    /// Updates weights from CPU layer.
    pub fn update_weights(&mut self, queue: &wgpu::Queue, cpu_layer: &KanLayer) {
        let padded_weights = Self::pad_weights(
            cpu_layer.weights.as_slice(),
            self.out_dim,
            self.in_dim,
            self.global_basis_size,
            self.basis_padded,
        );
        self.weights.update(queue, &padded_weights);
    }

    /// Updates bias from CPU layer.
    pub fn update_bias(&mut self, queue: &wgpu::Queue, cpu_layer: &KanLayer) {
        self.bias.update(queue, cpu_layer.bias.as_slice());
    }

    /// Returns the total weight count (padded).
    pub fn weight_count(&self) -> usize {
        self.out_dim * self.in_dim * self.basis_padded
    }

    /// Returns the bias count.
    pub fn bias_count(&self) -> usize {
        self.out_dim
    }

    /// Returns the total parameter count.
    pub fn param_count(&self) -> usize {
        self.weight_count() + self.bias_count()
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
            .field("weight_count", &self.weight_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_weights() {
        // 2x2x3 weights -> 2x2x4 padded
        let weights = vec![
            1.0, 2.0, 3.0, // [0,0,:]
            4.0, 5.0, 6.0, // [0,1,:]
            7.0, 8.0, 9.0, // [1,0,:]
            10.0, 11.0, 12.0, // [1,1,:]
        ];

        let padded = GpuLayer::pad_weights(&weights, 2, 2, 3, 4);

        assert_eq!(padded.len(), 2 * 2 * 4);
        assert_eq!(padded[0..4], [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(padded[4..8], [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(padded[8..12], [7.0, 8.0, 9.0, 0.0]);
        assert_eq!(padded[12..16], [10.0, 11.0, 12.0, 0.0]);
    }

    #[test]
    fn test_pad_weights_no_padding_needed() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let padded = GpuLayer::pad_weights(&weights, 1, 1, 4, 4);
        assert_eq!(padded, weights);
    }
}
