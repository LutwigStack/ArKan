//! GPU Workspace with dynamic buffer management.
//!
//! This module provides [`GpuWorkspace`] which manages input/output buffers
//! for GPU operations with automatic resizing.

use crate::error::{ArkanError, ArkanResult};
use crate::gpu::{exceeds_vram_limit, GpuTensor, MAX_VRAM_ALLOC};

/// GPU workspace for managing dynamic input/output buffers.
///
/// The workspace holds buffers for forward pass computation and implements
/// a resize policy that grows buffers as needed while respecting VRAM limits.
///
/// # Buffer Management
///
/// - Buffers are lazily allocated on first use
/// - Buffers grow when needed but never shrink (to avoid reallocation overhead)
/// - Cached bind groups are invalidated when buffers resize
///
/// # Bind Group Caching
///
/// The workspace caches bind groups (Group 1) to avoid recreation overhead:
/// - `cached_bind_group`: For single-layer or input/output only
/// - `cached_layer_bind_groups`: For multi-layer, maps (input_buf_idx, output_buf_idx) -> BindGroup
///
/// # Example
///
/// ```rust,ignore
/// use arkan::gpu::{GpuWorkspace, WgpuBackend};
///
/// let backend = WgpuBackend::init(Default::default())?;
/// let mut workspace = GpuWorkspace::new(&backend, 64, 21, 64)?;
///
/// // Resize for different batch size
/// workspace.ensure_capacity(&backend, 128)?;
/// ```
pub struct GpuWorkspace {
    /// Input buffer [batch, in_dim].
    pub input: Option<GpuTensor>,
    /// Output buffer [batch, out_dim].
    pub output: Option<GpuTensor>,

    /// Intermediate buffers for multi-layer networks.
    pub intermediates: Vec<GpuTensor>,

    /// Input dimension (fixed).
    pub in_dim: usize,
    /// Output dimension (fixed).
    pub out_dim: usize,
    /// Current maximum batch capacity.
    pub max_batch: usize,

    /// Bind group layout for dynamic resources (Group 1).
    pub bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Cached bind group for single-layer (input -> output).
    cached_bind_group: Option<wgpu::BindGroup>,
    /// Cached bind groups for multi-layer, indexed by (in_buffer_type, out_buffer_type).
    /// Types: 0 = input, 1 = output, 2+ = intermediate[idx-2]
    cached_layer_bind_groups: Vec<Option<wgpu::BindGroup>>,

    /// Generation counter for cache invalidation.
    generation: u64,
}

impl GpuWorkspace {
    /// Creates a new workspace with the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `max_batch` - Maximum batch size to support.
    /// * `in_dim` - Input dimension.
    /// * `out_dim` - Output dimension.
    ///
    /// # Returns
    ///
    /// A new workspace, or an error if allocation fails.
    pub fn new(
        device: &wgpu::Device,
        max_batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> ArkanResult<Self> {
        let input_size = max_batch * in_dim;
        let output_size = max_batch * out_dim;

        // Check VRAM limits
        let input_bytes = (input_size * std::mem::size_of::<f32>()) as u64;
        let output_bytes = (output_size * std::mem::size_of::<f32>()) as u64;

        if exceeds_vram_limit(input_bytes) || exceeds_vram_limit(output_bytes) {
            return Err(ArkanError::batch_too_large(
                max_batch,
                (MAX_VRAM_ALLOC / (in_dim.max(out_dim) * std::mem::size_of::<f32>()) as u64)
                    as usize,
            ));
        }

        let input = GpuTensor::storage_read_write(device, vec![max_batch, in_dim])?;
        let output = GpuTensor::storage_read_write(device, vec![max_batch, out_dim])?;

        Ok(Self {
            input: Some(input),
            output: Some(output),
            intermediates: Vec::new(),
            in_dim,
            out_dim,
            max_batch,
            bind_group_layout: None,
            cached_bind_group: None,
            cached_layer_bind_groups: Vec::new(),
            generation: 0,
        })
    }

    /// Creates an empty workspace (lazy allocation).
    pub fn empty(in_dim: usize, out_dim: usize) -> Self {
        Self {
            input: None,
            output: None,
            intermediates: Vec::new(),
            in_dim,
            out_dim,
            max_batch: 0,
            bind_group_layout: None,
            cached_bind_group: None,
            cached_layer_bind_groups: Vec::new(),
            generation: 0,
        }
    }

    /// Ensures the workspace can handle at least `batch_size` samples.
    ///
    /// If the current capacity is insufficient, buffers are reallocated
    /// and cached bind groups are invalidated.
    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        batch_size: usize,
    ) -> ArkanResult<bool> {
        if batch_size <= self.max_batch && self.input.is_some() {
            return Ok(false); // No resize needed
        }

        // Calculate new capacity with some headroom (1.5x requested)
        let new_capacity = (batch_size * 3 / 2).max(batch_size);

        let input_size = new_capacity * self.in_dim;
        let output_size = new_capacity * self.out_dim;

        // Check VRAM limits
        let input_bytes = (input_size * std::mem::size_of::<f32>()) as u64;
        let output_bytes = (output_size * std::mem::size_of::<f32>()) as u64;

        if exceeds_vram_limit(input_bytes) || exceeds_vram_limit(output_bytes) {
            return Err(ArkanError::batch_too_large(
                batch_size,
                (MAX_VRAM_ALLOC / (self.in_dim.max(self.out_dim) * std::mem::size_of::<f32>()) as u64)
                    as usize,
            ));
        }

        // Allocate new buffers
        self.input = Some(GpuTensor::storage_read_write(
            device,
            vec![new_capacity, self.in_dim],
        )?);
        self.output = Some(GpuTensor::storage_read_write(
            device,
            vec![new_capacity, self.out_dim],
        )?);

        self.max_batch = new_capacity;

        // Invalidate cached bind group
        self.invalidate_cache();

        Ok(true) // Resized
    }

    /// Ensures intermediate buffers for multi-layer forward pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `layer_dims` - Dimensions between layers (excluding input/output).
    /// * `batch_size` - Current batch size.
    pub fn ensure_intermediates(
        &mut self,
        device: &wgpu::Device,
        layer_dims: &[usize],
        batch_size: usize,
    ) -> ArkanResult<()> {
        // Need n-1 intermediate buffers for n layers
        let needed = layer_dims.len().saturating_sub(1);

        // Check if we need to resize existing intermediates
        let needs_resize = self.intermediates.len() < needed
            || self.intermediates.iter().enumerate().any(|(i, t)| {
                i < needed && (t.shape[0] < batch_size || t.shape[1] != layer_dims[i + 1])
            });

        if needs_resize {
            self.intermediates.clear();

            for i in 0..needed {
                let dim = layer_dims[i + 1]; // Output dim of layer i
                let tensor = GpuTensor::storage_read_write(device, vec![batch_size, dim])?;
                self.intermediates.push(tensor);
            }

            self.invalidate_cache();
        }

        Ok(())
    }

    /// Invalidates cached bind groups.
    fn invalidate_cache(&mut self) {
        self.cached_bind_group = None;
        self.cached_layer_bind_groups.clear();
        self.generation += 1;
    }

    /// Returns the current generation (for cache validation).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Gets or creates the bind group for input/output buffers.
    pub fn get_or_create_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> ArkanResult<&wgpu::BindGroup> {
        if self.cached_bind_group.is_none() {
            let input = self
                .input
                .as_ref()
                .ok_or_else(|| ArkanError::buffer("Input buffer not allocated"))?;
            let output = self
                .output
                .as_ref()
                .ok_or_else(|| ArkanError::buffer("Output buffer not allocated"))?;

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GpuWorkspace BindGroup"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.buffer.as_entire_binding(),
                    },
                ],
            });

            self.cached_bind_group = Some(bind_group);
        }

        Ok(self.cached_bind_group.as_ref().unwrap())
    }

    /// Gets or creates the bind group for a specific layer in multi-layer forward.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `layout` - The bind group layout.
    /// * `layer_idx` - Index of the current layer.
    /// * `num_layers` - Total number of layers.
    ///
    /// # Buffer Routing
    ///
    /// - Layer 0: input -> intermediate[0]
    /// - Layer i (middle): intermediate[i-1] -> intermediate[i]
    /// - Layer n-1: intermediate[n-2] -> output
    pub fn get_or_create_layer_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        layer_idx: usize,
        num_layers: usize,
    ) -> ArkanResult<&wgpu::BindGroup> {
        // Ensure cache vector is large enough
        if self.cached_layer_bind_groups.len() < num_layers {
            self.cached_layer_bind_groups.resize_with(num_layers, || None);
        }

        // Check if already cached
        if self.cached_layer_bind_groups[layer_idx].is_some() {
            return Ok(self.cached_layer_bind_groups[layer_idx].as_ref().unwrap());
        }

        // Determine input and output buffers
        let (input_buffer, output_buffer) = if layer_idx == 0 {
            // First layer: input from workspace.input, output to intermediate[0]
            (
                self.input.as_ref()
                    .ok_or_else(|| ArkanError::buffer("No input buffer"))?
                    .buffer.as_entire_binding(),
                self.intermediates.get(0)
                    .ok_or_else(|| ArkanError::buffer("No intermediate buffer 0"))?
                    .buffer.as_entire_binding(),
            )
        } else if layer_idx == num_layers - 1 {
            // Last layer: input from intermediate[n-2], output to workspace.output
            (
                self.intermediates.get(layer_idx - 1)
                    .ok_or_else(|| ArkanError::buffer(&format!("No intermediate buffer {}", layer_idx - 1)))?
                    .buffer.as_entire_binding(),
                self.output.as_ref()
                    .ok_or_else(|| ArkanError::buffer("No output buffer"))?
                    .buffer.as_entire_binding(),
            )
        } else {
            // Middle layer: intermediate[i-1] -> intermediate[i]
            (
                self.intermediates.get(layer_idx - 1)
                    .ok_or_else(|| ArkanError::buffer(&format!("No intermediate buffer {}", layer_idx - 1)))?
                    .buffer.as_entire_binding(),
                self.intermediates.get(layer_idx)
                    .ok_or_else(|| ArkanError::buffer(&format!("No intermediate buffer {}", layer_idx)))?
                    .buffer.as_entire_binding(),
            )
        };

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} I/O BindGroup", layer_idx)),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer,
                },
            ],
        });

        self.cached_layer_bind_groups[layer_idx] = Some(bind_group);
        Ok(self.cached_layer_bind_groups[layer_idx].as_ref().unwrap())
    }

    /// Uploads input data to the GPU.
    pub fn upload_input(&self, queue: &wgpu::Queue, data: &[f32]) -> ArkanResult<()> {
        let input = self
            .input
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Input buffer not allocated"))?;

        if data.len() > input.num_elements() {
            return Err(ArkanError::shape_mismatch(
                &input.shape,
                &[data.len() / self.in_dim, self.in_dim],
            ));
        }

        input.update(queue, data);
        Ok(())
    }

    /// Downloads output data from the GPU.
    pub fn download_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: usize,
    ) -> ArkanResult<Vec<f32>> {
        let output = self
            .output
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Output buffer not allocated"))?;

        // Download only the used portion
        let full_data = output.download(device, queue)?;
        let used_elements = batch_size * self.out_dim;
        Ok(full_data[..used_elements].to_vec())
    }

    /// Returns the input buffer reference.
    pub fn input_buffer(&self) -> Option<&wgpu::Buffer> {
        self.input.as_ref().map(|t| &t.buffer)
    }

    /// Returns the output buffer reference.
    pub fn output_buffer(&self) -> Option<&wgpu::Buffer> {
        self.output.as_ref().map(|t| &t.buffer)
    }
}

impl std::fmt::Debug for GpuWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuWorkspace")
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("max_batch", &self.max_batch)
            .field("generation", &self.generation)
            .field("has_input", &self.input.is_some())
            .field("has_output", &self.output.is_some())
            .field("num_intermediates", &self.intermediates.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    // GPU tests require actual GPU, run with: cargo test --features gpu -- --ignored
}
