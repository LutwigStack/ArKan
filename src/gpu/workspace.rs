//! GPU Workspace with dynamic buffer management.
//!
//! This module provides [`GpuWorkspace`] which manages input/output buffers
//! for GPU operations with automatic resizing.

use crate::error::{ArkanError, ArkanResult};
use crate::gpu::{GpuTensor, DEFAULT_MAX_VRAM_ALLOC};

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
/// # Training Buffers
///
/// For backward pass, the workspace holds:
/// - `z_values`: Saved normalized inputs [batch, in_dim] per layer
/// - `span_indices`: Saved span indices [batch, in_dim] per layer
/// - `grad_output`: Gradient of loss w.r.t. output [batch, out_dim]
/// - `grad_input`: Gradient w.r.t. input for backprop [batch, in_dim]
///
/// # Example
///
/// ```rust,no_run
/// use arkan::gpu::{GpuWorkspace, WgpuBackend, WgpuOptions};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let backend = WgpuBackend::init(WgpuOptions::default())?;
/// let mut workspace = GpuWorkspace::new(&backend.device, 64, 21, 64)?;
///
/// // Resize for different batch size
/// workspace.ensure_capacity(&backend.device, 128)?;
/// # Ok(())
/// # }
/// ```
pub struct GpuWorkspace {
    /// Input buffer [batch, in_dim].
    pub input: Option<GpuTensor>,
    /// Output buffer [batch, out_dim].
    pub output: Option<GpuTensor>,

    /// Intermediate buffers for multi-layer networks.
    pub intermediates: Vec<GpuTensor>,

    // === Training buffers ===
    pub(crate) training_batch: Option<usize>,
    /// Saved normalized inputs per layer [layer][batch * in_dim].
    pub z_values: Vec<GpuTensor>,
    /// Saved span indices per layer [layer][batch * in_dim].
    pub span_indices: Vec<wgpu::Buffer>,
    /// Gradient of loss w.r.t. network output [batch, out_dim].
    pub grad_output: Option<GpuTensor>,
    /// Gradient w.r.t. input (for layer backprop) [batch, max_dim].
    pub grad_input: Option<GpuTensor>,
    /// Optional inverse-std storage for direct kernel users. `GpuNetwork` backward
    /// uses the CPU-derived normalization owned by each `GpuLayer`.
    pub std_inv: Vec<GpuTensor>,

    /// Gradient of weights per layer `(layer, out_dim * in_dim * basis_padded)`.
    pub grad_weights: Vec<GpuTensor>,
    /// Gradient of biases per layer `(layer, out_dim)`.
    pub grad_bias: Vec<GpuTensor>,

    /// Input dimension (fixed).
    pub in_dim: usize,
    /// Output dimension (fixed).
    pub out_dim: usize,
    /// Current maximum batch capacity.
    pub max_batch: usize,
    /// Maximum VRAM allocation per buffer in bytes.
    /// Configurable via `WgpuBackend::max_vram_alloc()`.
    max_vram_alloc: u64,

    /// Cached bind group for single-layer (input -> output).
    cached_bind_group: Option<wgpu::BindGroup>,
    /// Cached bind groups for multi-layer, indexed by (in_buffer_type, out_buffer_type).
    /// Types: 0 = input, 1 = output, 2+ = intermediate[idx-2]
    cached_layer_bind_groups: Vec<Option<wgpu::BindGroup>>,
    /// Cached bind groups for training forward pass.
    cached_training_bind_groups: Vec<Option<wgpu::BindGroup>>,
    /// Cached bind groups for backward pass.
    cached_backward_bind_groups: Vec<Option<wgpu::BindGroup>>,
}

impl GpuWorkspace {
    /// Check legacy public tensor metadata before preparation indexes shapes.
    /// Hidden widths may differ from the next model: preparation reallocates them.
    pub(crate) fn validate_metadata(&self) -> ArkanResult<()> {
        for tensor in self
            .input
            .iter()
            .chain(&self.output)
            .chain(&self.intermediates)
            .chain(&self.z_values)
            .chain(&self.grad_output)
            .chain(&self.grad_input)
        {
            Self::validate_tensor(tensor, 2)?;
        }
        for tensor in self
            .std_inv
            .iter()
            .chain(&self.grad_weights)
            .chain(&self.grad_bias)
        {
            Self::validate_tensor(tensor, 1)?;
        }
        match (&self.input, &self.output) {
            (None, None) if self.max_batch == 0 => {}
            (Some(input), Some(output))
                if input.shape[1] == self.in_dim
                    && output.shape[1] == self.out_dim
                    && input.shape[0] >= self.max_batch
                    && output.shape[0] >= self.max_batch => {}
            _ => return Err(ArkanError::validation("Invalid GPU workspace I/O capacity")),
        }
        for (values, spans) in self.z_values.iter().zip(&self.span_indices) {
            if spans.size() < values.size_bytes() {
                return Err(ArkanError::validation("Invalid GPU saved-span capacity"));
            }
        }
        Ok(())
    }

    fn validate_tensor(tensor: &GpuTensor, rank: usize) -> ArkanResult<()> {
        let bytes = tensor
            .shape
            .iter()
            .try_fold(4u64, |bytes, &dim| bytes.checked_mul(dim as u64));
        if tensor.shape.len() != rank
            || bytes.is_none()
            || bytes.unwrap_or(u64::MAX) > tensor.capacity_bytes
            || tensor.capacity_bytes > tensor.buffer.size()
        {
            return Err(ArkanError::validation(
                "Invalid GPU tensor shape or byte capacity",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_io_capacity(&self, batch_size: usize) -> ArkanResult<()> {
        self.validate_metadata()?;
        if self.input.as_ref().is_none_or(|t| t.shape[0] < batch_size)
            || self.output.as_ref().is_none_or(|t| t.shape[0] < batch_size)
        {
            return Err(ArkanError::validation(
                "GPU workspace is not prepared for this batch",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_training_capacity(
        &self,
        batch_size: usize,
        layer_dims: &[usize],
    ) -> ArkanResult<()> {
        self.validate_io_capacity(batch_size)?;
        let max_dim = layer_dims.iter().copied().max().unwrap_or(self.in_dim);
        if self.z_values.len() != layer_dims.len().saturating_sub(1)
            || self.span_indices.len() != self.z_values.len()
            || self
                .z_values
                .iter()
                .zip(layer_dims)
                .any(|(t, &width)| t.shape[0] < batch_size || t.shape[1] != width)
            || [&self.grad_input, &self.grad_output].iter().any(|tensor| {
                tensor
                    .as_ref()
                    .is_none_or(|t| t.shape[0] < batch_size || t.shape[1] < max_dim)
            })
        {
            return Err(ArkanError::validation(
                "GPU training buffers are not prepared for this batch",
            ));
        }
        Ok(())
    }

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
    ///
    /// # Note
    ///
    /// Uses default VRAM limit (2GB). For custom limits, use `new_with_limit()`.
    pub fn new(
        device: &wgpu::Device,
        max_batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> ArkanResult<Self> {
        Self::new_with_limit(device, max_batch, in_dim, out_dim, DEFAULT_MAX_VRAM_ALLOC)
    }

    /// Creates a new workspace with custom VRAM limit.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `max_batch` - Maximum batch size to support.
    /// * `in_dim` - Input dimension.
    /// * `out_dim` - Output dimension.
    /// * `max_vram_alloc` - Maximum VRAM per buffer in bytes.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use arkan::gpu::{GpuWorkspace, WgpuBackend, WgpuOptions};
    ///
    /// let backend = WgpuBackend::init(WgpuOptions::with_max_vram(8))?;
    /// let workspace = GpuWorkspace::new_with_limit(
    ///     &backend.device, 64, 21, 64,
    ///     backend.max_vram_alloc()
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_with_limit(
        device: &wgpu::Device,
        max_batch: usize,
        in_dim: usize,
        out_dim: usize,
        max_vram_alloc: u64,
    ) -> ArkanResult<Self> {
        GpuTensor::allocation_bytes(device, &[max_batch, in_dim], Some(max_vram_alloc))?;
        GpuTensor::allocation_bytes(device, &[max_batch, out_dim], Some(max_vram_alloc))?;

        let input = GpuTensor::uninit_with_limit(
            device,
            vec![max_batch, in_dim],
            wgpu::BufferUsages::empty(),
            Some(max_vram_alloc),
        )?;
        let output = GpuTensor::uninit_with_limit(
            device,
            vec![max_batch, out_dim],
            wgpu::BufferUsages::empty(),
            Some(max_vram_alloc),
        )?;

        Ok(Self {
            input: Some(input),
            output: Some(output),
            intermediates: Vec::new(),
            training_batch: None,
            z_values: Vec::new(),
            span_indices: Vec::new(),
            grad_output: None,
            grad_input: None,
            std_inv: Vec::new(),
            grad_weights: Vec::new(),
            grad_bias: Vec::new(),
            in_dim,
            out_dim,
            max_batch,
            max_vram_alloc,
            cached_bind_group: None,
            cached_layer_bind_groups: Vec::new(),
            cached_training_bind_groups: Vec::new(),
            cached_backward_bind_groups: Vec::new(),
        })
    }

    /// Creates an empty workspace (lazy allocation).
    ///
    /// Uses default VRAM limit. For custom limits, use `empty_with_limit()`.
    pub fn empty(in_dim: usize, out_dim: usize) -> Self {
        Self::empty_with_limit(in_dim, out_dim, DEFAULT_MAX_VRAM_ALLOC)
    }

    /// Creates an empty workspace with custom VRAM limit.
    pub fn empty_with_limit(in_dim: usize, out_dim: usize, max_vram_alloc: u64) -> Self {
        Self {
            input: None,
            output: None,
            intermediates: Vec::new(),
            training_batch: None,
            z_values: Vec::new(),
            span_indices: Vec::new(),
            grad_output: None,
            grad_input: None,
            std_inv: Vec::new(),
            grad_weights: Vec::new(),
            grad_bias: Vec::new(),
            in_dim,
            out_dim,
            max_batch: 0,
            max_vram_alloc,
            cached_bind_group: None,
            cached_layer_bind_groups: Vec::new(),
            cached_training_bind_groups: Vec::new(),
            cached_backward_bind_groups: Vec::new(),
        }
    }

    /// Returns the maximum VRAM allocation per buffer.
    pub fn max_vram_alloc(&self) -> u64 {
        self.max_vram_alloc
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
        self.validate_metadata()?;
        if batch_size <= self.max_batch && self.input.is_some() {
            return Ok(false); // No resize needed
        }

        // Calculate new capacity with some headroom (1.5x requested)
        let new_capacity = batch_size
            .checked_add(batch_size / 2)
            .ok_or_else(|| ArkanError::buffer("GPU batch capacity overflow"))?;
        GpuTensor::allocation_bytes(
            device,
            &[new_capacity, self.in_dim],
            Some(self.max_vram_alloc),
        )?;
        GpuTensor::allocation_bytes(
            device,
            &[new_capacity, self.out_dim],
            Some(self.max_vram_alloc),
        )?;

        // Allocate new buffers
        self.input = Some(GpuTensor::uninit_with_limit(
            device,
            vec![new_capacity, self.in_dim],
            wgpu::BufferUsages::empty(),
            Some(self.max_vram_alloc),
        )?);
        self.output = Some(GpuTensor::uninit_with_limit(
            device,
            vec![new_capacity, self.out_dim],
            wgpu::BufferUsages::empty(),
            Some(self.max_vram_alloc),
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
        self.validate_metadata()?;
        // Need n-1 intermediate buffers for n layers
        let needed = layer_dims.len().saturating_sub(2);

        // Check if we need to resize existing intermediates
        let needs_resize = self.intermediates.len() != needed
            || self.intermediates.iter().enumerate().any(|(i, t)| {
                i < needed && (t.shape[0] < batch_size || t.shape[1] != layer_dims[i + 1])
            });

        if needs_resize {
            for &dim in layer_dims.iter().skip(1).take(needed) {
                GpuTensor::allocation_bytes(device, &[batch_size, dim], Some(self.max_vram_alloc))?;
            }
            self.intermediates.clear();

            for i in 0..needed {
                let dim = layer_dims[i + 1]; // Output dim of layer i
                let tensor = GpuTensor::uninit_with_limit(
                    device,
                    vec![batch_size, dim],
                    wgpu::BufferUsages::empty(),
                    Some(self.max_vram_alloc),
                )?;
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
        self.cached_training_bind_groups.clear();
        self.cached_backward_bind_groups.clear();
    }

    /// Prepares training buffers for backward pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `layer_dims` - Dimensions of each layer (in_dim for layer 0, then out_dims).
    /// * `batch_size` - Batch size.
    pub fn prepare_training(
        &mut self,
        device: &wgpu::Device,
        layer_dims: &[usize],
        batch_size: usize,
    ) -> ArkanResult<()> {
        self.validate_metadata()?;
        self.training_batch = None;
        let num_layers = layer_dims.len().saturating_sub(1);
        for &dim in layer_dims {
            GpuTensor::allocation_bytes(device, &[batch_size, dim], Some(self.max_vram_alloc))?;
        }

        // Allocate z_values and span_indices for each layer
        if self.z_values.len() != num_layers
            || self.span_indices.len() != num_layers
            || self
                .z_values
                .iter()
                .zip(layer_dims)
                .any(|(z, &dim)| z.shape[0] < batch_size || z.shape[1] != dim)
        {
            self.z_values.clear();
            self.span_indices.clear();

            for (i, &in_dim) in layer_dims.iter().take(num_layers).enumerate() {
                // z_values: normalized inputs
                let z = GpuTensor::uninit_with_limit(
                    device,
                    vec![batch_size, in_dim],
                    wgpu::BufferUsages::empty(),
                    Some(self.max_vram_alloc),
                )?;
                self.z_values.push(z);

                // span_indices: u32 indices
                let span_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("SpanIndices layer {}", i)),
                    size: GpuTensor::allocation_bytes(
                        device,
                        &[batch_size, in_dim],
                        Some(self.max_vram_alloc),
                    )?,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.span_indices.push(span_buf);
            }

            self.invalidate_cache();
        }

        // Max dim across all layers for gradient buffer reuse during backward pass
        // Backward pass uses grad_output to propagate gradients between layers,
        // so it needs to hold gradients of any intermediate layer size
        let max_dim = layer_dims.iter().max().copied().unwrap_or(self.in_dim);

        // Allocate grad_output if needed (uses max_dim for intermediate gradient propagation)
        // Use is_none_or to avoid unwrap on Option
        let needs_grad_output_realloc = self
            .grad_output
            .as_ref()
            .is_none_or(|g| g.shape[0] < batch_size || g.shape[1] < max_dim);
        if needs_grad_output_realloc {
            self.grad_output = Some(GpuTensor::uninit_with_limit(
                device,
                vec![batch_size, max_dim],
                wgpu::BufferUsages::empty(),
                Some(self.max_vram_alloc),
            )?);
            self.invalidate_cache();
        }

        // Allocate grad_input (max dim across layers for reuse)
        let needs_grad_input_realloc = self
            .grad_input
            .as_ref()
            .is_none_or(|g| g.shape[0] < batch_size || g.shape[1] < max_dim);
        if needs_grad_input_realloc {
            self.grad_input = Some(GpuTensor::uninit_with_limit(
                device,
                vec![batch_size, max_dim],
                wgpu::BufferUsages::empty(),
                Some(self.max_vram_alloc),
            )?);
            self.invalidate_cache();
        }

        Ok(())
    }

    /// Prepares gradient buffers for backward pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `layer_specs` - For each layer: (in_dim, out_dim, basis_padded).
    pub fn prepare_grad_buffers(
        &mut self,
        device: &wgpu::Device,
        layer_specs: &[(usize, usize, usize)],
    ) -> ArkanResult<()> {
        self.validate_metadata()?;
        let num_layers = layer_specs.len();
        for &(input, output, basis) in layer_specs {
            GpuTensor::allocation_bytes(
                device,
                &[output, input, basis],
                Some(self.max_vram_alloc),
            )?;
            GpuTensor::allocation_bytes(device, &[output], Some(self.max_vram_alloc))?;
        }

        // Check if we need to reallocate
        let needs_realloc = self.grad_bias.len() != num_layers
            || self
                .grad_bias
                .iter()
                .zip(layer_specs)
                .any(|(gb, &(_, out, _))| gb.num_elements() != out)
            || self.grad_weights.len() != num_layers
            || self
                .grad_weights
                .iter()
                .zip(layer_specs.iter())
                .any(|(gw, &(in_d, out_d, basis))| gw.num_elements() != out_d * in_d * basis);

        if needs_realloc {
            self.grad_weights.clear();
            self.grad_bias.clear();

            for &(in_dim, out_dim, basis_padded) in layer_specs.iter() {
                let weight_size = out_dim * in_dim * basis_padded;
                let gw = GpuTensor::uninit_with_limit(
                    device,
                    vec![weight_size],
                    wgpu::BufferUsages::empty(),
                    Some(self.max_vram_alloc),
                )?;
                self.grad_weights.push(gw);

                let gb = GpuTensor::uninit_with_limit(
                    device,
                    vec![out_dim],
                    wgpu::BufferUsages::empty(),
                    Some(self.max_vram_alloc),
                )?;
                self.grad_bias.push(gb);
            }

            self.invalidate_cache();
        }

        Ok(())
    }

    /// Clears gradient buffers to zero (must be called before backward pass).
    pub fn zero_grad_buffers(&self, queue: &wgpu::Queue) {
        for gw in &self.grad_weights {
            let zeros = vec![0.0f32; gw.num_elements()];
            gw.update(queue, &zeros);
        }
        for gb in &self.grad_bias {
            let zeros = vec![0.0f32; gb.num_elements()];
            gb.update(queue, &zeros);
        }
    }

    /// Weight backward completely writes the validated extent; bias backward accumulates.
    pub(crate) fn zero_bias_grad_buffers(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Clear bias gradients"),
        });
        for gb in &self.grad_bias {
            encoder.clear_buffer(&gb.buffer, 0, None);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Returns references to gradient buffers for GPU optimizer.
    ///
    /// Returns Vec of (&grad_weights_buffer, &grad_bias_buffer) for each layer.
    ///
    /// # Panics
    ///
    /// Panics if gradient buffers are not allocated. Call `prepare_grad_buffers` first.
    pub fn get_layer_grad_buffers(&self) -> Vec<(&wgpu::Buffer, &wgpu::Buffer)> {
        assert!(
            !self.grad_weights.is_empty(),
            "Gradient buffers not allocated. Call prepare_grad_buffers first."
        );
        self.grad_weights
            .iter()
            .zip(self.grad_bias.iter())
            .map(|(gw, gb)| (&gw.buffer, &gb.buffer))
            .collect()
    }

    /// Downloads weight gradients for a specific layer.
    pub fn download_grad_weights(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer_idx: usize,
    ) -> ArkanResult<Vec<f32>> {
        let gw = self.grad_weights.get(layer_idx).ok_or_else(|| {
            ArkanError::buffer(format!("grad_weights[{}] not allocated", layer_idx))
        })?;
        gw.download(device, queue)
    }

    /// Downloads bias gradients for a specific layer.
    pub fn download_grad_bias(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer_idx: usize,
    ) -> ArkanResult<Vec<f32>> {
        let gb = self
            .grad_bias
            .get(layer_idx)
            .ok_or_else(|| ArkanError::buffer(format!("grad_bias[{}] not allocated", layer_idx)))?;
        gb.download(device, queue)
    }

    /// Sets the 1/std values for input normalization (used in backward pass).
    pub fn set_std_inv(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        layer_std_inv: &[Vec<f32>],
    ) -> ArkanResult<()> {
        for values in layer_std_inv {
            GpuTensor::allocation_bytes(device, &[values.len()], Some(self.max_vram_alloc))?;
        }
        self.std_inv.clear();

        for std_inv_vals in layer_std_inv {
            let tensor = GpuTensor::upload_with_limit(
                device,
                std_inv_vals,
                vec![std_inv_vals.len()],
                Some(self.max_vram_alloc),
            )?;
            self.std_inv.push(tensor);
        }

        self.invalidate_cache();
        Ok(())
    }

    /// Uploads gradient of loss w.r.t. output.
    pub fn upload_grad_output(&self, queue: &wgpu::Queue, grad: &[f32]) -> ArkanResult<()> {
        let grad_output = self
            .grad_output
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("grad_output not allocated"))?;

        grad_output.update(queue, grad);
        Ok(())
    }

    /// Downloads gradient w.r.t. input (after backward pass).
    pub fn download_grad_input(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: usize,
        in_dim: usize,
    ) -> ArkanResult<Vec<f32>> {
        let grad_input = self
            .grad_input
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("grad_input not allocated"))?;

        let full_data = grad_input.download(device, queue)?;
        let used = batch_size * in_dim;
        Ok(full_data[..used].to_vec())
    }

    /// Copies grad_input to grad_output on GPU (no CPU round-trip).
    ///
    /// This is used during backward pass to propagate gradients between layers
    /// without expensive GPU-to-CPU-to-GPU transfers.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue.
    /// * `batch_size` - Current batch size.
    /// * `dim` - Dimension of the gradient (in_dim of current layer).
    pub fn copy_grad_input_to_grad_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch_size: usize,
        dim: usize,
    ) -> ArkanResult<()> {
        let grad_input = self
            .grad_input
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("grad_input not allocated"))?;
        let grad_output = self
            .grad_output
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("grad_output not allocated"))?;

        let copy_size = (batch_size * dim * std::mem::size_of::<f32>()) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_grad_input_to_grad_output"),
        });

        encoder.copy_buffer_to_buffer(&grad_input.buffer, 0, &grad_output.buffer, 0, copy_size);

        queue.submit(std::iter::once(encoder.finish()));

        Ok(())
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

        self.cached_bind_group
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Failed to create cached bind group"))
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
    /// - Layer 0: input -> intermediate\[0\]
    /// - Layer i (middle): intermediate\[i-1\] -> intermediate\[i\]
    /// - Layer n-1: intermediate\[n-2\] -> output
    pub fn get_or_create_layer_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        layer_idx: usize,
        num_layers: usize,
    ) -> ArkanResult<&wgpu::BindGroup> {
        // Ensure cache vector is large enough
        if self.cached_layer_bind_groups.len() < num_layers {
            self.cached_layer_bind_groups
                .resize_with(num_layers, || None);
        }

        // Check if already cached
        if self.cached_layer_bind_groups[layer_idx].is_some() {
            return self.cached_layer_bind_groups[layer_idx]
                .as_ref()
                .ok_or_else(|| ArkanError::buffer("Cache inconsistency: expected bind group"));
        }

        // Determine input and output buffers
        let (input_buffer, output_buffer) = if layer_idx == 0 {
            // First layer: input from workspace.input, output to intermediate[0]
            (
                self.input
                    .as_ref()
                    .ok_or_else(|| ArkanError::buffer("No input buffer"))?
                    .buffer
                    .as_entire_binding(),
                self.intermediates
                    .first()
                    .ok_or_else(|| ArkanError::buffer("No intermediate buffer 0"))?
                    .buffer
                    .as_entire_binding(),
            )
        } else if layer_idx == num_layers - 1 {
            // Last layer: input from intermediate[n-2], output to workspace.output
            (
                self.intermediates
                    .get(layer_idx - 1)
                    .ok_or_else(|| {
                        ArkanError::buffer(format!("No intermediate buffer {}", layer_idx - 1))
                    })?
                    .buffer
                    .as_entire_binding(),
                self.output
                    .as_ref()
                    .ok_or_else(|| ArkanError::buffer("No output buffer"))?
                    .buffer
                    .as_entire_binding(),
            )
        } else {
            // Middle layer: intermediate[i-1] -> intermediate[i]
            (
                self.intermediates
                    .get(layer_idx - 1)
                    .ok_or_else(|| {
                        ArkanError::buffer(format!("No intermediate buffer {}", layer_idx - 1))
                    })?
                    .buffer
                    .as_entire_binding(),
                self.intermediates
                    .get(layer_idx)
                    .ok_or_else(|| {
                        ArkanError::buffer(format!("No intermediate buffer {}", layer_idx))
                    })?
                    .buffer
                    .as_entire_binding(),
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
        self.cached_layer_bind_groups[layer_idx]
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Failed to cache layer bind group"))
    }

    // ==================== Training Bind Groups ====================

    /// Gets or creates the training bind group for a single-layer network.
    ///
    /// Bindings: input, output, z_values, span_indices
    pub fn get_or_create_training_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        layer_idx: usize,
        _in_dim: usize,
        _batch_size: usize,
    ) -> ArkanResult<&wgpu::BindGroup> {
        // Ensure cache vector is large enough
        let num_layers = self.z_values.len().max(1);
        if self.cached_training_bind_groups.len() < num_layers {
            self.cached_training_bind_groups
                .resize_with(num_layers, || None);
        }

        // Check if already cached for this layer
        if self
            .cached_training_bind_groups
            .get(layer_idx)
            .is_some_and(|bg| bg.is_some())
        {
            return self.cached_training_bind_groups[layer_idx]
                .as_ref()
                .ok_or_else(|| {
                    ArkanError::buffer("Cache inconsistency: expected training bind group")
                });
        }

        // Get buffers
        let input = self
            .input
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Input buffer not allocated"))?;
        let output = self
            .output
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Output buffer not allocated"))?;
        let z_values = self.z_values.get(layer_idx).ok_or_else(|| {
            ArkanError::buffer(format!("z_values buffer {} not allocated", layer_idx))
        })?;
        let span_indices = self.span_indices.get(layer_idx).ok_or_else(|| {
            ArkanError::buffer(format!("span_indices buffer {} not allocated", layer_idx))
        })?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Training BindGroup Layer {}", layer_idx)),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: z_values.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: span_indices.as_entire_binding(),
                },
            ],
        });

        if layer_idx >= self.cached_training_bind_groups.len() {
            self.cached_training_bind_groups
                .resize_with(layer_idx + 1, || None);
        }
        self.cached_training_bind_groups[layer_idx] = Some(bind_group);
        self.cached_training_bind_groups[layer_idx]
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Failed to cache training bind group"))
    }

    /// Gets or creates the training bind group for a layer in multi-layer network.
    ///
    /// Bindings: input, output, z_values, span_indices
    pub fn get_or_create_training_layer_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        layer_idx: usize,
        num_layers: usize,
        _in_dim: usize,
        _batch_size: usize,
    ) -> ArkanResult<&wgpu::BindGroup> {
        // Ensure cache vector is large enough
        if self.cached_training_bind_groups.len() < num_layers {
            self.cached_training_bind_groups
                .resize_with(num_layers, || None);
        }

        // Check if already cached for this layer
        if self.cached_training_bind_groups[layer_idx].is_some() {
            return self.cached_training_bind_groups[layer_idx]
                .as_ref()
                .ok_or_else(|| {
                    ArkanError::buffer("Cache inconsistency: expected training layer bind group")
                });
        }

        // Determine input and output buffers (same routing as forward)
        let (input_buffer, output_buffer) = if layer_idx == 0 {
            (
                self.input
                    .as_ref()
                    .ok_or_else(|| ArkanError::buffer("No input buffer"))?
                    .buffer
                    .as_entire_binding(),
                self.intermediates
                    .first()
                    .ok_or_else(|| ArkanError::buffer("No intermediate buffer 0"))?
                    .buffer
                    .as_entire_binding(),
            )
        } else if layer_idx == num_layers - 1 {
            (
                self.intermediates
                    .get(layer_idx - 1)
                    .ok_or_else(|| {
                        ArkanError::buffer(format!("No intermediate buffer {}", layer_idx - 1))
                    })?
                    .buffer
                    .as_entire_binding(),
                self.output
                    .as_ref()
                    .ok_or_else(|| ArkanError::buffer("No output buffer"))?
                    .buffer
                    .as_entire_binding(),
            )
        } else {
            (
                self.intermediates
                    .get(layer_idx - 1)
                    .ok_or_else(|| {
                        ArkanError::buffer(format!("No intermediate buffer {}", layer_idx - 1))
                    })?
                    .buffer
                    .as_entire_binding(),
                self.intermediates
                    .get(layer_idx)
                    .ok_or_else(|| {
                        ArkanError::buffer(format!("No intermediate buffer {}", layer_idx))
                    })?
                    .buffer
                    .as_entire_binding(),
            )
        };

        // Get training buffers
        let z_values = self.z_values.get(layer_idx).ok_or_else(|| {
            ArkanError::buffer(format!("z_values buffer {} not allocated", layer_idx))
        })?;
        let span_indices = self.span_indices.get(layer_idx).ok_or_else(|| {
            ArkanError::buffer(format!("span_indices buffer {} not allocated", layer_idx))
        })?;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Training Layer {} BindGroup", layer_idx)),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: z_values.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: span_indices.as_entire_binding(),
                },
            ],
        });

        self.cached_training_bind_groups[layer_idx] = Some(bind_group);
        self.cached_training_bind_groups[layer_idx]
            .as_ref()
            .ok_or_else(|| ArkanError::buffer("Failed to cache training layer bind group"))
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
            .field("has_input", &self.input.is_some())
            .field("has_output", &self.output.is_some())
            .field("num_intermediates", &self.intermediates.len())
            .field("num_z_values", &self.z_values.len())
            .field("has_grad_output", &self.grad_output.is_some())
            .field("has_grad_input", &self.grad_input.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    // GPU tests require actual GPU, run with: cargo test --features gpu -- --ignored
}
