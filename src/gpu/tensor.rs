//! GPU Tensor abstraction with upload/download helpers.
//!
//! This module provides [`GpuTensor`], a GPU-resident tensor that wraps
//! a wgpu buffer with shape metadata.

use crate::error::{ArkanError, ArkanResult};
use crate::gpu::{exceeds_vram_limit, MAX_VRAM_ALLOC};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Infers a logical shape for `got` from data length and expected shape.
///
/// If shape has >1 dimensions, computes tail = product(shape[1..]).
/// If data.len() is divisible by tail, returns [data.len()/tail, shape[1..]].
/// Otherwise returns [data.len()] as flat shape.
fn infer_got_shape(data_len: usize, expected_shape: &[usize]) -> Vec<usize> {
    if expected_shape.len() > 1 {
        let tail: usize = expected_shape[1..].iter().product();
        if tail > 0 && data_len % tail == 0 {
            let mut got_shape = vec![data_len / tail];
            got_shape.extend_from_slice(&expected_shape[1..]);
            return got_shape;
        }
    }
    vec![data_len]
}

/// A GPU-resident tensor with shape metadata.
///
/// `GpuTensor` owns a wgpu buffer and stores shape information for
/// validation and debugging. It provides helpers for uploading data
/// to and downloading data from the GPU.
///
/// # Memory Layout
///
/// Data is stored in row-major order (C-style) matching CPU tensor layout.
/// For example, a tensor with shape `[batch, features]` has data laid out
/// as `[batch0_feat0, batch0_feat1, ..., batch1_feat0, ...]`.
///
/// # Example
///
/// ```rust,ignore
/// use arkan::gpu::GpuTensor;
///
/// // Create a tensor from CPU data
/// let data = vec![1.0f32, 2.0, 3.0, 4.0];
/// let tensor = GpuTensor::upload(&device, &queue, &data, vec![2, 2])?;
///
/// // Download data back to CPU
/// let result = tensor.download(&device, &queue)?;
/// assert_eq!(result, data);
/// ```
pub struct GpuTensor {
    /// The underlying wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// Shape of the tensor (e.g., [batch, features]).
    pub shape: Vec<usize>,
    /// Total capacity in bytes (may be larger than data size for alignment).
    pub capacity_bytes: u64,
}

impl GpuTensor {
    /// Creates a new GPU tensor by uploading CPU data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the buffer on.
    /// * `queue` - The wgpu queue for data upload.
    /// * `data` - The f32 data to upload.
    /// * `shape` - The logical shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `GpuTensor` with the uploaded data, or an error if allocation exceeds VRAM limits.
    ///
    /// # Errors
    ///
    /// Returns `ArkanError::BatchTooLarge` if the data size exceeds `MAX_VRAM_ALLOC`.
    pub fn upload(device: &wgpu::Device, _queue: &wgpu::Queue, data: &[f32], shape: Vec<usize>) -> ArkanResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            // Infer logical shape for got: e.g., expected [32, 64], got [31, 64]
            let got_shape = infer_got_shape(data.len(), &shape);
            return Err(ArkanError::ShapeMismatch {
                expected: shape,
                got: got_shape,
            });
        }

        let size_bytes = (data.len() * std::mem::size_of::<f32>()) as u64;

        if exceeds_vram_limit(size_bytes) {
            return Err(ArkanError::batch_too_large(
                data.len(),
                (MAX_VRAM_ALLOC / std::mem::size_of::<f32>() as u64) as usize,
            ));
        }

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuTensor"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(Self {
            buffer,
            shape,
            capacity_bytes: size_bytes,
        })
    }

    /// Creates a new GPU tensor with uninitialized data.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device to create the buffer on.
    /// * `shape` - The logical shape of the tensor.
    /// * `usage` - Additional buffer usage flags.
    ///
    /// # Returns
    ///
    /// A new `GpuTensor` with uninitialized data, or an error if allocation fails.
    pub fn uninit(
        device: &wgpu::Device,
        shape: Vec<usize>,
        usage: wgpu::BufferUsages,
    ) -> ArkanResult<Self> {
        let num_elements: usize = shape.iter().product();
        let size_bytes = (num_elements * std::mem::size_of::<f32>()) as u64;

        if exceeds_vram_limit(size_bytes) {
            return Err(ArkanError::batch_too_large(
                num_elements,
                (MAX_VRAM_ALLOC / std::mem::size_of::<f32>() as u64) as usize,
            ));
        }

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor (uninit)"),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | usage,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
            shape,
            capacity_bytes: size_bytes,
        })
    }

    /// Creates a GPU tensor for use as storage (read-only in shaders).
    ///
    /// # Errors
    ///
    /// Returns `ArkanError::BatchTooLarge` if the data size exceeds `MAX_VRAM_ALLOC`.
    pub fn storage_read(device: &wgpu::Device, data: &[f32], shape: Vec<usize>) -> ArkanResult<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            // Infer logical shape for got
            let got_shape = infer_got_shape(data.len(), &shape);
            return Err(ArkanError::ShapeMismatch {
                expected: shape,
                got: got_shape,
            });
        }

        let size_bytes = (data.len() * std::mem::size_of::<f32>()) as u64;

        if exceeds_vram_limit(size_bytes) {
            return Err(ArkanError::batch_too_large(
                data.len(),
                (MAX_VRAM_ALLOC / std::mem::size_of::<f32>() as u64) as usize,
            ));
        }

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuTensor (storage read)"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(Self {
            buffer,
            shape,
            capacity_bytes: size_bytes,
        })
    }

    /// Creates a GPU tensor for use as storage (read-write in shaders).
    pub fn storage_read_write(device: &wgpu::Device, shape: Vec<usize>) -> ArkanResult<Self> {
        Self::uninit(device, shape, wgpu::BufferUsages::empty())
    }

    /// Downloads tensor data from GPU to CPU.
    ///
    /// This operation is synchronous and will block until the data transfer
    /// is complete.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `queue` - The wgpu queue for command submission.
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` containing the tensor data, or an error if download fails.
    pub fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> ArkanResult<Vec<f32>> {
        let num_elements: usize = self.shape.iter().product();
        let size_bytes = (num_elements * std::mem::size_of::<f32>()) as u64;

        // Create staging buffer for readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor staging (download)"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuTensor download encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for GPU to finish
        device.poll(wgpu::Maintain::Wait);

        // Check mapping result
        rx.recv()
            .map_err(|e| ArkanError::buffer(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| ArkanError::buffer(format!("Buffer mapping failed: {:?}", e)))?;

        // Read data
        let data = {
            let mapped = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(&mapped).to_vec()
        };

        staging_buffer.unmap();

        Ok(data)
    }

    /// Downloads tensor data asynchronously using a callback.
    ///
    /// This is more efficient than `download` when you don't need to block.
    pub fn download_async<F>(&self, device: &wgpu::Device, queue: &wgpu::Queue, callback: F)
    where
        F: FnOnce(ArkanResult<Vec<f32>>) + Send + 'static,
    {
        let num_elements: usize = self.shape.iter().product();
        let size_bytes = (num_elements * std::mem::size_of::<f32>()) as u64;

        // Create staging buffer
        let staging_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor staging (download async)"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Copy to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuTensor download encoder (async)"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        // Map asynchronously
        let staging_clone = Arc::clone(&staging_buffer);
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let data = match result {
                Ok(()) => {
                    let mapped = staging_clone.slice(..).get_mapped_range();
                    let data: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
                    drop(mapped);
                    staging_clone.unmap();
                    Ok(data)
                }
                Err(e) => Err(ArkanError::buffer(format!("Buffer mapping failed: {:?}", e))),
            };
            callback(data);
        });
    }

    /// Updates the tensor data on GPU.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue for data upload.
    /// * `data` - New data to upload. Can be smaller than tensor capacity.
    ///
    /// # Panics
    ///
    /// Panics if the data size exceeds the tensor capacity.
    pub fn update(&self, queue: &wgpu::Queue, data: &[f32]) {
        let capacity = self.capacity_bytes as usize / std::mem::size_of::<f32>();
        assert!(
            data.len() <= capacity,
            "Data size {} exceeds tensor capacity {}",
            data.len(),
            capacity
        );
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// Returns the total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the size in bytes.
    pub fn size_bytes(&self) -> u64 {
        (self.num_elements() * std::mem::size_of::<f32>()) as u64
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Validates that the tensor has the expected shape.
    pub fn validate_shape(&self, expected: &[usize]) -> ArkanResult<()> {
        if self.shape != expected {
            return Err(ArkanError::shape_mismatch(expected, &self.shape));
        }
        Ok(())
    }
}

impl std::fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape)
            .field("capacity_bytes", &self.capacity_bytes)
            .field("num_elements", &self.num_elements())
            .finish()
    }
}

/// A borrowed view into a GPU tensor's buffer.
///
/// `GpuTensorView` provides zero-copy access to a portion of a GPU buffer,
/// useful for passing sub-tensors to compute pipelines without creating
/// new allocations.
///
/// # Example
///
/// ```rust,ignore
/// use arkan::gpu::{GpuTensor, GpuTensorView};
///
/// let tensor = GpuTensor::upload(&device, &queue, &data, vec![batch, features])?;
/// let view = tensor.view();
///
/// // Use view in bind group
/// let binding = view.as_entire_binding();
/// ```
#[derive(Debug)]
pub struct GpuTensorView<'a> {
    /// Reference to the underlying buffer.
    buffer: &'a wgpu::Buffer,
    /// Byte offset into the buffer.
    offset: u64,
    /// Size in bytes (None = entire buffer from offset).
    size: Option<u64>,
    /// Logical shape of this view.
    shape: Vec<usize>,
}

impl<'a> GpuTensorView<'a> {
    /// Creates a view of the entire tensor.
    pub fn new(tensor: &'a GpuTensor) -> Self {
        Self {
            buffer: &tensor.buffer,
            offset: 0,
            size: None,
            shape: tensor.shape.clone(),
        }
    }

    /// Creates a view with a specific byte range.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The source tensor.
    /// * `offset` - Byte offset into the buffer.
    /// * `size` - Size in bytes (None = rest of buffer).
    /// * `shape` - Logical shape of this view.
    pub fn slice(
        tensor: &'a GpuTensor,
        offset: u64,
        size: Option<u64>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            buffer: &tensor.buffer,
            offset,
            size,
            shape,
        }
    }

    /// Returns the buffer reference.
    pub fn buffer(&self) -> &'a wgpu::Buffer {
        self.buffer
    }

    /// Returns the byte offset.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the size in bytes (None = rest of buffer).
    pub fn size(&self) -> Option<u64> {
        self.size
    }

    /// Returns the logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Creates a wgpu binding for the entire view.
    pub fn as_entire_binding(&self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: self.size.map(|s| std::num::NonZeroU64::new(s).unwrap()),
        })
    }

    /// Creates a buffer slice for this view.
    pub fn as_buffer_slice(&self) -> wgpu::BufferSlice<'a> {
        match self.size {
            Some(size) => self.buffer.slice(self.offset..self.offset + size),
            None => self.buffer.slice(self.offset..),
        }
    }
}

impl GpuTensor {
    /// Creates a view of the entire tensor.
    pub fn view(&self) -> GpuTensorView<'_> {
        GpuTensorView::new(self)
    }

    /// Creates a view with a specific byte range.
    pub fn view_slice(&self, offset: u64, size: Option<u64>, shape: Vec<usize>) -> GpuTensorView<'_> {
        GpuTensorView::slice(self, offset, size, shape)
    }
}

#[cfg(test)]
mod tests {
    // Tests require GPU device, run with: cargo test --features gpu -- --ignored
}
