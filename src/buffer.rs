//! Aligned buffers and workspace for zero-allocation inference.

use crate::config::KanConfig;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

/// Cache line size for alignment.
pub const CACHE_LINE: usize = 64;

/// 64-byte aligned buffer for SIMD operations.
#[repr(C)]
pub struct AlignedBuffer {
    ptr: NonNull<f32>,
    len: usize,
    capacity: usize,
}

// Safety: AlignedBuffer owns its data and doesn't share it
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Creates a new empty aligned buffer.
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    /// Creates a buffer with the specified capacity (in f32 elements).
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }

        let layout = Self::layout(capacity);
        // SAFETY: Layout is valid and non-zero sized
        let ptr = unsafe {
            let raw = alloc_zeroed(layout);
            if raw.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            NonNull::new_unchecked(raw as *mut f32)
        };

        Self {
            ptr,
            len: 0,
            capacity,
        }
    }

    /// Ensures capacity is at least `new_cap`.
    /// Does not shrink. Only grows if needed.
    #[inline]
    pub fn reserve(&mut self, new_cap: usize) {
        if new_cap <= self.capacity {
            return;
        }

        // Allocate new buffer
        let new_layout = Self::layout(new_cap);
        let new_ptr = unsafe {
            let raw = alloc_zeroed(new_layout);
            if raw.is_null() {
                std::alloc::handle_alloc_error(new_layout);
            }
            NonNull::new_unchecked(raw as *mut f32)
        };

        // Copy old data if any
        if self.capacity > 0 && self.len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
        }

        // Deallocate old buffer
        if self.capacity > 0 {
            let old_layout = Self::layout(self.capacity);
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, old_layout);
            }
        }

        self.ptr = new_ptr;
        self.capacity = new_cap;
    }

    /// Resizes the buffer, filling new elements with zero.
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        self.reserve(new_len);
        if new_len > self.len {
            // Zero new elements
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr().add(self.len), 0, new_len - self.len);
            }
        }
        self.len = new_len;
    }

    /// Clears the buffer (sets len to 0, doesn't deallocate).
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Fills with zeros.
    #[inline]
    pub fn zero(&mut self) {
        if self.len > 0 {
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len);
            }
        }
    }

    /// Current length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Current capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Is empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a slice of the buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Returns a mutable slice of the buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Raw pointer (for SIMD).
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    /// Raw mutable pointer (for SIMD).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }

    fn layout(capacity: usize) -> Layout {
        Layout::from_size_align(capacity * std::mem::size_of::<f32>(), CACHE_LINE)
            .expect("Invalid layout")
    }
}

impl Default for AlignedBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Self::layout(self.capacity);
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut new = Self::with_capacity(self.capacity);
        new.len = self.len;
        if self.len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new.ptr.as_ptr(), self.len);
            }
        }
        new
    }
}

impl std::ops::Index<usize> for AlignedBuffer {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len, "Index out of bounds");
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl std::ops::IndexMut<usize> for AlignedBuffer {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len, "Index out of bounds");
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

/// Preallocated workspace for zero-allocation forward/backward pass.
#[derive(Default)]
pub struct Workspace {
    /// Normalized inputs: [Batch, Input]
    pub z_buffer: AlignedBuffer,

    /// Basis function values: [Batch, Input, Basis]
    pub basis_values: AlignedBuffer,

    /// Grid indices: [Batch, Input]
    pub grid_indices: Vec<u32>,

    /// Intermediate layer outputs: [Batch, MaxHiddenDim]
    pub layer_output: AlignedBuffer,

    /// Previous layer output (for multi-layer): [Batch, MaxHiddenDim]
    pub layer_input: AlignedBuffer,

    // --- Backward pass buffers ---
    /// Basis gradients: [Batch, Input, Basis]
    pub basis_grads: AlignedBuffer,

    /// Output gradients: [Batch, Output]
    pub output_grads: AlignedBuffer,

    /// Weight gradients accumulator: [MaxWeights]
    pub weight_grads: AlignedBuffer,

    // --- Tracking ---
    /// Current batch capacity
    batch_capacity: usize,

    /// Max dimension across all layers
    max_dim: usize,
}

impl Workspace {
    /// Creates a new workspace for the given config.
    pub fn new(config: &KanConfig) -> Self {
        let mut ws = Self {
            z_buffer: AlignedBuffer::new(),
            basis_values: AlignedBuffer::new(),
            grid_indices: Vec::new(),
            layer_output: AlignedBuffer::new(),
            layer_input: AlignedBuffer::new(),
            basis_grads: AlignedBuffer::new(),
            output_grads: AlignedBuffer::new(),
            weight_grads: AlignedBuffer::new(),
            batch_capacity: 0,
            max_dim: 0,
        };

        // Pre-allocate for typical batch size
        ws.reserve(config.multithreading_threshold, config);
        ws
    }

    /// Ensures workspace has capacity for the given batch size.
    /// Only allocates if batch_size > current capacity.
    #[inline]
    pub fn reserve(&mut self, batch_size: usize, config: &KanConfig) {
        if batch_size <= self.batch_capacity {
            return;
        }

        let dims = config.layer_dims();
        let max_dim = *dims.iter().max().unwrap_or(&1);
        let basis = config.basis_size_aligned();
        let input_dim = config.input_dim;

        // z_buffer: [batch, input]
        self.z_buffer.reserve(batch_size * input_dim);

        // basis_values: [batch, input, basis]
        self.basis_values.reserve(batch_size * input_dim * basis);

        // grid_indices: [batch, input]
        self.grid_indices.reserve(batch_size * input_dim);
        if self.grid_indices.len() < batch_size * input_dim {
            self.grid_indices.resize(batch_size * input_dim, 0);
        }

        // layer buffers: [batch, max_dim]
        self.layer_output.reserve(batch_size * max_dim);
        self.layer_input.reserve(batch_size * max_dim);

        // basis_grads: same as basis_values
        self.basis_grads.reserve(batch_size * input_dim * basis);

        // output_grads: [batch, output]
        self.output_grads.reserve(batch_size * config.output_dim);

        // weight_grads: max layer weights
        let max_weights = dims
            .windows(2)
            .map(|w| w[0] * w[1] * basis)
            .max()
            .unwrap_or(0);
        self.weight_grads.reserve(max_weights);

        self.batch_capacity = batch_size;
        self.max_dim = max_dim;
    }

    /// Prepares workspace for a forward pass with the given batch size.
    #[inline]
    pub fn prepare_forward(&mut self, batch_size: usize, config: &KanConfig) {
        self.reserve(batch_size, config);

        let input_dim = config.input_dim;
        let basis = config.basis_size_aligned();

        self.z_buffer.resize(batch_size * input_dim);
        self.basis_values.resize(batch_size * input_dim * basis);
        self.grid_indices.resize(batch_size * input_dim, 0);
    }

    /// Current batch capacity.
    #[inline]
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }

    /// Asserts that workspace is properly sized for the batch.
    #[inline]
    pub fn assert_capacity(&self, batch_size: usize) {
        assert!(
            batch_size <= self.batch_capacity,
            "Workspace capacity {} < batch size {}",
            self.batch_capacity,
            batch_size
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_basic() {
        let mut buf = AlignedBuffer::with_capacity(100);
        assert_eq!(buf.capacity(), 100);
        assert_eq!(buf.len(), 0);

        buf.resize(50);
        assert_eq!(buf.len(), 50);

        // Check alignment
        assert_eq!(buf.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[test]
    fn test_aligned_buffer_grow() {
        let mut buf = AlignedBuffer::with_capacity(10);
        buf.resize(10);
        for i in 0..10 {
            buf[i] = i as f32;
        }

        // Grow
        buf.reserve(100);
        assert_eq!(buf.capacity(), 100);

        // Data preserved
        for i in 0..10 {
            assert_eq!(buf[i], i as f32);
        }
    }

    #[test]
    fn test_workspace_reserve() {
        let config = KanConfig::default_poker();
        let mut ws = Workspace::new(&config);

        // Initial capacity
        assert!(ws.batch_capacity() >= config.multithreading_threshold);

        // Reserve more
        ws.reserve(1024, &config);
        assert!(ws.batch_capacity() >= 1024);

        // No allocation on smaller batch
        let old_cap = ws.batch_capacity();
        ws.reserve(512, &config);
        assert_eq!(ws.batch_capacity(), old_cap);
    }

    #[test]
    fn test_workspace_prepare_forward() {
        let config = KanConfig::default_poker();
        let mut ws = Workspace::new(&config);

        ws.prepare_forward(64, &config);

        assert_eq!(ws.z_buffer.len(), 64 * 21);
        assert_eq!(ws.grid_indices.len(), 64 * 21);
    }
}
