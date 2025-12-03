//! Aligned buffers and workspace for zero-allocation inference.
//!
//! This module provides two key types:
//!
//! - [`AlignedBuffer`] — 64-byte aligned buffer for SIMD operations
//! - [`Workspace`] — Preallocated buffers for forward/backward passes
//!
//! # Zero-Allocation Pattern
//!
//! ArKan achieves zero-allocation inference by preallocating all buffers
//! in the [`Workspace`]. Create it once and reuse:
//!
//! ```rust
//! use arkan::{KanConfig, KanNetwork, Workspace};
//!
//! let config = KanConfig::preset();
//! let network = KanNetwork::new(config.clone());
//!
//! // Allocate workspace once for max batch size
//! let mut workspace = network.create_workspace(64);
//!
//! // All subsequent calls are zero-allocation
//! let input = vec![0.5f32; config.input_dim];
//! let mut output = vec![0.0f32; config.output_dim];
//!
//! for _ in 0..1000 {
//!     network.forward_single(&input, &mut output, &mut workspace);
//! }
//! ```
//!
//! # Memory Alignment
//!
//! [`AlignedBuffer`] uses 64-byte alignment ([`CACHE_LINE`]) to ensure
//! optimal performance with AVX-512 SIMD instructions.

use crate::config::KanConfig;
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

/// Cache line size for memory alignment (64 bytes).
///
/// All [`AlignedBuffer`] allocations use this alignment for optimal
/// SIMD performance and cache behavior.
pub const CACHE_LINE: usize = 64;

/// 64-byte aligned buffer for SIMD operations.
///
/// This buffer guarantees 64-byte alignment, making it suitable for
/// AVX-512 instructions. It provides a `Vec<f32>`-like interface but
/// with cache-friendly memory layout.
///
/// # Example
///
/// ```rust
/// use arkan::AlignedBuffer;
///
/// let mut buf = AlignedBuffer::with_capacity(1024);
/// buf.resize(100);
/// buf.as_mut_slice()[0] = 1.0;
/// assert_eq!(buf[0], 1.0);
/// ```
///
/// # Safety
///
/// The buffer uses raw allocation with proper alignment. All unsafe
/// operations are encapsulated and the public API is safe.
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
        // SAFETY: layout is derived from a positive `capacity`, allocation handled via handle_alloc_error
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
        // SAFETY: new_layout is valid for `new_cap`, allocation failure handled
        let new_ptr = unsafe {
            let raw = alloc_zeroed(new_layout);
            if raw.is_null() {
                std::alloc::handle_alloc_error(new_layout);
            }
            NonNull::new_unchecked(raw as *mut f32)
        };

        // Copy old data if any
        if self.capacity > 0 && self.len > 0 {
            // SAFETY: source and destination are valid, non-overlapping, len=self.len
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
        }

        // Deallocate old buffer
        if self.capacity > 0 {
            let old_layout = Self::layout(self.capacity);
            // SAFETY: layout matches original allocation
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
            // SAFETY: destination is within allocated region; zero the tail
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
            // SAFETY: buffer is allocated and len > 0
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
            // SAFETY: ptr is valid for `len` contiguous elements
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Returns a mutable slice of the buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: ptr uniquely owned, valid for `len` contiguous elements
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
            // SAFETY: layout matches allocation, ptr is valid
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
            // SAFETY: source/dest are distinct allocations, len is within both
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
        // SAFETY: bounds checked above
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl std::ops::IndexMut<usize> for AlignedBuffer {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len, "Index out of bounds");
        // SAFETY: bounds checked above, unique access
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

#[cfg(feature = "serde")]
impl Serialize for AlignedBuffer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for AlignedBuffer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data: Vec<f32> = Vec::<f32>::deserialize(deserializer)?;
        let mut buf = AlignedBuffer::with_capacity(data.len());
        buf.resize(data.len());
        buf.as_mut_slice().copy_from_slice(&data);
        Ok(buf)
    }
}

/// Preallocated workspace for zero-allocation forward/backward passes.
///
/// The workspace holds all intermediate buffers needed during inference
/// and training. By preallocating these buffers, ArKan avoids heap
/// allocations in the hot path.
///
/// # Buffer Preparation
///
/// Before using the workspace, call the appropriate preparation method:
///
/// | Method | Use Case |
/// |--------|----------|
/// | [`reserve`](Self::reserve) | Ensure capacity for batch size |
/// | [`prepare_forward`](Self::prepare_forward) | Inference: resize z_buffer, basis_values, etc. |
/// | [`prepare_training`](Self::prepare_training) | Training: add history buffers for backward pass |
/// | [`prepare_grad_buffers`](Self::prepare_grad_buffers) | Training: allocate per-layer gradient vectors |
///
/// Typical flow:
/// ```text
/// network.create_workspace(batch_size)   // calls reserve() internally
///   \u2514\u2500> Inference: forward_batch()         // calls prepare_forward() automatically
///   \u2514\u2500> Training:  train_step()            // calls prepare_training() + prepare_grad_buffers()
/// ```
///
/// # Buffer Categories
///
/// **Forward buffers** (used during inference and training):
/// - `z_buffer`: Normalized inputs `[batch, input_dim]`
/// - `basis_values`: B-spline basis values `[batch, input_dim, basis_size]`
/// - `layer_output`, `layer_input`: Ping-pong buffers for layer activations
///
/// **Backward buffers** (training only):
/// - `layers_inputs`: Saved normalized inputs per layer for gradient computation
/// - `layers_grid_indices`: Saved spline segment indices per layer
/// - `staging_buffer`: Current layer's output gradient (ping-pong with `layer_grads`)
/// - `layer_grads`: Accumulated input gradients during backprop
///
/// **Gradient buffers** (training only):
/// - `weight_grads`: Per-layer weight gradients `[layer][weights.len()]`
/// - `bias_grads`: Per-layer bias gradients `[layer][bias.len()]`
/// - `grad_output`: Initial output gradient (dL/dy)
/// - `predictions_buffer`: Forward pass outputs before loss computation
///
/// # Usage
///
/// Create a workspace using [`KanNetwork::create_workspace`](crate::KanNetwork::create_workspace):
///
/// ```rust
/// use arkan::{KanConfig, KanNetwork};
///
/// let network = KanNetwork::new(KanConfig::preset());
/// let mut workspace = network.create_workspace(64);
///
/// // Reuse workspace for all calls
/// ```
///
/// # Thread Safety
///
/// Workspaces are NOT thread-safe. Each thread should have its own workspace.
/// The network itself can be shared (it's read-only during inference).
#[derive(Default)]
pub struct Workspace {
    /// Normalized inputs: `[Batch, Input]`
    pub z_buffer: AlignedBuffer,

    /// Basis function values: `[Batch, Input, Basis]`
    pub basis_values: AlignedBuffer,

    /// Basis function derivatives: `[Batch, Input, Basis]`
    pub basis_derivs: AlignedBuffer,

    /// Grid indices: `[Batch, Input]`
    pub grid_indices: Vec<u32>,

    /// Intermediate layer outputs: `[Batch, MaxHiddenDim]`
    pub layer_output: AlignedBuffer,

    /// Previous layer output (for multi-layer): `[Batch, MaxHiddenDim]`
    pub layer_input: AlignedBuffer,

    // --- Backward pass history ---
    /// Saved normalized inputs per layer: `[Layer][Batch * in_dim_layer]`.
    ///
    /// Recorded during `forward_batch_training` and used in backward pass
    /// to recompute B-spline basis derivatives. Each buffer holds the
    /// z-normalized input values for one layer.
    pub layers_inputs: Vec<AlignedBuffer>,

    /// Saved grid indices per layer: `[Layer][Batch * in_dim_layer]`.
    ///
    /// The spline segment index for each input, recorded during forward.
    /// Used in backward to index into the correct spline weights.
    pub layers_grid_indices: Vec<Vec<u32>>,

    /// Gradient buffer passed between layers during backprop: `[Batch, MaxDim]`.
    ///
    /// During backward pass, this buffer accumulates dL/d(layer_input) which
    /// becomes dL/d(layer_output) for the previous layer.
    pub layer_grads: AlignedBuffer,

    /// Staging buffer for ping-pong gradient propagation: `[Batch, MaxDim]`.
    ///
    /// Holds the current layer's output gradient (dL/dy). After each layer's
    /// backward pass, `layer_grads` is copied here for the next iteration.
    /// This enables zero-allocation backward without double-borrow issues.
    pub staging_buffer: AlignedBuffer,

    /// Predictions buffer for train_step: `[Batch, OutputDim]`.
    ///
    /// Stores forward pass outputs before loss computation. Used with
    /// `std::mem::take` to avoid borrow conflicts with workspace during
    /// forward_batch_training.
    pub predictions_buffer: AlignedBuffer,

    /// Weight gradients per layer: `[Layer][weights.len()]`.
    ///
    /// Accumulated during backward pass. Prepared by `prepare_grad_buffers()`
    /// with sizes matching each layer's weight count.
    pub weight_grads: Vec<Vec<f32>>,

    /// Bias gradients per layer: `[Layer][bias.len()]`.
    ///
    /// Accumulated during backward pass. Prepared by `prepare_grad_buffers()`
    /// with sizes matching each layer's bias count.
    pub bias_grads: Vec<Vec<f32>>,

    /// Output gradient buffer for backprop: `[Batch * output_dim]`.
    ///
    /// Initial gradient dL/d(network_output), computed from loss function.
    /// Seeded by `compute_masked_mse_loss_into()` and propagated backward.
    pub grad_output: AlignedBuffer,

    /// Tracking ---
    /// Current batch capacity
    batch_capacity: usize,

    /// Max dimension across all layers
    max_dim: usize,

    /// Batch size of the last recorded history (for assertions)
    history_batch_size: usize,
}

impl Workspace {
    /// Creates a new workspace for the given config.
    pub fn new(config: &KanConfig) -> Self {
        let mut ws = Self {
            z_buffer: AlignedBuffer::new(),
            basis_values: AlignedBuffer::new(),
            basis_derivs: AlignedBuffer::new(),
            grid_indices: Vec::new(),
            layer_output: AlignedBuffer::new(),
            layer_input: AlignedBuffer::new(),
            layers_inputs: Vec::new(),
            layers_grid_indices: Vec::new(),
            layer_grads: AlignedBuffer::new(),
            staging_buffer: AlignedBuffer::new(),
            predictions_buffer: AlignedBuffer::new(),
            weight_grads: Vec::new(),
            bias_grads: Vec::new(),
            grad_output: AlignedBuffer::new(),
            batch_capacity: 0,
            max_dim: 0,
            history_batch_size: 0,
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
        let output_dim = config.output_dim;

        // z_buffer: [batch, input]
        self.z_buffer.reserve(batch_size * input_dim);

        // basis_values: [batch, input, basis]
        self.basis_values.reserve(batch_size * input_dim * basis);
        self.basis_derivs.reserve(batch_size * input_dim * basis);

        // grid_indices: [batch, input]
        self.grid_indices.reserve(batch_size * input_dim);
        if self.grid_indices.len() < batch_size * input_dim {
            self.grid_indices.resize(batch_size * input_dim, 0);
        }

        // layer buffers: [batch, max_dim]
        self.layer_output.reserve(batch_size * max_dim);
        self.layer_input.reserve(batch_size * max_dim);

        // layer_grads: [batch, max_dim]
        self.layer_grads.reserve(batch_size * max_dim);

        // staging_buffer for ping-pong without alloc: [batch, max_dim]
        self.staging_buffer.reserve(batch_size * max_dim);

        // predictions_buffer for train_step: [batch, output_dim]
        self.predictions_buffer.reserve(batch_size * output_dim);

        // grad_output for backprop: [batch, output_dim]
        self.grad_output.reserve(batch_size * output_dim);

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
        self.basis_derivs.resize(batch_size * input_dim * basis);
        self.grid_indices.resize(batch_size * input_dim, 0);
    }

    /// Prepares workspace for training with history tracking.
    ///
    /// Allocates/resizes buffers needed for backward pass:
    /// - `layers_inputs`: one buffer per layer for saved normalized inputs
    /// - `layers_grid_indices`: one vec per layer for saved spline indices
    /// - `layer_grads`, `staging_buffer`: ping-pong gradient buffers
    /// - `basis_derivs`: B-spline derivative values
    /// - `predictions_buffer`, `grad_output`: loss computation buffers
    ///
    /// # Zero-Allocation Guarantee
    ///
    /// After the first call with a given batch size, subsequent calls with
    /// the same or smaller batch size perform zero allocations. Buffers
    /// grow monotonically and are reused.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples in the batch
    /// * `config` - Network configuration
    /// * `layer_dims` - Dimensions of all layers `[input, hidden..., output]`
    #[inline]
    pub fn prepare_training(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
        layer_dims: &[usize],
    ) {
        self.reserve(batch_size, config);

        let num_layers = layer_dims.len().saturating_sub(1);
        if self.layers_inputs.len() < num_layers {
            self.layers_inputs
                .resize_with(num_layers, AlignedBuffer::new);
        }
        if self.layers_grid_indices.len() < num_layers {
            self.layers_grid_indices.resize_with(num_layers, Vec::new);
        }

        let basis = config.basis_size_aligned();
        let max_in_dim = *layer_dims.iter().max().unwrap_or(&config.input_dim);

        // Ensure history buffers sized per layer
        for (layer_idx, in_dim) in layer_dims.iter().copied().enumerate().take(num_layers) {
            let needed = batch_size * in_dim;

            let buf = &mut self.layers_inputs[layer_idx];
            buf.reserve(needed);
            buf.resize(needed);

            let indices = &mut self.layers_grid_indices[layer_idx];
            if indices.len() < needed {
                indices.resize(needed, 0);
            } else {
                indices.truncate(needed);
            }
        }

        // Gradient ping-pong buffer
        self.layer_grads.reserve(batch_size * max_in_dim);
        self.layer_grads.resize(batch_size * max_in_dim);

        // Derivatives buffer (same layout as basis_values)
        self.basis_derivs.reserve(batch_size * max_in_dim * basis);
        self.basis_derivs.resize(batch_size * max_in_dim * basis);

        // Training ping-pong buffers
        let output_dim = config.output_dim;
        self.predictions_buffer.reserve(batch_size * output_dim);
        self.predictions_buffer.resize(batch_size * output_dim);

        self.grad_output.reserve(batch_size * output_dim);
        self.grad_output.resize(batch_size * output_dim);

        self.history_batch_size = batch_size;
    }

    /// Prepares gradient buffers for training with layer sizes.
    ///
    /// Allocates `weight_grads` and `bias_grads` vectors with correct sizes
    /// for each layer. These buffers accumulate gradients during backward pass.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` - Tuples of `(weight_count, bias_count)` per layer.
    ///   Obtained from `KanNetwork::layer_param_sizes`.
    ///
    /// # Zero-Allocation Note
    ///
    /// Buffers grow monotonically. After first call, subsequent calls with
    /// same or smaller layer sizes perform zero allocations.
    #[inline]
    pub fn prepare_grad_buffers(&mut self, layer_sizes: &[(usize, usize)]) {
        let num_layers = layer_sizes.len();

        // Resize gradient vectors if needed
        if self.weight_grads.len() < num_layers {
            self.weight_grads.resize_with(num_layers, Vec::new);
        }
        if self.bias_grads.len() < num_layers {
            self.bias_grads.resize_with(num_layers, Vec::new);
        }

        // Ensure each layer's gradient buffer has correct capacity
        for (i, (w_size, b_size)) in layer_sizes.iter().enumerate() {
            let buf = &mut self.weight_grads[i];
            if buf.len() < *w_size {
                buf.resize(*w_size, 0.0);
            }
            let buf = &mut self.bias_grads[i];
            if buf.len() < *b_size {
                buf.resize(*b_size, 0.0);
            }
        }
    }

    /// Current batch capacity.
    #[inline]
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }

    /// Asserts that workspace is properly sized for the batch.
    #[inline]
    pub fn assert_history_batch(&self, batch_size: usize) {
        assert!(
            self.history_batch_size == batch_size,
            "Workspace history batch {} != requested {}",
            self.history_batch_size,
            batch_size
        );
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
        let config = KanConfig::preset();
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
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);

        ws.prepare_forward(64, &config);

        assert_eq!(ws.z_buffer.len(), 64 * 21);
        assert_eq!(ws.grid_indices.len(), 64 * 21);
    }
}
