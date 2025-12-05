//! Aligned buffers and workspace for zero-allocation inference.
//!
//! This module provides two key types:
//!
//! - [`AlignedBuffer`] — 64-byte aligned buffer for SIMD operations
//! - [`Workspace`] — Preallocated buffers for forward/backward passes
//! - [`WorkspaceGuard`] — RAII guard for exception-safe buffer management
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
//!
//! # Exception Safety
//!
//! The [`WorkspaceGuard`] provides basic exception safety guarantee:
//! buffers are returned to workspace even if a panic occurs during
//! forward/backward passes.

use crate::config::KanConfig;
use crate::error::{ArkanError, ArkanResult};
#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

/// Cache line size for memory alignment (64 bytes).
///
/// All [`AlignedBuffer`] allocations use this alignment for optimal
/// SIMD performance and cache behavior.
pub const CACHE_LINE: usize = 64;

/// Maximum buffer size to prevent overflow in dimension calculations.
///
/// This limit ensures that `batch_size * dim * basis_size` cannot overflow.
/// With `MAX_BUFFER_ELEMENTS = 2^30`, we can handle:
/// - batch_size = 4096, dim = 1024, basis = 256 (4096 * 1024 * 256 = 2^30)
///
/// For larger workloads, use streaming or chunked processing.
pub const MAX_BUFFER_ELEMENTS: usize = 1 << 30; // 1 billion f32s = 4 GB

/// Computes buffer size with overflow checking.
///
/// Returns `Ok(product)` if the multiplication succeeds and doesn't exceed
/// `MAX_BUFFER_ELEMENTS`, otherwise returns an `ArkanError::Overflow`.
///
/// # Example
///
/// ```rust
/// use arkan::checked_buffer_size;
///
/// // Normal case
/// assert!(checked_buffer_size(64, 10).is_ok());
/// assert_eq!(checked_buffer_size(64, 10).unwrap(), 640);
///
/// // Overflow case
/// assert!(checked_buffer_size(usize::MAX, 2).is_err());
/// ```
#[inline]
pub fn checked_buffer_size(a: usize, b: usize) -> ArkanResult<usize> {
    let result = a
        .checked_mul(b)
        .ok_or_else(|| ArkanError::overflow("Buffer size overflow"))?;

    if result > MAX_BUFFER_ELEMENTS {
        return Err(ArkanError::overflow(format!(
            "Buffer size {} exceeds MAX_BUFFER_ELEMENTS ({})",
            result, MAX_BUFFER_ELEMENTS
        )));
    }

    Ok(result)
}

/// Computes buffer size from three dimensions with overflow checking.
///
/// Equivalent to `checked_buffer_size(checked_buffer_size(a, b)?, c)?`.
///
/// # Example
///
/// ```rust
/// use arkan::checked_buffer_size3;
///
/// // batch_size * dim * basis
/// assert!(checked_buffer_size3(64, 10, 8).is_ok());
/// assert_eq!(checked_buffer_size3(64, 10, 8).unwrap(), 5120);
/// ```
#[inline]
pub fn checked_buffer_size3(a: usize, b: usize, c: usize) -> ArkanResult<usize> {
    let ab = checked_buffer_size(a, b)?;
    checked_buffer_size(ab, c)
}

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
///
/// # Memory Layout
///
/// - All memory from `[0, capacity)` is always initialized to zero
/// - `len` tracks the "logical" length, but memory beyond `len` is still valid zeros
/// - This ensures `Clone` and `resize` never expose uninitialized memory
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
    #[must_use]
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    /// Creates a buffer with the specified capacity (in f32 elements).
    /// All elements are initialized to zero.
    ///
    /// # Panics
    ///
    /// Panics if `capacity > MAX_BUFFER_ELEMENTS` or on allocation failure.
    /// Use [`try_with_capacity`](Self::try_with_capacity) for fallible allocation.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).expect("AlignedBuffer allocation failed")
    }

    /// Tries to create a buffer with the specified capacity.
    ///
    /// Returns an error if:
    /// - `capacity > MAX_BUFFER_ELEMENTS` (overflow protection)
    /// - Memory allocation fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::AlignedBuffer;
    ///
    /// // Small allocation succeeds
    /// let buf = AlignedBuffer::try_with_capacity(1000)?;
    /// assert_eq!(buf.capacity(), 1000);
    ///
    /// // Very large allocation may fail
    /// let result = AlignedBuffer::try_with_capacity(usize::MAX);
    /// assert!(result.is_err());
    /// # Ok::<(), arkan::ArkanError>(())
    /// ```
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_with_capacity(capacity: usize) -> ArkanResult<Self> {
        if capacity == 0 {
            return Ok(Self::new());
        }

        if capacity > MAX_BUFFER_ELEMENTS {
            return Err(ArkanError::overflow(format!(
                "Buffer capacity {} exceeds MAX_BUFFER_ELEMENTS ({})",
                capacity, MAX_BUFFER_ELEMENTS
            )));
        }

        let layout = Self::try_layout(capacity)?;
        // SAFETY: layout is valid, allocation may fail
        let ptr = unsafe {
            let raw = alloc_zeroed(layout);
            if raw.is_null() {
                return Err(ArkanError::cpu("AlignedBuffer allocation failed"));
            }
            NonNull::new_unchecked(raw as *mut f32)
        };

        Ok(Self {
            ptr,
            len: 0,
            capacity,
        })
    }

    /// Ensures capacity is at least `new_cap`.
    /// Does not shrink. Only grows if needed.
    /// New capacity is always zero-initialized.
    #[inline]
    pub fn reserve(&mut self, new_cap: usize) {
        if new_cap <= self.capacity {
            return;
        }

        // Allocate new buffer (zero-initialized)
        let new_layout = Self::layout(new_cap);
        // SAFETY: new_layout is valid for `new_cap`, allocation failure handled
        let new_ptr = unsafe {
            let raw = alloc_zeroed(new_layout);
            if raw.is_null() {
                std::alloc::handle_alloc_error(new_layout);
            }
            NonNull::new_unchecked(raw as *mut f32)
        };

        // Copy old data if any (only up to len, rest is zeros)
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

    /// Tries to reserve capacity, returning error on overflow or allocation failure.
    #[inline]
    pub fn try_reserve(&mut self, new_cap: usize) -> ArkanResult<()> {
        if new_cap <= self.capacity {
            return Ok(());
        }

        // Check for overflow in layout calculation
        let size = new_cap
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| ArkanError::overflow("AlignedBuffer capacity overflow"))?;

        let layout = Layout::from_size_align(size, CACHE_LINE)
            .map_err(|_| ArkanError::overflow("Invalid layout for AlignedBuffer"))?;

        // SAFETY: layout is valid, allocation may fail
        let new_ptr = unsafe {
            let raw = alloc_zeroed(layout);
            if raw.is_null() {
                return Err(ArkanError::cpu("AlignedBuffer allocation failed"));
            }
            NonNull::new_unchecked(raw as *mut f32)
        };

        // Copy old data
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
        Ok(())
    }

    /// Resizes the buffer, filling new elements with zero.
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        self.reserve(new_len);
        // Since reserve() allocates zeroed memory and we maintain zero-initialized
        // invariant, we only need to zero elements that might have been written to
        if new_len > self.len {
            // SAFETY: destination is within allocated region; zero the tail
            // This is technically redundant if capacity just grew (already zeroed),
            // but necessary if len shrank and then grew again
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr().add(self.len), 0, new_len - self.len);
            }
        }
        self.len = new_len;
    }

    /// Resizes the buffer without initializing new elements.
    ///
    /// # Safety
    ///
    /// Caller must ensure that all elements in `[old_len, new_len)` are written
    /// before being read. This is useful for hot paths where the buffer will be
    /// completely overwritten (e.g., forward pass output).
    ///
    /// Note: Due to our zero-initialization invariant, this is actually safe,
    /// but we keep it marked unsafe as a reminder that the data may be stale.
    #[inline]
    pub unsafe fn resize_uninitialized(&mut self, new_len: usize) {
        self.reserve(new_len);
        self.len = new_len;
    }

    /// Tries to resize, returning error on overflow.
    #[inline]
    pub fn try_resize(&mut self, new_len: usize) -> ArkanResult<()> {
        self.try_reserve(new_len)?;
        if new_len > self.len {
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr().add(self.len), 0, new_len - self.len);
            }
        }
        self.len = new_len;
        Ok(())
    }

    /// Clears the buffer (sets len to 0, doesn't deallocate).
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Fills the current length with zeros.
    #[inline]
    pub fn zero(&mut self) {
        if self.len > 0 {
            // SAFETY: buffer is allocated and len > 0
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len);
            }
        }
    }

    /// Fills the entire capacity with zeros (useful after resize_uninitialized).
    #[inline]
    pub fn zero_all(&mut self) {
        if self.capacity > 0 {
            unsafe {
                std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.capacity);
            }
        }
    }

    /// Current length.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Current capacity.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Is empty?
    #[inline]
    #[must_use]
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

    /// Returns the layout for a given capacity, panicking on overflow.
    ///
    /// # Panics
    ///
    /// Panics if `capacity * size_of::<f32>()` overflows or layout is invalid.
    fn layout(capacity: usize) -> Layout {
        Layout::from_size_align(capacity * std::mem::size_of::<f32>(), CACHE_LINE)
            .expect("Invalid layout")
    }

    /// Returns the layout for a given capacity, returning an error on overflow.
    fn try_layout(capacity: usize) -> ArkanResult<Layout> {
        let size = capacity
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| ArkanError::overflow("AlignedBuffer capacity overflow in layout"))?;

        Layout::from_size_align(size, CACHE_LINE)
            .map_err(|_| ArkanError::overflow("Invalid layout for AlignedBuffer"))
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
        if self.capacity == 0 {
            return Self::new();
        }

        // Allocate new buffer with same capacity, zero-initialized
        let layout = Self::layout(self.capacity);
        let new_ptr = unsafe {
            let raw = alloc_zeroed(layout);
            if raw.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            NonNull::new_unchecked(raw as *mut f32)
        };

        // Copy only the logical length (rest is already zeros)
        if self.len > 0 {
            // SAFETY: source/dest are distinct allocations, len is within both
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
            }
        }

        Self {
            ptr: new_ptr,
            len: self.len,
            capacity: self.capacity,
        }
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

/// Type alias for tensor data on CPU.
///
/// This is an alias for [`AlignedBuffer`], providing a more semantic name
/// when working with tensor operations. The underlying buffer is 64-byte
/// aligned for optimal SIMD performance.
///
/// # Example
///
/// ```rust
/// use arkan::Tensor;
///
/// let mut tensor = Tensor::with_capacity(1024);
/// tensor.resize(100);
/// tensor.as_mut_slice()[0] = 1.0;
/// ```
pub type Tensor = AlignedBuffer;

/// A borrowed view into tensor data without copying.
///
/// `TensorView` provides zero-copy access to a slice of `f32` data,
/// allowing efficient read-only operations on tensor contents.
///
/// # Example
///
/// ```rust
/// use arkan::TensorView;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0];
/// let view = TensorView::new(&data);
///
/// assert_eq!(view.len(), 4);
/// assert_eq!(view[0], 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    data: &'a [f32],
}

impl<'a> TensorView<'a> {
    /// Creates a new tensor view from a slice.
    #[inline]
    pub fn new(data: &'a [f32]) -> Self {
        Self { data }
    }

    /// Returns the length of the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &'a [f32] {
        self.data
    }

    /// Returns a raw pointer to the data.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Creates a sub-view of this view.
    #[inline]
    pub fn slice(&self, range: std::ops::Range<usize>) -> Self {
        Self {
            data: &self.data[range],
        }
    }

    /// Iterates over elements.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'a, f32> {
        self.data.iter()
    }
}

impl<'a> std::ops::Index<usize> for TensorView<'a> {
    type Output = f32;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a> From<&'a [f32]> for TensorView<'a> {
    fn from(data: &'a [f32]) -> Self {
        Self::new(data)
    }
}

impl<'a> From<&'a AlignedBuffer> for TensorView<'a> {
    fn from(buffer: &'a AlignedBuffer) -> Self {
        Self::new(buffer.as_slice())
    }
}

impl<'a> From<&'a Vec<f32>> for TensorView<'a> {
    fn from(vec: &'a Vec<f32>) -> Self {
        Self::new(vec.as_slice())
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

    /// Ensures workspace has capacity for the given batch size (fallible version).
    ///
    /// This is the checked version that returns an error on overflow instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Overflow`] if any buffer size calculation overflows.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_reserve(&mut self, batch_size: usize, config: &KanConfig) -> ArkanResult<()> {
        if batch_size <= self.batch_capacity {
            return Ok(());
        }

        let dims = config.layer_dims();
        let max_dim = *dims.iter().max().unwrap_or(&1);
        let basis = config.basis_size_aligned();
        let output_dim = config.output_dim;

        // Check all size calculations for overflow using checked_buffer_size
        // z_buffer: [batch, max_dim] - needs max_dim for hidden layers wider than input
        let z_size = checked_buffer_size(batch_size, max_dim)?;

        // basis_values: [batch, max_dim, basis] - needs max_dim for hidden layers
        let basis_size = checked_buffer_size3(batch_size, max_dim, basis)?;

        // layer buffers: [batch, max_dim]
        let layer_size = checked_buffer_size(batch_size, max_dim)?;

        // predictions_buffer: [batch, output_dim]
        let pred_size = checked_buffer_size(batch_size, output_dim)?;

        // All checks passed, now allocate
        self.z_buffer.reserve(z_size);
        self.basis_values.reserve(basis_size);
        self.basis_derivs.reserve(basis_size);

        self.grid_indices.reserve(z_size);
        if self.grid_indices.len() < z_size {
            self.grid_indices.resize(z_size, 0);
        }

        self.layer_output.reserve(layer_size);
        self.layer_input.reserve(layer_size);
        self.layer_grads.reserve(layer_size);
        self.staging_buffer.reserve(layer_size);

        self.predictions_buffer.reserve(pred_size);
        self.grad_output.reserve(pred_size);

        self.batch_capacity = batch_size;
        self.max_dim = max_dim;

        Ok(())
    }

    /// Ensures workspace has capacity for the given batch size.
    /// Only allocates if batch_size > current capacity.
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_reserve`](Self::try_reserve)
    /// for a fallible version.
    #[inline]
    pub fn reserve(&mut self, batch_size: usize, config: &KanConfig) {
        self.try_reserve(batch_size, config)
            .expect("Workspace::reserve: buffer size overflow")
    }

    /// Prepares workspace for a forward pass with the given batch size (fallible version).
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Overflow`] if any buffer size calculation overflows.
    #[inline]
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_prepare_forward(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
    ) -> ArkanResult<()> {
        if batch_size == 0 {
            return Err(crate::ArkanError::shape_mismatch(&[1], &[0]));
        }
        self.try_reserve(batch_size, config)?;

        // Use max_dim for hidden layers wider than input
        let dims = config.layer_dims();
        let max_dim = *dims.iter().max().unwrap_or(&1);
        let basis = config.basis_size_aligned();

        // Use checked arithmetic with max_dim
        let z_size = checked_buffer_size(batch_size, max_dim)?;
        let basis_size = checked_buffer_size3(batch_size, max_dim, basis)?;

        self.z_buffer.resize(z_size);
        self.basis_values.resize(basis_size);
        self.basis_derivs.resize(basis_size);
        self.grid_indices.resize(z_size, 0);

        Ok(())
    }

    /// Prepares workspace for a forward pass with the given batch size.
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_prepare_forward`](Self::try_prepare_forward)
    /// for a fallible version.
    #[inline]
    pub fn prepare_forward(&mut self, batch_size: usize, config: &KanConfig) {
        self.try_prepare_forward(batch_size, config)
            .expect("Workspace::prepare_forward: buffer size overflow")
    }

    /// Prepares workspace for training with history tracking (fallible version).
    ///
    /// # Errors
    ///
    /// Returns [`ArkanError::Overflow`] if any buffer size calculation overflows.
    #[must_use = "this returns a Result that should be handled"]
    pub fn try_prepare_training(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
        layer_dims: &[usize],
    ) -> ArkanResult<()> {
        self.try_reserve(batch_size, config)?;

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

        // Ensure history buffers sized per layer with overflow checks
        for (layer_idx, in_dim) in layer_dims.iter().copied().enumerate().take(num_layers) {
            let needed = checked_buffer_size(batch_size, in_dim)?;

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
        let grad_size = checked_buffer_size(batch_size, max_in_dim)?;
        self.layer_grads.reserve(grad_size);
        self.layer_grads.resize(grad_size);

        // Derivatives buffer (same layout as basis_values)
        let deriv_size = checked_buffer_size3(batch_size, max_in_dim, basis)?;
        self.basis_derivs.reserve(deriv_size);
        self.basis_derivs.resize(deriv_size);

        // Training ping-pong buffers
        let output_dim = config.output_dim;
        let output_size = checked_buffer_size(batch_size, output_dim)?;
        self.predictions_buffer.reserve(output_size);
        self.predictions_buffer.resize(output_size);

        self.grad_output.reserve(output_size);
        self.grad_output.resize(output_size);

        self.history_batch_size = batch_size;

        Ok(())
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
    ///
    /// # Panics
    ///
    /// Panics if buffer size calculations overflow. Use [`try_prepare_training`](Self::try_prepare_training)
    /// for a fallible version.
    #[inline]
    pub fn prepare_training(
        &mut self,
        batch_size: usize,
        config: &KanConfig,
        layer_dims: &[usize],
    ) {
        self.try_prepare_training(batch_size, config, layer_dims)
            .expect("Workspace::prepare_training: buffer size overflow")
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
    ///
    /// # Panics
    ///
    /// May panic on allocation failure. Use [`try_prepare_grad_buffers`](Self::try_prepare_grad_buffers)
    /// for a fallible version.
    #[inline]
    pub fn prepare_grad_buffers(&mut self, layer_sizes: &[(usize, usize)]) {
        self.try_prepare_grad_buffers(layer_sizes)
            .expect("Workspace::prepare_grad_buffers: allocation failed")
    }

    /// Fallible version of [`prepare_grad_buffers`](Self::prepare_grad_buffers).
    ///
    /// Returns `Ok(())` on success, or `ArkanError::Overflow` if buffer size
    /// calculations overflow or exceed `MAX_BUFFER_ELEMENTS`.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` - Tuples of `(weight_count, bias_count)` per layer.
    #[inline]
    pub fn try_prepare_grad_buffers(&mut self, layer_sizes: &[(usize, usize)]) -> ArkanResult<()> {
        let num_layers = layer_sizes.len();

        // Validate sizes won't overflow
        for (w_size, b_size) in layer_sizes {
            if *w_size > MAX_BUFFER_ELEMENTS {
                return Err(ArkanError::overflow(format!(
                    "Weight gradient size {} exceeds MAX_BUFFER_ELEMENTS ({})",
                    w_size, MAX_BUFFER_ELEMENTS
                )));
            }
            if *b_size > MAX_BUFFER_ELEMENTS {
                return Err(ArkanError::overflow(format!(
                    "Bias gradient size {} exceeds MAX_BUFFER_ELEMENTS ({})",
                    b_size, MAX_BUFFER_ELEMENTS
                )));
            }
        }

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

        Ok(())
    }

    /// Zeros all gradient buffers in-place without reallocation.
    ///
    /// Call this at the start of each training step to clear gradients
    /// from the previous iteration. This is more efficient than reallocating
    /// the buffers.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfig, KanNetwork, Workspace};
    ///
    /// let config = KanConfig::preset();
    /// let network = KanNetwork::new(config.clone());
    /// let mut workspace = network.create_workspace(64);
    ///
    /// // After training step, zero grads for next iteration
    /// workspace.zero_grads();
    /// ```
    #[inline]
    pub fn zero_grads(&mut self) {
        for wg in &mut self.weight_grads {
            wg.fill(0.0);
        }
        for bg in &mut self.bias_grads {
            bg.fill(0.0);
        }
    }

    /// Zeros gradient buffers and grad_output buffer.
    ///
    /// Extended version that also clears the output gradient buffer
    /// used for backpropagation.
    #[inline]
    pub fn zero_all_grads(&mut self) {
        self.zero_grads();
        self.grad_output.zero();
    }

    /// Current batch capacity.
    #[inline]
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity
    }

    /// Returns the history batch size.
    #[inline]
    pub fn history_batch_size(&self) -> usize {
        self.history_batch_size
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

    /// Checks that workspace history matches the expected batch size.
    ///
    /// Returns `Ok(())` if `history_batch_size == batch_size`, otherwise
    /// returns `ArkanError::ShapeMismatch`.
    #[inline]
    pub fn check_history_batch(&self, batch_size: usize) -> ArkanResult<()> {
        if self.history_batch_size != batch_size {
            return Err(ArkanError::shape_mismatch(
                &[batch_size],
                &[self.history_batch_size],
            ));
        }
        Ok(())
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

    /// Validates workspace state, returning error if invalid.
    #[inline]
    pub fn validate(&self) -> ArkanResult<()> {
        // Check that ping-pong buffers have capacity
        if self.batch_capacity > 0 {
            if self.layer_output.capacity() == 0 && self.max_dim > 0 {
                return Err(ArkanError::invalid_workspace(
                    "layer_output buffer is empty after previous operation",
                ));
            }
            if self.layer_input.capacity() == 0 && self.max_dim > 0 {
                return Err(ArkanError::invalid_workspace(
                    "layer_input buffer is empty after previous operation",
                ));
            }
        }
        Ok(())
    }

    /// Checks workspace capacity, returning error if insufficient.
    #[inline]
    pub fn check_capacity(&self, batch_size: usize) -> ArkanResult<()> {
        if batch_size > self.batch_capacity {
            return Err(ArkanError::batch_too_large(batch_size, self.batch_capacity));
        }
        Ok(())
    }
}

/// RAII guard for workspace ping-pong buffers.
///
/// This guard ensures that buffers borrowed from a [`Workspace`] are returned
/// even if a panic occurs during computation. This provides basic exception
/// safety guarantee - the workspace remains valid (though possibly with
/// different buffer contents) after unwinding.
///
/// # Usage
///
/// ```rust
/// use arkan::{KanConfig, Workspace, WorkspaceGuard};
///
/// let config = KanConfig::preset();
/// let mut workspace = Workspace::new(&config);
/// workspace.reserve(64, &config);
///
/// {
///     let mut guard = WorkspaceGuard::new(&mut workspace);
///     let (buffer_a, buffer_b) = guard.buffers_mut();
///
///     // ... do computations ...
///     buffer_a.resize(100);
///
///     // Buffers are returned to workspace when guard is dropped
///     guard.finish();
/// }
///
/// // Workspace has the buffers back
/// assert!(workspace.layer_output.capacity() >= 100);
/// ```
///
/// # Panic Behavior
///
/// If a panic occurs while the guard is active:
/// - The `Drop` implementation will return the buffers to the workspace
/// - The workspace will be in a valid state (buffers have correct capacity)
/// - Buffer contents may be in an intermediate state
pub struct WorkspaceGuard<'a> {
    workspace: &'a mut Workspace,
    buffer_a: Option<AlignedBuffer>,
    buffer_b: Option<AlignedBuffer>,
}

impl<'a> WorkspaceGuard<'a> {
    /// Creates a new guard, taking ownership of the ping-pong buffers.
    #[inline]
    pub fn new(workspace: &'a mut Workspace) -> Self {
        let buffer_a = std::mem::take(&mut workspace.layer_output);
        let buffer_b = std::mem::take(&mut workspace.layer_input);
        Self {
            workspace,
            buffer_a: Some(buffer_a),
            buffer_b: Some(buffer_b),
        }
    }

    /// Returns mutable references to both buffers.
    #[inline]
    pub fn buffers_mut(&mut self) -> (&mut AlignedBuffer, &mut AlignedBuffer) {
        (
            self.buffer_a.as_mut().expect("buffer_a already taken"),
            self.buffer_b.as_mut().expect("buffer_b already taken"),
        )
    }

    /// Returns references to both buffers.
    #[inline]
    pub fn buffers(&self) -> (&AlignedBuffer, &AlignedBuffer) {
        (
            self.buffer_a.as_ref().expect("buffer_a already taken"),
            self.buffer_b.as_ref().expect("buffer_b already taken"),
        )
    }

    /// Takes buffer_a, leaving None in its place.
    /// The buffer will NOT be returned to workspace on drop.
    #[inline]
    pub fn take_buffer_a(&mut self) -> AlignedBuffer {
        self.buffer_a.take().expect("buffer_a already taken")
    }

    /// Takes buffer_b, leaving None in its place.
    /// The buffer will NOT be returned to workspace on drop.
    #[inline]
    pub fn take_buffer_b(&mut self) -> AlignedBuffer {
        self.buffer_b.take().expect("buffer_b already taken")
    }

    /// Explicitly returns buffers to workspace and consumes the guard.
    /// This is the normal completion path (no panic).
    #[inline]
    pub fn finish(mut self) {
        self.return_buffers();
    }

    /// Returns the workspace reference for accessing other fields.
    #[inline]
    pub fn workspace(&mut self) -> &mut Workspace {
        self.workspace
    }

    fn return_buffers(&mut self) {
        if let Some(buf) = self.buffer_a.take() {
            self.workspace.layer_output = buf;
        }
        if let Some(buf) = self.buffer_b.take() {
            self.workspace.layer_input = buf;
        }
    }
}

impl<'a> Drop for WorkspaceGuard<'a> {
    fn drop(&mut self) {
        // Return any buffers that weren't explicitly taken
        self.return_buffers();
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

        // After fix: workspace uses max_dim (64) for z_buffer and grid_indices
        let max_dim = *config.layer_dims().iter().max().unwrap();
        assert_eq!(ws.z_buffer.len(), 64 * max_dim);
        assert_eq!(ws.grid_indices.len(), 64 * max_dim);
    }

    #[test]
    fn test_aligned_buffer_clone() {
        let mut buf = AlignedBuffer::with_capacity(100);
        buf.resize(50);
        for i in 0..50 {
            buf[i] = i as f32 * 2.0;
        }

        let cloned = buf.clone();

        // Same logical length
        assert_eq!(cloned.len(), 50);
        assert_eq!(cloned.capacity(), 100);

        // Data matches
        for i in 0..50 {
            assert_eq!(cloned[i], i as f32 * 2.0);
        }

        // Check alignment of clone
        assert_eq!(cloned.as_ptr() as usize % CACHE_LINE, 0);
    }

    #[test]
    fn test_aligned_buffer_zero_all() {
        let mut buf = AlignedBuffer::with_capacity(100);
        buf.resize(50);
        for i in 0..50 {
            buf[i] = 1.0;
        }

        buf.zero();

        for i in 0..50 {
            assert_eq!(buf[i], 0.0);
        }
    }

    #[test]
    fn test_aligned_buffer_try_reserve() {
        let mut buf = AlignedBuffer::new();

        // Normal reservation should succeed
        assert!(buf.try_reserve(100).is_ok());
        assert!(buf.capacity() >= 100);

        // Very large reservation might fail (depends on system)
        // We just test that it returns Result, not panics
        let result = buf.try_reserve(usize::MAX / 2);
        // Either succeeds or returns error, but doesn't panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_workspace_guard_normal_flow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);

        // Get initial capacities
        let initial_output_cap = ws.layer_output.capacity();
        let initial_input_cap = ws.layer_input.capacity();

        {
            let mut guard = WorkspaceGuard::new(&mut ws);
            let (buf_a, buf_b) = guard.buffers_mut();

            // Buffers should have the original capacity
            assert_eq!(buf_a.capacity(), initial_output_cap);
            assert_eq!(buf_b.capacity(), initial_input_cap);

            // Modify buffers
            buf_a.resize(100);
            buf_b.resize(100);

            guard.finish();
        }

        // Buffers returned to workspace
        assert!(ws.layer_output.capacity() >= 100);
        assert!(ws.layer_input.capacity() >= 100);
    }

    #[test]
    fn test_workspace_guard_drop_returns_buffers() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);

        {
            let mut guard = WorkspaceGuard::new(&mut ws);
            let (buf_a, _buf_b) = guard.buffers_mut();
            buf_a.resize(200);
            // Guard dropped without calling finish()
        }

        // Buffers should still be returned
        assert!(ws.layer_output.capacity() >= 200);
        assert!(ws.layer_input.capacity() > 0);
    }

    #[test]
    fn test_workspace_validate() {
        let config = KanConfig::preset();
        let ws = Workspace::new(&config);

        // Fresh workspace should be valid
        assert!(ws.validate().is_ok());
    }

    #[test]
    fn test_workspace_check_capacity() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);

        let cap = ws.batch_capacity();

        // Within capacity - OK
        assert!(ws.check_capacity(32).is_ok());
        assert!(ws.check_capacity(cap).is_ok());

        // Exceeds capacity - Error
        assert!(ws.check_capacity(cap + 1).is_err());
    }

    #[test]
    fn test_checked_buffer_size() {
        // Normal multiplication
        assert_eq!(super::checked_buffer_size(64, 10).unwrap(), 640);
        assert_eq!(super::checked_buffer_size(1, 1).unwrap(), 1);
        assert_eq!(super::checked_buffer_size(0, 1000).unwrap(), 0);

        // Overflow case
        let result = super::checked_buffer_size(usize::MAX, 2);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            super::super::error::ArkanError::Overflow { .. }
        ));
    }

    #[test]
    fn test_checked_buffer_size3() {
        // Normal multiplication
        assert_eq!(super::checked_buffer_size3(64, 10, 8).unwrap(), 5120);

        // Overflow case
        let result = super::checked_buffer_size3(usize::MAX / 2, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_checked_buffer_size_exceeds_max() {
        // Exceeds MAX_BUFFER_ELEMENTS
        let result = super::checked_buffer_size(super::MAX_BUFFER_ELEMENTS + 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_with_capacity_normal() {
        let buf = AlignedBuffer::try_with_capacity(1000).unwrap();
        assert_eq!(buf.capacity(), 1000);
    }

    #[test]
    fn test_try_with_capacity_overflow() {
        // Should fail - exceeds MAX_BUFFER_ELEMENTS
        let result = AlignedBuffer::try_with_capacity(super::MAX_BUFFER_ELEMENTS + 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_with_capacity_zero() {
        let buf = AlignedBuffer::try_with_capacity(0).unwrap();
        assert_eq!(buf.capacity(), 0);
    }

    #[test]
    fn test_try_reserve_success() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        assert!(ws.try_reserve(100, &config).is_ok());
    }

    #[test]
    fn test_try_reserve_overflow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        // Overflow: huge batch × reasonable dims
        let result = ws.try_reserve(usize::MAX / 2, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_prepare_forward_success() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);
        let result = ws.try_prepare_forward(32, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_prepare_forward_overflow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        // Overflow in size calculation
        let result = ws.try_prepare_forward(usize::MAX / 2, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_prepare_training_success() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        ws.reserve(64, &config);
        let dims = config.layer_dims();
        let result = ws.try_prepare_training(32, &config, &dims);
        assert!(result.is_ok());
    }

    #[test]
    fn test_try_prepare_training_overflow() {
        let config = KanConfig::preset();
        let mut ws = Workspace::new(&config);
        let dims = config.layer_dims();
        // Overflow in gradient size calculation
        let result = ws.try_prepare_training(usize::MAX / 4, &config, &dims);
        assert!(result.is_err());
    }
}
