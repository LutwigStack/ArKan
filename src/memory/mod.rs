//! Aligned allocation, checked buffer extents and tensor views.

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
/// - All memory from `[0, capacity)` is initialized
/// - `len` tracks the logical length; regrown elements are zeroed before exposure
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
        self.try_reserve(new_cap)
            .expect("AlignedBuffer::reserve failed");
    }

    /// Tries to reserve capacity, returning error on overflow or allocation failure.
    #[inline]
    pub fn try_reserve(&mut self, new_cap: usize) -> ArkanResult<()> {
        if new_cap <= self.capacity {
            return Ok(());
        }

        let layout = Self::try_layout(new_cap)?;

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
        Self::try_layout(capacity).expect("Invalid AlignedBuffer layout")
    }

    /// Returns the layout for a given capacity, returning an error on overflow.
    fn try_layout(capacity: usize) -> ArkanResult<Layout> {
        checked_buffer_size(capacity, 1)?;
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
        let mut buf =
            AlignedBuffer::try_with_capacity(data.len()).map_err(serde::de::Error::custom)?;
        buf.try_resize(data.len())
            .map_err(serde::de::Error::custom)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_layout_paths_reject_limit_before_allocation() {
        assert!(AlignedBuffer::try_layout(MAX_BUFFER_ELEMENTS + 1).is_err());
        assert!(AlignedBuffer::try_layout(usize::MAX).is_err());
        let mut buffer = AlignedBuffer::with_capacity(4);
        buffer.resize(2);
        buffer[0] = 7.0;
        assert!(buffer.try_reserve(usize::MAX).is_err());
        assert!(buffer.try_resize(usize::MAX).is_err());
        assert_eq!(buffer.as_slice(), &[7.0, 0.0]);
        assert_eq!(buffer.capacity(), 4);
        let panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| buffer.reserve(usize::MAX)));
        assert!(panic.is_err());
        assert_eq!(buffer.as_slice(), &[7.0, 0.0]);
    }

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
}
