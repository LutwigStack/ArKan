//! Data normalization utilities.
//!
//! CRITICAL: KAN networks are very sensitive to input normalization!
//! The default grid range is [-1, 1], so inputs must be normalized to this range.

use crate::game::Board;

/// Zero-allocation one-hot encoding - writes directly to pre-allocated buffer.
/// Buffer must be at least 256 elements.
#[inline]
pub fn board_to_onehot_inplace(board: &Board, buffer: &mut [f32]) {
    debug_assert!(buffer.len() >= 256, "Buffer must be at least 256 elements");
    
    // Clear the buffer (optimized: only clear if needed)
    buffer[..256].fill(0.0);
    
    for row in 0..4 {
        for col in 0..4 {
            let cell_idx = row * 4 + col;
            let val = board.get(row, col) as usize;
            buffer[cell_idx * 16 + val] = 1.0;
        }
    }
}

