//! Data normalization utilities.
//!
//! CRITICAL: KAN networks are very sensitive to input normalization!
//! The default grid range is [-1, 1], so inputs must be normalized to this range.

use crate::game::{Board, Game};

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

/// Bootstrap only from moves the rollout policy can select.
pub fn bellman_target(next_state: &[f32], next_q: &[f32], reward: f32, gamma: f32) -> f32 {
    let mut board = Board::empty();
    for (cell, values) in next_state.chunks_exact(16).take(16).enumerate() {
        let exponent = values.iter().position(|&value| value == 1.0).unwrap_or(0);
        board.set(cell / 4, cell % 4, exponent as u8);
    }
    let game = Game { board, score: 0, game_over: false };
    let max_q = game.valid_moves().into_iter().map(|dir| next_q[dir as usize])
        .reduce(f32::max);
    max_q.map_or(reward, |value| reward + gamma * value)
}

