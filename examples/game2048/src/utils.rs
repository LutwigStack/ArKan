//! Data normalization utilities.
//!
//! CRITICAL: KAN networks are very sensitive to input normalization!
//! The default grid range is [-1, 1], so inputs must be normalized to this range.

use crate::game::Board;

/// Converts board to normalized feature vector.
///
/// Uses one-hot encoding for each cell position, which works better for KAN
/// than raw values or simple log normalization.
///
/// For each cell, we encode:
/// - The log2 of tile value (0-15) normalized to [-1, 1]
///
/// This gives 16 features (one per cell).
pub fn board_to_features(board: &Board) -> Vec<f32> {
    let mut features = Vec::with_capacity(16);
    
    for row in 0..4 {
        for col in 0..4 {
            let val = board.get(row, col) as f32;
            // Normalize to [-1, 1] range
            // val is 0-15 (log2 of tile), max practical is ~11 (2048)
            // We use tanh-like mapping: (val / 8) - 1, clamped to [-1, 1]
            let normalized = if val == 0.0 {
                -1.0 // Empty cell
            } else {
                ((val / 8.0) - 0.5).clamp(-1.0, 1.0)
            };
            features.push(normalized);
        }
    }
    
    features
}

/// Alternative: One-hot encoding per cell (16 * 16 = 256 features).
/// More expressive but larger network needed.
#[allow(dead_code)]
pub fn board_to_onehot(board: &Board) -> Vec<f32> {
    let mut features = vec![0.0f32; 16 * 16]; // 16 cells * 16 possible values
    
    for row in 0..4 {
        for col in 0..4 {
            let cell_idx = row * 4 + col;
            let val = board.get(row, col) as usize;
            features[cell_idx * 16 + val] = 1.0;
        }
    }
    
    features
}

/// Extended features with additional game-relevant information.
/// Total: 16 (board) + 8 (extra) = 24 features.
#[allow(dead_code)]
pub fn board_to_extended_features(board: &Board) -> Vec<f32> {
    let mut features = board_to_features(board);
    
    // Add extra features
    let empty_count = board.count_empty() as f32;
    let max_tile = board.max_tile() as f32;
    
    // Empty count normalized (0-16 -> [-1, 1])
    features.push((empty_count / 8.0) - 1.0);
    
    // Max tile normalized (log2 scale)
    let max_log = if max_tile > 0.0 { max_tile.log2() } else { 0.0 };
    features.push((max_log / 8.0) - 1.0);
    
    // Monotonicity features (how well tiles are ordered)
    let (mono_lr, mono_ud) = compute_monotonicity(board);
    features.push(mono_lr);
    features.push(mono_ud);
    
    // Smoothness (how similar adjacent tiles are)
    let smoothness = compute_smoothness(board);
    features.push(smoothness);
    
    // Corner bonus (largest tile in corner)
    let corner = compute_corner_score(board);
    features.push(corner);
    
    // Edge density (tiles on edges)
    let edge = compute_edge_density(board);
    features.push(edge);
    
    // Merge potential
    let merges = compute_merge_potential(board);
    features.push(merges);
    
    features
}

/// Computes monotonicity score (preference for ordered tiles).
fn compute_monotonicity(board: &Board) -> (f32, f32) {
    let mut mono_lr = 0.0f32;
    let mut mono_ud = 0.0f32;
    
    for row in 0..4 {
        for col in 0..3 {
            let curr = board.get(row, col) as f32;
            let next = board.get(row, col + 1) as f32;
            if curr >= next {
                mono_lr += 0.1;
            } else {
                mono_lr -= 0.1;
            }
        }
    }
    
    for row in 0..3 {
        for col in 0..4 {
            let curr = board.get(row, col) as f32;
            let next = board.get(row + 1, col) as f32;
            if curr >= next {
                mono_ud += 0.1;
            } else {
                mono_ud -= 0.1;
            }
        }
    }
    
    (mono_lr.clamp(-1.0, 1.0), mono_ud.clamp(-1.0, 1.0))
}

/// Computes smoothness (how similar adjacent tiles are).
fn compute_smoothness(board: &Board) -> f32 {
    let mut smoothness = 0.0f32;
    
    for row in 0..4 {
        for col in 0..4 {
            let val = board.get(row, col) as f32;
            if val == 0.0 {
                continue;
            }
            
            // Check right neighbor
            if col < 3 {
                let right = board.get(row, col + 1) as f32;
                if right > 0.0 {
                    smoothness -= (val - right).abs() / 15.0;
                }
            }
            
            // Check down neighbor
            if row < 3 {
                let down = board.get(row + 1, col) as f32;
                if down > 0.0 {
                    smoothness -= (val - down).abs() / 15.0;
                }
            }
        }
    }
    
    (smoothness / 24.0 + 1.0).clamp(-1.0, 1.0)
}

/// Computes corner score (bonus if max tile is in corner).
fn compute_corner_score(board: &Board) -> f32 {
    let max_val = (0..16)
        .map(|i| board.get(i / 4, i % 4))
        .max()
        .unwrap_or(0);
    
    let corners = [
        board.get(0, 0),
        board.get(0, 3),
        board.get(3, 0),
        board.get(3, 3),
    ];
    
    if corners.contains(&max_val) {
        1.0
    } else {
        -0.5
    }
}

/// Computes edge density (proportion of high tiles on edges).
fn compute_edge_density(board: &Board) -> f32 {
    let mut edge_sum = 0.0f32;
    let mut total_sum = 0.0f32;
    
    for row in 0..4 {
        for col in 0..4 {
            let val = board.get(row, col) as f32;
            total_sum += val;
            
            if row == 0 || row == 3 || col == 0 || col == 3 {
                edge_sum += val;
            }
        }
    }
    
    if total_sum > 0.0 {
        (edge_sum / total_sum * 2.0 - 1.0).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Computes merge potential (number of adjacent equal tiles).
fn compute_merge_potential(board: &Board) -> f32 {
    let mut merges = 0.0f32;
    
    for row in 0..4 {
        for col in 0..4 {
            let val = board.get(row, col);
            if val == 0 {
                continue;
            }
            
            if col < 3 && board.get(row, col + 1) == val {
                merges += 1.0;
            }
            if row < 3 && board.get(row + 1, col) == val {
                merges += 1.0;
            }
        }
    }
    
    (merges / 12.0 * 2.0 - 1.0).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_range() {
        let board = Board::empty();
        let features = board_to_features(&board);
        
        assert_eq!(features.len(), 16);
        for f in &features {
            assert!(*f >= -1.0 && *f <= 1.0, "Feature {} out of range", f);
        }
    }

    #[test]
    fn test_extended_features_range() {
        let mut board = Board::empty();
        board.set(0, 0, 10); // 1024
        board.set(0, 1, 9);  // 512
        
        let features = board_to_extended_features(&board);
        
        for (i, f) in features.iter().enumerate() {
            assert!(
                *f >= -1.0 && *f <= 1.0,
                "Feature {} = {} out of range",
                i,
                f
            );
        }
    }
}
