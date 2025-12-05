//! Fast 2048 game implementation using bitboard representation.
//!
//! Each tile is stored as 4 bits (0-15), representing power of 2.
//! The entire 4x4 board fits in a single u64.

use rand::Rng;

/// Bitboard representation of 2048 game.
/// Each nibble (4 bits) stores log2(tile_value), or 0 for empty.
/// Layout: bits 0-3 = cell (0,0), bits 4-7 = cell (0,1), etc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Board(pub u64);

/// Direction for moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3,
}

impl Direction {
    pub fn from_index(idx: usize) -> Self {
        match idx & 3 {
            0 => Direction::Up,
            1 => Direction::Right,
            2 => Direction::Down,
            _ => Direction::Left,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Direction::Up => "UP",
            Direction::Right => "RIGHT",
            Direction::Down => "DOWN",
            Direction::Left => "LEFT",
        }
    }
}

/// 2048 Game state.
pub struct Game {
    pub board: Board,
    pub score: u32,
    pub game_over: bool,
}

impl Board {
    /// Creates an empty board.
    pub const fn empty() -> Self {
        Board(0)
    }

    /// Gets the value at position (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        let shift = (row * 4 + col) * 4;
        ((self.0 >> shift) & 0xF) as u8
    }

    /// Sets the value at position (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: u8) {
        let shift = (row * 4 + col) * 4;
        self.0 = (self.0 & !(0xF << shift)) | ((val as u64 & 0xF) << shift);
    }

    /// Gets the actual tile value (2^n or 0).
    #[inline]
    pub fn tile_value(&self, row: usize, col: usize) -> u32 {
        let n = self.get(row, col);
        if n == 0 { 0 } else { 1 << n }
    }

    /// Returns the maximum tile value on the board.
    pub fn max_tile(&self) -> u32 {
        let mut max = 0u8;
        for i in 0..16 {
            let val = ((self.0 >> (i * 4)) & 0xF) as u8;
            if val > max {
                max = val;
            }
        }
        if max == 0 { 0 } else { 1 << max }
    }

    /// Counts empty cells.
    pub fn count_empty(&self) -> usize {
        let mut count = 0;
        for i in 0..16 {
            if ((self.0 >> (i * 4)) & 0xF) == 0 {
                count += 1;
            }
        }
        count
    }

    /// Returns list of empty cell positions.
    pub fn empty_cells(&self) -> Vec<(usize, usize)> {
        let mut cells = Vec::with_capacity(16);
        for row in 0..4 {
            for col in 0..4 {
                if self.get(row, col) == 0 {
                    cells.push((row, col));
                }
            }
        }
        cells
    }

    /// Transpose the board (swap rows and columns).
    #[inline]
    pub fn transpose(&self) -> Board {
        let mut result = Board::empty();
        for row in 0..4 {
            for col in 0..4 {
                result.set(col, row, self.get(row, col));
            }
        }
        result
    }

    /// Reverse each row (mirror horizontally).
    #[inline]
    pub fn reverse_rows(&self) -> Board {
        let mut result = Board::empty();
        for row in 0..4 {
            for col in 0..4 {
                result.set(row, 3 - col, self.get(row, col));
            }
        }
        result
    }
}

impl Game {
    /// Creates a new game with two random tiles.
    pub fn new() -> Self {
        let mut game = Self {
            board: Board::empty(),
            score: 0,
            game_over: false,
        };
        game.add_random_tile();
        game.add_random_tile();
        game
    }

    /// Adds a random tile (2 with 90% probability, 4 with 10%).
    pub fn add_random_tile(&mut self) {
        let empty = self.board.empty_cells();
        if empty.is_empty() {
            return;
        }

        let mut rng = rand::thread_rng();
        let (row, col) = empty[rng.gen_range(0..empty.len())];
        let val = if rng.gen::<f32>() < 0.9 { 1 } else { 2 }; // 1 = tile 2, 2 = tile 4
        self.board.set(row, col, val);
    }

    /// Executes a move and returns (reward, changed).
    pub fn make_move(&mut self, dir: Direction) -> (f32, bool) {
        if self.game_over {
            return (0.0, false);
        }

        let old_board = self.board;
        let old_score = self.score;

        match dir {
            Direction::Left => self.slide_left(),
            Direction::Right => {
                self.board = self.board.reverse_rows();
                self.slide_left();
                self.board = self.board.reverse_rows();
            }
            Direction::Up => {
                self.board = self.board.transpose();
                self.slide_left();
                self.board = self.board.transpose();
            }
            Direction::Down => {
                self.board = self.board.transpose().reverse_rows();
                self.slide_left();
                self.board = self.board.reverse_rows().transpose();
            }
        }

        let changed = self.board != old_board;
        if changed {
            self.add_random_tile();
            self.check_game_over();
        }

        let score_gained = (self.score - old_score) as f32;
        
        // Enhanced reward shaping for better learning
        let reward = if changed {
            let mut r = 0.0f32;
            
            // 1. Reward for score gained (scaled)
            if score_gained > 0.0 {
                r += score_gained / 100.0; // Scale down large scores
            }
            
            // 2. Bonus for keeping max tile in corner
            let max_tile = self.board.max_tile();
            let corner_vals = [
                self.board.get(0, 0),
                self.board.get(0, 3),
                self.board.get(3, 0),
                self.board.get(3, 3),
            ];
            let max_in_corner = corner_vals.iter().any(|&v| v == max_tile as u8);
            if max_in_corner && max_tile >= 64 {
                r += 0.5;
            }
            
            // 3. Bonus for monotonic rows/columns
            let monotonic_bonus = self.compute_monotonicity_bonus();
            r += monotonic_bonus * 0.1;
            
            // 4. Penalty for scattered high tiles
            let empty_count = self.board.empty_cells().len();
            r += (empty_count as f32) * 0.05; // Bonus for keeping board clean
            
            r.max(0.01) // Minimum positive reward for valid move
        } else {
            -0.5 // Penalty for invalid move
        };

        (reward, changed)
    }
    
    /// Computes monotonicity bonus (higher tiles in corners/edges).
    fn compute_monotonicity_bonus(&self) -> f32 {
        let mut bonus = 0.0f32;
        
        // Check if values decrease from top-left corner
        // Row monotonicity
        for row in 0..4 {
            let mut increasing = true;
            let mut decreasing = true;
            for col in 1..4 {
                if self.board.get(row, col) > self.board.get(row, col - 1) {
                    decreasing = false;
                }
                if self.board.get(row, col) < self.board.get(row, col - 1) {
                    increasing = false;
                }
            }
            if increasing || decreasing {
                bonus += 1.0;
            }
        }
        
        // Column monotonicity  
        for col in 0..4 {
            let mut increasing = true;
            let mut decreasing = true;
            for row in 1..4 {
                if self.board.get(row, col) > self.board.get(row - 1, col) {
                    decreasing = false;
                }
                if self.board.get(row, col) < self.board.get(row - 1, col) {
                    increasing = false;
                }
            }
            if increasing || decreasing {
                bonus += 1.0;
            }
        }
        
        bonus / 8.0 // Normalize to [0, 1]
    }

    /// Slide all rows to the left and merge.
    fn slide_left(&mut self) {
        for row in 0..4 {
            let mut line = [0u8; 4];
            for col in 0..4 {
                line[col] = self.board.get(row, col);
            }

            // Slide and merge
            let merged = self.slide_and_merge_line(&mut line);
            self.score += merged;

            // Write back
            for col in 0..4 {
                self.board.set(row, col, line[col]);
            }
        }
    }

    /// Slides and merges a single line, returns score gained.
    fn slide_and_merge_line(&self, line: &mut [u8; 4]) -> u32 {
        let mut score = 0u32;

        // Compact: remove zeros
        let mut write = 0;
        for read in 0..4 {
            if line[read] != 0 {
                line[write] = line[read];
                write += 1;
            }
        }
        for i in write..4 {
            line[i] = 0;
        }

        // Merge adjacent equal tiles
        for i in 0..3 {
            if line[i] != 0 && line[i] == line[i + 1] {
                line[i] += 1; // Double the tile (add 1 to exponent)
                score += 1 << line[i]; // Score is the merged tile value
                line[i + 1] = 0;
            }
        }

        // Compact again
        write = 0;
        for read in 0..4 {
            if line[read] != 0 {
                line[write] = line[read];
                write += 1;
            }
        }
        for i in write..4 {
            line[i] = 0;
        }

        score
    }

    /// Checks if game is over (no valid moves).
    fn check_game_over(&mut self) {
        // Check for empty cells
        if self.board.count_empty() > 0 {
            return;
        }

        // Check for possible merges
        for row in 0..4 {
            for col in 0..3 {
                if self.board.get(row, col) == self.board.get(row, col + 1) {
                    return;
                }
            }
        }
        for row in 0..3 {
            for col in 0..4 {
                if self.board.get(row, col) == self.board.get(row + 1, col) {
                    return;
                }
            }
        }

        self.game_over = true;
    }

    /// Returns true if a move in the given direction is valid.
    pub fn is_valid_move(&self, dir: Direction) -> bool {
        let mut test = self.clone();
        let (_, changed) = test.make_move(dir);
        changed
    }

    /// Returns list of valid moves.
    pub fn valid_moves(&self) -> Vec<Direction> {
        [Direction::Up, Direction::Right, Direction::Down, Direction::Left]
            .iter()
            .copied()
            .filter(|&d| self.is_valid_move(d))
            .collect()
    }

    /// Prints the board to stdout.
    pub fn print(&self) {
        println!("┌──────┬──────┬──────┬──────┐");
        for row in 0..4 {
            print!("│");
            for col in 0..4 {
                let val = self.board.tile_value(row, col);
                if val == 0 {
                    print!("      │");
                } else {
                    print!("{:^6}│", val);
                }
            }
            println!();
            if row < 3 {
                println!("├──────┼──────┼──────┼──────┤");
            }
        }
        println!("└──────┴──────┴──────┴──────┘");
        println!("Score: {}  Max tile: {}", self.score, self.board.max_tile());
    }
}

impl Clone for Game {
    fn clone(&self) -> Self {
        Self {
            board: self.board,
            score: self.score,
            game_over: self.game_over,
        }
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = Game::new();
        assert_eq!(game.score, 0);
        assert!(!game.game_over);
        assert_eq!(16 - game.board.count_empty(), 2); // 2 tiles placed
    }

    #[test]
    fn test_merge() {
        let mut game = Game::new();
        game.board = Board::empty();
        game.board.set(0, 0, 1); // 2
        game.board.set(0, 1, 1); // 2

        let (_, changed) = game.make_move(Direction::Left);
        assert!(changed);
        assert_eq!(game.board.get(0, 0), 2); // Merged to 4
        assert_eq!(game.score, 4);
    }

    #[test]
    fn test_bitboard_get_set() {
        let mut board = Board::empty();
        board.set(1, 2, 5);
        assert_eq!(board.get(1, 2), 5);
        assert_eq!(board.tile_value(1, 2), 32);
    }
}
