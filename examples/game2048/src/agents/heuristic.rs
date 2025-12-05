//! Heuristic agent using corner strategy.

use crate::env::Env;
use crate::game::{Direction, Game};
use super::Agent;

/// Runs episodes with heuristic agent and prints statistics.
pub fn run_episodes(episodes: usize, verbose: bool) {
    let mut total_score = 0u64;
    let mut best_score = 0u32;
    let mut best_tile = 0u32;
    let mut tile_counts = [0usize; 12];

    for ep in 1..=episodes {
        let mut env = Env::new();
        let mut agent = HeuristicAgent::new();
        
        while !env.is_done() {
            let state = env.get_state();
            let action = agent.select_action(&state, &env, 0.0);
            env.step(action);
        }

        let score = env.score();
        let max_tile = env.max_tile();
        
        total_score += score as u64;
        if score > best_score {
            best_score = score;
        }
        if max_tile > best_tile {
            best_tile = max_tile;
        }

        let tile_idx = if max_tile > 0 {
            (max_tile as f32).log2() as usize - 1
        } else {
            0
        };
        if tile_idx < tile_counts.len() {
            tile_counts[tile_idx] += 1;
        }

        if verbose || ep % 100 == 0 || ep == episodes {
            println!(
                "Episode {:4}/{}: Score {:5}, Max Tile {:4}, Avg: {:.1}",
                ep,
                episodes,
                score,
                max_tile,
                total_score as f64 / ep as f64
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                   HEURISTIC AGENT RESULTS                     ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Episodes:       {}", episodes);
    println!("Average score:  {:.1}", total_score as f64 / episodes as f64);
    println!("Best score:     {}", best_score);
    println!("Best tile:      {}", best_tile);
    println!();
    println!("Tile distribution:");
    for (i, &count) in tile_counts.iter().enumerate() {
        if count > 0 {
            let tile = 1u32 << (i + 1);
            let pct = count as f64 / episodes as f64 * 100.0;
            println!("  {:5}: {:4} ({:5.1}%)", tile, count, pct);
        }
    }
}

/// Heuristic agent using corner strategy with simple lookahead.
pub struct HeuristicAgent {
    /// Preferred corner (0 = top-left)
    corner: usize,
}

impl HeuristicAgent {
    pub fn new() -> Self {
        Self { corner: 0 } // Top-left corner strategy
    }

    /// Evaluates board state using heuristics.
    fn evaluate(&self, game: &Game) -> f32 {
        if game.game_over {
            return -100000.0;
        }

        let board = &game.board;
        let mut score = 0.0f32;

        // 1. Empty cells (more is better)
        let empty = board.count_empty() as f32;
        score += empty * 100.0;

        // 2. Monotonicity (tiles should decrease from corner)
        score += self.monotonicity_score(board);

        // 3. Smoothness (adjacent tiles should be similar)
        score += self.smoothness_score(board);

        // 4. Max tile in corner bonus
        score += self.corner_bonus(board);

        // 5. Merge potential
        score += self.merge_score(board);

        score
    }

    fn monotonicity_score(&self, board: &crate::game::Board) -> f32 {
        // Weight matrix favoring top-left corner
        const WEIGHTS: [[f32; 4]; 4] = [
            [15.0, 14.0, 13.0, 12.0],
            [8.0, 9.0, 10.0, 11.0],
            [7.0, 6.0, 5.0, 4.0],
            [0.0, 1.0, 2.0, 3.0],
        ];

        let mut score = 0.0f32;
        for row in 0..4 {
            for col in 0..4 {
                let val = board.get(row, col) as f32;
                score += val * WEIGHTS[row][col];
            }
        }
        score * 10.0
    }

    fn smoothness_score(&self, board: &crate::game::Board) -> f32 {
        let mut score = 0.0f32;
        
        for row in 0..4 {
            for col in 0..4 {
                let val = board.get(row, col);
                if val == 0 {
                    continue;
                }
                
                // Penalize difference with neighbors
                if col < 3 {
                    let right = board.get(row, col + 1);
                    if right > 0 {
                        score -= ((val as i32 - right as i32).abs() as f32) * 5.0;
                    }
                }
                if row < 3 {
                    let down = board.get(row + 1, col);
                    if down > 0 {
                        score -= ((val as i32 - down as i32).abs() as f32) * 5.0;
                    }
                }
            }
        }
        
        score
    }

    fn corner_bonus(&self, board: &crate::game::Board) -> f32 {
        let max_val = (0..16)
            .map(|i| board.get(i / 4, i % 4))
            .max()
            .unwrap_or(0);

        // Check if max is in top-left corner
        if board.get(0, 0) == max_val {
            500.0
        } else if board.get(0, 3) == max_val || board.get(3, 0) == max_val || board.get(3, 3) == max_val {
            200.0
        } else {
            -100.0
        }
    }

    fn merge_score(&self, board: &crate::game::Board) -> f32 {
        let mut score = 0.0f32;
        
        for row in 0..4 {
            for col in 0..4 {
                let val = board.get(row, col);
                if val == 0 {
                    continue;
                }
                
                if col < 3 && board.get(row, col + 1) == val {
                    score += (1 << val) as f32;
                }
                if row < 3 && board.get(row + 1, col) == val {
                    score += (1 << val) as f32;
                }
            }
        }
        
        score
    }

    /// Simple 1-ply lookahead.
    fn best_action(&self, game: &Game) -> usize {
        let directions = [Direction::Up, Direction::Left, Direction::Down, Direction::Right];
        let mut best_score = f32::NEG_INFINITY;
        let mut best_action = 0;

        for (i, &dir) in directions.iter().enumerate() {
            let mut test_game = game.clone();
            let (_, changed) = test_game.make_move(dir);
            
            if changed {
                let score = self.evaluate(&test_game);
                if score > best_score {
                    best_score = score;
                    best_action = i;
                }
            }
        }

        best_action
    }
}

impl super::Agent for HeuristicAgent {
    fn select_action(&mut self, _state: &[f32], env: &Env, _epsilon: f32) -> usize {
        self.best_action(env.game())
    }

    fn name(&self) -> &str {
        "Heuristic"
    }
}
