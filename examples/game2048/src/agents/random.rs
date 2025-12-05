//! Random agent (baseline).

use crate::env::Env;
use rand::Rng;

/// Runs episodes with random agent and prints statistics.
pub fn run_episodes(episodes: usize, verbose: bool) {
    let mut total_score = 0u64;
    let mut best_score = 0u32;
    let mut best_tile = 0u32;
    let mut tile_counts = [0usize; 12]; // Count of max tiles reached (2, 4, 8, ..., 2048, 4096)

    for ep in 1..=episodes {
        let mut env = Env::new();
        
        while !env.is_done() {
            let valid = env.valid_actions();
            if valid.is_empty() {
                break;
            }
            
            let action = valid[rand::thread_rng().gen_range(0..valid.len())];
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

        // Track tile distribution
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
    println!("                    RANDOM AGENT RESULTS                       ");
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

/// Random agent that selects valid actions uniformly.
pub struct RandomAgent;

impl super::Agent for RandomAgent {
    fn select_action(&mut self, _state: &[f32], env: &Env, _epsilon: f32) -> usize {
        let valid = env.valid_actions();
        if valid.is_empty() {
            0
        } else {
            valid[rand::thread_rng().gen_range(0..valid.len())]
        }
    }

    fn name(&self) -> &str {
        "Random"
    }
}
