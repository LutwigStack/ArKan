//! CPU-based DQN training with parallel experience collection.

use super::TrainingHistory;
use crate::agents::kan_dqn::KanDqnAgent;
use crate::agents::Agent;
use crate::env::{Env, Experience, ReplayBuffer};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Number of parallel environments for experience collection.
const NUM_ENVS: usize = 8;

/// Trains DQN agent on CPU with parallel environment rollouts.
pub fn train(
    episodes: usize,
    batch_size: usize,
    lr: f32,
    gamma: f32,
    replay_capacity: usize,
    target_update: usize,
    epsilon_start: f32,
    epsilon_end: f32,
    epsilon_decay: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize main agent (used for training)
    let agent = Arc::new(RwLock::new(KanDqnAgent::new(lr)?));
    let replay_buffer = Arc::new(RwLock::new(ReplayBuffer::new(replay_capacity)));
    let epsilon = Arc::new(RwLock::new(epsilon_start));

    // Statistics
    let mut total_score = 0u64;
    let mut best_score = 0u32;
    let mut best_tile = 0u32;
    let mut recent_scores: Vec<u32> = Vec::with_capacity(100);
    let mut recent_losses: Vec<f32> = Vec::with_capacity(100);
    let mut history = TrainingHistory::new();

    // Progress bar
    let pb = ProgressBar::new(episodes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let start = Instant::now();
    let mut total_steps = 0usize;
    let mut train_steps = 0usize;

    println!();
    println!("Training started ({} parallel envs)...", NUM_ENVS);
    println!("─────────────────────────────────────────────────────────────");

    // Process episodes in batches of NUM_ENVS
    let num_batches = (episodes + NUM_ENVS - 1) / NUM_ENVS;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * NUM_ENVS;
        let batch_end = (batch_start + NUM_ENVS).min(episodes);
        let batch_count = batch_end - batch_start;

        // Get current weights and epsilon for workers
        let current_epsilon = *epsilon.read().unwrap();
        
        // Collect weights from main agent for workers
        let weights: Vec<(Vec<f32>, Vec<f32>)> = {
            let agent_guard = agent.read().unwrap();
            agent_guard.policy().layers.iter()
                .map(|l| (l.weights.clone(), l.bias.clone()))
                .collect()
        };

        // Run batch_count episodes in parallel
        let episode_results: Vec<_> = (0..batch_count)
            .into_par_iter()
            .map(|_| {
                let mut env = Env::new();
                let mut experiences = Vec::with_capacity(200);
                let mut local_agent = KanDqnAgent::new(lr).unwrap();
                
                // Copy weights from main agent
                for (layer_idx, (w, b)) in weights.iter().enumerate() {
                    local_agent.policy_net.layers[layer_idx].weights.copy_from_slice(w);
                    local_agent.policy_net.layers[layer_idx].bias.copy_from_slice(b);
                }

                // Pre-allocated state buffers
                let mut state_buffer = [0.0f32; 256];
                let mut next_state_buffer = [0.0f32; 256];

                while !env.is_done() {
                    // Copy state to fixed-size buffer
                    let state = env.get_state();
                    state_buffer.copy_from_slice(&state);
                    
                    let action = local_agent.select_action(&state, &env, current_epsilon);
                    let (next_state, reward, done) = env.step(action);
                    next_state_buffer.copy_from_slice(&next_state);

                    experiences.push(Experience::from_arrays(
                        &state_buffer,
                        action,
                        reward,
                        &next_state_buffer,
                        done,
                    ));
                }

                (env.score(), env.max_tile(), experiences)
            })
            .collect();

        // Aggregate results
        for (score, max_tile, experiences) in episode_results {
            total_score += score as u64;
            total_steps += experiences.len();

            if score > best_score {
                best_score = score;
            }
            if max_tile > best_tile {
                best_tile = max_tile;
            }

            recent_scores.push(score);
            if recent_scores.len() > 100 {
                recent_scores.remove(0);
            }

            // Add experiences to replay buffer
            {
                let mut buffer = replay_buffer.write().unwrap();
                for exp in experiences {
                    buffer.push(exp);
                }
            }
        }

        // Training step: multiple updates per batch
        let updates_per_batch = batch_count * 8; // Aggressive training
        let mut batch_loss = 0.0f32;
        let mut batch_train_steps = 0;

        {
            let buffer = replay_buffer.read().unwrap();
            let mut agent_guard = agent.write().unwrap();
            
            if buffer.len() >= batch_size {
                for _ in 0..updates_per_batch {
                    if let Some((states, actions, rewards, next_states, dones)) =
                        buffer.sample_batch(batch_size)
                    {
                        let loss = agent_guard.train_batch(
                            &states,
                            &actions,
                            &rewards,
                            &next_states,
                            &dones,
                            gamma,
                        );
                        batch_loss += loss;
                        batch_train_steps += 1;
                        train_steps += 1;
                    }
                }
            }
        }

        if batch_train_steps > 0 {
            recent_losses.push(batch_loss / batch_train_steps as f32);
            if recent_losses.len() > 100 {
                recent_losses.remove(0);
            }
        }

        // Decay epsilon
        {
            let mut eps = epsilon.write().unwrap();
            for _ in 0..batch_count {
                *eps = (*eps * epsilon_decay).max(epsilon_end);
            }
        }

        // Update target network
        let current_episode = batch_end;
        if current_episode % target_update < batch_count || batch_idx == 0 {
            let mut agent_guard = agent.write().unwrap();
            agent_guard.update_target_network();
        }

        // Progress update
        pb.set_position(batch_end as u64);

        // Periodic logging
        if current_episode % 50 < batch_count || batch_idx == 0 || batch_end == episodes {
            let avg_score: f32 =
                recent_scores.iter().sum::<u32>() as f32 / recent_scores.len().max(1) as f32;
            let avg_loss: f32 = if recent_losses.is_empty() {
                0.0
            } else {
                recent_losses.iter().sum::<f32>() / recent_losses.len() as f32
            };
            let elapsed = start.elapsed().as_secs_f64();
            let eps_per_sec = current_episode as f64 / elapsed;
            let current_eps = *epsilon.read().unwrap();

            // Record for graph
            history.record(current_episode, avg_score, avg_loss);

            pb.println(format!(
                "Ep {:5} | Best: {:5} | Tile: {:4} | Avg: {:6.1} | Loss: {:7.4} | ε: {:.3} | {:.1} ep/s",
                current_episode, best_score, best_tile, avg_score, avg_loss, current_eps, eps_per_sec
            ));
        }
    }

    pb.finish_with_message("Training complete!");

    // Final statistics
    let elapsed = start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    TRAINING COMPLETE                          ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Parallel envs:      {}", NUM_ENVS);
    println!("Total episodes:     {}", episodes);
    println!("Total steps:        {}", total_steps);
    println!("Training steps:     {}", train_steps);
    println!("Training time:      {:.1}s", elapsed.as_secs_f64());
    println!(
        "Episodes/second:    {:.1}",
        episodes as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Steps/second:       {:.1}",
        total_steps as f64 / elapsed.as_secs_f64()
    );
    println!();
    println!("Best score:         {}", best_score);
    println!("Best tile:          {}", best_tile);
    println!(
        "Average score:      {:.1}",
        total_score as f64 / episodes as f64
    );
    println!();

    // Play demo game
    println!("🎮 Demo game with trained agent (greedy policy):");
    println!("─────────────────────────────────────────────────────────────");

    let mut agent_guard = agent.write().unwrap();
    let mut env = Env::new();
    let mut moves = 0;

    while !env.is_done() {
        let state = env.get_state();
        let action = agent_guard.select_action(&state, &env, 0.0); // Greedy
        env.step(action);
        moves += 1;
    }

    env.render();
    println!("Moves: {}", moves);
    drop(agent_guard);

    // Test 100 games to get statistics
    println!();
    println!("📊 Testing trained agent (100 games):");
    println!("─────────────────────────────────────────────────────────────");
    
    let test_results: Vec<(u32, u32)> = (0..100)
        .into_par_iter()
        .map(|_| {
            let weights: Vec<(Vec<f32>, Vec<f32>)> = {
                let agent_guard = agent.read().unwrap();
                agent_guard.policy().layers.iter()
                    .map(|l| (l.weights.clone(), l.bias.clone()))
                    .collect()
            };
            
            let mut local_agent = KanDqnAgent::new(lr).unwrap();
            for (layer_idx, (w, b)) in weights.iter().enumerate() {
                local_agent.policy_net.layers[layer_idx].weights.copy_from_slice(w);
                local_agent.policy_net.layers[layer_idx].bias.copy_from_slice(b);
            }
            
            let mut env = Env::new();
            while !env.is_done() {
                let state = env.get_state();
                let action = local_agent.select_action(&state, &env, 0.0);
                env.step(action);
            }
            (env.score(), env.max_tile())
        })
        .collect();
    
    let test_scores: Vec<u32> = test_results.iter().map(|(s, _)| *s).collect();
    let test_tiles: Vec<u32> = test_results.iter().map(|(_, t)| *t).collect();
    
    let avg_test_score = test_scores.iter().sum::<u32>() as f64 / 100.0;
    let max_test_score = *test_scores.iter().max().unwrap();
    let max_test_tile = *test_tiles.iter().max().unwrap();
    
    // Count tile achievements
    let tile_256 = test_tiles.iter().filter(|&&t| t >= 256).count();
    let tile_512 = test_tiles.iter().filter(|&&t| t >= 512).count();
    let tile_1024 = test_tiles.iter().filter(|&&t| t >= 1024).count();
    let tile_2048 = test_tiles.iter().filter(|&&t| t >= 2048).count();
    
    println!("Average score:      {:.1}", avg_test_score);
    println!("Best score:         {}", max_test_score);
    println!("Best tile:          {}", max_test_tile);
    println!();
    println!("Tile achievements (out of 100 games):");
    println!("  256+:  {} games ({}%)", tile_256, tile_256);
    println!("  512+:  {} games ({}%)", tile_512, tile_512);
    println!("  1024+: {} games ({}%)", tile_1024, tile_1024);
    println!("  2048:  {} games ({}%)", tile_2048, tile_2048);

    // Print training graphs
    history.print_graph();

    Ok(())
}
