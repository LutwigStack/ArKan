//! GPU-based DQN training.
//!
//! Two modes:
//! - Hybrid: GPU forward/backward + CPU optimizer (stable, has gradient clipping)
//! - Native: All on GPU with GpuAdam (fastest, but no gradient clipping)

use super::TrainingHistory;
use crate::agents::kan_dqn::KanDqnAgent;
use crate::agents::Agent;
use crate::env::{Env, Experience, ReplayBuffer};

use arkan::optimizer::{Adam, AdamConfig};
use arkan::{KanConfigBuilder, KanNetwork, TrainOptions, Workspace};
use arkan::gpu::{GpuAdam, GpuAdamConfig, GpuNetwork, GpuWorkspace, WgpuBackend, WgpuOptions};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Number of parallel environments for experience collection.
const NUM_ENVS: usize = 8;

/// GPU DQN Agent for hybrid/native training.
pub struct GpuDqnAgent {
    /// WGPU backend (device/queue).
    backend: WgpuBackend,
    /// CPU policy network (for weight storage and hybrid updates).
    pub cpu_policy: KanNetwork,
    /// CPU target network.
    pub cpu_target: KanNetwork,
    /// GPU policy network.
    pub gpu_policy: GpuNetwork,
    /// CPU workspace for inference.
    cpu_workspace: Workspace,
    /// GPU workspace for training.
    gpu_workspace: GpuWorkspace,
    /// CPU optimizer (for hybrid mode).
    cpu_optimizer: Adam,
    /// GPU optimizer (for native mode).
    gpu_optimizer: Option<GpuAdam>,
    /// Output buffer.
    output: Vec<f32>,
    /// Training options.
    train_opts: TrainOptions,
    /// Is native mode.
    native_mode: bool,
}

impl GpuDqnAgent {
    /// Creates a new GPU DQN agent.
    pub fn new(lr: f32, native_mode: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let config = KanConfigBuilder::new()
            .input_dim(16)
            .hidden_dims(vec![32, 16])
            .output_dim(4)
            .spline_order(3)
            .grid_size(5)
            .grid_range(-1.0, 1.0)
            .build()?;

        let cpu_policy = KanNetwork::new(config.clone());
        let cpu_target = KanNetwork::new(config.clone());
        let cpu_workspace = Workspace::new(&config);

        // Initialize WGPU backend
        let wgpu_opts = WgpuOptions::compute();
        let backend = WgpuBackend::init(wgpu_opts)?;

        // Create GPU network from CPU
        let gpu_policy = GpuNetwork::from_cpu(&backend, &cpu_policy)?;

        // GPU workspace for batch training (max batch 64)
        let gpu_workspace = gpu_policy.create_workspace(64)?;

        // CPU optimizer for hybrid mode
        let cpu_optimizer = Adam::new(&cpu_policy, AdamConfig::with_lr(lr));

        // GPU optimizer for native mode
        let gpu_optimizer = if native_mode {
            let layer_sizes = gpu_policy.layer_param_sizes();
            let gpu_adam_config = GpuAdamConfig {
                lr,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
            };
            Some(GpuAdam::new(
                backend.device_arc(),
                backend.queue_arc(),
                &layer_sizes,
                gpu_adam_config,
            ))
        } else {
            None
        };

        let train_opts = TrainOptions {
            max_grad_norm: Some(1.0),
            weight_decay: 0.0,
        };

        Ok(Self {
            backend,
            cpu_policy,
            cpu_target,
            gpu_policy,
            cpu_workspace,
            gpu_workspace,
            cpu_optimizer,
            gpu_optimizer,
            output: vec![0.0f32; 4],
            train_opts,
            native_mode,
        })
    }

    /// Gets Q-values using CPU (for inference during gameplay).
    pub fn get_q_values(&mut self, state: &[f32]) -> &[f32] {
        self.cpu_policy
            .forward_single(state, &mut self.output, &mut self.cpu_workspace);
        &self.output
    }

    /// Gets Q-values from target network.
    pub fn get_target_q_values(&mut self, state: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; 4];
        self.cpu_target
            .forward_single(state, &mut output, &mut self.cpu_workspace);
        output
    }

    /// Selects action using epsilon-greedy policy.
    pub fn select_action_eps(&mut self, state: &[f32], env: &Env, epsilon: f32) -> usize {
        let valid_actions = env.valid_actions();

        if valid_actions.is_empty() {
            return 0;
        }

        let rng_val: f32 = rand::random();
        if rng_val < epsilon {
            let idx = rand::random::<usize>() % valid_actions.len();
            valid_actions[idx]
        } else {
            let q_values = self.get_q_values(state);
            valid_actions
                .iter()
                .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap())
                .copied()
                .unwrap_or(0)
        }
    }

    /// Hybrid training: GPU forward/backward + CPU weight update.
    /// Uses train_step_with_options which has gradient clipping.
    pub fn train_batch_hybrid(
        &mut self,
        states: &[f32],
        actions: &[usize],
        rewards: &[f32],
        next_states: &[f32],
        dones: &[bool],
        gamma: f32,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let batch_size = actions.len();
        let state_dim = 16;
        let action_dim = 4;

        // Compute target Q-values using CPU target network
        let mut targets = vec![0.0f32; batch_size * action_dim];

        for i in 0..batch_size {
            let state = &states[i * state_dim..(i + 1) * state_dim];

            // Current Q-values from CPU policy
            let mut current_q = vec![0.0f32; 4];
            self.cpu_policy
                .forward_single(state, &mut current_q, &mut self.cpu_workspace);

            for a in 0..action_dim {
                targets[i * action_dim + a] = current_q[a];
            }

            let action = actions[i];
            let reward = rewards[i];
            let done = dones[i];

            let target_q = if done {
                reward
            } else {
                let next_state = &next_states[i * state_dim..(i + 1) * state_dim];
                let next_q = self.get_target_q_values(next_state);
                let max_next_q = next_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                reward + gamma * max_next_q
            };

            targets[i * action_dim + action] = target_q;
        }

        // Use GPU hybrid training: GPU forward/backward, CPU optimizer
        let loss = self.gpu_policy.train_step_with_options(
            states,
            &targets,
            None, // No mask
            batch_size,
            &mut self.gpu_workspace,
            &mut self.cpu_optimizer,
            &mut self.cpu_policy,
            &self.train_opts,
        )?;

        Ok(loss)
    }

    /// Native GPU training: all computation on GPU.
    /// WARNING: No gradient clipping! May be unstable.
    pub fn train_batch_native(
        &mut self,
        states: &[f32],
        actions: &[usize],
        rewards: &[f32],
        next_states: &[f32],
        dones: &[bool],
        gamma: f32,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let batch_size = actions.len();
        let state_dim = 16;
        let action_dim = 4;

        // For native mode, we still compute targets on CPU for simplicity
        let mut targets = vec![0.0f32; batch_size * action_dim];

        for i in 0..batch_size {
            let state = &states[i * state_dim..(i + 1) * state_dim];

            let mut current_q = vec![0.0f32; 4];
            self.cpu_policy
                .forward_single(state, &mut current_q, &mut self.cpu_workspace);

            for a in 0..action_dim {
                targets[i * action_dim + a] = current_q[a];
            }

            let action = actions[i];
            let reward = rewards[i];
            let done = dones[i];

            let target_q = if done {
                reward
            } else {
                let next_state = &next_states[i * state_dim..(i + 1) * state_dim];
                let next_q = self.get_target_q_values(next_state);
                let max_next_q = next_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                reward + gamma * max_next_q
            };

            targets[i * action_dim + action] = target_q;
        }

        // Use native GPU training with GpuAdam (no gradient clipping!)
        let gpu_opt = self
            .gpu_optimizer
            .as_mut()
            .expect("GpuAdam required for native mode");

        let loss = self.gpu_policy.train_step_gpu_native(
            states,
            &targets,
            batch_size,
            &mut self.gpu_workspace,
            gpu_opt,
        )?;

        // Sync weights back to CPU for inference
        self.gpu_policy.sync_weights_to_cpu(&mut self.cpu_policy)?;

        Ok(loss)
    }

    /// Computes Q-learning targets for a batch.
    pub fn compute_targets(
        &mut self,
        states: &[f32],
        actions: &[usize],
        rewards: &[f32],
        next_states: &[f32],
        dones: &[bool],
        gamma: f32,
    ) -> Vec<f32> {
        let batch_size = actions.len();
        let state_dim = 16;
        let action_dim = 4;

        let mut targets = vec![0.0f32; batch_size * action_dim];

        for i in 0..batch_size {
            let state = &states[i * state_dim..(i + 1) * state_dim];

            // Current Q-values
            let mut current_q = vec![0.0f32; 4];
            self.cpu_policy
                .forward_single(state, &mut current_q, &mut self.cpu_workspace);

            for a in 0..action_dim {
                targets[i * action_dim + a] = current_q[a];
            }

            let action = actions[i];
            let reward = rewards[i];
            let done = dones[i];

            let target_q = if done {
                reward
            } else {
                let next_state = &next_states[i * state_dim..(i + 1) * state_dim];
                let next_q = self.get_target_q_values(next_state);
                let max_next_q = next_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                reward + gamma * max_next_q
            };

            targets[i * action_dim + action] = target_q;
        }

        targets
    }

    /// Train on pre-computed targets using hybrid mode.
    pub fn train_on_targets_hybrid(
        &mut self,
        states: &[f32],
        targets: &[f32],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let batch_size = targets.len() / 4;

        let loss = self.gpu_policy.train_step_with_options(
            states,
            targets,
            None,
            batch_size,
            &mut self.gpu_workspace,
            &mut self.cpu_optimizer,
            &mut self.cpu_policy,
            &self.train_opts,
        )?;

        Ok(loss)
    }

    /// Train on pre-computed targets using native GPU mode.
    pub fn train_on_targets_native(
        &mut self,
        states: &[f32],
        targets: &[f32],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let batch_size = targets.len() / 4;

        let gpu_opt = self
            .gpu_optimizer
            .as_mut()
            .expect("GpuAdam required for native mode");

        let loss = self.gpu_policy.train_step_gpu_native(
            states,
            targets,
            batch_size,
            &mut self.gpu_workspace,
            gpu_opt,
        )?;

        // Sync weights back to CPU for inference
        self.gpu_policy.sync_weights_to_cpu(&mut self.cpu_policy)?;

        Ok(loss)
    }

    /// Updates target network.
    pub fn update_target_network(&mut self) {
        for (policy_layer, target_layer) in self
            .cpu_policy
            .layers
            .iter()
            .zip(self.cpu_target.layers.iter_mut())
        {
            target_layer.weights.copy_from_slice(&policy_layer.weights);
            target_layer.bias.copy_from_slice(&policy_layer.bias);
        }
    }
}

impl Agent for GpuDqnAgent {
    fn select_action(&mut self, state: &[f32], env: &Env, epsilon: f32) -> usize {
        self.select_action_eps(state, env, epsilon)
    }

    fn name(&self) -> &str {
        "GPU-DQN"
    }
}

/// Trains DQN agent using GPU hybrid mode.
pub fn train_hybrid(
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
    train_impl(
        episodes, batch_size, lr, gamma, replay_capacity, target_update,
        epsilon_start, epsilon_end, epsilon_decay, false,
    )
}

/// Trains DQN agent using GPU native mode.
pub fn train_native(
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
    train_impl(
        episodes, batch_size, lr, gamma, replay_capacity, target_update,
        epsilon_start, epsilon_end, epsilon_decay, true,
    )
}

/// Implementation of GPU training with parallel experience collection.
fn train_impl(
    episodes: usize,
    batch_size: usize,
    lr: f32,
    gamma: f32,
    replay_capacity: usize,
    target_update: usize,
    epsilon_start: f32,
    epsilon_end: f32,
    epsilon_decay: f32,
    native_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mode_name = if native_mode { "Native" } else { "Hybrid" };
    
    // Initialize GPU agent for training
    let mut gpu_agent = GpuDqnAgent::new(lr, native_mode)?;
    let replay_buffer = Arc::new(RwLock::new(ReplayBuffer::new(replay_capacity)));
    let epsilon = Arc::new(RwLock::new(epsilon_start));

    // Statistics
    let mut total_score = 0u64;
    let mut best_score = 0u32;
    let mut best_tile = 0u32;
    let mut recent_scores: Vec<u32> = Vec::with_capacity(100);
    let mut recent_losses: Vec<f32> = Vec::with_capacity(100);
    let mut nan_count = 0usize;
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
    println!("Training started (GPU {} + {} parallel envs)...", mode_name, NUM_ENVS);
    if native_mode {
        println!("⚠️  WARNING: Native mode has NO gradient clipping!");
    }
    println!("─────────────────────────────────────────────────────────────");

    // Process episodes in batches of NUM_ENVS
    let num_batches = (episodes + NUM_ENVS - 1) / NUM_ENVS;

    for batch_idx in 0..num_batches {
        let batch_start = batch_idx * NUM_ENVS;
        let batch_end = (batch_start + NUM_ENVS).min(episodes);
        let batch_count = batch_end - batch_start;

        // Get current epsilon
        let current_epsilon = *epsilon.read().unwrap();

        // Get current weights for workers (from CPU policy in GPU agent)
        let weights: Vec<(Vec<f32>, Vec<f32>)> = gpu_agent.cpu_policy.layers.iter()
            .map(|l| (l.weights.clone(), l.bias.clone()))
            .collect();

        // Run episodes in parallel using lightweight CPU agents for experience collection
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

                while !env.is_done() {
                    let state = env.get_state();
                    let action = local_agent.select_action(&state, &env, current_epsilon);
                    let (next_state, reward, done) = env.step(action);

                    experiences.push(Experience {
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    });
                }

                (env.score(), env.max_tile(), experiences)
            })
            .collect();

        // Aggregate results and add to replay buffer
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

        // GPU Training: multiple updates per batch
        let updates_per_batch = batch_count * 4;
        let mut batch_loss = 0.0f32;
        let mut batch_train_steps = 0;

        {
            let buffer = replay_buffer.read().unwrap();

            if buffer.len() >= batch_size {
                for _ in 0..updates_per_batch {
                    if let Some((states, actions, rewards, next_states, dones)) =
                        buffer.sample_batch(batch_size)
                    {
                        // Compute targets using CPU target network
                        let targets = gpu_agent.compute_targets(
                            &states, &actions, &rewards, &next_states, &dones, gamma
                        );

                        // Train on GPU
                        let loss_result = if native_mode {
                            gpu_agent.train_on_targets_native(&states, &targets)
                        } else {
                            gpu_agent.train_on_targets_hybrid(&states, &targets)
                        };

                        match loss_result {
                            Ok(loss) => {
                                if loss.is_nan() || loss.is_infinite() {
                                    nan_count += 1;
                                } else {
                                    batch_loss += loss;
                                    batch_train_steps += 1;
                                }
                            }
                            Err(_) => {
                                nan_count += 1;
                            }
                        }
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
            gpu_agent.update_target_network();
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
    println!("              TRAINING COMPLETE (GPU {})                ", mode_name);
    println!("═══════════════════════════════════════════════════════════════");
    println!("Parallel envs:      {}", NUM_ENVS);
    println!("Total episodes:     {}", episodes);
    println!("Total steps:        {}", total_steps);
    println!("Training steps:     {}", train_steps);
    println!("NaN/Inf events:     {}", nan_count);
    println!("Training time:      {:.1}s", elapsed.as_secs_f64());
    println!("Episodes/second:    {:.1}", episodes as f64 / elapsed.as_secs_f64());
    println!("Steps/second:       {:.1}", total_steps as f64 / elapsed.as_secs_f64());
    println!();
    println!("Best score:         {}", best_score);
    println!("Best tile:          {}", best_tile);
    println!("Average score:      {:.1}", total_score as f64 / episodes as f64);
    
    if nan_count > 0 {
        println!();
        println!("⚠️  {} NaN/Inf events detected.", nan_count);
        if native_mode {
            println!("   Consider using --mode train-gpu-hybrid for stability.");
        }
    }
    println!();

    // Demo game
    println!("🎮 Demo game with trained agent (greedy policy):");
    println!("─────────────────────────────────────────────────────────────");
    
    let mut env = Env::new();
    let mut moves = 0;
    
    while !env.is_done() {
        let state = env.get_state();
        let action = gpu_agent.select_action(&state, &env, 0.0);
        env.step(action);
        moves += 1;
    }
    
    env.render();
    println!("Final Score: {}", env.score());
    println!("Moves: {}", moves);

    // Print training graphs
    history.print_graph();

    Ok(())
}
