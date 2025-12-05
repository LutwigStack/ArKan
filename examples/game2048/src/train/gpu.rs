//! GPU-based DQN training.
//!
//! Two modes:
//! - Hybrid: GPU forward/backward + CPU optimizer (stable, has gradient clipping)
//! - Native: All on GPU with GpuAdam (fastest, but no gradient clipping)
//!
//! Optimized for high throughput with large batch parallel training.

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
/// Increased for better throughput.
const NUM_ENVS: usize = 32;

/// Batch size for GPU training (larger = better GPU utilization).
const GPU_BATCH_SIZE: usize = 256;

/// How often to sync weights to worker agents (in episodes).
const WEIGHT_SYNC_INTERVAL: usize = 16;

/// Pre-allocated batch size for compute_targets optimization.
const TARGET_BATCH_SIZE: usize = 256;

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
    /// CPU workspace for single-sample inference.
    cpu_workspace: Workspace,
    /// CPU workspace for batched forward (compute_targets).
    cpu_batch_workspace: Workspace,
    /// GPU workspace for training.
    gpu_workspace: GpuWorkspace,
    /// CPU optimizer (for hybrid mode).
    cpu_optimizer: Adam,
    /// GPU optimizer (for native mode).
    gpu_optimizer: Option<GpuAdam>,
    /// Output buffer for single sample.
    output: Vec<f32>,
    /// Pre-allocated buffer for batch current Q-values.
    batch_current_q: Vec<f32>,
    /// Pre-allocated buffer for batch next Q-values.
    batch_next_q: Vec<f32>,
    /// Training options.
    train_opts: TrainOptions,
    /// Is native mode.
    native_mode: bool,
}

impl GpuDqnAgent {
    /// Creates a new GPU DQN agent.
    pub fn new(lr: f32, native_mode: bool) -> Result<Self, Box<dyn std::error::Error>> {
        // Network architecture must match KanDqnAgent!
        // 256 inputs (one-hot: 16 cells * 16 possible values) -> hidden -> 4 outputs
        let config = KanConfigBuilder::new()
            .input_dim(256)        // 16 cells * 16 possible values (one-hot)
            .hidden_dims(vec![64, 32])  // Hidden layers (same as KanDqnAgent)
            .output_dim(4)         // 4 actions
            .spline_order(3)       // Cubic splines
            .grid_size(5)          // 5 grid points
            .grid_range(0.0, 1.0)  // One-hot values are 0 or 1
            .build()?;

        let cpu_policy = KanNetwork::new(config.clone());
        let cpu_target = KanNetwork::new(config.clone());
        let cpu_workspace = Workspace::new(&config);
        // Batch workspace for compute_targets optimization
        let cpu_batch_workspace = cpu_policy.create_workspace(TARGET_BATCH_SIZE);

        // Initialize WGPU backend
        let wgpu_opts = WgpuOptions::compute();
        let backend = WgpuBackend::init(wgpu_opts)?;

        // Create GPU network from CPU
        let gpu_policy = GpuNetwork::from_cpu(&backend, &cpu_policy)?;

        // GPU workspace for batch training (larger batch = better GPU utilization)
        let gpu_workspace = gpu_policy.create_workspace(GPU_BATCH_SIZE)?;

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
            cpu_batch_workspace,
            gpu_workspace,
            cpu_optimizer,
            gpu_optimizer,
            output: vec![0.0f32; 4],
            batch_current_q: vec![0.0f32; TARGET_BATCH_SIZE * 4],
            batch_next_q: vec![0.0f32; TARGET_BATCH_SIZE * 4],
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
        let state_dim = 256;  // one-hot: 16 cells * 16 values
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
        let state_dim = 256;  // one-hot: 16 cells * 16 values
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

    /// Computes Q-learning targets for a batch using PARALLEL forward passes.
    /// Uses ArKan's built-in forward_batch_parallel for optimal performance.
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
        let action_dim = 4;

        // Resize pre-allocated buffers if needed
        if self.batch_current_q.len() < batch_size * action_dim {
            self.batch_current_q.resize(batch_size * action_dim, 0.0);
        }
        if self.batch_next_q.len() < batch_size * action_dim {
            self.batch_next_q.resize(batch_size * action_dim, 0.0);
        }

        // PARALLEL forward pass for current Q-values (policy network)
        self.cpu_policy.forward_batch_parallel(
            states,
            &mut self.batch_current_q[..batch_size * action_dim],
        );

        // PARALLEL forward pass for next Q-values (target network)
        self.cpu_target.forward_batch_parallel(
            next_states,
            &mut self.batch_next_q[..batch_size * action_dim],
        );

        // Build targets: copy current Q, then update the action taken
        let mut targets = self.batch_current_q[..batch_size * action_dim].to_vec();

        for i in 0..batch_size {
            let action = actions[i];
            let reward = rewards[i];
            let done = dones[i];

            let target_q = if done {
                reward
            } else {
                // Max Q from target network for next state
                let next_q_start = i * action_dim;
                let max_next_q = self.batch_next_q[next_q_start..next_q_start + action_dim]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
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
/// Optimized for high throughput with large batches.
fn train_impl(
    episodes: usize,
    _batch_size: usize, // Ignored, we use GPU_BATCH_SIZE
    lr: f32,
    gamma: f32,
    replay_capacity: usize,
    target_update: usize,
    epsilon_start: f32,
    epsilon_end: f32,
    epsilon_decay: f32,
    native_mode: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::cell::RefCell;
    
    let mode_name = if native_mode { "Native" } else { "Hybrid" };
    
    // Initialize GPU agent for training
    let mut gpu_agent = GpuDqnAgent::new(lr, native_mode)?;
    let replay_buffer = Arc::new(RwLock::new(ReplayBuffer::new(replay_capacity)));

    // Statistics
    let mut total_score = 0u64;
    let mut best_score = 0u32;
    let mut best_tile = 0u32;
    let mut recent_scores: Vec<u32> = Vec::with_capacity(100);
    let mut recent_losses: Vec<f32> = Vec::with_capacity(100);
    let mut nan_count = 0usize;
    let mut history = TrainingHistory::new();

    // Shared weights for workers (updated less frequently for speed)
    let shared_weights: Arc<RwLock<Vec<(Vec<f32>, Vec<f32>)>>> = Arc::new(RwLock::new(
        gpu_agent.cpu_policy.layers.iter()
            .map(|l| (l.weights.clone(), l.bias.clone()))
            .collect()
    ));

    // Thread-local agent pool to avoid recreating agents
    thread_local! {
        static LOCAL_AGENT: RefCell<Option<KanDqnAgent>> = const { RefCell::new(None) };
        static LOCAL_STATE_BUFFER: RefCell<[f32; 256]> = const { RefCell::new([0.0f32; 256]) };
        static LOCAL_NEXT_STATE_BUFFER: RefCell<[f32; 256]> = const { RefCell::new([0.0f32; 256]) };
    }

    // Pre-allocated batch buffers for sampling
    let mut batch_states = Vec::with_capacity(GPU_BATCH_SIZE * 256);
    let mut batch_actions = Vec::with_capacity(GPU_BATCH_SIZE);
    let mut batch_rewards = Vec::with_capacity(GPU_BATCH_SIZE);
    let mut batch_next_states = Vec::with_capacity(GPU_BATCH_SIZE * 256);
    let mut batch_dones = Vec::with_capacity(GPU_BATCH_SIZE);

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
    let mut current_epsilon = epsilon_start;

    println!();
    println!("Training started (GPU {} + {} parallel envs, batch {})...", 
             mode_name, NUM_ENVS, GPU_BATCH_SIZE);
    if native_mode {
        println!("⚠️  WARNING: Native mode has NO gradient clipping!");
    }
    println!("─────────────────────────────────────────────────────────────");

    // Process episodes in batches of NUM_ENVS
    let num_batches = (episodes + NUM_ENVS - 1) / NUM_ENVS;
    let mut completed_episodes = 0usize;

    for batch_idx in 0..num_batches {
        let batch_count = NUM_ENVS.min(episodes - completed_episodes);

        // Update shared weights periodically (not every batch for speed)
        if batch_idx % (WEIGHT_SYNC_INTERVAL / NUM_ENVS).max(1) == 0 {
            let mut weights = shared_weights.write().unwrap();
            *weights = gpu_agent.cpu_policy.layers.iter()
                .map(|l| (l.weights.clone(), l.bias.clone()))
                .collect();
        }

        // Clone for parallel use
        let weights_snapshot = shared_weights.read().unwrap().clone();
        let eps = current_epsilon;
        let lr_copy = lr;

        // Run episodes in parallel using thread-local reusable CPU agents
        let episode_results: Vec<_> = (0..batch_count)
            .into_par_iter()
            .map(|_| {
                use crate::utils::board_to_onehot_inplace;
                
                let mut env = Env::new();
                let mut experiences = Vec::with_capacity(200);
                
                // Reuse thread-local agent (create only once per thread)
                LOCAL_AGENT.with(|agent_cell| {
                    let mut agent_ref = agent_cell.borrow_mut();
                    if agent_ref.is_none() {
                        *agent_ref = Some(KanDqnAgent::new(lr_copy).unwrap());
                    }
                    let local_agent = agent_ref.as_mut().unwrap();

                    // Copy weights from snapshot
                    for (layer_idx, (w, b)) in weights_snapshot.iter().enumerate() {
                        local_agent.policy_net.layers[layer_idx].weights.copy_from_slice(w);
                        local_agent.policy_net.layers[layer_idx].bias.copy_from_slice(b);
                    }

                    // Use thread-local state buffers
                    LOCAL_STATE_BUFFER.with(|state_buf| {
                        LOCAL_NEXT_STATE_BUFFER.with(|next_state_buf| {
                            let mut state_buffer = state_buf.borrow_mut();
                            let mut next_state_buffer = next_state_buf.borrow_mut();
                            
                            while !env.is_done() {
                                // Zero-allocation state encoding
                                board_to_onehot_inplace(env.board(), &mut state_buffer[..]);
                                
                                let action = local_agent.select_action(&state_buffer[..], &env, eps);
                                let (_, reward, done) = env.step(action);
                                
                                // Zero-allocation next state encoding
                                board_to_onehot_inplace(env.board(), &mut next_state_buffer[..]);

                                experiences.push(Experience::from_arrays(
                                    &*state_buffer,
                                    action,
                                    reward,
                                    &*next_state_buffer,
                                    done,
                                ));
                            }
                        });
                    });
                });

                (env.score(), env.max_tile(), experiences)
            })
            .collect();

        // Aggregate results and add to replay buffer
        for (score, max_tile, experiences) in episode_results {
            total_score += score as u64;
            total_steps += experiences.len();
            completed_episodes += 1;

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

            // Decay epsilon per episode
            current_epsilon = (current_epsilon * epsilon_decay).max(epsilon_end);
        }

        // GPU Training: multiple large-batch updates
        // More updates when buffer is fuller for better sample efficiency
        let buffer_len = replay_buffer.read().unwrap().len();
        let updates_per_batch = if buffer_len >= GPU_BATCH_SIZE * 4 {
            batch_count * 2  // More updates when buffer is full
        } else {
            batch_count
        };

        let mut batch_loss = 0.0f32;
        let mut batch_train_steps = 0;

        {
            let buffer = replay_buffer.read().unwrap();

            if buffer.len() >= GPU_BATCH_SIZE {
                for _ in 0..updates_per_batch {
                    // Use optimized batch sampling into pre-allocated buffers
                    if buffer.sample_batch_into(
                        GPU_BATCH_SIZE,
                        &mut batch_states,
                        &mut batch_actions,
                        &mut batch_rewards,
                        &mut batch_next_states,
                        &mut batch_dones,
                    ).is_some() {
                        // Compute targets using CPU target network
                        let targets = gpu_agent.compute_targets(
                            &batch_states, &batch_actions, &batch_rewards, &batch_next_states, &batch_dones, gamma
                        );

                        // Train on GPU with large batch
                        let loss_result = if native_mode {
                            gpu_agent.train_on_targets_native(&batch_states, &targets)
                        } else {
                            gpu_agent.train_on_targets_hybrid(&batch_states, &targets)
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

        // Update target network
        if completed_episodes % target_update < batch_count || batch_idx == 0 {
            gpu_agent.update_target_network();
        }

        // Progress update
        pb.set_position(completed_episodes as u64);

        // Periodic logging (every ~100 episodes for cleaner output)
        if completed_episodes % 100 < batch_count || batch_idx == 0 || completed_episodes >= episodes {
            let avg_score: f32 =
                recent_scores.iter().sum::<u32>() as f32 / recent_scores.len().max(1) as f32;
            let avg_loss: f32 = if recent_losses.is_empty() {
                0.0
            } else {
                recent_losses.iter().sum::<f32>() / recent_losses.len() as f32
            };
            let elapsed = start.elapsed().as_secs_f64();
            let eps_per_sec = completed_episodes as f64 / elapsed;

            // Record for graph
            history.record(completed_episodes, avg_score, avg_loss);

            pb.println(format!(
                "Ep {:5} | Best: {:5} | Tile: {:4} | Avg: {:6.1} | Loss: {:7.4} | ε: {:.3} | {:.1} ep/s",
                completed_episodes, best_score, best_tile, avg_score, avg_loss, current_epsilon, eps_per_sec
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
    println!("GPU batch size:     {}", GPU_BATCH_SIZE);
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
