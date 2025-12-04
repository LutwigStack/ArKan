//! KAN-based DQN agent.

use crate::env::Env;
use arkan::{KanConfigBuilder, KanNetwork, Workspace, TrainOptions};

/// KAN DQN Agent for CPU training.
pub struct KanDqnAgent {
    /// Policy network (online network).
    pub policy_net: KanNetwork,
    /// Target network (for stable Q-value estimation).
    pub target_net: KanNetwork,
    /// CPU workspace for inference.
    workspace: Workspace,
    /// Output buffer.
    output: Vec<f32>,
    /// Training options.
    #[allow(dead_code)]
    train_opts: TrainOptions,
    /// Learning rate.
    lr: f32,
}

impl KanDqnAgent {
    /// Creates a new KAN DQN agent.
    pub fn new(lr: f32) -> Result<Self, Box<dyn std::error::Error>> {
        // Network architecture: 256 inputs (one-hot) -> hidden -> 4 outputs
        // Larger network for better representation with one-hot encoding
        let config = KanConfigBuilder::new()
            .input_dim(256)        // 16 cells * 16 possible values (one-hot)
            .hidden_dims(vec![64, 32])  // Hidden layers
            .output_dim(4)         // 4 actions
            .spline_order(3)       // Cubic splines
            .grid_size(5)          // 5 grid points
            .grid_range(0.0, 1.0)  // One-hot values are 0 or 1
            .build()?;

        let policy_net = KanNetwork::new(config.clone());
        let target_net = KanNetwork::new(config.clone());
        let workspace = Workspace::new(&config);
        let output = vec![0.0f32; 4];

        let train_opts = TrainOptions {
            max_grad_norm: Some(1.0), // Gradient clipping
            weight_decay: 0.0,
        };

        Ok(Self {
            policy_net,
            target_net,
            workspace,
            output,
            train_opts,
            lr,
        })
    }

    /// Gets Q-values for a state.
    pub fn get_q_values(&mut self, state: &[f32]) -> &[f32] {
        self.policy_net.forward_single(state, &mut self.output, &mut self.workspace);
        &self.output
    }

    /// Gets Q-values from target network.
    pub fn get_target_q_values(&mut self, state: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; 4];
        self.target_net.forward_single(state, &mut output, &mut self.workspace);
        output
    }

    /// Selects action using epsilon-greedy policy.
    pub fn select_action_eps(&mut self, state: &[f32], env: &Env, epsilon: f32) -> usize {
        let valid_actions = env.valid_actions();
        
        if valid_actions.is_empty() {
            return 0;
        }

        // Epsilon-greedy
        let rng_val: f32 = rand::random();
        if rng_val < epsilon {
            // Random valid action
            let idx = rand::random::<usize>() % valid_actions.len();
            valid_actions[idx]
        } else {
            // Greedy action from Q-values
            let q_values = self.get_q_values(state);
            
            // Find best valid action
            valid_actions
                .iter()
                .max_by(|&&a, &&b| {
                    q_values[a].partial_cmp(&q_values[b]).unwrap()
                })
                .copied()
                .unwrap_or(0)
        }
    }

    /// Performs a training step with batch of experiences.
    /// Returns the loss.
    pub fn train_batch(
        &mut self,
        states: &[f32],
        actions: &[usize],
        rewards: &[f32],
        next_states: &[f32],
        dones: &[bool],
        gamma: f32,
    ) -> f32 {
        let batch_size = actions.len();
        let state_dim = self.policy_net.layers[0].in_dim;  // Get from network config
        let action_dim = 4;

        // Compute target Q-values
        let mut targets = vec![0.0f32; batch_size * action_dim];
        
        for i in 0..batch_size {
            // Get current Q-values
            let state = &states[i * state_dim..(i + 1) * state_dim];
            let current_q = self.get_q_values(state).to_vec();
            
            // Copy current Q-values as baseline
            for a in 0..action_dim {
                targets[i * action_dim + a] = current_q[a];
            }
            
            // Compute target for taken action using Bellman equation
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

        // Train with SGD-style update using train_step
        // We'll accumulate loss over the batch
        let mut total_loss = 0.0f32;
        
        for i in 0..batch_size {
            let state = &states[i * state_dim..(i + 1) * state_dim];
            let target = &targets[i * action_dim..(i + 1) * action_dim];
            
            let loss = self.policy_net.train_step(
                state,
                target,
                None,  // No mask
                self.lr,
                &mut self.workspace,
            );
            total_loss += loss;
        }

        // Note: train_step already applies gradient updates internally
        // For more advanced optimization (Adam), we would need to use
        // backward() separately and then call optimizer.step()

        total_loss / batch_size as f32
    }

    /// Updates target network with policy network weights.
    pub fn update_target_network(&mut self) {
        for (policy_layer, target_layer) in self.policy_net.layers.iter()
            .zip(self.target_net.layers.iter_mut())
        {
            target_layer.weights.copy_from_slice(&policy_layer.weights);
            target_layer.bias.copy_from_slice(&policy_layer.bias);
        }
    }

    /// Copies weights from another agent's policy network.
    pub fn copy_weights_from(&mut self, other: &KanDqnAgent) {
        for (src_layer, dst_layer) in other.policy_net.layers.iter()
            .zip(self.policy_net.layers.iter_mut())
        {
            dst_layer.weights.copy_from_slice(&src_layer.weights);
            dst_layer.bias.copy_from_slice(&src_layer.bias);
        }
    }

    /// Returns the policy network (for weight access).
    pub fn policy(&self) -> &KanNetwork {
        &self.policy_net
    }

    /// Soft update of target network (Polyak averaging).
    #[allow(dead_code)]
    pub fn soft_update_target(&mut self, tau: f32) {
        for (policy_layer, target_layer) in self.policy_net.layers.iter()
            .zip(self.target_net.layers.iter_mut())
        {
            for (tp, pp) in target_layer.weights.iter_mut().zip(policy_layer.weights.iter()) {
                *tp = tau * pp + (1.0 - tau) * *tp;
            }
            for (tb, pb) in target_layer.bias.iter_mut().zip(policy_layer.bias.iter()) {
                *tb = tau * pb + (1.0 - tau) * *tb;
            }
        }
    }
}

impl super::Agent for KanDqnAgent {
    fn select_action(&mut self, state: &[f32], env: &Env, epsilon: f32) -> usize {
        self.select_action_eps(state, env, epsilon)
    }

    fn name(&self) -> &str {
        "KAN-DQN"
    }
}
