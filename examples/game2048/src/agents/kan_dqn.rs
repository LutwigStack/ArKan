//! KAN-based DQN agent.

use crate::env::Env;
use arkan::{KanConfigBuilder, KanNetwork, Workspace};

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
            // grid_range applies to EVERY layer, not just the input layer. Only layer 0
            // gets input_mean/input_std; hidden layers use identity normalization, so a
            // hidden layer's input is the previous layer's raw activation — which is not
            // bounded to [0,1] just because the one-hot inputs are.
            //
            // This used to be (0.0, 1.0) "because one-hot values are 0 or 1". Measured
            // consequence: layer0 0% saturated, but layer1 43.6% and layer2 48.9%, because
            // every negative activation collapsed onto the lower bound (observed z range
            // was [0.000, 0.801] — that exact 0.000 minimum is the clamp, not the data).
            // Nearly half of each hidden layer was dead: constant output, zero gradient.
            //
            // With (-1.0, 1.0) saturation is 0% on all three layers and activations sit
            // comfortably inside at [-0.771, 0.755] and [-0.604, 0.508].
            .grid_range(-1.0, 1.0)
            .build()?;

        let policy_net = KanNetwork::new(config.clone());
        let target_net = policy_net.clone();
        let workspace = Workspace::new(&config);
        let output = vec![0.0f32; 4];

        Ok(Self {
            policy_net,
            target_net,
            workspace,
            output,
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
                crate::utils::bellman_target(next_state, &next_q, reward, gamma)
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

}

impl super::Agent for KanDqnAgent {
    fn select_action(&mut self, state: &[f32], env: &Env, epsilon: f32) -> usize {
        self.select_action_eps(state, env, epsilon)
    }

    fn name(&self) -> &str {
        "KAN-DQN"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{game::Board, utils::board_to_onehot_inplace};

    #[test]
    fn initial_target_is_policy_snapshot() {
        let agent = KanDqnAgent::new(0.0).unwrap();
        for (policy, target) in agent.policy_net.layers.iter().zip(&agent.target_net.layers) {
            assert_eq!(policy.weights, target.weights);
            assert_eq!(policy.bias, target.bias);
        }
    }

    #[test]
    fn bellman_loss_excludes_board_preserving_actions() {
        let mut agent = KanDqnAgent::new(0.0).unwrap();
        for layer in agent.policy_net.layers.iter_mut().chain(&mut agent.target_net.layers) {
            layer.weights.fill(0.0);
            layer.bias.fill(0.0);
        }
        agent.target_net.layers.last_mut().unwrap().bias.copy_from_slice(&[100.0, 1.0, 2.0, 3.0]);
        let mut board = Board::empty();
        board.set(0, 0, 1);
        board.set(0, 3, 2);
        let mut state = [0.0; 256];
        board_to_onehot_inplace(&board, &mut state);
        assert_eq!(agent.train_batch(&state, &[1], &[0.0], &state, &[false], 1.0), 2.25);
        assert_eq!(agent.train_batch(&state, &[1], &[2.0], &state, &[true], 1.0), 1.0);
        board_to_onehot_inplace(&Board::empty(), &mut state);
        assert_eq!(agent.train_batch(&state, &[1], &[2.0], &state, &[false], 1.0), 1.0);
    }
}
