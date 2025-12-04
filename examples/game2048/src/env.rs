//! RL Environment wrapper for 2048.
//!
//! Provides a clean interface for agents to interact with the game.

use crate::game::{Direction, Game};
use crate::utils;

/// Experience tuple for replay buffer.
#[derive(Clone)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// RL Environment wrapper.
pub struct Env {
    game: Game,
    state_dim: usize,
}

impl Env {
    /// Creates a new environment.
    pub fn new() -> Self {
        Self {
            game: Game::new(),
            state_dim: 16, // 4x4 board
        }
    }

    /// Resets the environment and returns initial state.
    pub fn reset(&mut self) -> Vec<f32> {
        self.game = Game::new();
        self.get_state()
    }

    /// Gets the current state as one-hot encoded vector (256 features).
    pub fn get_state(&self) -> Vec<f32> {
        utils::board_to_onehot(&self.game.board)
    }

    /// Takes an action and returns (next_state, reward, done).
    pub fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let dir = Direction::from_index(action);
        let (reward, _changed) = self.game.make_move(dir);
        
        let next_state = self.get_state();
        let done = self.game.game_over;

        (next_state, reward, done)
    }

    /// Returns current score.
    pub fn score(&self) -> u32 {
        self.game.score
    }

    /// Returns maximum tile value.
    pub fn max_tile(&self) -> u32 {
        self.game.board.max_tile()
    }

    /// Returns true if game is over.
    pub fn is_done(&self) -> bool {
        self.game.game_over
    }

    /// Returns the state dimension.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Returns the action dimension (4 directions).
    pub fn action_dim(&self) -> usize {
        4
    }

    /// Returns valid actions.
    pub fn valid_actions(&self) -> Vec<usize> {
        self.game.valid_moves().iter().map(|d| *d as usize).collect()
    }

    /// Prints the current board.
    pub fn render(&self) {
        self.game.print();
    }

    /// Returns a reference to the underlying game.
    pub fn game(&self) -> &Game {
        &self.game
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

/// Replay buffer for experience replay.
pub struct ReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
    position: usize,
}

impl ReplayBuffer {
    /// Creates a new replay buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    /// Stores an experience.
    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(exp);
        } else {
            self.buffer[self.position] = exp;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    /// Returns the number of experiences stored.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Samples a random batch.
    pub fn sample(&self, batch_size: usize) -> Option<Vec<Experience>> {
        if self.buffer.len() < batch_size {
            return None;
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        Some(
            self.buffer
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect(),
        )
    }

    /// Samples a batch and returns as separate vectors.
    pub fn sample_batch(
        &self,
        batch_size: usize,
    ) -> Option<(Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>, Vec<bool>)> {
        let batch = self.sample(batch_size)?;
        let state_dim = batch[0].state.len();

        let mut states = Vec::with_capacity(batch_size * state_dim);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size * state_dim);
        let mut dones = Vec::with_capacity(batch_size);

        for exp in batch {
            states.extend(&exp.state);
            actions.push(exp.action);
            rewards.push(exp.reward);
            next_states.extend(&exp.next_state);
            dones.push(exp.done);
        }

        Some((states, actions, rewards, next_states, dones))
    }
}
