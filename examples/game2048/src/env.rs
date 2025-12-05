//! RL Environment wrapper for 2048.
//!
//! Provides a clean interface for agents to interact with the game.

use crate::game::{Board, Direction, Game};
use crate::utils;

/// State dimension for one-hot encoding (16 cells * 16 values).
pub const STATE_DIM: usize = 256;

/// Experience tuple for replay buffer.
/// Uses fixed-size arrays to avoid allocations.
#[derive(Clone)]
pub struct Experience {
    pub state: [f32; STATE_DIM],
    pub action: usize,
    pub reward: f32,
    pub next_state: [f32; STATE_DIM],
    pub done: bool,
}

impl Experience {
    /// Creates a new experience with pre-allocated arrays.
    pub fn new() -> Self {
        Self {
            state: [0.0; STATE_DIM],
            action: 0,
            reward: 0.0,
            next_state: [0.0; STATE_DIM],
            done: false,
        }
    }

    /// Creates an experience from array references.
    #[inline]
    pub fn from_arrays(
        state: &[f32; STATE_DIM],
        action: usize,
        reward: f32,
        next_state: &[f32; STATE_DIM],
        done: bool,
    ) -> Self {
        Self {
            state: *state,
            action,
            reward,
            next_state: *next_state,
            done,
        }
    }
}

impl Default for Experience {
    fn default() -> Self {
        Self::new()
    }
}

/// RL Environment wrapper with pre-allocated state buffers.
pub struct Env {
    game: Game,
    /// Pre-allocated buffer for current state.
    state_buffer: [f32; STATE_DIM],
}

impl Env {
    /// Creates a new environment.
    pub fn new() -> Self {
        let mut env = Self {
            game: Game::new(),
            state_buffer: [0.0; STATE_DIM],
        };
        env.update_state_buffer();
        env
    }

    /// Updates internal state buffer from game board.
    #[inline]
    fn update_state_buffer(&mut self) {
        utils::board_to_onehot_inplace(&self.game.board, &mut self.state_buffer);
    }

    /// Resets the environment and returns initial state.
    pub fn reset(&mut self) -> Vec<f32> {
        self.game = Game::new();
        self.update_state_buffer();
        self.state_buffer.to_vec()
    }

    /// Gets the current state as one-hot encoded vector (256 features).
    /// Note: Returns a clone. Use get_state_ref for zero-copy access.
    pub fn get_state(&self) -> Vec<f32> {
        self.state_buffer.to_vec()
    }

    /// Gets a reference to the current state buffer (zero-copy).
    #[inline]
    pub fn get_state_ref(&self) -> &[f32; STATE_DIM] {
        &self.state_buffer
    }

    /// Copies current state into provided buffer.
    #[inline]
    pub fn copy_state_to(&self, dest: &mut [f32; STATE_DIM]) {
        *dest = self.state_buffer;
    }

    /// Takes an action and returns (next_state, reward, done).
    pub fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        let dir = Direction::from_index(action);
        let (reward, _changed) = self.game.make_move(dir);
        
        self.update_state_buffer();
        let done = self.game.game_over;

        (self.state_buffer.to_vec(), reward, done)
    }

    /// Gets a reference to the game board.
    #[inline]
    pub fn board(&self) -> &Board {
        &self.game.board
    }

    /// Takes an action and fills Experience struct (zero-copy friendly).
    /// Returns reward and done flag.
    #[inline]
    pub fn step_into(&mut self, action: usize, exp: &mut Experience) -> (f32, bool) {
        // Copy current state before move
        exp.state = self.state_buffer;
        exp.action = action;
        
        let dir = Direction::from_index(action);
        let (reward, _changed) = self.game.make_move(dir);
        
        self.update_state_buffer();
        
        exp.reward = reward;
        exp.next_state = self.state_buffer;
        exp.done = self.game.game_over;
        
        (reward, self.game.game_over)
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
        STATE_DIM
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
/// Optimized to minimize allocations during sampling.
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
    #[inline]
    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(exp);
        } else {
            self.buffer[self.position] = exp;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    /// Returns the number of experiences stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Samples random indices for a batch (for lock-free pattern).
    pub fn sample_indices(&self, batch_size: usize) -> Option<Vec<usize>> {
        if self.buffer.len() < batch_size {
            return None;
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let len = self.buffer.len();
        
        Some((0..batch_size).map(|_| rng.gen_range(0..len)).collect())
    }

    /// Samples a batch into pre-allocated buffers (minimal allocations).
    /// Returns batch_size on success.
    pub fn sample_batch_into(
        &self,
        batch_size: usize,
        states: &mut Vec<f32>,
        actions: &mut Vec<usize>,
        rewards: &mut Vec<f32>,
        next_states: &mut Vec<f32>,
        dones: &mut Vec<bool>,
    ) -> Option<usize> {
        if self.buffer.len() < batch_size {
            return None;
        }

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let len = self.buffer.len();

        // Clear and ensure capacity
        states.clear();
        actions.clear();
        rewards.clear();
        next_states.clear();
        dones.clear();
        
        states.reserve(batch_size * STATE_DIM);
        actions.reserve(batch_size);
        rewards.reserve(batch_size);
        next_states.reserve(batch_size * STATE_DIM);
        dones.reserve(batch_size);

        for _ in 0..batch_size {
            let idx = rng.gen_range(0..len);
            let exp = &self.buffer[idx];
            
            states.extend_from_slice(&exp.state);
            actions.push(exp.action);
            rewards.push(exp.reward);
            next_states.extend_from_slice(&exp.next_state);
            dones.push(exp.done);
        }

        Some(batch_size)
    }

    /// Samples a batch and returns as separate vectors.
    /// Legacy API - prefer sample_batch_into for performance.
    pub fn sample_batch(
        &self,
        batch_size: usize,
    ) -> Option<(Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>, Vec<bool>)> {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut next_states = Vec::new();
        let mut dones = Vec::new();
        
        self.sample_batch_into(
            batch_size,
            &mut states,
            &mut actions,
            &mut rewards,
            &mut next_states,
            &mut dones,
        )?;

        Some((states, actions, rewards, next_states, dones))
    }
}
