//! Agent implementations.

pub mod random;
pub mod heuristic;
pub mod kan_dqn;

use crate::env::Env;

/// Trait for all agents.
pub trait Agent {
    /// Selects an action given the current state.
    fn select_action(&mut self, state: &[f32], env: &Env, epsilon: f32) -> usize;
    
    /// Returns the agent name.
    fn name(&self) -> &str;
}
