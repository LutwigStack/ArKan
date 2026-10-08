//! Learning-rate schedules.

/// Learning rate scheduler trait.
///
/// Implement this trait to create custom learning rate schedules.
pub trait LrScheduler {
    /// Returns the learning rate for the given epoch.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Current epoch number (0-indexed)
    /// * `current_lr` - Current learning rate (may be ignored)
    fn get_lr(&self, epoch: usize, current_lr: f32) -> f32;
}

/// Step decay learning rate scheduler.
///
/// Multiplies the learning rate by `gamma` every `step_size` epochs.
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::{StepLR, LrScheduler};
///
/// let scheduler = StepLR::new(0.1, 10, 0.5);
///
/// assert!((scheduler.get_lr(0, 0.1) - 0.1).abs() < 1e-6);
/// assert!((scheduler.get_lr(10, 0.1) - 0.05).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Decay factor (multiplied each step).
    pub gamma: f32,
    /// Number of epochs between decays.
    pub step_size: usize,
}

impl StepLR {
    /// Creates a new step decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate
    /// * `step_size` - Epochs between decays
    /// * `gamma` - Decay factor
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size,
        }
    }
}

impl LrScheduler for StepLR {
    fn get_lr(&self, epoch: usize, _: f32) -> f32 {
        let n_steps = epoch / self.step_size;
        self.initial_lr * self.gamma.powi(n_steps as i32)
    }
}

/// Cosine annealing learning rate scheduler.
///
/// Smoothly decreases the learning rate from `initial_lr` to `min_lr`
/// following a cosine curve over `t_max` epochs.
///
/// # Formula
///
/// $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t \pi}{T_{max}}))$$
///
/// # Example
///
/// ```rust
/// use arkan::optimizer::{CosineAnnealingLR, LrScheduler};
///
/// let scheduler = CosineAnnealingLR::new(0.1, 100, 0.001);
///
/// // At start: ~0.1
/// let lr_start = scheduler.get_lr(0, 0.1);
///
/// // At end: ~0.001
/// let lr_end = scheduler.get_lr(100, 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    /// Initial (maximum) learning rate.
    pub initial_lr: f32,
    /// Minimum learning rate at the end of annealing.
    pub min_lr: f32,
    /// Total number of epochs for one cycle.
    pub t_max: usize,
}

impl CosineAnnealingLR {
    /// Creates a new cosine annealing scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate
    /// * `t_max` - Total epochs for decay
    /// * `min_lr` - Minimum learning rate
    pub fn new(initial_lr: f32, t_max: usize, min_lr: f32) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn get_lr(&self, epoch: usize, _: f32) -> f32 {
        let epoch = epoch.min(self.t_max);
        let cos_inner = std::f32::consts::PI * epoch as f32 / self.t_max as f32;
        self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1.0 + cos_inner.cos())
    }
}
