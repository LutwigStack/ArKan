//! Transactional L-BFGS evaluation and line search.

use super::*;

// Accepted trial parameters, gradient, and loss.
type AcceptedTrial = (Vec<f32>, Vec<f32>, f64);

// =============================================================================
// L-BFGS OPTIMIZER (Second-Order)
// =============================================================================

/// L-BFGS optimizer configuration.
///
/// Limited-memory BFGS is a quasi-Newton method that approximates the
/// inverse Hessian matrix using a limited number of past gradients.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LBFGSConfig {
    /// Learning rate (step size).
    pub lr: f32,

    /// Maximum number of iterations per step.
    pub max_iter: usize,

    /// Maximum number of function evaluations per step, including the initial evaluation.
    /// None removes this budget; each line search still has its convergence limit.
    pub max_eval: Option<usize>,

    /// Termination tolerance on gradient norm.
    pub tolerance_grad: f64,

    /// Termination tolerance on maximum parameter change or absolute loss change.
    pub tolerance_change: f64,

    /// Number of corrections to approximate inverse Hessian.
    /// Higher = more memory, better approximation.
    pub history_size: usize,

    /// Line search method.
    pub line_search_fn: LineSearchMethod,

    /// Safety configuration.
    #[cfg_attr(feature = "serde", serde(default))]
    pub safety: SafetyConfig,
}

impl Default for LBFGSConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            max_iter: 20,
            max_eval: Some(25),
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
            line_search_fn: LineSearchMethod::StrongWolfe,
            safety: SafetyConfig::default(),
        }
    }
}

/// Line search methods for L-BFGS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LineSearchMethod {
    /// Strong Wolfe conditions (default, most robust).
    #[default]
    StrongWolfe,
    /// Backtracking with Armijo condition.
    Backtracking,
    /// No line search (fixed step size). Use with caution.
    NoLineSearch,
}

/// L-BFGS optimizer for KAN networks.
///
/// Implements the Limited-memory BFGS algorithm for second-order optimization.
/// Best suited for small to medium networks where function evaluations are cheap.
///
/// # Atomicity and Rollback
///
/// On an objective error or failed line search, the optimizer:
/// 1. Restores parameters and history to their pre-step values
/// 2. Returns the original error
///
/// This ensures no partial updates occur.
///
/// # Thread Safety
///
/// `LBFGS` implements `Send + Sync`.
///
/// See [`Self::step_lbfgs`] for an executable gradient-evaluation example.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LBFGS {
    /// Configuration.
    pub config: LBFGSConfig,

    /// History of s vectors (parameter differences).
    s_history: Vec<Vec<f32>>,

    /// History of y vectors (gradient differences).
    y_history: Vec<Vec<f32>>,

    /// History of rho values (1 / (y^T s)).
    rho_history: Vec<f64>,

    /// Previous parameters (for computing s).
    prev_params: Option<Vec<f32>>,

    /// Previous gradients (for computing y).
    prev_grads: Option<Vec<f32>>,

    /// State version for topology tracking.
    state_version: u64,

    /// Number of function evaluations.
    n_eval: usize,
}

// SAFETY: LBFGS uses only thread-safe types
unsafe impl Send for LBFGS {}
unsafe impl Sync for LBFGS {}

enum LBFGSEvaluation {
    Finite(f64, Vec<f32>),
    RejectedNonFinite,
    BudgetExhausted,
}

impl LBFGS {
    /// Creates a new L-BFGS optimizer.
    pub fn new(_network: &KanNetwork, config: LBFGSConfig) -> Self {
        Self {
            config,
            s_history: Vec::with_capacity(config.history_size),
            y_history: Vec::with_capacity(config.history_size),
            rho_history: Vec::with_capacity(config.history_size),
            prev_params: None,
            prev_grads: None,
            state_version: 0,
            n_eval: 0,
        }
    }

    /// Resets all history buffers.
    pub fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
        self.prev_params = None;
        self.prev_grads = None;
        self.n_eval = 0;
    }

    /// Returns the number of function evaluations.
    pub fn num_evals(&self) -> usize {
        self.n_eval
    }

    /// Flattens network parameters into a single vector.
    pub fn flatten_params(network: &KanNetwork) -> Vec<f32> {
        let total: usize = network
            .layers
            .iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum();
        let mut params = Vec::with_capacity(total);
        for layer in &network.layers {
            params.extend_from_slice(&layer.weights);
            params.extend_from_slice(&layer.bias);
        }
        params
    }

    /// Restores parameters from a flat vector.
    pub fn restore_params(network: &mut KanNetwork, params: &[f32]) {
        let mut offset = 0;
        for layer in network
            .try_parameters_mut()
            .expect("LBFGS parameter layout changed")
            .iter_mut()
        {
            let w_len = layer.weights.len();
            layer
                .weights
                .copy_from_slice(&params[offset..offset + w_len]);
            offset += w_len;

            let b_len = layer.bias.len();
            layer.bias.copy_from_slice(&params[offset..offset + b_len]);
            offset += b_len;
        }
    }

    /// Flattens gradients into a single vector.
    pub fn flatten_grads(weight_grads: &[Vec<f32>], bias_grads: &[Vec<f32>]) -> Vec<f32> {
        let total: usize = weight_grads.iter().map(|v| v.len()).sum::<usize>()
            + bias_grads.iter().map(|v| v.len()).sum::<usize>();
        let mut grads = Vec::with_capacity(total);
        for (wg, bg) in weight_grads.iter().zip(bias_grads.iter()) {
            grads.extend_from_slice(wg);
            grads.extend_from_slice(bg);
        }
        grads
    }

    /// Updates history with new s and y vectors.
    fn update_history(&mut self, s: Vec<f32>, y: Vec<f32>) {
        let sy: f64 = s
            .iter()
            .zip(y.iter())
            .map(|(&si, &yi)| si as f64 * yi as f64)
            .sum();

        // Skip if curvature condition not satisfied
        if !sy.is_finite() || sy <= 1e-10 {
            return;
        }

        let rho = 1.0 / sy;

        // Maintain history_size limit
        if self.s_history.len() >= self.config.history_size {
            self.s_history.remove(0);
            self.y_history.remove(0);
            self.rho_history.remove(0);
        }

        self.s_history.push(s);
        self.y_history.push(y);
        self.rho_history.push(rho);
    }

    /// Two-loop recursion for computing search direction.
    ///
    /// Returns -H⁻¹g where H⁻¹ is approximated using L-BFGS history.
    pub fn two_loop_recursion(&self, grad: &[f32]) -> Vec<f32> {
        let m = self.s_history.len();

        if m == 0 {
            // No history: steepest descent
            return grad.iter().map(|&g| -g).collect();
        }

        // q = grad
        let mut q: Vec<f64> = grad.iter().map(|&g| g as f64).collect();
        let mut alpha = vec![0.0f64; m];

        // First loop (backward)
        // Note: We need to access alpha[i] and history[i] in tandem, clippy allow is correct here
        #[allow(clippy::needless_range_loop)]
        for i in (0..m).rev() {
            alpha[i] = self.rho_history[i]
                * self.s_history[i]
                    .iter()
                    .zip(q.iter())
                    .map(|(&s, &q)| s as f64 * q)
                    .sum::<f64>();
            for (q_j, &y_ij) in q.iter_mut().zip(self.y_history[i].iter()) {
                *q_j -= alpha[i] * y_ij as f64;
            }
        }

        // Scale initial Hessian approximation: H0 = γI where γ = sᵀy / yᵀy
        let s_last = &self.s_history[m - 1];
        let y_last = &self.y_history[m - 1];
        let yy: f64 = y_last.iter().map(|&y| (y as f64).powi(2)).sum();
        let sy: f64 = s_last
            .iter()
            .zip(y_last.iter())
            .map(|(&s, &y)| s as f64 * y as f64)
            .sum();
        let gamma = if yy > 1e-10 { sy / yy } else { 1.0 };

        // r = γ * q
        let mut r = q;
        // Preserve gamma as the left operand, including NaN propagation order.
        #[allow(clippy::assign_op_pattern)]
        for entry in &mut r {
            *entry = gamma * *entry;
        }

        // Second loop (forward)
        // Note: We need to access alpha[i] and history[i] in tandem, clippy allow is correct here
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            let beta = self.rho_history[i]
                * self.y_history[i]
                    .iter()
                    .zip(r.iter())
                    .map(|(&y, &r)| y as f64 * r)
                    .sum::<f64>();
            for (r_j, &s_ij) in r.iter_mut().zip(self.s_history[i].iter()) {
                *r_j += s_ij as f64 * (alpha[i] - beta);
            }
        }

        // Return -r (descent direction)
        r.iter().map(|&r| -r as f32).collect()
    }

    /// Computes directional derivative: g ⋅ d
    fn directional_derivative(grad: &[f32], direction: &[f32]) -> f64 {
        grad.iter()
            .zip(direction.iter())
            .map(|(&g, &d)| g as f64 * d as f64)
            .sum()
    }

    /// Computes gradient norm: ||g||
    fn grad_norm(grad: &[f32]) -> f64 {
        grad.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt()
    }

    // Numerical exhaustion is separate from closure errors for the caller's safety policy.
    fn exhausted_line_search(
        numerical_rejection: bool,
        reason: &str,
    ) -> ArkanResult<Option<AcceptedTrial>> {
        if numerical_rejection {
            Ok(None)
        } else {
            Err(ArkanError::line_search_failed(reason))
        }
    }

    /// Strong Wolfe line search.
    ///
    /// Finds step size α satisfying Strong Wolfe conditions:
    /// 1. Sufficient decrease: f(x + αd) ≤ f(x) + c₁·α·(∇f·d)
    /// 2. Curvature: |∇f(x + αd)·d| ≤ c₂·|∇f(x)·d|
    ///
    /// Parameters from PyTorch LBFGS defaults:
    /// - c1 = 1e-4 (Armijo constant)
    /// - c2 = 0.9 (curvature constant for L-BFGS, 0.1 for CG)
    /// - max_iter = 25
    fn strong_wolfe_line_search<F>(
        &mut self,
        network: &mut KanNetwork,
        closure: &mut F,
        x0: &[f32],
        f0: f64,
        g0: &[f32],
        direction: &[f32],
    ) -> ArkanResult<Option<AcceptedTrial>>
    where
        F: FnMut(&KanNetwork) -> ArkanResult<LBFGSEvaluation>,
    {
        const C1: f64 = 1e-4;
        const C2: f64 = 0.9;
        const MAX_LS_ITER: usize = 25;
        const ALPHA_MAX: f64 = 50.0;

        let dg0 = Self::directional_derivative(g0, direction);

        // dg0 should be negative (descent direction)
        if dg0 >= 0.0 {
            return Err(ArkanError::line_search_failed(
                "Search direction is not a descent direction",
            ));
        }

        let mut alpha = self.config.lr as f64;
        let mut numerical_rejection = false;
        let mut alpha_lo: f64 = 0.0;
        let mut alpha_hi: f64 = ALPHA_MAX;
        let mut f_lo = f0;

        for iter in 0..MAX_LS_ITER {
            // x = x0 + alpha * d
            let x_new: Vec<f32> = x0
                .iter()
                .zip(direction.iter())
                .map(|(&x, &d)| (x as f64 + alpha * d as f64) as f32)
                .collect();

            Self::restore_params(network, &x_new);
            let (f_new, g_new) = match closure(network)? {
                LBFGSEvaluation::Finite(loss, gradient) => (loss, gradient),
                LBFGSEvaluation::RejectedNonFinite => {
                    numerical_rejection = true;
                    alpha_hi = alpha;
                    alpha = (alpha_lo + alpha_hi) / 2.0;
                    continue;
                }
                LBFGSEvaluation::BudgetExhausted => {
                    return Self::exhausted_line_search(
                        numerical_rejection,
                        "LBFGS evaluation budget exhausted",
                    );
                }
            };

            let dg_new = Self::directional_derivative(&g_new, direction);

            // Check Armijo condition (sufficient decrease)
            if f_new > f0 + C1 * alpha * dg0 || (iter > 0 && f_new >= f_lo) {
                // Zoom into [alpha_lo, alpha]
                alpha_hi = alpha;
            } else {
                // Check Strong Wolfe curvature condition
                if dg_new.abs() <= -C2 * dg0 {
                    // Both conditions satisfied
                    return Ok(Some((x_new, g_new, f_new)));
                }

                if dg_new >= 0.0 {
                    // Zoom into [alpha, alpha_lo]
                    alpha_hi = alpha_lo;
                    alpha_lo = alpha;
                    f_lo = f_new;
                } else {
                    // Move to higher alpha
                    alpha_lo = alpha;
                    f_lo = f_new;
                    alpha = (alpha + alpha_hi) / 2.0;
                    continue;
                }
            }

            // Zoom phase: binary search in [alpha_lo, alpha_hi]
            if (alpha_hi - alpha_lo).abs() < 1e-10 {
                // Interval too small
                return Self::exhausted_line_search(
                    numerical_rejection,
                    "Strong Wolfe interval exhausted",
                );
            }

            alpha = (alpha_lo + alpha_hi) / 2.0;
        }

        Self::exhausted_line_search(numerical_rejection, "Strong Wolfe search exhausted")
    }

    /// Backtracking line search with Armijo condition.
    fn backtracking_line_search<F>(
        &mut self,
        network: &mut KanNetwork,
        closure: &mut F,
        x0: &[f32],
        f0: f64,
        g0: &[f32],
        direction: &[f32],
    ) -> ArkanResult<Option<AcceptedTrial>>
    where
        F: FnMut(&KanNetwork) -> ArkanResult<LBFGSEvaluation>,
    {
        const C1: f64 = 1e-4;
        const RHO: f64 = 0.5; // Backtrack factor
        const MAX_LS_ITER: usize = 20;

        let dg0 = Self::directional_derivative(g0, direction);
        let mut alpha = self.config.lr as f64;
        let mut numerical_rejection = false;

        for _ in 0..MAX_LS_ITER {
            let x_new: Vec<f32> = x0
                .iter()
                .zip(direction.iter())
                .map(|(&x, &d)| (x as f64 + alpha * d as f64) as f32)
                .collect();

            Self::restore_params(network, &x_new);
            let (f_new, g_new) = match closure(network)? {
                LBFGSEvaluation::Finite(loss, gradient) => (loss, gradient),
                LBFGSEvaluation::RejectedNonFinite => {
                    numerical_rejection = true;
                    alpha *= RHO;
                    continue;
                }
                LBFGSEvaluation::BudgetExhausted => {
                    return Self::exhausted_line_search(
                        numerical_rejection,
                        "LBFGS evaluation budget exhausted",
                    );
                }
            };

            // Check Armijo condition
            if f_new <= f0 + C1 * alpha * dg0 {
                return Ok(Some((x_new, g_new, f_new)));
            }

            alpha *= RHO;
        }

        Self::exhausted_line_search(
            numerical_rejection,
            "Backtracking line search did not converge",
        )
    }
}

impl Clone for LBFGS {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            s_history: self.s_history.clone(),
            y_history: self.y_history.clone(),
            rho_history: self.rho_history.clone(),
            prev_params: self.prev_params.clone(),
            prev_grads: self.prev_grads.clone(),
            state_version: self.state_version,
            n_eval: self.n_eval,
        }
    }
}

impl Optimizer for LBFGS {
    fn step(
        &mut self,
        _network: &mut KanNetwork,
        _weight_grads: &[Vec<f32>],
        _bias_grads: &[Vec<f32>],
        _max_grad_norm: Option<f32>,
    ) -> ArkanResult<()> {
        // L-BFGS requires closure-based API
        Err(ArkanError::optimizer(
            "LBFGS requires step_lbfgs() with closure. Use step() only for first-order optimizers.",
        ))
    }

    fn step_with_closure<F>(&mut self, _closure: F) -> ArkanResult<f64>
    where
        F: FnMut() -> ArkanResult<f64>,
    {
        // This default impl doesn't work for LBFGS because we need mutable network access
        // Use step_lbfgs() instead
        Err(ArkanError::optimizer(
            "Use step_lbfgs() for L-BFGS optimization with network access.",
        ))
    }

    fn zero_grad(&mut self, network: &mut KanNetwork) -> ArkanResult<()> {
        let _ = network;
        Ok(())
    }

    fn get_state_version(&self) -> u64 {
        self.state_version
    }

    fn bump_version(&mut self) {
        self.state_version += 1;
        self.reset();
    }

    fn get_lr(&self, group_index: usize) -> ArkanResult<f64> {
        if group_index == 0 {
            Ok(self.config.lr as f64)
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }

    fn set_lr(&mut self, group_index: usize, new_lr: f64) -> ArkanResult<()> {
        if group_index == 0 {
            self.config.lr = new_lr as f32;
            Ok(())
        } else {
            Err(ArkanError::group_index_out_of_bounds(group_index, 1))
        }
    }
}

impl LBFGS {
    /// Performs L-BFGS optimization step.
    ///
    /// The closure evaluates loss and gradients through a shared network borrow, so it
    /// cannot change model parameters or topology. It retains `FnMut`: captured
    /// workspace, gradient buffers and external state may be mutated between calls.
    /// It may be called multiple times during line search; returning `Err` cancels the step.
    ///
    /// # Arguments
    ///
    /// * `network` - Network to optimize
    /// * `closure` - Closure that computes `(loss, gradient_vector)` for current params
    ///
    /// # Returns
    ///
    /// Final loss value after the optimization step.
    ///
    /// # Rollback Guarantee
    ///
    /// On any returned evaluation or line-search error, parameters and history are restored.
    /// The original error is returned. Evaluation counts include failed objective calls.
    /// `max_iter` bounds accepted iterations; `max_eval` includes the initial evaluation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use arkan::{KanConfigBuilder, KanNetwork, LBFGS, LBFGSConfig, masked_mse_into};
    /// let config = KanConfigBuilder::new().input_dim(1).output_dim(1)
    ///     .hidden_dims(vec![]).normalization(vec![0.0], vec![1.0]).seed(7).build()?;
    /// let mut network = KanNetwork::new(config);
    /// let mut optimizer = LBFGS::new(&network, LBFGSConfig::default());
    /// let mut workspace = network.create_workspace(2);
    /// let mut output = [0.0; 2];
    /// let mut derivative = [0.0; 2];
    /// let mut evaluations = 0;
    /// let loss = optimizer.step_lbfgs(&mut network, |net: &KanNetwork| {
    ///     evaluations += 1;
    ///     let pass = net.try_forward_for_backward(&[-0.25, 0.5], &mut output, &mut workspace)?;
    ///     let loss = masked_mse_into(&output, &[0.0, 0.5], None, &mut derivative)?;
    ///     let gradients = pass.backward(&derivative)?;
    ///     Ok((loss as f64, LBFGS::flatten_grads(gradients.weights, gradients.biases)))
    /// })?;
    /// assert!(loss.is_finite());
    /// assert!(evaluations > 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// A callback cannot mutate the model it evaluates:
    ///
    /// ```compile_fail,E0596
    /// use arkan::{KanConfig, KanNetwork, LBFGS, LBFGSConfig};
    /// let mut network = KanNetwork::new(KanConfig::default());
    /// let mut optimizer = LBFGS::new(&network, LBFGSConfig::default());
    /// let _ = optimizer.step_lbfgs(&mut network, |net| {
    ///     net.layers[0].weights[0] = 0.0;
    ///     net.layers.clear();
    ///     Ok((0.0, Vec::new()))
    /// });
    /// ```
    pub fn step_lbfgs<F>(&mut self, network: &mut KanNetwork, mut closure: F) -> ArkanResult<f64>
    where
        F: FnMut(&KanNetwork) -> ArkanResult<(f64, Vec<f32>)>,
    {
        network.checked_layout()?;
        validate_safety(&self.config.safety, None)?;
        if !self.config.lr.is_finite()
            || self.config.lr <= 0.0
            || self.config.max_iter == 0
            || self.config.history_size == 0
            || self
                .config
                .max_eval
                .map(|budget| budget < 2)
                .unwrap_or(false)
        {
            return Err(ArkanError::optimizer(
                "LBFGS requires positive lr, max_iter and history_size, and max_eval >= 2",
            ));
        }
        validate_nonnegative(self.config.tolerance_grad, "tolerance_grad")?;
        validate_nonnegative(self.config.tolerance_change, "tolerance_change")?;
        let original_params = Self::flatten_params(network);
        let size = original_params.len();
        validate_shape(self.s_history.len(), self.y_history.len())?;
        validate_shape(self.s_history.len(), self.rho_history.len())?;
        for history in self
            .s_history
            .iter()
            .chain(&self.y_history)
            .chain(self.prev_params.iter())
            .chain(self.prev_grads.iter())
        {
            validate_shape(size, history.len())?;
            if find_nan_in_grads(history).is_some() {
                return Err(ArkanError::optimizer("LBFGS history must be finite"));
            }
        }
        if self
            .rho_history
            .iter()
            .any(|rho| !rho.is_finite() || *rho <= 0.0)
        {
            return Err(ArkanError::optimizer(
                "LBFGS curvature state must be finite and positive",
            ));
        }

        // Keep parameter and history commits in one transaction, including fixed steps.
        let original_state = self.clone();
        let budget = self.config.max_eval.unwrap_or(usize::MAX);
        let evaluations = std::cell::Cell::new(0usize);
        let mut numerical_failure = false;
        let mut initial_loss = None;
        let result = (|| {
            let mut evaluate = |net: &KanNetwork| -> ArkanResult<LBFGSEvaluation> {
                if evaluations.get() >= budget {
                    return Ok(LBFGSEvaluation::BudgetExhausted);
                }
                if net
                    .layers
                    .iter()
                    .flat_map(|layer| layer.weights.iter().chain(&layer.bias))
                    .any(|p| !p.is_finite())
                {
                    return Ok(LBFGSEvaluation::RejectedNonFinite);
                }
                evaluations.set(evaluations.get() + 1);
                let (loss, gradient) = closure(net)?;
                initial_loss.get_or_insert(loss);
                validate_shape(size, gradient.len())?;
                if !loss.is_finite() || find_nan_in_grads(&gradient).is_some() {
                    return Ok(LBFGSEvaluation::RejectedNonFinite);
                }
                Ok(LBFGSEvaluation::Finite(loss, gradient))
            };
            let (mut loss, mut gradient) = match evaluate(network)? {
                LBFGSEvaluation::Finite(loss, gradient) => (loss, gradient),
                LBFGSEvaluation::RejectedNonFinite => {
                    numerical_failure = true;
                    return Err(ArkanError::nan_encountered(
                        0,
                        "LBFGS initial objective or parameters",
                    ));
                }
                LBFGSEvaluation::BudgetExhausted => {
                    return Err(ArkanError::line_search_failed(
                        "LBFGS evaluation budget exhausted",
                    ))
                }
            };
            let mut params = original_params.clone();
            let mut accepted_any = false;
            for _ in 0..self.config.max_iter {
                if Self::grad_norm(&gradient) <= self.config.tolerance_grad {
                    break;
                }
                // The initial evaluation consumes one budget unit; a new trial needs another.
                if evaluations.get() >= budget {
                    break;
                }
                let direction = self.two_loop_recursion(&gradient);
                if find_nan_in_grads(&direction).is_some() {
                    return Err(ArkanError::optimizer("LBFGS search direction is nonfinite"));
                }
                let accepted_point = match self.config.line_search_fn {
                    LineSearchMethod::StrongWolfe => self.strong_wolfe_line_search(
                        network,
                        &mut evaluate,
                        &params,
                        loss,
                        &gradient,
                        &direction,
                    )?,
                    LineSearchMethod::Backtracking => self.backtracking_line_search(
                        network,
                        &mut evaluate,
                        &params,
                        loss,
                        &gradient,
                        &direction,
                    )?,
                    LineSearchMethod::NoLineSearch => {
                        let alpha = self.config.lr as f64;
                        let trial: Vec<f32> = params
                            .iter()
                            .zip(&direction)
                            .map(|(&x, &d)| (x as f64 + alpha * d as f64) as f32)
                            .collect();
                        Self::restore_params(network, &trial);
                        match evaluate(network)? {
                            LBFGSEvaluation::Finite(new_loss, new_gradient) => {
                                Some((trial, new_gradient, new_loss))
                            }
                            LBFGSEvaluation::RejectedNonFinite => None,
                            LBFGSEvaluation::BudgetExhausted => {
                                return Err(ArkanError::line_search_failed(
                                    "LBFGS evaluation budget exhausted",
                                ))
                            }
                        }
                    }
                };
                let Some((accepted, new_gradient, new_loss)) = accepted_point else {
                    numerical_failure = true;
                    return Err(ArkanError::nan_encountered(
                        0,
                        "LBFGS numerical trials exhausted",
                    ));
                };
                let s: Vec<f32> = accepted
                    .iter()
                    .zip(&params)
                    .map(|(&x, &old)| x - old)
                    .collect();
                let y = new_gradient
                    .iter()
                    .zip(&gradient)
                    .map(|(&g, &old)| g - old)
                    .collect();
                let parameter_change = s.iter().map(|&v| (v as f64).abs()).fold(0.0, f64::max);
                let loss_change = (new_loss - loss).abs();
                self.update_history(s, y);
                if !accepted_any {
                    drop(self.prev_params.take());
                    drop(self.prev_grads.take());
                }
                accepted_any = true;
                params = accepted;
                gradient = new_gradient;
                loss = new_loss;
                if parameter_change <= self.config.tolerance_change
                    || loss_change <= self.config.tolerance_change
                {
                    break;
                }
            }
            if accepted_any {
                self.prev_params = Some(if params.capacity() == params.len() {
                    params
                } else {
                    params.clone()
                });
                self.prev_grads = Some(if gradient.capacity() == gradient.len() {
                    gradient
                } else {
                    gradient.clone()
                });
            }
            Ok(loss)
        })();
        self.n_eval = original_state.n_eval.saturating_add(evaluations.get());
        match result {
            Ok(loss) => Ok(loss),
            Err(error) => {
                Self::restore_params(network, &original_params);
                let n_eval = self.n_eval;
                *self = original_state;
                self.n_eval = n_eval;
                if numerical_failure && self.config.safety.skip_step_on_nan {
                    Ok(initial_loss.unwrap_or(f64::NAN))
                } else {
                    Err(error)
                }
            }
        }
    }
}
