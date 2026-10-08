/// Euclidean norm across all gradient tensors, with squares accumulated in f64.
pub(crate) fn global_grad_norm(weight_grads: &[Vec<f32>], bias_grads: &[Vec<f32>]) -> f64 {
    weight_grads
        .iter()
        .chain(bias_grads)
        .flatten()
        .map(|&g| (g as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Multiply gradients in f64 before casting back, so tiny scales do not underflow.
pub(crate) fn global_clip_scale(norm: f64, max_norm: Option<f32>) -> f64 {
    match max_norm {
        Some(max) if norm > max as f64 && norm > 0.0 => max as f64 / norm,
        _ => 1.0,
    }
}

/// Disabled clipping does not traverse gradients or compute their norm.
pub(crate) fn clip_in_place(
    weights: &mut [Vec<f32>],
    biases: &mut [Vec<f32>],
    max_norm: Option<f32>,
) {
    if max_norm.is_none() {
        return;
    }
    let scale = global_clip_scale(global_grad_norm(weights, biases), max_norm);
    if scale != 1.0 {
        for g in weights.iter_mut().chain(biases).flatten() {
            *g = (*g as f64 * scale) as f32;
        }
    }
}
