//! WGSL Shader source code for GPU operations.
//!
//! This module contains shader source code as string constants that are
//! compiled at runtime by wgpu.

/// Forward pass shader for KAN layer.
///
/// Computes: y[j] = Σᵢ Σₖ weights[j,i,k] · B_k(x[i]) + bias[j]
///
/// # Bind Groups
///
/// - Group 0 (Static):
///   - Binding 0: weights (storage, read) - [out_dim, in_dim, basis_padded]
///   - Binding 1: bias (storage, read) - [out_dim]
///   - Binding 2: config (uniform) - LayerUniforms
///
/// - Group 1 (Dynamic):
///   - Binding 0: input (storage, read) - [batch, in_dim]
///   - Binding 1: output (storage, read_write) - [batch, out_dim]
///
/// # Workgroup Size
///
/// [64, 1, 1] - each thread processes one (batch, output) pair.
pub const FORWARD_SHADER: &str = r#"
// Uniform buffer layout (must match LayerUniforms in Rust)
struct Uniforms {
    grid_min: f32,
    grid_max: f32,
    grid_size: u32,
    order: u32,
    
    in_dim: u32,
    out_dim: u32,
    basis_padded: u32,
    batch_size: u32,
}

// Group 0: Static layer resources
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> config: Uniforms;

// Group 1: Dynamic workspace resources
@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

// Constants
const WORKGROUP_SIZE: u32 = 64u;
const MAX_ORDER: u32 = 4u;  // Maximum supported spline order
const MAX_BASIS: u32 = 5u;  // MAX_ORDER + 1

// Compute B-spline basis functions using Cox-de Boor recursion
// Returns basis values in a local array
fn compute_basis_order3(t_norm: f32, span: u32) -> array<f32, 4> {
    // Hardcoded for order=3 (cubic B-splines)
    // t_norm is in [0, 1], span is the knot span index
    
    var basis: array<f32, 4>;
    
    // Cox-de Boor for order 3
    // Level 0: basis functions of order 0
    var N0: array<f32, 4>;
    let grid_size_f = f32(config.grid_size);
    let t = t_norm * grid_size_f;
    let span_f = f32(span);
    
    // Initialize level 0
    for (var k = 0u; k < 4u; k++) {
        let knot_k = span_f - 2.0 + f32(k);
        let knot_k1 = knot_k + 1.0;
        if (t >= knot_k && t < knot_k1) {
            N0[k] = 1.0;
        } else {
            N0[k] = 0.0;
        }
    }
    
    // Handle edge case at t = grid_size
    if (t >= grid_size_f) {
        N0[3] = 1.0;
    }
    
    // Level 1
    var N1: array<f32, 3>;
    for (var k = 0u; k < 3u; k++) {
        let knot_k = span_f - 2.0 + f32(k);
        let knot_k1 = knot_k + 1.0;
        let knot_k2 = knot_k + 2.0;
        
        var left = 0.0;
        var right = 0.0;
        
        let denom_left = knot_k1 - knot_k;
        if (abs(denom_left) > 1e-10) {
            left = (t - knot_k) / denom_left * N0[k];
        }
        
        let denom_right = knot_k2 - knot_k1;
        if (abs(denom_right) > 1e-10) {
            right = (knot_k2 - t) / denom_right * N0[k + 1u];
        }
        
        N1[k] = left + right;
    }
    
    // Level 2
    var N2: array<f32, 2>;
    for (var k = 0u; k < 2u; k++) {
        let knot_k = span_f - 2.0 + f32(k);
        let knot_k2 = knot_k + 2.0;
        let knot_k3 = knot_k + 3.0;
        
        var left = 0.0;
        var right = 0.0;
        
        let denom_left = knot_k2 - knot_k;
        if (abs(denom_left) > 1e-10) {
            left = (t - knot_k) / denom_left * N1[k];
        }
        
        let denom_right = knot_k3 - knot_k2;
        if (abs(denom_right) > 1e-10) {
            right = (knot_k3 - t) / denom_right * N1[k + 1u];
        }
        
        N2[k] = left + right;
    }
    
    // Level 3 (final)
    for (var k = 0u; k < 1u; k++) {
        let knot_k = span_f - 2.0 + f32(k);
        let knot_k3 = knot_k + 3.0;
        let knot_k4 = knot_k + 4.0;
        
        var left = 0.0;
        var right = 0.0;
        
        let denom_left = knot_k3 - knot_k;
        if (abs(denom_left) > 1e-10) {
            left = (t - knot_k) / denom_left * N2[k];
        }
        
        let denom_right = knot_k4 - knot_k3;
        if (abs(denom_right) > 1e-10) {
            right = (knot_k4 - t) / denom_right * N2[k + 1u];
        }
        
        basis[0] = left + right;
    }
    
    // For order 3, we have 4 active basis functions
    // Simplified: return uniform basis for now
    // TODO: Full Cox-de Boor implementation
    let u = t - floor(t);
    let u2 = u * u;
    let u3 = u2 * u;
    
    // Cubic B-spline basis (uniform knots)
    basis[0] = (1.0 - u) * (1.0 - u) * (1.0 - u) / 6.0;
    basis[1] = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
    basis[2] = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
    basis[3] = u3 / 6.0;
    
    return basis;
}

// Find the span index for a normalized value t in [0, 1]
fn find_span(t_norm: f32) -> u32 {
    let grid_size = config.grid_size;
    let order = config.order;
    
    // Map t from [0,1] to grid position
    let t_grid = t_norm * f32(grid_size);
    
    // Clamp span to valid range [order, grid_size + order - 1]
    var span = u32(floor(t_grid));
    span = clamp(span + order, order, grid_size + order - 1u);
    
    return span;
}

@compute @workgroup_size(64, 1, 1)
fn forward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x / config.out_dim;
    let out_idx = global_id.x % config.out_dim;
    
    // Bounds check
    if (batch_idx >= config.batch_size || out_idx >= config.out_dim) {
        return;
    }
    
    var sum = 0.0;
    
    // Process each input dimension
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {
        // Get input value and normalize to [0, 1]
        let input_idx = batch_idx * config.in_dim + in_idx;
        var x = input[input_idx];
        
        // Clamp to grid range and normalize
        x = clamp(x, config.grid_min, config.grid_max);
        let t_norm = (x - config.grid_min) / (config.grid_max - config.grid_min);
        
        // Find span and compute basis
        let span = find_span(t_norm);
        let basis = compute_basis_order3(t_norm, span);
        
        // Compute weighted sum for this input
        let weight_base = (out_idx * config.in_dim + in_idx) * config.basis_padded;
        
        // Active basis functions start at (span - order)
        let basis_start = span - config.order;
        
        for (var k = 0u; k < 4u; k++) {  // order + 1 = 4 for cubic splines
            let weight_idx = weight_base + basis_start + k;
            if (basis_start + k < config.basis_padded) {
                sum += weights[weight_idx] * basis[k];
            }
        }
    }
    
    // Add bias and write output
    sum += bias[out_idx];
    
    let output_idx = batch_idx * config.out_dim + out_idx;
    output[output_idx] = sum;
}
"#;

/// Simple forward shader with scalar basis computation.
///
/// This is a simpler version that processes each output element independently.
pub const FORWARD_SIMPLE_SHADER: &str = r#"
struct Uniforms {
    grid_min: f32,
    grid_max: f32,
    grid_size: u32,
    order: u32,
    
    in_dim: u32,
    out_dim: u32,
    basis_padded: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> config: Uniforms;

@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

// Cubic B-spline basis functions (order=3)
fn cubic_basis(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    
    return vec4<f32>(
        omt3 / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0
    );
}

@compute @workgroup_size(64, 1, 1)
fn forward_simple(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch_idx = idx / config.out_dim;
    let out_idx = idx % config.out_dim;
    
    if (batch_idx >= config.batch_size) {
        return;
    }
    
    var sum = 0.0;
    let grid_size_f = f32(config.grid_size);
    
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {
        let input_idx = batch_idx * config.in_dim + in_idx;
        var x = input[input_idx];
        
        // Normalize to [0, 1]
        x = clamp(x, config.grid_min, config.grid_max);
        let t = (x - config.grid_min) / (config.grid_max - config.grid_min);
        
        // Scale to grid
        let t_grid = t * grid_size_f;
        let span = clamp(u32(floor(t_grid)), 0u, config.grid_size - 1u);
        let t_local = t_grid - f32(span);
        
        // Get basis values
        let basis = cubic_basis(t_local);
        
        // Weight base index
        let weight_base = (out_idx * config.in_dim + in_idx) * config.basis_padded + span;
        
        // Accumulate weighted basis
        sum += weights[weight_base] * basis.x;
        if (span + 1u < config.basis_padded) {
            sum += weights[weight_base + 1u] * basis.y;
        }
        if (span + 2u < config.basis_padded) {
            sum += weights[weight_base + 2u] * basis.z;
        }
        if (span + 3u < config.basis_padded) {
            sum += weights[weight_base + 3u] * basis.w;
        }
    }
    
    sum += bias[out_idx];
    output[batch_idx * config.out_dim + out_idx] = sum;
}
"#;

/// Softmax shader for output normalization.
pub const SOFTMAX_SHADER: &str = r#"
struct Uniforms {
    num_elements: u32,
    dim: u32,
    batch_size: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> config: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn softmax_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= config.batch_size) {
        return;
    }
    
    let offset = batch_idx * config.dim;
    
    // Find max for numerical stability
    var max_val = data[offset];
    for (var i = 1u; i < config.dim; i++) {
        max_val = max(max_val, data[offset + i]);
    }
    
    // Compute exp and sum
    var sum = 0.0;
    for (var i = 0u; i < config.dim; i++) {
        let exp_val = exp(data[offset + i] - max_val);
        data[offset + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    let inv_sum = 1.0 / sum;
    for (var i = 0u; i < config.dim; i++) {
        data[offset + i] *= inv_sum;
    }
}
"#;

/// ReLU activation shader.
pub const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn relu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&data)) {
        data[idx] = max(data[idx], 0.0);
    }
}
"#;

/// Element-wise addition shader.
pub const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn add_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&result)) {
        result[idx] = a[idx] + b[idx];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_sources_not_empty() {
        assert!(!FORWARD_SHADER.is_empty());
        assert!(!FORWARD_SIMPLE_SHADER.is_empty());
        assert!(!SOFTMAX_SHADER.is_empty());
        assert!(!RELU_SHADER.is_empty());
        assert!(!ADD_SHADER.is_empty());
    }

    #[test]
    fn test_shader_contains_entry_points() {
        assert!(FORWARD_SHADER.contains("forward_main"));
        assert!(FORWARD_SIMPLE_SHADER.contains("forward_simple"));
        assert!(SOFTMAX_SHADER.contains("softmax_main"));
        assert!(RELU_SHADER.contains("relu_main"));
        assert!(ADD_SHADER.contains("add_main"));
    }
}
