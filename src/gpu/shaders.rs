//! WGSL Shader source code for GPU operations.
//!
//! This module contains shader source code as string constants that are
//! compiled at runtime by wgpu.

/// Forward pass shader for KAN layer (cubic B-spline, order=3).
///
/// Computes: y[j] = Σᵢ Σₖ weights[j,i,k] · B_k(x[i]) + bias[j]
///
/// # Bind Groups
///
/// - Group 0 (Static):
///   - Binding 0: weights (storage, read) - array<vec4<f32>>, layout [out_dim, in_dim, basis_vec4s]
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
///
/// # Weight Layout
///
/// Weights are stored as `array<vec4<f32>>` where basis_vec4s = ceil(basis_padded / 4).
/// Each vec4 contains 4 consecutive basis weights. Access pattern:
/// - vec4_idx = basis_idx / 4
/// - component = basis_idx % 4 (use indexing: v[0], v[1], v[2], v[3])
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
// Weights stored as vec4 for efficient GPU access
@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> config: Uniforms;

// Group 1: Dynamic workspace resources
@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

// Cubic B-spline basis functions (hardcoded order=3)
// Input: t_local in [0, 1) - local parameter within span
// Output: vec4 with 4 basis values B_{i-1}, B_i, B_{i+1}, B_{i+2}
fn cubic_basis(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    
    return vec4<f32>(
        omt3 / 6.0,                                    // B_{i-1}
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,            // B_i
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, // B_{i+1}
        t3 / 6.0                                       // B_{i+2}
    );
}

// Get weight at specific basis index from vec4 array
fn get_weight(base_vec4: u32, basis_idx: u32) -> f32 {
    let vec4_idx = base_vec4 + basis_idx / 4u;
    let component = basis_idx % 4u;
    let w = weights[vec4_idx];
    
    // Component selection (WGSL doesn't support dynamic indexing on vec4)
    if (component == 0u) { return w.x; }
    else if (component == 1u) { return w.y; }
    else if (component == 2u) { return w.z; }
    else { return w.w; }
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
    let grid_size_f = f32(config.grid_size);
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    
    // Process each input dimension
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {
        // Get input value and normalize to [0, 1]
        let input_idx = batch_idx * config.in_dim + in_idx;
        var x = input[input_idx];
        
        // Clamp to grid range and normalize
        x = clamp(x, config.grid_min, config.grid_max);
        let t_norm = (x - config.grid_min) / (config.grid_max - config.grid_min);
        
        // Scale to grid coordinates
        let t_grid = t_norm * grid_size_f;
        let span = clamp(u32(floor(t_grid)), 0u, config.grid_size - 1u);
        let t_local = t_grid - f32(span);
        
        // Compute cubic basis values
        let basis = cubic_basis(t_local);
        
        // Weight base in vec4 units for this (out, in) pair
        let weight_base_vec4 = (out_idx * config.in_dim + in_idx) * basis_vec4s;
        
        // Accumulate weighted basis (4 active basis functions for cubic)
        sum += get_weight(weight_base_vec4, span) * basis.x;
        if (span + 1u < config.basis_padded) {
            sum += get_weight(weight_base_vec4, span + 1u) * basis.y;
        }
        if (span + 2u < config.basis_padded) {
            sum += get_weight(weight_base_vec4, span + 2u) * basis.z;
        }
        if (span + 3u < config.basis_padded) {
            sum += get_weight(weight_base_vec4, span + 3u) * basis.w;
        }
    }
    
    // Add bias and write output
    sum += bias[out_idx];
    
    let output_idx = batch_idx * config.out_dim + out_idx;
    output[output_idx] = sum;
}
"#;

/// Simple forward shader with vec4 weight access (cubic B-spline, order=3).
///
/// This is a simpler version that processes each output element independently.
/// Weights are stored as `array<vec4<f32>>` for efficient memory access.
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

// Weights stored as vec4 for efficient GPU access
@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
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

// Get weight at specific basis index from vec4 array
fn get_weight(base_vec4: u32, basis_idx: u32) -> f32 {
    let vec4_idx = base_vec4 + basis_idx / 4u;
    let component = basis_idx % 4u;
    let w = weights[vec4_idx];
    
    if (component == 0u) { return w.x; }
    else if (component == 1u) { return w.y; }
    else if (component == 2u) { return w.z; }
    else { return w.w; }
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
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    
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
        
        // Weight base in vec4 units
        let weight_base_vec4 = (out_idx * config.in_dim + in_idx) * basis_vec4s;
        
        // Accumulate weighted basis using vec4 access
        sum += get_weight(weight_base_vec4, span) * basis.x;
        if (span + 1u < config.basis_padded) {
            sum += get_weight(weight_base_vec4, span + 1u) * basis.y;
        }
        if (span + 2u < config.basis_padded) {
            sum += get_weight(weight_base_vec4, span + 2u) * basis.z;
        }
        if (span + 3u < config.basis_padded) {
            sum += get_weight(weight_base_vec4, span + 3u) * basis.w;
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
