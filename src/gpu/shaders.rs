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
///
/// **Status: Experimental** - Available but not integrated into forward pipeline.
/// Can be used for custom post-processing or added manually to compute passes.
#[allow(dead_code)]
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
///
/// **Status: Experimental** - Available but not integrated into forward pipeline.
/// KAN networks typically don't use ReLU (the B-splines provide non-linearity).
#[allow(dead_code)]
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
///
/// **Status: Experimental** - Available for custom tensor operations.
/// Not used in standard KAN forward/backward passes.
#[allow(dead_code)]
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

/// Backward pass shader for KAN layer - weight gradients.
///
/// **Strategy: Per-weight-element parallelism with batch reduction**
///
/// Each thread handles ONE weight element grad_weights[j,i,k] and reduces
/// across all batch samples. No race conditions since each weight is updated
/// by exactly one thread.
///
/// Computes:
/// - grad_weights[j,i,k] = Σ_batch grad_output[b,j] * basis_values[b,i,k]
///
/// # Bind Groups
///
/// - Group 0 (Static):
///   - Binding 0: weights (storage, read) - for input gradient computation
///   - Binding 1: config (uniform) - BackwardUniforms
///
/// - Group 1 (Dynamic):
///   - Binding 0: z_values (storage, read) - clamped inputs [batch, in_dim]
///   - Binding 1: span_indices (storage, read) - span indices [batch, in_dim]
///   - Binding 2: grad_output (storage, read) - [batch, out_dim]
///   - Binding 3: grad_weights (storage, read_write) - [out_dim, in_dim, basis_padded]
///   - Binding 4: grad_bias (storage, read_write) - [out_dim] (unused here, see bias shader)
///   - Binding 5: grad_input (storage, read_write) - [batch, in_dim] (unused here, see input_grad shader)
///   - Binding 6: std_inv (storage, read) - 1/std for each input dim [in_dim]
///
/// # Workgroup
///
/// Dispatch: (out_dim * in_dim * basis_padded + 63) / 64
/// Each thread: one (j, i, k) weight element
pub const BACKWARD_WEIGHTS_SHADER: &str = r#"
struct BackwardUniforms {
    grid_min: f32,
    grid_max: f32,
    grid_size: u32,
    order: u32,
    
    in_dim: u32,
    out_dim: u32,
    basis_padded: u32,
    batch_size: u32,
    
    compute_input_grad: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> config: BackwardUniforms;

@group(1) @binding(0) var<storage, read> z_values: array<f32>;
@group(1) @binding(1) var<storage, read> span_indices: array<u32>;
@group(1) @binding(2) var<storage, read> grad_output: array<f32>;
@group(1) @binding(3) var<storage, read_write> grad_weights: array<f32>;
@group(1) @binding(4) var<storage, read_write> grad_bias: array<f32>;
@group(1) @binding(5) var<storage, read_write> grad_input: array<f32>;
@group(1) @binding(6) var<storage, read> std_inv: array<f32>;

// Cubic B-spline basis (order=3)
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

// Get basis value for active index k (0-3) at span+k
fn get_active_basis(basis: vec4<f32>, local_k: u32) -> f32 {
    if (local_k == 0u) { return basis.x; }
    else if (local_k == 1u) { return basis.y; }
    else if (local_k == 2u) { return basis.z; }
    else { return basis.w; }
}

// Per-weight-element kernel: each thread computes grad for ONE weight[j,i,k]
@compute @workgroup_size(64, 1, 1)
fn backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_weights = config.out_dim * config.in_dim * config.basis_padded;
    let weight_idx = global_id.x;
    
    if (weight_idx >= total_weights) {
        return;
    }
    
    // Decode (j, i, k) from flat weight index
    let k = weight_idx % config.basis_padded;
    let temp = weight_idx / config.basis_padded;
    let i = temp % config.in_dim;
    let j = temp / config.in_dim;
    
    let grid_size_f = f32(config.grid_size);
    let grid_range = config.grid_max - config.grid_min;
    
    var grad_sum = 0.0;
    
    // Reduce over batch
    for (var b = 0u; b < config.batch_size; b++) {
        let g_out = grad_output[b * config.out_dim + j];
        
        // Skip masked samples
        if (g_out == 0.0) {
            continue;
        }
        
        // Get saved z and span for this (batch, input)
        let input_idx = b * config.in_dim + i;
        let z = z_values[input_idx];
        let span = span_indices[input_idx];
        
        // Check if this weight k is active for this span
        // Active basis indices are: span, span+1, span+2, span+3
        if (k < span || k > span + 3u) {
            continue; // This basis not active for this sample
        }
        
        // Compute local parameter
        let t_norm = (z - config.grid_min) / grid_range;
        let t_grid = t_norm * grid_size_f;
        let t_local = t_grid - f32(span);
        
        // Get basis value
        let basis = cubic_basis(t_local);
        let local_k = k - span; // 0, 1, 2, or 3
        let basis_val = get_active_basis(basis, local_k);
        
        grad_sum += g_out * basis_val;
    }
    
    grad_weights[weight_idx] = grad_sum;
}
"#;

/// Backward pass shader for input gradients.
///
/// **Strategy: Per-input-element parallelism with output reduction**
///
/// Each thread handles ONE input element grad_input[b,i] and reduces
/// across all output dimensions j.
///
/// Computes:
/// - grad_input[b,i] = Σ_j Σ_k grad_output[b,j] * weights[j,i,k] * basis_deriv[k] * scale
///
/// # Bind Groups (same as BACKWARD_WEIGHTS_SHADER)
///
/// # Workgroup
///
/// Dispatch: (batch_size * in_dim + 63) / 64
pub const BACKWARD_INPUT_SHADER: &str = r#"
struct BackwardUniforms {
    grid_min: f32,
    grid_max: f32,
    grid_size: u32,
    order: u32,
    
    in_dim: u32,
    out_dim: u32,
    basis_padded: u32,
    batch_size: u32,
    
    compute_input_grad: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> config: BackwardUniforms;

@group(1) @binding(0) var<storage, read> z_values: array<f32>;
@group(1) @binding(1) var<storage, read> span_indices: array<u32>;
@group(1) @binding(2) var<storage, read> grad_output: array<f32>;
@group(1) @binding(3) var<storage, read> grad_weights: array<f32>; // read-only here
@group(1) @binding(4) var<storage, read> grad_bias: array<f32>;     // unused
@group(1) @binding(5) var<storage, read_write> grad_input: array<f32>;
@group(1) @binding(6) var<storage, read> std_inv: array<f32>;

// Cubic B-spline derivatives
fn cubic_basis_deriv(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    return vec4<f32>(
        -omt2 / 2.0,
        (3.0 * t2 - 4.0 * t) / 2.0,
        (-3.0 * t2 + 2.0 * t + 1.0) / 2.0,
        t2 / 2.0
    );
}

fn get_deriv(deriv: vec4<f32>, local_k: u32) -> f32 {
    if (local_k == 0u) { return deriv.x; }
    else if (local_k == 1u) { return deriv.y; }
    else if (local_k == 2u) { return deriv.z; }
    else { return deriv.w; }
}

fn get_weight(out_idx: u32, in_idx: u32, basis_idx: u32) -> f32 {
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    let base = (out_idx * config.in_dim + in_idx) * basis_vec4s;
    let vec4_idx = base + basis_idx / 4u;
    let component = basis_idx % 4u;
    let w = weights[vec4_idx];
    if (component == 0u) { return w.x; }
    else if (component == 1u) { return w.y; }
    else if (component == 2u) { return w.z; }
    else { return w.w; }
}

// Per-input-element kernel
@compute @workgroup_size(64, 1, 1)
fn backward_input_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_inputs = config.batch_size * config.in_dim;
    let input_idx = global_id.x;
    
    if (input_idx >= total_inputs) {
        return;
    }
    
    let b = input_idx / config.in_dim;
    let i = input_idx % config.in_dim;
    
    let z = z_values[input_idx];
    let span = span_indices[input_idx];
    
    let grid_size_f = f32(config.grid_size);
    let grid_range = config.grid_max - config.grid_min;
    let t_norm = (z - config.grid_min) / grid_range;
    let t_grid = t_norm * grid_size_f;
    let t_local = t_grid - f32(span);
    
    let deriv = cubic_basis_deriv(t_local);
    let scale = grid_size_f / grid_range * std_inv[i];
    
    var grad_sum = 0.0;
    
    // Sum over all output dimensions
    for (var j = 0u; j < config.out_dim; j++) {
        let g_out = grad_output[b * config.out_dim + j];
        if (g_out == 0.0) {
            continue;
        }
        
        // Sum over 4 active basis functions
        for (var local_k = 0u; local_k < 4u; local_k++) {
            let k = span + local_k;
            if (k < config.basis_padded) {
                let w = get_weight(j, i, k);
                let d = get_deriv(deriv, local_k);
                grad_sum += g_out * w * d * scale;
            }
        }
    }
    
    grad_input[input_idx] = grad_sum;
}
"#;

/// Bias gradient reduction shader.
///
/// Computes: grad_bias[j] = Σ_batch grad_output[batch, j]
/// 
/// This is separated to avoid atomic contention in the main backward pass.
pub const BACKWARD_BIAS_SHADER: &str = r#"
struct BiasUniforms {
    out_dim: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> config: BiasUniforms;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_bias: array<f32>;

// Each thread handles one output dimension
@compute @workgroup_size(64, 1, 1)
fn backward_bias_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    
    if (out_idx >= config.out_dim) {
        return;
    }
    
    var sum = 0.0;
    for (var b = 0u; b < config.batch_size; b++) {
        sum += grad_output[b * config.out_dim + out_idx];
    }
    
    grad_bias[out_idx] += sum;
}
"#;

/// Forward pass with training mode - saves intermediate values.
///
/// Same as FORWARD_SHADER but also saves:
/// - z_values: normalized inputs for backward
/// - span_indices: span indices for backward
pub const FORWARD_TRAINING_SHADER: &str = r#"
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

@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> config: Uniforms;

@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;
@group(1) @binding(2) var<storage, read_write> z_values: array<f32>;
@group(1) @binding(3) var<storage, read_write> span_indices: array<u32>;

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
fn forward_training_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x / config.out_dim;
    let out_idx = global_id.x % config.out_dim;
    
    if (batch_idx >= config.batch_size || out_idx >= config.out_dim) {
        return;
    }
    
    var sum = 0.0;
    let grid_size_f = f32(config.grid_size);
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    let grid_range = config.grid_max - config.grid_min;
    
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {
        let input_idx = batch_idx * config.in_dim + in_idx;
        var x = input[input_idx];
        
        x = clamp(x, config.grid_min, config.grid_max);
        let t_norm = (x - config.grid_min) / grid_range;
        let t_grid = t_norm * grid_size_f;
        let span = clamp(u32(floor(t_grid)), 0u, config.grid_size - 1u);
        let t_local = t_grid - f32(span);
        
        // Save for backward pass (only first output thread per batch writes)
        if (out_idx == 0u) {
            z_values[input_idx] = x;
            span_indices[input_idx] = span;
        }
        
        let basis = cubic_basis(t_local);
        let weight_base_vec4 = (out_idx * config.in_dim + in_idx) * basis_vec4s;
        
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
        assert!(!BACKWARD_WEIGHTS_SHADER.is_empty());
        assert!(!BACKWARD_INPUT_SHADER.is_empty());
        assert!(!BACKWARD_BIAS_SHADER.is_empty());
        assert!(!FORWARD_TRAINING_SHADER.is_empty());
    }

    #[test]
    fn test_shader_contains_entry_points() {
        assert!(FORWARD_SHADER.contains("forward_main"));
        assert!(FORWARD_SIMPLE_SHADER.contains("forward_simple"));
        assert!(SOFTMAX_SHADER.contains("softmax_main"));
        assert!(RELU_SHADER.contains("relu_main"));
        assert!(ADD_SHADER.contains("add_main"));
        assert!(BACKWARD_WEIGHTS_SHADER.contains("backward_main"));
        assert!(BACKWARD_INPUT_SHADER.contains("backward_input_main"));
        assert!(BACKWARD_BIAS_SHADER.contains("backward_bias_main"));
        assert!(FORWARD_TRAINING_SHADER.contains("forward_training_main"));
    }
}
