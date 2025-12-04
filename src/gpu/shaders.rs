//! WGSL Shader source code for GPU operations.
//!
//! This module contains shader source code as string constants that are
//! compiled at runtime by wgpu.
//!
//! # Safety
//!
//! All shaders include bounds checking via `arrayLength()` to prevent
//! out-of-bounds access which could crash the GPU driver or read garbage.

/// Forward pass shader for KAN layer (cubic B-spline, order=3).
///
/// Computes: y[j] = Σᵢ Σₖ weights[j,i,k] · B_k(x[i]) + bias[j]
///
/// # Bind Groups
///
/// - Group 0 (Static):
///   - Binding 0: weights (storage, read) - array<vec4<f32>>, layout [out_dim, in_dim, basis_vec4s]
///   - Binding 1: bias (storage, read) - `(out_dim,)`
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
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()` to prevent
/// out-of-bounds reads which could crash the GPU driver.
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

// Safe weight access with bounds checking
fn get_weight_safe(base_vec4: u32, basis_idx: u32) -> f32 {
    let vec4_idx = base_vec4 + basis_idx / 4u;
    
    // Bounds check
    if (vec4_idx >= arrayLength(&weights)) {
        return 0.0;
    }
    
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
    
    // Bounds check on batch and output indices
    if (batch_idx >= config.batch_size || out_idx >= config.out_dim) {
        return;
    }
    
    // Additional bounds check on output buffer
    let output_idx = batch_idx * config.out_dim + out_idx;
    if (output_idx >= arrayLength(&output)) {
        return;
    }
    
    // Bounds check on bias
    if (out_idx >= arrayLength(&bias)) {
        return;
    }
    
    var sum = 0.0;
    let grid_size_f = f32(config.grid_size);
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    
    // Process each input dimension
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {
        // Bounds check on input buffer
        let input_idx = batch_idx * config.in_dim + in_idx;
        if (input_idx >= arrayLength(&input)) {
            continue;
        }
        
        // Get input value and normalize to [0, 1]
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
        // Using safe weight access to prevent OOB
        sum += get_weight_safe(weight_base_vec4, span) * basis.x;
        if (span + 1u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 1u) * basis.y;
        }
        if (span + 2u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 2u) * basis.z;
        }
        if (span + 3u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 3u) * basis.w;
        }
    }
    
    // Add bias and write output
    sum += bias[out_idx];
    output[output_idx] = sum;
}
"#;

/// Simple forward shader with vec4 weight access (cubic B-spline, order=3).
///
/// This is a simpler version that processes each output element independently.
/// Weights are stored as `array<vec4<f32>>` for efficient memory access.
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
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

// Safe weight access with bounds checking
fn get_weight_safe(base_vec4: u32, basis_idx: u32) -> f32 {
    let vec4_idx = base_vec4 + basis_idx / 4u;
    
    // Bounds check
    if (vec4_idx >= arrayLength(&weights)) {
        return 0.0;
    }
    
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
    
    // Bounds check on batch and output indices
    if (batch_idx >= config.batch_size || out_idx >= config.out_dim) {
        return;
    }
    
    // Bounds check on output buffer
    let output_idx = batch_idx * config.out_dim + out_idx;
    if (output_idx >= arrayLength(&output)) {
        return;
    }
    
    // Bounds check on bias
    if (out_idx >= arrayLength(&bias)) {
        return;
    }
    
    var sum = 0.0;
    let grid_size_f = f32(config.grid_size);
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {
        let input_idx = batch_idx * config.in_dim + in_idx;
        
        // Bounds check on input
        if (input_idx >= arrayLength(&input)) {
            continue;
        }
        
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
        
        // Accumulate weighted basis using safe vec4 access
        sum += get_weight_safe(weight_base_vec4, span) * basis.x;
        if (span + 1u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 1u) * basis.y;
        }
        if (span + 2u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 2u) * basis.z;
        }
        if (span + 3u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 3u) * basis.w;
        }
    }
    
    sum += bias[out_idx];
    output[output_idx] = sum;
}
"#;

/// Softmax shader for output normalization.
///
/// **Status: Experimental** - Available but not integrated into forward pipeline.
/// Can be used for custom post-processing or added manually to compute passes.
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
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
    let data_len = arrayLength(&data);
    
    // Bounds check on first element
    if (offset >= data_len) {
        return;
    }
    
    // Find max for numerical stability
    var max_val = data[offset];
    for (var i = 1u; i < config.dim; i++) {
        let idx = offset + i;
        if (idx >= data_len) { break; }
        max_val = max(max_val, data[idx]);
    }
    
    // Compute exp and sum
    var sum = 0.0;
    for (var i = 0u; i < config.dim; i++) {
        let idx = offset + i;
        if (idx >= data_len) { break; }
        let exp_val = exp(data[idx] - max_val);
        data[idx] = exp_val;
        sum += exp_val;
    }
    
    // Normalize (guard against division by zero)
    if (sum <= 0.0) { return; }
    let inv_sum = 1.0 / sum;
    for (var i = 0u; i < config.dim; i++) {
        let idx = offset + i;
        if (idx >= data_len) { break; }
        data[idx] *= inv_sum;
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
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
#[allow(dead_code)]
pub const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn add_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let result_len = arrayLength(&result);
    let a_len = arrayLength(&a);
    let b_len = arrayLength(&b);
    
    // All three arrays must have this index
    if (idx < result_len && idx < a_len && idx < b_len) {
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
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
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
    
    // Bounds check on output buffer
    if (weight_idx >= arrayLength(&grad_weights)) {
        return;
    }
    
    // Decode (j, i, k) from flat weight index
    let k = weight_idx % config.basis_padded;
    let temp = weight_idx / config.basis_padded;
    let i = temp % config.in_dim;
    let j = temp / config.in_dim;
    
    let grid_size_f = f32(config.grid_size);
    let grid_range = config.grid_max - config.grid_min;
    
    let z_len = arrayLength(&z_values);
    let span_len = arrayLength(&span_indices);
    let grad_out_len = arrayLength(&grad_output);
    
    var grad_sum = 0.0;
    
    // Reduce over batch
    for (var b = 0u; b < config.batch_size; b++) {
        let out_idx = b * config.out_dim + j;
        
        // Bounds check
        if (out_idx >= grad_out_len) {
            continue;
        }
        
        let g_out = grad_output[out_idx];
        
        // Skip masked samples
        if (g_out == 0.0) {
            continue;
        }
        
        // Get saved z and span for this (batch, input)
        let input_idx = b * config.in_dim + i;
        
        // Bounds check on saved values
        if (input_idx >= z_len || input_idx >= span_len) {
            continue;
        }
        
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
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
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

// Safe weight access with bounds checking
fn get_weight_safe(out_idx: u32, in_idx: u32, basis_idx: u32) -> f32 {
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    let base = (out_idx * config.in_dim + in_idx) * basis_vec4s;
    let vec4_idx = base + basis_idx / 4u;
    
    // Bounds check
    if (vec4_idx >= arrayLength(&weights)) {
        return 0.0;
    }
    
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
    
    // Bounds check on output buffer
    if (input_idx >= arrayLength(&grad_input)) {
        return;
    }
    
    let b = input_idx / config.in_dim;
    let i = input_idx % config.in_dim;
    
    // Bounds check on z_values and span_indices
    let z_len = arrayLength(&z_values);
    let span_len = arrayLength(&span_indices);
    if (input_idx >= z_len || input_idx >= span_len) {
        return;
    }
    
    let z = z_values[input_idx];
    let span = span_indices[input_idx];
    
    let grid_size_f = f32(config.grid_size);
    let grid_range = config.grid_max - config.grid_min;
    let t_norm = (z - config.grid_min) / grid_range;
    let t_grid = t_norm * grid_size_f;
    let t_local = t_grid - f32(span);
    
    let deriv = cubic_basis_deriv(t_local);
    
    // Bounds check on std_inv
    var scale_factor = 1.0;
    if (i < arrayLength(&std_inv)) {
        scale_factor = grid_size_f / grid_range * std_inv[i];
    }
    
    let grad_out_len = arrayLength(&grad_output);
    
    var grad_sum = 0.0;
    
    // Sum over all output dimensions
    for (var j = 0u; j < config.out_dim; j++) {
        let out_idx = b * config.out_dim + j;
        
        // Bounds check
        if (out_idx >= grad_out_len) {
            continue;
        }
        
        let g_out = grad_output[out_idx];
        if (g_out == 0.0) {
            continue;
        }
        
        // Sum over 4 active basis functions
        for (var local_k = 0u; local_k < 4u; local_k++) {
            let k = span + local_k;
            if (k < config.basis_padded) {
                let w = get_weight_safe(j, i, k);
                let d = get_deriv(deriv, local_k);
                grad_sum += g_out * w * d * scale_factor;
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
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
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
    
    // Bounds check on output buffer
    if (out_idx >= arrayLength(&grad_bias)) {
        return;
    }
    
    let grad_out_len = arrayLength(&grad_output);
    
    var sum = 0.0;
    for (var b = 0u; b < config.batch_size; b++) {
        let idx = b * config.out_dim + out_idx;
        // Bounds check
        if (idx >= grad_out_len) {
            break;
        }
        sum += grad_output[idx];
    }
    
    grad_bias[out_idx] += sum;
}
"#;

/// Forward pass with training mode - saves intermediate values.
///
/// Same as FORWARD_SHADER but also saves:
/// - z_values: normalized inputs for backward
/// - span_indices: span indices for backward
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
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

// Safe weight access with bounds checking
fn get_weight_safe(base_vec4: u32, basis_idx: u32) -> f32 {
    let vec4_idx = base_vec4 + basis_idx / 4u;
    
    // Bounds check
    if (vec4_idx >= arrayLength(&weights)) {
        return 0.0;
    }
    
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
    
    // Bounds check on output buffer
    let output_idx = batch_idx * config.out_dim + out_idx;
    if (output_idx >= arrayLength(&output)) {
        return;
    }
    
    // Bounds check on bias
    if (out_idx >= arrayLength(&bias)) {
        return;
    }
    
    var sum = 0.0;
    let grid_size_f = f32(config.grid_size);
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    let grid_range = config.grid_max - config.grid_min;
    
    let input_len = arrayLength(&input);
    let z_len = arrayLength(&z_values);
    let span_len = arrayLength(&span_indices);
    
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {
        let input_idx = batch_idx * config.in_dim + in_idx;
        
        // Bounds check on input
        if (input_idx >= input_len) {
            continue;
        }
        
        var x = input[input_idx];
        
        x = clamp(x, config.grid_min, config.grid_max);
        let t_norm = (x - config.grid_min) / grid_range;
        let t_grid = t_norm * grid_size_f;
        let span = clamp(u32(floor(t_grid)), 0u, config.grid_size - 1u);
        let t_local = t_grid - f32(span);
        
        // Save for backward pass (only first output thread per batch writes)
        if (out_idx == 0u) {
            if (input_idx < z_len) {
                z_values[input_idx] = x;
            }
            if (input_idx < span_len) {
                span_indices[input_idx] = span;
            }
        }
        
        let basis = cubic_basis(t_local);
        let weight_base_vec4 = (out_idx * config.in_dim + in_idx) * basis_vec4s;
        
        sum += get_weight_safe(weight_base_vec4, span) * basis.x;
        if (span + 1u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 1u) * basis.y;
        }
        if (span + 2u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 2u) * basis.z;
        }
        if (span + 3u < config.basis_padded) {
            sum += get_weight_safe(weight_base_vec4, span + 3u) * basis.w;
        }
    }
    
    sum += bias[out_idx];
    output[output_idx] = sum;
}
"#;

// =============================================================================
// GPU Optimizer Shaders
// =============================================================================

/// Adam optimizer compute shader.
///
/// Updates weights using Adam algorithm with bias correction.
///
/// Computes for each parameter i:
/// - m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
/// - v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
/// - m_hat = m[i] / (1 - beta1^t)
/// - v_hat = v[i] / (1 - beta2^t)
/// - param[i] -= lr * m_hat / (sqrt(v_hat) + epsilon)
/// - param[i] -= weight_decay * param[i]  (decoupled weight decay)
///
/// # Bind Groups
///
/// - Binding 0: params (storage, read_write) - weights to update
/// - Binding 1: grads (storage, read) - gradients
/// - Binding 2: m (storage, read_write) - first moment
/// - Binding 3: v (storage, read_write) - second moment
/// - Binding 4: config (uniform) - optimizer configuration
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
pub const ADAM_SHADER: &str = r#"
struct AdamUniforms {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    // Bias correction factors (precomputed on CPU):
    // beta1_correction = 1 / (1 - beta1^t)
    // beta2_correction = 1 / (1 - beta2^t)
    beta1_correction: f32,
    beta2_correction: f32,
    num_params: u32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> config: AdamUniforms;

@compute @workgroup_size(256, 1, 1)
fn adam_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds checks
    if (idx >= config.num_params) {
        return;
    }
    
    let params_len = arrayLength(&params);
    let grads_len = arrayLength(&grads);
    let m_len = arrayLength(&m);
    let v_len = arrayLength(&v);
    
    if (idx >= params_len || idx >= grads_len || idx >= m_len || idx >= v_len) {
        return;
    }
    
    let grad = grads[idx];
    
    // Update biased first moment estimate
    let m_new = config.beta1 * m[idx] + (1.0 - config.beta1) * grad;
    m[idx] = m_new;
    
    // Update biased second moment estimate
    let v_new = config.beta2 * v[idx] + (1.0 - config.beta2) * grad * grad;
    v[idx] = v_new;
    
    // Bias-corrected estimates
    let m_hat = m_new * config.beta1_correction;
    let v_hat = v_new * config.beta2_correction;
    
    // Update parameters
    let param = params[idx];
    var new_param = param - config.lr * m_hat / (sqrt(v_hat) + config.epsilon);
    
    // Decoupled weight decay (AdamW)
    if (config.weight_decay > 0.0) {
        new_param -= config.weight_decay * config.lr * param;
    }
    
    params[idx] = new_param;
}
"#;

/// SGD optimizer compute shader with momentum.
///
/// Computes:
/// - velocity[i] = momentum * velocity[i] + grad[i]
/// - param[i] -= lr * velocity[i]
/// - param[i] -= weight_decay * param[i]  (decoupled weight decay)
///
/// # Bounds Safety
///
/// All array accesses are bounds-checked via `arrayLength()`.
pub const SGD_SHADER: &str = r#"
struct SGDUniforms {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    num_params: u32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(3) var<uniform> config: SGDUniforms;

@compute @workgroup_size(256, 1, 1)
fn sgd_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds checks
    if (idx >= config.num_params) {
        return;
    }
    
    let params_len = arrayLength(&params);
    let grads_len = arrayLength(&grads);
    let vel_len = arrayLength(&velocity);
    
    if (idx >= params_len || idx >= grads_len || idx >= vel_len) {
        return;
    }
    
    let grad = grads[idx];
    
    // Update velocity with momentum
    let vel_new = config.momentum * velocity[idx] + grad;
    velocity[idx] = vel_new;
    
    // Update parameters
    let param = params[idx];
    var new_param = param - config.lr * vel_new;
    
    // Decoupled weight decay
    if (config.weight_decay > 0.0) {
        new_param -= config.weight_decay * config.lr * param;
    }
    
    params[idx] = new_param;
}
"#;

/// Gradient clipping shader (max norm).
///
/// Clips gradients so that their global L2 norm doesn't exceed max_norm.
/// This is a two-pass operation:
/// 1. First pass: compute squared norm (using reduce pattern)
/// 2. Second pass: scale gradients if needed (this shader)
///
/// For simplicity, we assume norm is computed on CPU and passed as uniform.
pub const GRAD_CLIP_SHADER: &str = r#"
struct ClipUniforms {
    scale: f32,        // min(1.0, max_norm / grad_norm)
    num_params: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> grads: array<f32>;
@group(0) @binding(1) var<uniform> config: ClipUniforms;

@compute @workgroup_size(256, 1, 1)
fn grad_clip_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= config.num_params) {
        return;
    }
    
    if (idx >= arrayLength(&grads)) {
        return;
    }
    
    // Scale gradient if norm exceeds threshold
    if (config.scale < 1.0) {
        grads[idx] *= config.scale;
    }
}
"#;

// =============================================================================
// Dynamic Shader Generation for Variable Spline Orders (2-5)
// =============================================================================

use crate::config::{MAX_GPU_SPLINE_ORDER, MIN_GPU_SPLINE_ORDER};
use crate::error::{ArkanError, ArkanResult};

/// Generates B-spline basis function WGSL code for a given order.
///
/// # Arguments
///
/// * `order` - Spline order (2-5). Order 2 = quadratic, 3 = cubic, etc.
///
/// # Returns
///
/// WGSL function definition as a string.
fn generate_basis_function(order: usize) -> String {
    match order {
        2 => {
            // Quadratic B-spline (order=2): 3 basis functions
            r#"
// Quadratic B-spline basis functions (order=2)
// Returns vec3 with 3 basis values
fn bspline_basis(t: f32) -> vec3<f32> {
    let t2 = t * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    
    return vec3<f32>(
        omt2 / 2.0,           // B_0
        -t2 + t + 0.5,        // B_1
        t2 / 2.0              // B_2
    );
}

fn get_basis_value(basis: vec3<f32>, idx: u32) -> f32 {
    if (idx == 0u) { return basis.x; }
    else if (idx == 1u) { return basis.y; }
    else { return basis.z; }
}

const ACTIVE_BASIS_COUNT: u32 = 3u;
"#
            .to_string()
        }
        3 => {
            // Cubic B-spline (order=3): 4 basis functions
            r#"
// Cubic B-spline basis functions (order=3)
// Returns vec4 with 4 basis values
fn bspline_basis(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    
    return vec4<f32>(
        omt3 / 6.0,                                    // B_0
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,            // B_1
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, // B_2
        t3 / 6.0                                       // B_3
    );
}

fn get_basis_value(basis: vec4<f32>, idx: u32) -> f32 {
    if (idx == 0u) { return basis.x; }
    else if (idx == 1u) { return basis.y; }
    else if (idx == 2u) { return basis.z; }
    else { return basis.w; }
}

const ACTIVE_BASIS_COUNT: u32 = 4u;
"#
            .to_string()
        }
        4 => {
            // Quartic B-spline (order=4): 5 basis functions
            // Need to store in array since vec5 doesn't exist
            r#"
// Quartic B-spline basis functions (order=4)
// Returns array of 5 basis values
fn bspline_basis(t: f32) -> array<f32, 5> {
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    let omt4 = omt3 * omt;
    
    var result: array<f32, 5>;
    result[0] = omt4 / 24.0;
    result[1] = (4.0 * t4 - 12.0 * t3 + 6.0 * t2 + 12.0 * t + 1.0) / 24.0;
    result[2] = (-6.0 * t4 + 12.0 * t3 + 6.0 * t2 - 12.0 * t + 11.0) / 24.0;
    result[3] = (4.0 * t4 - 4.0 * t3 - 6.0 * t2 - 4.0 * t + 11.0) / 24.0;
    result[4] = t4 / 24.0;
    return result;
}

fn get_basis_value(basis: array<f32, 5>, idx: u32) -> f32 {
    return basis[idx];
}

const ACTIVE_BASIS_COUNT: u32 = 5u;
"#
            .to_string()
        }
        5 => {
            // Quintic B-spline (order=5): 6 basis functions
            r#"
// Quintic B-spline basis functions (order=5)
// Returns array of 6 basis values
fn bspline_basis(t: f32) -> array<f32, 6> {
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    let omt4 = omt3 * omt;
    let omt5 = omt4 * omt;
    
    var result: array<f32, 6>;
    result[0] = omt5 / 120.0;
    result[1] = (5.0 * t5 - 20.0 * t4 + 20.0 * t3 + 20.0 * t2 - 20.0 * t + 26.0) / 120.0;
    result[2] = (-10.0 * t5 + 30.0 * t4 - 20.0 * t2 + 66.0) / 120.0;
    result[3] = (10.0 * t5 - 20.0 * t4 - 20.0 * t3 + 20.0 * t2 + 20.0 * t + 26.0) / 120.0;
    result[4] = (-5.0 * t5 + 5.0 * t4 + 10.0 * t3 + 10.0 * t2 + 5.0 * t + 1.0) / 120.0;
    result[5] = t5 / 120.0;
    return result;
}

fn get_basis_value(basis: array<f32, 6>, idx: u32) -> f32 {
    return basis[idx];
}

const ACTIVE_BASIS_COUNT: u32 = 6u;
"#
            .to_string()
        }
        _ => panic!("Unsupported spline order: {}", order),
    }
}

/// Generates the basis derivative function for backward pass.
fn generate_basis_derivative(order: usize) -> String {
    match order {
        2 => {
            r#"
// Quadratic B-spline derivatives (order=2)
fn bspline_deriv(t: f32) -> vec3<f32> {
    let omt = 1.0 - t;
    return vec3<f32>(
        -omt,              // dB_0/dt
        1.0 - 2.0 * t,     // dB_1/dt
        t                  // dB_2/dt
    );
}

fn get_deriv_value(deriv: vec3<f32>, idx: u32) -> f32 {
    if (idx == 0u) { return deriv.x; }
    else if (idx == 1u) { return deriv.y; }
    else { return deriv.z; }
}
"#
            .to_string()
        }
        3 => {
            r#"
// Cubic B-spline derivatives (order=3)
fn bspline_deriv(t: f32) -> vec4<f32> {
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

fn get_deriv_value(deriv: vec4<f32>, idx: u32) -> f32 {
    if (idx == 0u) { return deriv.x; }
    else if (idx == 1u) { return deriv.y; }
    else if (idx == 2u) { return deriv.z; }
    else { return deriv.w; }
}
"#
            .to_string()
        }
        4 => {
            r#"
// Quartic B-spline derivatives (order=4)
fn bspline_deriv(t: f32) -> array<f32, 5> {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    
    var result: array<f32, 5>;
    result[0] = -omt3 / 6.0;
    result[1] = (16.0 * t3 - 36.0 * t2 + 12.0 * t + 12.0) / 24.0;
    result[2] = (-24.0 * t3 + 36.0 * t2 + 12.0 * t - 12.0) / 24.0;
    result[3] = (16.0 * t3 - 12.0 * t2 - 12.0 * t - 4.0) / 24.0;
    result[4] = t3 / 6.0;
    return result;
}

fn get_deriv_value(deriv: array<f32, 5>, idx: u32) -> f32 {
    return deriv[idx];
}
"#
            .to_string()
        }
        5 => {
            r#"
// Quintic B-spline derivatives (order=5)
fn bspline_deriv(t: f32) -> array<f32, 6> {
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    let omt4 = omt3 * omt;
    
    var result: array<f32, 6>;
    result[0] = -omt4 / 24.0;
    result[1] = (25.0 * t4 - 80.0 * t3 + 60.0 * t2 + 40.0 * t - 20.0) / 120.0;
    result[2] = (-50.0 * t4 + 120.0 * t3 - 40.0 * t) / 120.0;
    result[3] = (50.0 * t4 - 80.0 * t3 - 60.0 * t2 + 40.0 * t + 20.0) / 120.0;
    result[4] = (-25.0 * t4 + 20.0 * t3 + 30.0 * t2 + 20.0 * t + 5.0) / 120.0;
    result[5] = t4 / 24.0;
    return result;
}

fn get_deriv_value(deriv: array<f32, 6>, idx: u32) -> f32 {
    return deriv[idx];
}
"#
            .to_string()
        }
        _ => panic!("Unsupported spline order: {}", order),
    }
}

/// Generates a forward pass shader for a specific spline order.
///
/// # Arguments
///
/// * `order` - Spline order (2-5)
///
/// # Returns
///
/// Complete WGSL shader source code.
///
/// # Errors
///
/// Returns `ArkanError::UnsupportedOrder` if order is outside [2, 5] range.
///
/// # Example
///
/// ```rust
/// use arkan::gpu::generate_forward_shader;
///
/// let shader = generate_forward_shader(4).unwrap(); // Quartic splines
/// assert!(shader.contains("order=4"));
/// ```
pub fn generate_forward_shader(order: usize) -> ArkanResult<String> {
    if order < MIN_GPU_SPLINE_ORDER || order > MAX_GPU_SPLINE_ORDER {
        return Err(ArkanError::unsupported_order(order));
    }

    let basis_fn = generate_basis_function(order);

    Ok(format!(
        r#"
// Auto-generated forward shader for order={order} B-splines
// Generated by ArKan dynamic shader system

struct Uniforms {{
    grid_min: f32,
    grid_max: f32,
    grid_size: u32,
    order: u32,
    
    in_dim: u32,
    out_dim: u32,
    basis_padded: u32,
    batch_size: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> config: Uniforms;

@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

{basis_fn}

// Safe weight access with bounds checking
fn get_weight_safe(base_vec4: u32, basis_idx: u32) -> f32 {{
    let vec4_idx = base_vec4 + basis_idx / 4u;
    
    if (vec4_idx >= arrayLength(&weights)) {{
        return 0.0;
    }}
    
    let component = basis_idx % 4u;
    let w = weights[vec4_idx];
    
    if (component == 0u) {{ return w.x; }}
    else if (component == 1u) {{ return w.y; }}
    else if (component == 2u) {{ return w.z; }}
    else {{ return w.w; }}
}}

@compute @workgroup_size(64, 1, 1)
fn forward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x / config.out_dim;
    let out_idx = global_id.x % config.out_dim;
    
    if (batch_idx >= config.batch_size || out_idx >= config.out_dim) {{
        return;
    }}
    
    let output_idx = batch_idx * config.out_dim + out_idx;
    if (output_idx >= arrayLength(&output)) {{
        return;
    }}
    
    if (out_idx >= arrayLength(&bias)) {{
        return;
    }}
    
    var sum = 0.0;
    let grid_size_f = f32(config.grid_size);
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {{
        let input_idx = batch_idx * config.in_dim + in_idx;
        if (input_idx >= arrayLength(&input)) {{
            continue;
        }}
        
        var x = input[input_idx];
        
        x = clamp(x, config.grid_min, config.grid_max);
        let t_norm = (x - config.grid_min) / (config.grid_max - config.grid_min);
        let t_grid = t_norm * grid_size_f;
        let span = clamp(u32(floor(t_grid)), 0u, config.grid_size - 1u);
        let t_local = t_grid - f32(span);
        
        let basis = bspline_basis(t_local);
        let weight_base_vec4 = (out_idx * config.in_dim + in_idx) * basis_vec4s;
        
        for (var k = 0u; k < ACTIVE_BASIS_COUNT; k++) {{
            let basis_idx = span + k;
            if (basis_idx < config.basis_padded) {{
                sum += get_weight_safe(weight_base_vec4, basis_idx) * get_basis_value(basis, k);
            }}
        }}
    }}
    
    sum += bias[out_idx];
    output[output_idx] = sum;
}}
"#,
        order = order,
        basis_fn = basis_fn
    ))
}

/// Generates a backward pass shader for weight gradients for a specific spline order.
///
/// # Arguments
///
/// * `order` - Spline order (2-5)
///
/// # Returns
///
/// Complete WGSL shader source code for weight gradient computation.
pub fn generate_backward_weights_shader(order: usize) -> ArkanResult<String> {
    if order < MIN_GPU_SPLINE_ORDER || order > MAX_GPU_SPLINE_ORDER {
        return Err(ArkanError::unsupported_order(order));
    }

    let basis_fn = generate_basis_function(order);

    Ok(format!(
        r#"
// Auto-generated backward weights shader for order={order} B-splines

struct BackwardUniforms {{
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
}}

@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> config: BackwardUniforms;

@group(1) @binding(0) var<storage, read> z_values: array<f32>;
@group(1) @binding(1) var<storage, read> span_indices: array<u32>;
@group(1) @binding(2) var<storage, read> grad_output: array<f32>;
@group(1) @binding(3) var<storage, read_write> grad_weights: array<f32>;
@group(1) @binding(4) var<storage, read_write> grad_bias: array<f32>;
@group(1) @binding(5) var<storage, read_write> grad_input: array<f32>;
@group(1) @binding(6) var<storage, read> std_inv: array<f32>;

{basis_fn}

@compute @workgroup_size(64, 1, 1)
fn backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let total_weights = config.out_dim * config.in_dim * config.basis_padded;
    let weight_idx = global_id.x;
    
    if (weight_idx >= total_weights) {{
        return;
    }}
    
    if (weight_idx >= arrayLength(&grad_weights)) {{
        return;
    }}
    
    let k = weight_idx % config.basis_padded;
    let temp = weight_idx / config.basis_padded;
    let i = temp % config.in_dim;
    let j = temp / config.in_dim;
    
    let grid_size_f = f32(config.grid_size);
    let grid_range = config.grid_max - config.grid_min;
    
    let z_len = arrayLength(&z_values);
    let span_len = arrayLength(&span_indices);
    let grad_out_len = arrayLength(&grad_output);
    
    var grad_sum = 0.0;
    
    for (var b = 0u; b < config.batch_size; b++) {{
        let out_idx = b * config.out_dim + j;
        
        if (out_idx >= grad_out_len) {{
            continue;
        }}
        
        let g_out = grad_output[out_idx];
        
        if (g_out == 0.0) {{
            continue;
        }}
        
        let input_idx = b * config.in_dim + i;
        
        if (input_idx >= z_len || input_idx >= span_len) {{
            continue;
        }}
        
        let z = z_values[input_idx];
        let span = span_indices[input_idx];
        
        // Check if this weight k is active for this span
        if (k < span || k >= span + ACTIVE_BASIS_COUNT) {{
            continue;
        }}
        
        let t_norm = (z - config.grid_min) / grid_range;
        let t_grid = t_norm * grid_size_f;
        let t_local = t_grid - f32(span);
        
        let basis = bspline_basis(t_local);
        let local_k = k - span;
        let basis_val = get_basis_value(basis, local_k);
        
        grad_sum += g_out * basis_val;
    }}
    
    grad_weights[weight_idx] = grad_sum;
}}
"#,
        order = order,
        basis_fn = basis_fn
    ))
}

/// Generates a backward pass shader for input gradients for a specific spline order.
pub fn generate_backward_input_shader(order: usize) -> ArkanResult<String> {
    if order < MIN_GPU_SPLINE_ORDER || order > MAX_GPU_SPLINE_ORDER {
        return Err(ArkanError::unsupported_order(order));
    }

    let basis_fn = generate_basis_function(order);
    let deriv_fn = generate_basis_derivative(order);

    Ok(format!(
        r#"
// Auto-generated backward input shader for order={order} B-splines

struct BackwardUniforms {{
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
}}

@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> config: BackwardUniforms;

@group(1) @binding(0) var<storage, read> z_values: array<f32>;
@group(1) @binding(1) var<storage, read> span_indices: array<u32>;
@group(1) @binding(2) var<storage, read> grad_output: array<f32>;
@group(1) @binding(3) var<storage, read> grad_weights: array<f32>;
@group(1) @binding(4) var<storage, read> grad_bias: array<f32>;
@group(1) @binding(5) var<storage, read_write> grad_input: array<f32>;
@group(1) @binding(6) var<storage, read> std_inv: array<f32>;

{basis_fn}
{deriv_fn}

fn get_weight_safe(out_idx: u32, in_idx: u32, basis_idx: u32) -> f32 {{
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    let base = (out_idx * config.in_dim + in_idx) * basis_vec4s;
    let vec4_idx = base + basis_idx / 4u;
    
    if (vec4_idx >= arrayLength(&weights)) {{
        return 0.0;
    }}
    
    let component = basis_idx % 4u;
    let w = weights[vec4_idx];
    if (component == 0u) {{ return w.x; }}
    else if (component == 1u) {{ return w.y; }}
    else if (component == 2u) {{ return w.z; }}
    else {{ return w.w; }}
}}

@compute @workgroup_size(64, 1, 1)
fn backward_input_main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let total_inputs = config.batch_size * config.in_dim;
    let input_idx = global_id.x;
    
    if (input_idx >= total_inputs) {{
        return;
    }}
    
    if (input_idx >= arrayLength(&grad_input)) {{
        return;
    }}
    
    let b = input_idx / config.in_dim;
    let i = input_idx % config.in_dim;
    
    let z_len = arrayLength(&z_values);
    let span_len = arrayLength(&span_indices);
    if (input_idx >= z_len || input_idx >= span_len) {{
        return;
    }}
    
    let z = z_values[input_idx];
    let span = span_indices[input_idx];
    
    let grid_size_f = f32(config.grid_size);
    let grid_range = config.grid_max - config.grid_min;
    let t_norm = (z - config.grid_min) / grid_range;
    let t_grid = t_norm * grid_size_f;
    let t_local = t_grid - f32(span);
    
    let deriv = bspline_deriv(t_local);
    
    var scale_factor = 1.0;
    if (i < arrayLength(&std_inv)) {{
        scale_factor = grid_size_f / grid_range * std_inv[i];
    }}
    
    let grad_out_len = arrayLength(&grad_output);
    
    var grad_sum = 0.0;
    
    for (var j = 0u; j < config.out_dim; j++) {{
        let out_idx = b * config.out_dim + j;
        
        if (out_idx >= grad_out_len) {{
            continue;
        }}
        
        let g_out = grad_output[out_idx];
        if (g_out == 0.0) {{
            continue;
        }}
        
        for (var local_k = 0u; local_k < ACTIVE_BASIS_COUNT; local_k++) {{
            let k = span + local_k;
            if (k < config.basis_padded) {{
                let w = get_weight_safe(j, i, k);
                let d = get_deriv_value(deriv, local_k);
                grad_sum += g_out * w * d * scale_factor;
            }}
        }}
    }}
    
    grad_input[input_idx] = grad_sum;
}}
"#,
        order = order,
        basis_fn = basis_fn,
        deriv_fn = deriv_fn
    ))
}

/// Generates a forward training shader that saves intermediate values.
pub fn generate_forward_training_shader(order: usize) -> ArkanResult<String> {
    if order < MIN_GPU_SPLINE_ORDER || order > MAX_GPU_SPLINE_ORDER {
        return Err(ArkanError::unsupported_order(order));
    }

    let basis_fn = generate_basis_function(order);

    Ok(format!(
        r#"
// Auto-generated forward training shader for order={order} B-splines

struct Uniforms {{
    grid_min: f32,
    grid_max: f32,
    grid_size: u32,
    order: u32,
    
    in_dim: u32,
    out_dim: u32,
    basis_padded: u32,
    batch_size: u32,
}}

@group(0) @binding(0) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> config: Uniforms;

@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;
@group(1) @binding(2) var<storage, read_write> z_values: array<f32>;
@group(1) @binding(3) var<storage, read_write> span_indices: array<u32>;

{basis_fn}

fn get_weight_safe(base_vec4: u32, basis_idx: u32) -> f32 {{
    let vec4_idx = base_vec4 + basis_idx / 4u;
    
    if (vec4_idx >= arrayLength(&weights)) {{
        return 0.0;
    }}
    
    let component = basis_idx % 4u;
    let w = weights[vec4_idx];
    
    if (component == 0u) {{ return w.x; }}
    else if (component == 1u) {{ return w.y; }}
    else if (component == 2u) {{ return w.z; }}
    else {{ return w.w; }}
}}

@compute @workgroup_size(64, 1, 1)
fn forward_training_main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x / config.out_dim;
    let out_idx = global_id.x % config.out_dim;
    
    if (batch_idx >= config.batch_size || out_idx >= config.out_dim) {{
        return;
    }}
    
    let output_idx = batch_idx * config.out_dim + out_idx;
    if (output_idx >= arrayLength(&output)) {{
        return;
    }}
    
    if (out_idx >= arrayLength(&bias)) {{
        return;
    }}
    
    var sum = 0.0;
    let grid_size_f = f32(config.grid_size);
    let basis_vec4s = (config.basis_padded + 3u) / 4u;
    let grid_range = config.grid_max - config.grid_min;
    
    let input_len = arrayLength(&input);
    let z_len = arrayLength(&z_values);
    let span_len = arrayLength(&span_indices);
    
    for (var in_idx = 0u; in_idx < config.in_dim; in_idx++) {{
        let input_idx = batch_idx * config.in_dim + in_idx;
        
        if (input_idx >= input_len) {{
            continue;
        }}
        
        var x = input[input_idx];
        
        x = clamp(x, config.grid_min, config.grid_max);
        let t_norm = (x - config.grid_min) / grid_range;
        let t_grid = t_norm * grid_size_f;
        let span = clamp(u32(floor(t_grid)), 0u, config.grid_size - 1u);
        let t_local = t_grid - f32(span);
        
        // Save for backward pass (only first output thread per batch writes)
        if (out_idx == 0u) {{
            if (input_idx < z_len) {{
                z_values[input_idx] = x;
            }}
            if (input_idx < span_len) {{
                span_indices[input_idx] = span;
            }}
        }}
        
        let basis = bspline_basis(t_local);
        let weight_base_vec4 = (out_idx * config.in_dim + in_idx) * basis_vec4s;
        
        for (var k = 0u; k < ACTIVE_BASIS_COUNT; k++) {{
            let basis_idx = span + k;
            if (basis_idx < config.basis_padded) {{
                sum += get_weight_safe(weight_base_vec4, basis_idx) * get_basis_value(basis, k);
            }}
        }}
    }}
    
    sum += bias[out_idx];
    output[output_idx] = sum;
}}
"#,
        order = order,
        basis_fn = basis_fn
    ))
}

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

    #[test]
    fn test_shaders_have_bounds_checking() {
        // All production shaders must use arrayLength for bounds safety
        assert!(
            FORWARD_SHADER.contains("arrayLength"),
            "FORWARD_SHADER must have arrayLength bounds checks"
        );
        assert!(
            FORWARD_SIMPLE_SHADER.contains("arrayLength"),
            "FORWARD_SIMPLE_SHADER must have arrayLength bounds checks"
        );
        assert!(
            FORWARD_TRAINING_SHADER.contains("arrayLength"),
            "FORWARD_TRAINING_SHADER must have arrayLength bounds checks"
        );
        assert!(
            BACKWARD_WEIGHTS_SHADER.contains("arrayLength"),
            "BACKWARD_WEIGHTS_SHADER must have arrayLength bounds checks"
        );
        assert!(
            BACKWARD_INPUT_SHADER.contains("arrayLength"),
            "BACKWARD_INPUT_SHADER must have arrayLength bounds checks"
        );
        assert!(
            BACKWARD_BIAS_SHADER.contains("arrayLength"),
            "BACKWARD_BIAS_SHADER must have arrayLength bounds checks"
        );
        assert!(
            SOFTMAX_SHADER.contains("arrayLength"),
            "SOFTMAX_SHADER must have arrayLength bounds checks"
        );
        assert!(
            ADD_SHADER.contains("arrayLength"),
            "ADD_SHADER must have arrayLength bounds checks"
        );
        // RELU already had arrayLength
        assert!(
            RELU_SHADER.contains("arrayLength"),
            "RELU_SHADER must have arrayLength bounds checks"
        );
    }

    #[test]
    fn test_generate_forward_shader_order2() {
        let shader = generate_forward_shader(2).unwrap();
        assert!(shader.contains("order=2"));
        assert!(shader.contains("Quadratic B-spline"));
        assert!(shader.contains("ACTIVE_BASIS_COUNT: u32 = 3u"));
        assert!(shader.contains("arrayLength"));
    }

    #[test]
    fn test_generate_forward_shader_order3() {
        let shader = generate_forward_shader(3).unwrap();
        assert!(shader.contains("order=3"));
        assert!(shader.contains("Cubic B-spline"));
        assert!(shader.contains("ACTIVE_BASIS_COUNT: u32 = 4u"));
    }

    #[test]
    fn test_generate_forward_shader_order4() {
        let shader = generate_forward_shader(4).unwrap();
        assert!(shader.contains("order=4"));
        assert!(shader.contains("Quartic B-spline"));
        assert!(shader.contains("ACTIVE_BASIS_COUNT: u32 = 5u"));
    }

    #[test]
    fn test_generate_forward_shader_order5() {
        let shader = generate_forward_shader(5).unwrap();
        assert!(shader.contains("order=5"));
        assert!(shader.contains("Quintic B-spline"));
        assert!(shader.contains("ACTIVE_BASIS_COUNT: u32 = 6u"));
    }

    #[test]
    fn test_generate_forward_shader_invalid_order() {
        assert!(generate_forward_shader(1).is_err());
        assert!(generate_forward_shader(6).is_err());
        assert!(generate_forward_shader(0).is_err());
    }

    #[test]
    fn test_generate_backward_shaders() {
        for order in 2..=5 {
            let weights_shader = generate_backward_weights_shader(order).unwrap();
            let input_shader = generate_backward_input_shader(order).unwrap();
            let training_shader = generate_forward_training_shader(order).unwrap();

            assert!(weights_shader.contains("backward_main"));
            assert!(input_shader.contains("backward_input_main"));
            assert!(training_shader.contains("forward_training_main"));

            // All must have bounds checking
            assert!(weights_shader.contains("arrayLength"));
            assert!(input_shader.contains("arrayLength"));
            assert!(training_shader.contains("arrayLength"));
        }
    }

    #[test]
    fn test_optimizer_shaders_exist() {
        assert!(!ADAM_SHADER.is_empty());
        assert!(!SGD_SHADER.is_empty());
        assert!(!GRAD_CLIP_SHADER.is_empty());
    }

    #[test]
    fn test_optimizer_shaders_entry_points() {
        assert!(ADAM_SHADER.contains("adam_main"));
        assert!(SGD_SHADER.contains("sgd_main"));
        assert!(GRAD_CLIP_SHADER.contains("grad_clip_main"));
    }

    #[test]
    fn test_optimizer_shaders_have_bounds_checking() {
        assert!(
            ADAM_SHADER.contains("arrayLength"),
            "ADAM_SHADER must have bounds checks"
        );
        assert!(
            SGD_SHADER.contains("arrayLength"),
            "SGD_SHADER must have bounds checks"
        );
        assert!(
            GRAD_CLIP_SHADER.contains("arrayLength"),
            "GRAD_CLIP_SHADER must have bounds checks"
        );
    }
}
