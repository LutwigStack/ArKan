# ArKan Architecture

This document describes the internal architecture of ArKan, a high-performance
Kolmogorov-Arnold Network (KAN) library with CPU SIMD and GPU backends.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Public API                               │
│  KanNetwork, KanConfig, Workspace, TrainOptions, Adam          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│    CPU Backend    │                     │    GPU Backend    │
│  (SIMD-optimized) │                     │  (wgpu + WGSL)    │
└───────────────────┘                     └───────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│   KanLayer        │                     │   GpuKanLayer     │
│   B-spline eval   │                     │   Compute shaders │
└───────────────────┘                     └───────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│  AlignedBuffer    │                     │   GpuTensor       │
│  (64-byte align)  │                     │   (GPU buffers)   │
└───────────────────┘                     └───────────────────┘
```

## Core Components

### 1. Network Layer (`src/network.rs`)

`KanNetwork` is the main entry point. It manages:
- Network configuration (`KanConfig`)
- Layer stack (`Vec<KanLayer>`)
- Default training options
- Serialization/deserialization

Key methods:
- `forward_single` - Single sample inference (lowest latency)
- `forward_batch` - Batch inference (highest throughput)
- `train_step` - Complete training iteration with SGD
- `try_*` variants - Result-returning versions for error handling

### 2. Layer (`src/layer.rs`)

`KanLayer` implements a single KAN layer with learnable B-spline basis functions.

**Weight layout:**
```
weights[i * out * basis + j * basis + k]
  where:
    i = input dimension
    j = output dimension
    k = basis function index
```

**Forward pass:**
1. Normalize inputs to grid range
2. Find active grid span for each input
3. Evaluate B-spline basis (SIMD vectorized)
4. Weighted sum with learned weights

### 3. Spline Module (`src/spline.rs`)

Implements B-spline mathematics:

**Basis function evaluation (De Boor recursion):**
```
B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0

B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x)
           + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
```

SIMD-optimized for orders 2-7 (`MAX_SPLINE_ORDER`).

### 4. Buffer Management (`src/buffer.rs`)

**AlignedBuffer:**
- 64-byte aligned (CACHE_LINE) for AVX-512
- Zero-allocation resize within capacity
- `try_reserve` for fallible allocation
- Overflow protection with `MAX_BUFFER_ELEMENTS`

**Workspace:**
- Preallocates all buffers for forward/backward passes
- Enables zero-allocation inference
- Thread-local usage pattern

**WorkspaceGuard (RAII):**
```rust
let mut guard = WorkspaceGuard::new(&mut workspace);
// Use guard.buffers_mut() for computations
// Buffers returned automatically on drop
```

### 5. Error Handling (`src/error.rs`)

Unified error type `ArkanError` with variants:
- `ShapeMismatch` - Dimension errors
- `ConfigError` - Invalid configuration
- `CpuError` - CPU computation failures
- `GpuError` - GPU/wgpu errors
- `SerializationError` - Save/load failures
- `Overflow` - Integer overflow protection
- `BatchTooLarge` - Workspace capacity exceeded

## GPU Backend (`src/gpu/`)

### Module Structure

```
src/gpu/
├── mod.rs          # Public exports
├── backend.rs      # wgpu device/queue initialization
├── layer.rs        # GpuKanLayer (GPU layer wrapper)
├── network.rs      # GpuKanNetwork (GPU network)
├── pipeline.rs     # Compute pipeline management
├── shaders.rs      # WGSL shader sources
├── tensor.rs       # GpuTensor (GPU buffer wrapper)
├── uniforms.rs     # Shader uniform structs
├── workspace.rs    # GpuWorkspace
└── optimizer.rs    # GpuAdam, GpuSgd
```

### Shader Architecture

All WGSL shaders follow this pattern:

```wgsl
// Uniforms in binding group 0
@group(0) @binding(0) var<uniform> params: LayerParams;

// Storage buffers in binding group 1
@group(1) @binding(0) var<storage, read> input: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

// Bounds checking
let idx = global_id.x;
if idx >= arrayLength(&input) { return; }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }
```

### Shaders

| Shader | Purpose | Workgroup |
|--------|---------|-----------|
| `FORWARD_SHADER` | B-spline forward pass | 64×1×1 |
| `FORWARD_SIMPLE_SHADER` | Simple forward (no splines) | 64×1×1 |
| `FORWARD_TRAINING_SHADER` | Forward with history capture | 64×1×1 |
| `BACKWARD_WEIGHTS_SHADER` | Weight gradient accumulation | 64×1×1 |
| `BACKWARD_INPUT_SHADER` | Input gradient computation | 64×1×1 |
| `BACKWARD_BIAS_SHADER` | Bias gradient accumulation | 64×1×1 |
| `SOFTMAX_SHADER` | Softmax activation | 64×1×1 |
| `ADD_SHADER` | Element-wise addition | 64×1×1 |
| `ADAM_SHADER` | Adam optimizer step | 64×1×1 |
| `SGD_SHADER` | SGD with momentum step | 64×1×1 |
| `GRAD_CLIP_SHADER` | Global gradient clipping | 64×1×1 |

### Dynamic Shader Generation

For spline orders 2-5, shaders are generated at runtime:
```rust
let shader = generate_forward_shader(spline_order)?;
```

This inlines the B-spline basis computation for each order,
avoiding runtime branching and enabling compiler optimizations.

## Memory Layout

### CPU Tensors

Row-major layout: `[batch, dim]`

```
Input:  [batch_size × input_dim]
Output: [batch_size × output_dim]
Basis:  [batch_size × input_dim × basis_size]
```

### GPU Buffers

All GPU buffers use `f32` with 4-byte alignment.
Weights are packed into `vec4` for coalesced memory access:

```rust
// CPU: weights[i * out * basis + j * basis + k]
// GPU: weights_vec4[(i * out * basis + j * basis + k) / 4]
```

## Training Pipeline

```
1. forward_batch_training()
   └── Captures layer inputs and grid indices for backward

2. compute_masked_mse_loss_into()
   └── Computes loss and output gradients

3. backward() for each layer (reverse order)
   ├── Weight gradients: sum over batch
   ├── Bias gradients: sum over batch
   └── Input gradients: propagate to previous layer

4. Gradient clipping (optional)
   └── Global norm clipping across all parameters

5. Parameter update
   ├── Weight decay (decoupled)
   └── SGD step: w -= lr * grad
```

## Performance Optimizations

### CPU
- 64-byte aligned buffers for AVX-512
- SIMD-vectorized B-spline evaluation
- Zero-allocation inference with Workspace
- Cache-friendly memory layout

### GPU
- `vec4` weight packing for coalesced access
- Persistent compute pipelines
- Workgroup size 64 (GPU wavefront friendly)
- Bounds checking with `arrayLength()` (no OOB crashes)

## Serialization

Binary format with versioning:
```
[MAGIC: 5 bytes "ARKAN"]
[VERSION: 4 bytes u32]
[CONFIG: bincode-serialized KanConfig]
[LAYERS: bincode-serialized Vec<KanLayer>]
```

Version 1 is the initial versioned format (ArKan 0.3.0+).

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_SPLINE_ORDER` | 7 | Maximum B-spline order (CPU) |
| `MAX_GPU_SPLINE_ORDER` | 5 | Maximum B-spline order (GPU) |
| `MIN_GPU_SPLINE_ORDER` | 2 | Minimum B-spline order (GPU) |
| `CACHE_LINE` | 64 | Buffer alignment in bytes |
| `MAX_BUFFER_ELEMENTS` | 2^30 | Maximum buffer size (overflow protection) |

## Thread Safety

- `KanNetwork` is `Send + Sync` (immutable forward pass)
- `Workspace` should be thread-local (one per thread)
- GPU operations are single-threaded (wgpu limitation)

## Error Handling Strategy

Two API styles:
1. **Panic-on-error** (default): `forward_batch()`, `train_step()`
2. **Result-returning**: `try_forward_batch()`, `try_train_step()`

Use panic-style for performance-critical code with validated inputs.
Use Result-style when inputs may be malformed or for graceful error handling.
