# ArKan Architecture

This document describes the internal architecture of ArKan, a high-performance
Kolmogorov-Arnold Network (KAN) library with CPU SIMD and GPU backends.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Public API                               │
│  KanNetwork, KanConfig, Workspace, TrainOptions,               │
│  Adam/SGD/LBFGS + LR schedulers, GpuNetwork (gpu feature)     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│    CPU Backend    │                     │    GPU Backend    │
│  (SIMD + rayon)   │                     │  (wgpu 23 + WGSL) │
└───────────────────┘                     └───────────────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│   KanLayer        │                     │   GpuLayer        │
│   B-spline eval   │                     │   Compute shaders │
│   orders 2–7      │                     │   orders 2–5      │
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
- `train_step` - Complete training iteration (SGD/Adam/LBFGS)
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

SIMD-optimized for orders 2–7 (`MAX_SPLINE_ORDER`). GPU shaders support orders 2–5 (`MIN_GPU_SPLINE_ORDER`–`MAX_GPU_SPLINE_ORDER`).

### 4. Buffer Management (`src/buffer.rs`)

**AlignedBuffer:**
- 64-byte aligned (CACHE_LINE) for AVX-512
- Zero-allocation resize within capacity
- `try_reserve` for fallible allocation
- Overflow protection with `MAX_BUFFER_ELEMENTS`

**Workspace:**
- Preallocates all buffers for forward/backward passes
- Enables zero-allocation inference (reuse across calls)
- Thread-local usage pattern (one `Workspace` per thread)

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
├── mod.rs          # Public exports + utility fns (align_to, pad_to_vec4, etc.)
├── backend.rs      # WgpuBackend — device/queue/adapter init, VramLimit
├── layer.rs        # GpuLayer — bind groups, weight upload, per-layer pipelines
├── network.rs      # GpuNetwork — full GPU forward/backward/training + GpuForwardHandle
├── pipeline.rs     # PipelineCache — compiled compute pipelines, workgroup_count
├── shaders.rs      # WGSL shader sources + dynamic shader generation
├── tensor.rs       # GpuTensor / GpuTensorView — GPU buffer upload/download
├── uniforms.rs     # LayerUniforms (std140) — shader uniform structs
├── workspace.rs    # GpuWorkspace — resizable workspace buffers
└── optimizer.rs    # GpuAdam / GpuSgd — on-device optimizer steps
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
| `generate_forward_shader(order)` | B-spline forward pass (order 2–5) | 64×1×1 |
| `generate_forward_training_shader(order)` | Forward with history capture | 64×1×1 |
| `generate_backward_weights_shader(order)` | Weight gradient accumulation | 64×1×1 |
| `generate_backward_input_shader(order)` | Input gradient computation | 64×1×1 |
| `ADAM_SHADER` | On-device Adam optimizer step | 64×1×1 |
| `SGD_SHADER` | On-device SGD with momentum step | 64×1×1 |
| `GRAD_CLIP_SHADER` | Global gradient clipping | 64×1×1 |

### Dynamic Shader Generation

All primary compute shaders are generated at runtime for each spline order (2–5):
```rust
let fwd = generate_forward_shader(spline_order)?;
let fwd_train = generate_forward_training_shader(spline_order)?;
let bwd_w = generate_backward_weights_shader(spline_order)?;
let bwd_i = generate_backward_input_shader(spline_order)?;
```

This inlines the B-spline basis computation for each order,
avoiding runtime branching and enabling compiler optimizations.
Compiled pipelines are cached in `PipelineCache`.

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

2. Loss computation (masked MSE/BCE/cross-entropy/etc.)
   └── Computes loss and output gradients

3. backward() for each layer (reverse order)
   ├── Weight gradients: sum over batch
   ├── Bias gradients: sum over batch
   └── Input gradients: propagate to previous layer

4. Gradient clipping (optional)
   └── Global norm clipping across all parameters

5. Parameter update via Optimizer trait
   ├── SGD: w -= lr * (μ*v + g)  [optional Nesterov, momentum, weight decay]
   ├── Adam: bias-corrected moment estimates + decoupled weight decay (AdamW)
   └── LBFGS: two-loop recursion + Strong-Wolfe line search

6. Safety layer (per AdamConfig/SGDConfig/LBFGSConfig.safety: SafetyConfig)
   ├── NaN detection (fail_on_nan / skip_step_on_nan)
   └── AMP gradient scaling (grad_scaling_factor)
```

### LR Schedulers

- `StepLR` — decay by factor every N epochs
- `CosineAnnealingLR` — cosine schedule between `lr_max` and `lr_min`

## Performance Optimizations

### CPU
- 64-byte aligned buffers for AVX-512 (`AlignedBuffer`)
- SIMD-vectorized B-spline evaluation (`wide` crate, `simd` feature)
- Rayon parallelism over batch samples (`parallel` feature)
- Zero-allocation inference with `Workspace` (reuse across calls)
- Cache-friendly memory layout

### GPU
- `vec4` weight packing for coalesced access (`pad_to_vec4`)
- Persistent compute pipelines via `PipelineCache`
- Workgroup size 64 (GPU wavefront friendly)
- Bounds checking with `arrayLength()` (no OOB crashes)
- On-device `GpuAdam` / `GpuSgd` — optimizer step runs entirely on GPU, no CPU round-trip
- Split bind-group strategy: Group 0 (layer weights/bias/uniforms, static) + Group 1 (workspace IO, dynamic)
- `GpuForwardHandle` for async dispatch without blocking the CPU

## Serialization

Binary format with versioning:
```
[MAGIC: 5 bytes "ARKAN"]
[VERSION: 4 bytes u32]
[CONFIG: bincode-serialized KanConfig]
[LAYERS: bincode-serialized Vec<KanLayer>]
```

Version 1 is the initial versioned format (ArKan 0.3.0+).

## Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `default` | No extra features | On |
| `simd` | Explicit SIMD via `wide` crate | Off |
| `parallel` | Rayon batch parallelism | Off |
| `serde` | Serialization via `serde` + `bincode` | Off |
| `quantization` | Half-precision (f16) support | Off |
| `nightly` | Nightly-only optimizations | Off |
| `gpu` | GPU backend via wgpu 23 (Vulkan/DX12/Metal) | Off |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_SPLINE_ORDER` | 7 | Maximum B-spline order (CPU) |
| `MAX_GPU_SPLINE_ORDER` | 5 | Maximum B-spline order (GPU dynamic shaders) |
| `MIN_GPU_SPLINE_ORDER` | 2 | Minimum B-spline order (GPU dynamic shaders) |
| `CACHE_LINE` | 64 | Buffer alignment in bytes (AVX-512) |
| `MAX_BUFFER_ELEMENTS` | 2^30 | Maximum buffer size (overflow protection) |
| `WORKGROUP_SIZE` | 64 | GPU compute workgroup size |
| `GPU_BUFFER_ALIGNMENT` | 256 | GPU buffer alignment (uniform offsets) |
| `MAX_VRAM_ALLOC` | 2 GB | Default per-buffer VRAM limit |

## Deprecated / Stub Modules

- `src/baked.rs` — `BakedModel` is deprecated since v0.2.0 and fully unimplemented (`forward()` always panics). Full quantization planned for v0.4.0. Do not use.

## Thread Safety

- `KanNetwork` is `Send + Sync` (immutable forward pass)
- `Workspace` should be thread-local (one per thread); CPU training uses rayon internally
- `Adam`, `SGD`, `LBFGS` are `Send + Sync` (manual impls; use `AlignedBuffer` internally)
- `GpuNetwork` / `WgpuBackend` — GPU dispatch uses a single wgpu queue; do not share across threads without external synchronization

## Error Handling Strategy

Two API styles:
1. **Panic-on-error** (default): `forward_batch()`, `train_step()`
2. **Result-returning**: `try_forward_batch()`, `try_train_step()`

Use panic-style for performance-critical code with validated inputs.
Use Result-style when inputs may be malformed or for graceful error handling.
