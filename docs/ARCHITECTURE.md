# ArKan Architecture

This document describes the internal architecture of ArKan, a high-performance
Kolmogorov-Arnold Network (KAN) library with CPU SIMD and GPU backends.

Checked against the code at **0.4.0** (2026-07-26). Line references point at the
file they name; if one has drifted, the code wins. See
[Design constraints](#design-constraints) for the properties that will bite you
before the fast paths help you.

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
│ SIMD + opt. rayon │                     │  (wgpu 23 + WGSL) │
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

## Module Map

Every file under `src/`. Nothing here is a stub.

| Module | Lines | What lives there |
|---|---|---|
| `src/lib.rs` | 227 | Crate docs, module declarations, re-exports, `VERSION`, `MAGIC_BAKED` |
| `src/config.rs` | 939 | `KanConfig`, `KanConfigBuilder`, `LayerConfig`, `validate()`, order/grid limits |
| `src/network.rs` | 2276 | `KanNetwork`, `TrainOptions`, forward/backward/train-step family, serialization |
| `src/layer.rs` | 1461 | `KanLayer` — normalization, span lookup, basis evaluation, weight/bias/input gradients |
| `src/spline.rs` | 569 | Cox-de Boor basis and derivative, knot vectors, `find_span`, `SPAN_CLAMPED_FLAG` |
| `src/buffer.rs` | 1536 | `AlignedBuffer`, `Tensor`/`TensorView`, `Workspace`, `WorkspaceGuard` |
| `src/optimizer.rs` | 2527 | `Optimizer` trait, `Adam`, `SGD`, `LBFGS`, `StepLR`, `CosineAnnealingLR`, `SafetyConfig` |
| `src/loss.rs` | 1871 | Masked MSE/MAE/RMSE/Huber/BCE/cross-entropy, softmax, KAN and physics regularizers |
| `src/baked.rs` | 1519 | `BakedModel` — int8 fixed-point inference-only path (see below) |
| `src/error.rs` | 376 | `ArkanError`, `ArkanResult` |
| `src/gpu/` | 8598 | wgpu backend, behind `feature = "gpu"` (see [GPU Backend](#gpu-backend-srcgpu)) |

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

**Weight layout** (`KanLayer::weight_index`, src/layer.rs:344) — **output
outermost**, not input:
```
weights[(j * in_dim + i) * global_basis_size + k]
  where:
    j = output dimension
    i = input dimension
    k = basis function index
```

**Forward pass:**
1. `z = clamp((x - mean) / std, grid_min, grid_max)` — normalize, then clamp
2. Find the active grid span for each input (`find_span`)
3. Evaluate the B-spline basis over the local `order + 1` window (SIMD vectorized)
4. Weighted sum with the learned weights, plus bias

**Normalization is per-layer, and only layer 0 gets real statistics.**
`KanNetwork::new` calls `set_normalization` on layer 0 only. Every hidden layer
is constructed with identity normalization (`mean = 0`, `std = 1`) and there is
no running-statistics update anywhere in the crate. So a hidden layer's `z` *is*
the previous layer's raw activation, clamped to the **shared** `grid_range`.
Nothing bounds a KAN layer's output to its own grid range — see
[Design constraints](#design-constraints).

**Saturation flag (`SPAN_CLAMPED_FLAG`).** Step 1 is a clamp, so `dz/dx` is
`1/std` inside the range and exactly `0` outside it. Backward has to know which
one happened, and it cannot recover that by comparing `z` to the endpoint —
that would confuse a saturated input with one that legitimately landed on the
boundary. So the forward pass records it:

```rust
// src/layer.rs:511 — forward_batch, while storing the span index
let clamped = if z == unclamped { 0 } else { SPAN_CLAMPED_FLAG };
grid_indices[idx] = span as u32 | clamped;
```

`SPAN_CLAMPED_FLAG` is `0x8000_0000` (src/spline.rs:102) — the **high bit of
the stored span index** in `Workspace::layers_grid_indices`. Backward reads it
and forces `dz/dx = 0` (src/layer.rs:914 and :1090); consumers of the span
itself must mask with `SPAN_INDEX_MASK`. Costs no extra buffer and no signature
change. The GPU shaders carry the identical scheme
(src/gpu/shaders.rs:914, :1866).

`KanLayer::forward_single` does **not** record the flag — it is the inference
path and never feeds a backward pass (`// ponytail:` at src/layer.rs:413).

### 3. Spline Module (`src/spline.rs`)

Implements B-spline mathematics:

**Basis function evaluation (De Boor recursion):**
```
B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0

B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x)
           + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
```

SIMD-optimized for orders 2–7 (`MAX_SPLINE_ORDER`). GPU shaders support orders
2–5 (`MIN_GPU_SPLINE_ORDER`–`MAX_GPU_SPLINE_ORDER`). `BakedModel` also supports
2–5 only, and **panics** on anything else — `KanConfig::validate()` accepts up
to 7, so that gap is a documented panic, not a `Result`.

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

### 6. Baked Inference (`src/baked.rs`)

`BakedModel` is an inference-only, fixed-point rewrite of a trained
`KanNetwork`. No f32 appears in the hot path between the entry and exit
conversions. It is available in a **default build** — there is no feature flag.

**Representation**

| Piece | Type | Scale |
|---|---|---|
| Weights | `i8`, per output channel | `s_w[j] = 127 / max|w[j,*,*]|` |
| B-spline basis | `u16` | Q0.15, sums to exactly 32768 |
| MAC accumulator | `i64` | — |
| Requantization | `i128` product, per-channel `M0[j] >> shift[j]` | — |
| Inter-layer activations | `i32` | `s_act = 2^28 / p99.9` of the calibration set |
| Inter-layer normalization | `i32` `A_FIXED`/`B_FIXED` | Q32, `>> 16` in the hot path |

**Data flow** (`BakedModel::forward`)

```
f32 input
  └─> entry: q_z = z * 2^16, clamped to [q_rmin, q_rmax]      (Q15.16)
        └─> extract_span_t: span (clamped to [0, G-1]) and t_q16
              └─> eval_basis_fixed(order, t_q16)              -> u16 Q0.15
                    └─> acc: i64 = Σ_i Σ_k  w_i8 · basis_u16
                          └─> requant: ((acc·M0[j]) + round) >> shift[j]
                                └─> clamp(±ACT_CLAMP = 2^28)  -> i32 activation
                                      └─> inter-layer: q_z = (q·A_FIXED + B_FIXED) >> 16,
                                          clamped to the NEXT layer's [q_rmin, q_rmax]
  └─> exit: output[j] = act[j] / s_act_last                   -> f32
```

Calibration (`from_network(net, Some(&calib))`) sets `s_act` from the 99.9th
percentile of observed activation magnitudes per layer. Without it the scale
falls back to a coarse heuristic and accuracy degrades badly.

**Two consequences of this design that you must know about** (both unfixed):

1. `ACT_CLAMP` is applied to the **output** layer's activations as well, and the
   exit scale is `2^28 / p99.9`. So `|output[j]| <= p99.9` of the calibration
   set — a hard ceiling. Nothing can produce a larger magnitude.
2. `A_FIXED[i] = round(2^32 / (s_act_prev · std_i))` lands on small integers
   (measured 5–10 on the parity fixtures, i.e. about 3 bits), so rounding it
   costs a systematic 2.6–6.7% scale error at every inter-layer hop.

Together these are the likely origin of the per-output tail error reported in
[BENCHMARKS.md](BENCHMARKS.md#baked-int8-inference). Baked is for ranking and
argmax, not for per-output absolute accuracy.

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
// CPU: weights[(j * in_dim + i) * basis + k]
// GPU: weights_vec4[((j * in_dim + i) * basis + k) / 4]
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
       └── dz/dx = 0 where SPAN_CLAMPED_FLAG is set, else 1/std

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
- SIMD-vectorized B-spline evaluation (`wide` crate — **always on**, not a feature)
- Rayon parallelism over batch samples (`parallel` feature; absent without it)
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

Both formats require `feature = "serde"`. Two independent formats, two magics.

**`KanNetwork::to_bytes` / `from_bytes`** (`SERIALIZATION_MAGIC`, src/network.rs:66):
```
[MAGIC: 5 bytes "ARKAN"]
[VERSION: 4 bytes u32]
[CONFIG: bincode-serialized KanConfig]
[LAYERS: bincode-serialized Vec<KanLayer>]
```
Version 1 is the initial versioned format (ArKan 0.3.0+). `KanLayer` has a
custom `Deserialize` that recomputes the knot vector after load.

**`BakedModel::to_bytes` / `from_bytes`** (`MAGIC_BAKED`, src/lib.rs:200):
```
[MAGIC: 12 bytes "KAN_BAKED_v1"]
[VERSION: 4 bytes u32]
[BODY: bincode-serialized BakedModel]
```
`from_bytes` validates magic and version **before** deserializing and returns a
descriptive `Err` on mismatch or truncation, not a panic. `FORMAT_VERSION` is 1
(src/baked.rs:864). The two formats are not interchangeable.

## Feature Flags

Three, and each one gates real code. `grep -rn 'feature = "X"' src/` is the
check that keeps this table honest.

| Flag | Description | Default | Extra deps |
|------|-------------|---------|------------|
| `parallel` | `KanLayer::backward_parallel`, `KanNetwork::forward_batch_parallel`, and the parallel branch of `train_step`'s backward pass above `multithreading_threshold`. Without it those two methods **do not exist** and `multithreading_threshold` is ignored. Gradients are identical either way. | Off | `rayon` |
| `serde` | `to_bytes()` / `from_bytes()` for `KanNetwork` and `BakedModel` | Off | `serde`, `bincode` |
| `gpu` | GPU backend via wgpu 23 (Vulkan/DX12/Metal/WebGPU) | Off | `wgpu`, `bytemuck`, `pollster`, `log` |

SIMD is **not** a feature flag — B-spline vectorization through `wide` is
unconditional. A default build pulls `wide`, `rand` and `thiserror`, nothing
else. `simd`, `nightly` and `quantization` were removed in 0.4.0; all three
gated nothing.

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

## Design constraints

Real properties of the current design, not bugs with a ticket. Read these before
choosing a config.

### `grid_range` is shared by every layer, but only layer 0 is normalized

`KanNetwork::new` sets `input_mean` / `input_std` on layer 0 only. Hidden layers
get identity normalization and nothing updates it during training. A hidden
layer therefore sees the previous layer's raw activation, clamped to the same
`grid_range` you picked for the inputs — and nothing in a KAN layer bounds its
output to its own grid range.

Choosing `grid_range` from the *inputs* silently kills the hidden layers.
Measured on the shape `examples/game2048` shipped with (256 → [64, 32] → 4,
one-hot inputs, `tests/hidden_layer_saturation.rs`):

| `grid_range` | layer 0 | layer 1 | layer 2 |
|---|---|---|---|
| `(0.0, 1.0)` | 0% | **43.6%** | **48.9%** |
| `(-1.0, 1.0)` | 0% | 0% | 0% |
| `(-3.0, 3.0)` | 0% | 0% | 0% |

A saturated input has zero derivative, so those features emit a constant and —
correctly, since the clamp/gradient fix — a zero gradient. They stop learning.
Prefer a symmetric range sized for the *activations*.

### Saturation is silent

There is no `out_of_grid_fraction`, no drift warning, no configurable
extrapolation and no grid recalibration. `SPAN_CLAMPED_FLAG` records saturation
per element for the backward pass, but nothing surfaces it to the caller.
Distribution drift shows up as unexplained accuracy loss, not as a diagnostic.

### `BakedModel` accuracy has a per-output tail

Aggregate NRMSE is 0.6–2.7%; the worst-case error on decision-relevant outputs
(≥1σ) is 34–54% on a 2-hidden net. Two identified causes, both unfixed, are
described in [Baked Inference](#6-baked-inference-srcbakedrs) above. Use baked
for ranking / argmax, not for per-output absolute accuracy.

### `BakedModel` is slower than f32 at batch=1

1.4–2.1× on the shipped bench configs. Its win today is size (2.2–3.0×). A
design that reverses this — a weight-layout change to output-innermost plus
hoisting the basis evaluation out of the output loop, **not** SIMD — has been
prototyped and measured, and is **not implemented**.

## Thread Safety

- `KanNetwork` is `Send + Sync` (immutable forward pass)
- `Workspace` should be thread-local (one per thread); with `feature = "parallel"` CPU training uses rayon internally, without it everything is single-threaded
- `Adam`, `SGD`, `LBFGS` are `Send + Sync` (manual impls; use `AlignedBuffer` internally)
- `GpuNetwork` / `WgpuBackend` — GPU dispatch uses a single wgpu queue; do not share across threads without external synchronization

## Error Handling Strategy

Two API styles:
1. **Panic-on-error** (default): `forward_batch()`, `train_step()`
2. **Result-returning**: `try_forward_batch()`, `try_train_step()`

Use panic-style for performance-critical code with validated inputs.
Use Result-style when inputs may be malformed or for graceful error handling.
