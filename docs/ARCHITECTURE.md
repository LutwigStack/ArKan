# ArKan Architecture

This document describes the internal architecture of ArKan, a high-performance
Kolmogorov-Arnold Network (KAN) library with CPU SIMD and GPU backends.

The implementation remains a single `arkan` crate. Legacy public imports and
valid saved-model formats remain compatible across the internal ownership moves.
See [Design constraints](#design-constraints) for numerical limits.

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

Each responsibility owns executable code, with one definition of every public type.

| Owner | Responsibility |
|---|---|
| `src/model/` | KanNetwork construction/clone, configuration, immutable checked topology, normalization specifications and parameter-only borrows |
| `src/math/spline.rs` | Knots, span packing, scalar/SIMD spline basis and derivatives |
| `src/memory/` | Aligned allocation, checked extents, Tensor/TensorView and aligned-buffer serde |
| `src/cpu/` | Layer kernels, forward orchestration, workspace/history storage and bounded parallel scratch |
| `src/training/` | TrainOptions, borrowed ForwardPass/Gradients, one reverse-layer loop, trainers and shared f64 global clipping |
| `src/optimizer/` | Optimizer trait/safety policy, Adam/SGD, transactional LBFGS/search, schedulers |
| `src/loss/` | Regression, classification and regularization formulas, including reusable MSE fill |
| `src/baked/` | Validated calibration/quantization, fixed-point arithmetic and workspace inference |
| `src/gpu/` | Device/tensor/pipeline/shaders, GPU execution/workspace and native optimizers |
| `src/format/` | Private stable DTOs, borrowed serializers, checked import and versioned envelopes |
| `src/error.rs` | Shared public errors, including existing feature-gated GPU variants |
| `src/lib.rs` | Public module declarations and convenience exports |

`config`, `network`, `layer`, `spline`, and `buffer` remain compatibility facades.
Their reexports preserve old paths and type identity; no conversion wrappers or
second copy of a public type is introduced. `gpu`, `optimizer`, `loss` and `baked`
keep their existing public module paths.

## Core Components

### 1. Model and checked layout (`src/model/`)

`KanNetwork` owns configuration, the layer stack, default training options and a
private immutable layout snapshot. The snapshot includes dimensions, parameter
extents, spline geometry/range and SIMD alignment. Checked execution, baking,
GPU conversion and synchronization validate legacy public fields against it.
Normalization values may still be updated through existing setters or public
fields; their shape and validity are checked, and conversions consume each
layer's actual statistics.

`try_parameters_mut` returns fixed-length weight/bias slices. Adam, SGD, direct
training and LBFGS restoration consume this boundary; it cannot resize layers,
change geometry or mutate normalization. Legacy public structural fields remain
available until a future breaking release, so incompatible edits return errors
at checked boundaries. Raw layer calls retain their documented caller-managed
buffer contract.

### 2. CPU layer (`src/cpu/layer.rs`)

`KanLayer` implements a single KAN layer with learnable B-spline basis functions.

**Weight layout** (`KanLayer::weight_index`, src/cpu/layer.rs) — **output
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

**Normalization is explicit at construction.** The network builder supplies
configured statistics to layer zero and identity statistics to every hidden
layer, including hidden layers whose width equals the input width. Standalone
`KanLayer::new` retains its matching-width behavior; `try_new_at` selects by layer
position. Setters can change each layer's statistics, and no running-statistics
estimation occurs during training. Hidden identity normalization therefore leaves
the previous layer's activation unchanged before the shared grid-range clamp.

**Saturation flag (`SPAN_CLAMPED_FLAG`).** Step 1 is a clamp, so `dz/dx` is
`1/std` inside the range and exactly `0` outside it. Backward has to know which
one happened, and it cannot recover that by comparing `z` to the endpoint —
that would confuse a saturated input with one that legitimately landed on the
boundary. So the forward pass records it:

```rust
// src/cpu/layer.rs — forward_batch, while storing the span index
let clamped = if z == unclamped { 0 } else { SPAN_CLAMPED_FLAG };
grid_indices[idx] = span as u32 | clamped;
```

`SPAN_CLAMPED_FLAG` is `0x8000_0000` (src/math/spline.rs) — the **high bit of
the stored span index** in `Workspace::layers_grid_indices`. Backward reads it
and forces `dz/dx = 0` (src/cpu/layer.rs); consumers of the span
itself must mask with `SPAN_INDEX_MASK`. Costs no extra buffer and no signature
change. The GPU shaders carry the identical scheme
(src/gpu/shaders.rs).

`KanLayer::forward_single` does **not** record the flag — it is the inference
path and never feeds a backward pass (`// ponytail:` at src/cpu/layer.rs).

### 3. Spline mathematics (`src/math/spline.rs`)

Implements B-spline mathematics:

**Basis function evaluation (De Boor recursion):**
```
B_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0

B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x)
           + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
```

SIMD-optimized for orders 2–7 (`MAX_SPLINE_ORDER`). GPU shaders support orders
2–5 (`MIN_GPU_SPLINE_ORDER`–`MAX_GPU_SPLINE_ORDER`). `BakedModel` also supports
2–5 only. `BakedModel::try_from_network` returns an error for unsupported orders;
its legacy `from_network` wrapper panics on conversion errors.

### 4. Memory and CPU workspace (`src/memory/`, `src/cpu/workspace.rs`)

**AlignedBuffer:**
- 64-byte aligned (CACHE_LINE)
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

### 6. Baked inference (`src/baked/`)

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
| Inter-layer normalization | `i32` `A_FIXED`/`B_FIXED` | `A_FIXED` at Q(16+`norm_shift`) with a per-layer shift; `B_FIXED` at Q15.16 |

**Data flow** (`BakedModel::forward`)

```
f32 input
  └─> entry: q_z = z * 2^16, clamped to [q_rmin, q_rmax]      (Q15.16)
        └─> extract_span_t: span (clamped to [0, G-1]) and t_q16
              └─> eval_basis_fixed(order, t_q16)              -> u16 Q0.15
                    └─> acc: i64 = Σ_i Σ_k  w_i8 · basis_u16
                          └─> requant: ((acc·M0[j]) + round) >> shift[j]
                                └─> clamp to i32 (no p99.9 clip)  -> i32 activation
                                      └─> inter-layer: q_z = ((q·A_FIXED + round) >> SH) + B_FIXED,
                                          clamped to the NEXT layer's [q_rmin, q_rmax]
  └─> exit: output[j] = act[j] / s_act_last                   -> f32
```

Calibration (`try_from_network(net, Some(&calib))`) sets `s_act` from the 99.9th
percentile of observed activation magnitudes per layer. Missing or empty
calibration uses the heuristic and marks the model `uncalibrated`; the percentile
sets a scale, not an output ceiling. Conversion preflights the complete CPU
layout, finite calibration, supported order and fixed-point bounds.

`BakedWorkspace` owns integer activation, span and basis scratch. Create it once
and call `forward_with_workspace` for allocation-free repeated inference. The
legacy `forward` convenience method creates scratch on each call. Public baked
fields retain the validated bake/import invariant required by low-level inference.

**Two things about this pipeline that used to be wrong and are worth knowing**
(both fixed; see [BENCHMARKS.md](BENCHMARKS.md#baked-int8-inference)):

1. Activations were clipped at `ACT_TARGET = 2^28` on **every** layer. Since the
   exit scale is `2^28 / p99.9`, that made `|output[j]| <= p99.9` of the
   calibration set a hard ceiling; on hidden layers it was a saturation the f32
   path does not have, on top of the grid-range clamp that both paths share. The
   percentile still sets `s_act`; only the clip is gone.
2. `A_FIXED[i] = round(2^32 / (s_act_prev · std_i))` landed on the integers 7–9
   (about 3 bits) with a hardcoded `>> 16`, and rounded to **0** — collapsing
   the next layer's inputs to a constant — whenever a layer's `p99.9` fell below
   `std/32`. The per-layer `norm_shift` puts `A_FIXED` in `[2^29, 2^30)` and
   makes 0 unreachable.

The remaining ≥1σ worst case is 0.8–7.9% and is gated by
`tests/baked_parity.rs`.

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
| `GRAD_CLIP_SHADER` | Available shader helper; high-level clipping currently downloads gradients | 64×1×1 |

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

## Training pipeline

`try_forward_for_backward` returns a `ForwardPass` only after a successful checked
training forward. It holds an immutable model/layout borrow and an exclusive
Workspace borrow, so parameters, normalization and saved inputs/spans cannot
change before backward. Callers compute any loss derivative from the output,
then consume the pass with `backward`. Returned `Gradients` borrow only workspace
storage, releasing the model for an optimizer update.

MSE trainers and custom losses share one reverse-layer loop in `training`.
Shape/history checks and all fallible scratch preparation happen before zeroing
gradients or taking workspace buffers. Empty batches produce zero parameter
gradients and skip updates. Reservation alone does not establish a valid pass.
Legacy public workspace fields and raw backward methods remain available with
caller-managed validity; the borrowed interface is the safe high-level path.

`masked_mse_into` validates lengths before writes, clears inactive mask entries,
and fills caller-owned gradients. The allocating MSE API delegates to it; its
reciprocal multiplication matches training and can differ by a final rounding bit
from the former allocating division. Logits and probability-domain classification
APIs are named separately; legacy fused gradient contracts remain available.

| Training path | Clipping and decay ownership |
|---|---|
| CPU direct SGD | Raw backward, optional f64 global clip, TrainOptions weight-only decoupled decay, SGD update |
| CPU standalone optimizer | Raw gradients and threshold go to optimizer; AMP unscale, finite checks, clip, update/configured decay |
| Hybrid GPU + CPU optimizer | Optimizer owns unscale/clip; explicit TrainOptions decay remains additional to optimizer-configured decay |
| Native GPU with options | Optional downloaded-gradient f64 clipping; native optimizer owns configured decay; TrainOptions decay is not applied again |

Disabled clipping skips the global norm pass and native GPU clipping downloads.
The GPU options entry points validate the same finite positive clipping threshold.
Parallel CPU backward keeps bounded reusable scratch and deterministic chunk
reduction; default warmed allocation guarantees cover the existing no-clipping,
no-AMP optimizer configuration. Active optimizer preprocessing may allocate.

LBFGS evaluates a closure at trial parameters, restores parameters/history on
errors, and shrinks rejected numerical trials within its bounded line search.
Its example uses the same borrowed backward primitive without a zero-rate update.

### LR Schedulers

- `StepLR` — decay by factor every N epochs
- `CosineAnnealingLR` — cosine schedule between `lr_max` and `lr_min`

## Performance Optimizations

### CPU
- 64-byte cache-line aligned buffers (`AlignedBuffer`)
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

Both formats require `serde`. Private owned DTOs decode saved values, and
borrowed serialization records preserve field names/order without cloning the
runtime model. Runtime caches are reconstructed and checked at import.

| Format | Envelope | Stable body |
|---|---|---|
| Network V1 | `ARKAN` (5 bytes), little-endian u32 version 1 | config, layers, advisory layer_dims/parameter sizes, default TrainOptions |
| Legacy network | Raw bincode body | Same network field order; accepted by legacy import |
| Baked V2 | `KAN_BAKED_v1` (12 bytes), little-endian u32 version 2 | config, fixed-point layers, uncalibrated |

Layer records retain weights/bias/statistics/geometry/SIMD width; knot caches are
rebuilt. Network advisory caches are rebuilt rather than trusted. Baked import
checks shape, normalization and integer bounds; version 1 is rejected because
it predates the current normalization representation. Direct JSON field names
are preserved. Pre-refactor golden bytes and JSON live in
`tests/fixtures/formats/` and are checked without regenerating them.

## Feature Flags

Three feature flags gate their implementation and dependencies.

| Flag | Description | Default | Extra deps |
|------|-------------|---------|------------|
| `parallel` | `KanLayer::backward_parallel`, `KanNetwork::forward_batch_parallel`, and the parallel branch of `train_step`'s backward pass above `multithreading_threshold`. Without it those two methods **do not exist** and `multithreading_threshold` is ignored. Reduction is deterministic across thread counts; sequential/parallel numerical parity is tolerance-tested. | Off | `rayon` |
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
| `CACHE_LINE` | 64 | Buffer alignment in bytes (cache line) |
| `MAX_BUFFER_ELEMENTS` | 2^30 | Maximum buffer size (overflow protection) |
| `WORKGROUP_SIZE` | 64 | GPU compute workgroup size |
| `GPU_BUFFER_ALIGNMENT` | 256 | GPU buffer alignment (uniform offsets) |
| `MAX_VRAM_ALLOC` | 2 GB | Default per-buffer VRAM limit |

## Design constraints

Real properties of the current design, not bugs with a ticket. Read these before
choosing a config.

### Hidden normalization starts as identity within a shared grid range

`KanNetwork::new` assigns `input_mean` / `input_std` to layer 0 only. Hidden layers
start with identity normalization. Per-layer setters are supported, but training
does not estimate statistics. With identity normalization, a hidden
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

### The clamp is an absorbing state, and training can walk into it

The table above is the *initialization-time* version: a bad `grid_range` starves
hidden layers from step 0. The worse version arrives during training from a
config that looked fine at step 0.

Saturation is not static. Weights grow, activations grow with them, and nothing
bounds a layer's output to the next layer's `grid_range`. Once **every**
`(sample, feature)` pair at a boundary is clamped, `grad_input` upstream of it is
exactly zero — not small — so every layer before it is frozen. Only the output
layer can still move, and all it can learn is a constant.

Measured on `[2, 16, 16, 1]`, grid 8, order 3, `grid_range = (-1.5, 1.5)`, Adam
`lr = 0.1`, batch 64, `sin(pi a) cos(pi b)` (`tests/training_dynamics.rs`):

| epoch | test MSE | saturation per layer |
|---|---|---|
| 25 | 0.0139 | `[0%, 26%, 96%]` — healthy, still learning |
| 35 | 0.2800 | `[0%, 86%, 100%]` — boundary fully clamped |
| 40+ | 0.2616 frozen | `[0%, 86%, 100%]` — every prediction identical |

Three things that do **not** help, all measured:

- **Lowering the learning rate.** 400 epochs at 1e-3 / 1e-4 / 1e-2 from the dead
  state give 0.2548 / 0.2548 / 0.2634; the mean-predictor baseline is 0.2548.
  There is no gradient to scale.
- **Widening `grid_range`.** `[2, 16×6, 1]` at Adam `lr = 0.01` with `(-6, 6)`
  never learned at all — 0.2575 at epoch 0, 0.2504 at epoch 119.
- **Gradient clipping.** Every run in the 216-run frequency grid used
  `max_grad_norm = 1.0`; 18.5% still ended dead.

Adam is the risky optimizer here, not the safe one: its normalized step drives
activations out of range regardless of gradient magnitude. Same net and task,
200 epochs, as a ratio against the mean baseline / peak saturation — Adam 0.01 →
7512× / 61%, Adam 0.07 → 138× / 95%, Adam 0.1 → 0.9× / 100%; SGD 0.1 → 5917× /
0%, SGD 0.5 → 2.5× / 28%, SGD 2.0 → 0.0× / 100%. SGD degrades gracefully, Adam
falls off a cliff. Depth sharpens it: 1 hidden layer never died in 54 runs, 4
hidden layers died at Adam 0.03, an entirely ordinary learning rate.

**The one thing that does work** is decoupled weight decay
(`AdamConfig::weight_decay`, applied as `w *= 1 - lr * decay` *before* the
gradient update). It is the only knob that moves a parameter whose gradient is
zero. What matters is `lr * decay * steps` being large enough to shrink the
weights back inside the grid, not `decay` crossing a threshold. On the collapsed
network above, 200 epochs at `lr = 1e-3`: `decay = 0.1` leaves it at 0.2571 and
100% saturated, `decay = 0.5` reaches 0.000026 and 29.7%. Note this has to go on
the optimizer config — `TrainOptions::weight_decay` is ignored by
`train_step_with_optimizer` by design, to avoid double-counting.

### Saturation is reported, but only if you ask

[`KanNetwork::clamped_fraction`] returns the clamped fraction per layer from the
`SPAN_CLAMPED_FLAG` bits a training forward pass already records, so watching for
the collapse above costs one pass over the span indices and no extra forward.

Nothing calls it for you. There is still no drift warning, no configurable
extrapolation and no grid recalibration, and `forward_batch` (the inference path)
does not record the flag at all — so distribution drift *in deployment* still
shows up as unexplained accuracy loss rather than as a diagnostic.

### `BakedModel` accuracy still has a per-output tail

Aggregate NRMSE is 0.17–0.59%; the worst-case error on decision-relevant outputs
(≥1σ) is 0.8–7.9% on a 2-hidden net, down from 34–54% before the three
fixed-point fixes described in
[Baked Inference](#6-baked-inference-srcbaked) above. `tests/baked_parity.rs`
gates the ≥1σ figure at 15%. What is left is absolute-error outliers from int8
weight quantization, not a systematic scale error.

### Historical baked measurements

Published timing and size figures in [BENCHMARKS.md](BENCHMARKS.md) describe the
recorded benchmark revision and environment. The current implementation uses
reusable workspace and computes basis values once per input. Rerun the corrected
benchmarks before comparing current latency; this architecture refactor makes
no new speedup claim.

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

## GPU freshness and later crate extraction

GPU conversion captures the CPU's checked layout and actual per-layer statistics.
CPU→GPU synchronization refreshes weights, bias and normalization; GPU→CPU checks
layout/storage and completes all snapshot readbacks before changing the destination.
Native GPU optimizer updates are not automatically reflected in the CPU model. Call the
explicit sync method before CPU inference, serialization or baking.

GPU execution checks legacy model metadata, tensor ranks and actual storage
extents. Lazy workspaces allocate before use; valid growth/shrink and hidden-width
reuse recreate buffers with cache invalidation. Backward uses the saved training
batch even after another workspace performs inference. Arbitrary replacement of
raw public GPU buffer/bind-group handles remains caller-controlled. Software
llvmpipe tests establish correctness evidence, not physical GPU performance.

A later `arkan-core` / `arkan-wgpu` split can build on these ownership boundaries.
Before extraction, separate GPU-specific error/dependency coupling, define the
cross-crate checked snapshot/parameter interface, and rerun source/wire fixtures,
feature/MSRV/package checks and GPU parity. Legacy public structural fields need
a future breaking release to become private. This change introduces no new
crate, universal backend trait, model-ID/hash framework or wire version.
