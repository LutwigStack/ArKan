# Benchmark methodology

The historical ArKan/PyTorch comparison ratios are **invalid pending a matched
rerun**. The Python baselines used spans inconsistent with the grid range, and
the competitor baseline stored only `order + 1` coefficients instead of
`grid_size + order`. Training option timings also carried model and optimizer
state between samples; the CPU/GPU training comparison used different optimizers.
These defects invalidate the old speedup and option-overhead conclusions. The AMP
allocation qualification below is separate from those historical comparisons.

Historical Rust constants remain explicitly tagged in `scripts/bench_competitors.py`
for provenance, separate from the current Python results. Their estimated
`backward + update` column also includes loss computation. It is not a measured
forward-plus-backward time and must not be compared with the Python `fwd_bwd`
column. The original machine/date attribution is unverified; a current Python
run does not remeasure those constants.

Historical baked latency numbers measured the allocating `BakedModel::forward`
convenience method. The current benchmark uses `forward_with_workspace`, so those
latency ratios do not describe the current benchmark. Rerun on the same hardware
and record the revision before making a new deployment latency claim.

## Cases and state

The CPU raw-training, learning-rate and option comparisons use seed 42.
Training benchmarks clone the seeded base model for each
measured step, and prepare reusable scratch buffers outside the timer. Learning
rate and option cases start from the same weights, rather than from a model
trained by a preceding case. The CPU optimizer suite measures raw SGD training,
optimizer construction, and active/inactive clipping thresholds at one half/twice
the measured initial gradient norm. The norm is asserted finite and nonzero and
printed before timing.

The `amp_identity` group measures whole public Adam/SGD `step` calls on seed-11
`[8, 16, 16, 4]` models with grid 5/order 3 and prebuilt gradients. Each sample
clones a model and optimizer warmed by five steps; cloning and fixture drops
are outside the timer. Checked strict/skip factor `1.0` cases have unchecked
identity, active clipping and non-unit factor `2.0` controls. Clipping uses a
fixed `0.25` threshold below the independently checked fixture norm.

The matched allocation qualification counted every request over fifty warmed
public steps for each of Adam/SGD strict/skip identity AMP. Each baseline row
made 400 requests for 731,200 bytes; each candidate row made zero requests.
This applies to the finite, unclipped fixture above, not every safety result or
AMP configuration. A separate public-step qualification checked 208 fixed cases
per role with exact original/candidate bits, including exceptional gradients,
clipping, caller immutability, warmed rollback, optimizer state and whole-window
live/peak/drop accounting.

The fixed ABBA timing attempt ended `INVALID_STOP`: 16 of 40 confidence widths
and 9 of 20 paired drifts exceeded the unchanged 3% validity limits. It supplies
no qualified latency result. The change is adopted for the allocation reduction
above; it makes no speedup promise. The benchmark remains available to reproduce
the whole-step workload.

A separate optimizer change omits the second finite-gradient scan for validated
AMP factors at least `1.0`: dividing finite f32 values by a finite factor at least
one cannot overflow the f32 result. With finite checking enabled, this avoids
`P` additional `is_finite` checks and a logical `4 * P`-byte traversal, where `P`
is the total number of weight and bias gradient coefficients. This is an
operation-count reduction, not measured memory bandwidth or a latency claim.
`None` already skipped that scan; with both safety checks disabled, the old
helper already performed no per-element finite scan. Factors below one retain
the post-unscale check. Matched debug/release optimizer suites and the permanent
regression cover neighboring factors around one, large factors, signed zeros,
subnormals, explicit unscale parity and tiny-factor strict/skip rollback.

A combined allocation proposal was rejected because matched release results
differed in NaN payload bits. Its memory savings were not adopted; only the
independently qualified optimizer guard above was retained.

GPU training uses `BatchSize::PerIteration`: each reset of the shared GPU model
finishes before its measured step. `SmallInput` would run multiple setup closures
before multiple routines and therefore reset the shared device model only before
the first routine. Native GPU Adam resets both weights and moments for every
sample. Training buffers are initialized before timing. CPU/GPU Adam comparison
cases use the same seeded network, inputs, targets, and `AdamConfig::with_lr(0.001)`.
Hybrid GPU timing includes transfers, backward readback, the CPU optimizer step,
and updated-weight upload. Native timing includes its synchronous loss readback.
These are different workflows; publish them with their operation boundaries.

The GPU option cases use Adam plus `TrainOptions` clipping and weights-only
option decay. The raw CPU option cases use SGD; they are not a backend speedup
comparison. Standalone optimizer decay policies are separate from these options.

`baked` measures batch-one inference for two calibrated seeded networks, using
reusable workspaces for both f32 and baked inference. Workspace creation and
calibration are outside timing. `size_bytes()` is an approximate baked
payload count; it excludes some metadata, capacities, allocator overhead, and the
workspace. The f32 column counts only weights and biases. Neither reports total
process memory or serialized size.

An experimental order-4/5 integer-basis rewrite preserved the complete coefficient
vectors for all 65,536 Q16 coordinates in both debug and release qualification.
The retained release assembly removed two wide signed-division helper calls for
order 4 and three for order 5 from whole baked inference. These checks establish
exact arithmetic and the intended codegen change.

Its CPU-2 ABBA timing used the medium `8 → 16 → 8 → 4`, grid-5 fixture, with
order-4/5 targets and order-3 baked/f32 controls, 100 samples, a 5-second warmup
and a 15-second measurement. The frozen analysis ended `INVALID_STOP`: 1 of 16
confidence widths, 5 of 8 repeat drift guards and 3 of 16 comparison guards failed.
There was no eligible target or confirmed paired slowdown. The rewrite was not
adopted, and this campaign supplies no qualified latency result. Small exact
coefficient boundary regressions remain in the ordinary test suite.

A separate CPU inference experiment reduced hidden-layer staging copies while
retaining every workspace reserve/resize, final full copy and initialized
spare-capacity value. Four debug/release role profiles passed exact public-state,
error, workspace-reuse and whole-call allocation/live/peak comparisons. Three
ordinary integration regressions preserve initialized tails and output sentinels,
including reuse across different layer shapes and a rejected input shape.

Its paired timing used seed-42, grid-5/order-3 models on CPU 2, with one-/two-layer
controls and wide/deep hidden-layer targets. An A-only calibration fixed the batch
size before 32 balanced ABBA/BAAB bundles; the same-binary A/A null campaign passed.
The subsequent A/B campaign ended `INVALID_STOP`: the wide-hidden half-to-half
drift guard failed and control equivalence was unresolved. The predeclared
analysis supplied no eligible target or qualified nonregression result. The
production rewrite was not adopted; these regressions establish correctness
contracts without a speed claim.

The PDE residual loss now uses implicit positive-zero targets in the shared MSE
implementation while preserving subtraction, mask reduction and shape validation.
Matched public-call qualification removes one temporary zero-target vector:
valid nonempty calls request one vector for `4 * n` bytes instead of two for
`8 * n` bytes, retaining the same returned gradient. Empty calls request none.
The ordinary allocation test pins the seven-element case at one request/28 bytes.

Entropy gradients store the per-coefficient derivative in the existing returned
buffer and reuse it in the final pass, avoiding repeated active probability and
logarithm work without another allocation. Debug/release comparisons covered
complete loss/gradient bits, including signed zeros and NaN payloads, input
immutability, malformed masks and whole-call allocation/live/peak behavior.
These changes are adopted for their demonstrated storage and operation reductions;
no latency measurement or speedup is claimed.

Python baseline operations share `scripts/bench_reference.py`: normalize, clamp,
find the span from the actual knots, evaluate local basis functions, and gather
active global coefficients from `[output, input, global_basis]`. Hidden layers
use identity normalization. The benchmark defaults use identity input
normalization; the dictionary forward helpers also accept per-layer `mean` and
`std`. Same-weight tests use the five checked-in training fixtures, including
multilayer and order-four cases, and check clamped gradients and coefficient
gathering. The GPU script's tensor helper can be tested on CPU without CUDA.

Legacy baselines initialize unit-normal coefficients; the competitor pure-spline
model scales its initial coefficients by 0.1. These are separate workloads.
Python reports medians. The legacy CPU scripts default to five samples; the
competitor runner uses 50 samples after 10 warmups. Repeat counts are part of the
result and must match for a comparison. Legacy CPU full-step timing uses SGD;
the competitor runner and third-party CUDA training runner use Adam. Competitor
Adam buffers are prepared outside timing, and model weights plus zero optimizer
state are restored before every warmup and measured step. CUDA synchronizes
before and after timing. Runtime metadata records UTC timestamp, platform,
Python/PyTorch versions, thread count, revision, compiler, and Rust flags.

`efficient-kan` adds a base/residual term; FastKAN uses RBFs. Their timings are
workload comparisons, not costs of equivalent mathematical functions. A seed
alone does not create identical weights across Rust and PyTorch RNGs. Use a
shared exported model and identical data for an actual same-function speedup
claim; the parity fixtures establish mathematical agreement, not speedup.

## Metrics and hardware

`Throughput::Elements(batch * input_dim)` counts input elements, not samples or
FLOPS. Report batch latency and latency per sample alongside it. The CPU and GPU
memory groups report an approximate selected-buffer working-set size divided by
elapsed time. Each selected buffer is counted once; this model is neither a
complete allocation inventory nor measured cache/DRAM traffic. It cannot support
roofline arithmetic intensity or a comparison with peak DDR bandwidth. Use
hardware counters or a profiler for memory traffic.

Record OS, CPU/GPU adapter, physical versus software GPU, revision, toolchain,
features, flags, input/model fixtures, repeat counts, and thread count with every
result. `wide::f32x8` and `simd_width = 8` do not establish AVX2 code generation.
Portable builds and `RUSTFLAGS='-C target-cpu=native'` builds are separate cases;
record `rustc --print cfg` with matching target flags. Software llvmpipe runs can
validate correctness but do not establish physical GPU performance.

## Run and validate

```bash
# Compile/link benchmark harnesses without taking performance measurements.
cargo test --benches --no-run
cargo test --benches --no-run --features serde,parallel,gpu

# CPU suites (Criterion needs several minutes; run on an otherwise idle machine).
cargo bench --bench forward --bench backward --bench optimizer
cargo bench --bench scaling --bench spline_config --bench latency
cargo bench --bench memory --bench baked
cargo bench --features parallel --bench forward --bench backward

# Physical GPU only; unset ARKAN_GPU_BENCH skips runtime benchmark cases.
ARKAN_GPU_BENCH=1 cargo bench --features gpu --bench gpu_forward --bench gpu_backward

# Install CPU PyTorch into a project/workspace-owned venv if needed.
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python -m unittest discover -s scripts -p test_bench_reference.py
.venv/bin/python scripts/bench_pytorch.py
.venv/bin/python scripts/bench_pytorch_train.py
.venv/bin/python scripts/bench_competitors.py

# Example runtime smoke: build outside the runtime deadline.
cargo build --example basic --example baked_inference
python3 scripts/smoke_examples.py
cargo build --features serde --example basic --example baked_inference
python3 scripts/smoke_examples.py --serde
```

CI runs the CPU smoke with and without serde. Each example has a 30-second
process timeout, and each CI smoke step has a two-minute timeout. The smoke checks
finite single/batch inference output, completed baked inference, and bit-identical
baked serialization output. It requires no datasets, downloads, GPU, or benchmark
sampling. GPU CI compilation alone does not validate GPU runtime behavior.

## Historical archive

The tables below preserve the earlier published record. **All historical
comparison ratios and performance conclusions are invalid pending a matched
rerun under the methodology above.** Platform/date attribution describes the
old report, not this checkout or a fresh run. Earlier baked accuracy and payload
counts are retained as historical observations; their latency uses the allocating
API. The old byte-rate tables do not measure DRAM bandwidth.

<details>
<summary>Superseded tables and conclusions (invalid for current comparisons)</summary>

## Archived benchmark results

**Platform:** Windows 11, AMD Ryzen (AVX2), NVIDIA GeForce RTX 4070 SUPER
(Vulkan via wgpu 0.23)
**CPU Config:** Poker preset `[21, 64, 64, 24]`, Grid 5, Spline Order 3 (cubic)
**Rust:** `cargo bench`, release profile, default features (SIMD via `wide` is
always on; `parallel` is **not** enabled — these are single-threaded numbers)
**Python:** PyTorch 2.x

### Which numbers are how old

Every table below is labelled. Read the label before you quote the number.

| Section | Last measured | Note |
|---|---|---|
| [Baked (int8)](#baked-int8-inference) | **2026-07-26** | Re-measured after the order-4/5 basis fix |
| CPU forward / train / latency / scaling | 2026-06-27 | `cargo bench --bench forward --bench backward` |
| GPU forward | 2026-06-27 | `gpu_forward` bench with `ARKAN_GPU_BENCH=1` |
| GPU train step (hybrid, native, options) | **not re-measured** | Predates 2026-06-27; `gpu_backward` was not run |
| PyTorch CPU comparison | 2026-06-27 | Some entries are *extrapolated*, marked "(est.)" |
| PyTorch GPU comparison | 2026-06-27 | Config mismatch, see that section |

> **Void:** any baked accuracy figure for **spline order 4 or 5** published
> before commit `576fbc7` (2026-07-26) was measured on a numerically wrong
> fixed-point basis — max absolute basis error 0.208 (order 4) and 0.775
> (order 5), end-to-end NRMSE 13.98% and 89.30%. Those numbers are not
> conservative, they are meaningless. The tables in this file are post-fix.

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Single inference latency (P50)** | **15.0 µs** |
| **Single inference throughput** | **~66,000 inferences/sec** |
| **vs faithful-PyTorch CPU (batch=1)** | **~44-54x faster** (same pure B-spline math) |
| **vs faithful-PyTorch CPU (batch=64)** | **~2x faster** |
| **vs faithful-PyTorch CPU (batch=256+)** | **PyTorch wins** (BLAS advantage at large batch) |
| **vs efficient-kan GPU (batch=64)** | **ArKan GPU ~1.7x faster** (ArKan 1.30ms vs efficient-kan 2.24ms) |
| **vs FastKAN GPU** | **FastKAN wins** (RBF not B-splines — different math) |
| **Memory footprint** | 218.6 KB (weights only) |
| **Reusable training storage** | Warmed train steps reuse ArKan execution storage; see runtime caveat below |
| **Native GPU training (batch=64)** | **3.96 ms** (see GPU section; not re-measured) |
| **Baked (int8) vs f32 at batch=1** | **1.4–2.1x SLOWER** — the win is size (2.2–3.0x), not speed |
| **Baked worst-case error at ≥1σ** | **34–54%** on a 2-hidden net — ranking/argmax only |

Allocation counts depend on call context. Serial/workspace reuse and baked-workspace
guarantees remain applicable. With `parallel`, repeated calls from an external
thread can allocate Rayon scheduling-queue blocks, even after warmup. Strict zero
counts were observed with repeated work inside one enclosing, warmed four-worker
Rayon pool; this does not guarantee zero allocations for arbitrary pools. Short
external-call windows that count zero do not establish an indefinite guarantee.

---

## 🖥️ GPU Backend Performance

> **Note:** GPU benchmarks require the `gpu` feature flag, a compatible GPU, and the `ARKAN_GPU_BENCH=1` environment variable (CI-safe guard — benchmarks are silently skipped when the variable is absent).
> Run with:
> ```bash
> # Windows PowerShell
> $env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_forward --features gpu
> $env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_backward --features gpu
>
> # Linux/macOS
> ARKAN_GPU_BENCH=1 cargo bench --bench gpu_forward --features gpu
> ARKAN_GPU_BENCH=1 cargo bench --bench gpu_backward --features gpu
> ```

### GPU vs CPU Forward Pass

**GPU:** NVIDIA GeForce RTX 4070 SUPER (Vulkan)

**Re-measured 2026-06-27** via `cargo bench --bench gpu_forward --features gpu` with `ARKAN_GPU_BENCH=1`. Criterion median (100 samples).

| Batch | CPU (Rust) | GPU (wgpu) | Speedup | Notes |
|-------|-----------|-----------|---------|-------|
| 1 | 26.8 µs | **1.154 ms** | 0.023x | CPU wins decisively (GPU dispatch overhead) |
| 8 | 213 µs | **1.214 ms** | 0.18x | CPU still faster |
| 16 | 431 µs | **1.242 ms** | 0.35x | CPU still faster |
| 64 | 1.694 ms | **1.296 ms** | **1.31x GPU** | GPU starts winning |
| 256 | 6.787 ms | **1.421 ms** | **4.78x GPU** | GPU wins decisively |
| 1024 | ~25 ms (est.) | **1.436 ms** | **~17x GPU** | GPU advantage maximized |

**Key Insight:** GPU crossover point is around batch size 32-64. For single-sample latency-critical applications (e.g., real-time MCTS), CPU is preferred.

### GPU Train Step — Hybrid mode (Adam optimizer)

> **Note:** These figures are from the *hybrid* `train_step_mse` path (forward on GPU, backward + optimizer on CPU with weight sync). Criterion group: `gpu_train_step_adam`. **Numbers not re-measured in 2026-06-27 run** (gpu_backward bench not run; the gpu_forward bench does not cover train step).

| Batch | Time | Throughput |
|-------|------|------------|
| 1 | 7.68 ms | 2.7 K elem/s |
| 8 | 6.55 ms | 25.7 K elem/s |
| 16 | 9.22 ms | 36.4 K elem/s |
| 64 | 9.87 ms | 136 K elem/s |
| 256 | 10.1 ms | 530 K elem/s |

### GPU Train Step — Hybrid mode (SGD optimizer)

> **Note:** Hybrid path (`train_step_sgd`). Criterion group: `gpu_train_step_sgd`. **Numbers not re-measured in 2026-06-27 run** (same caveat as Adam hybrid above).

| Batch | Time | Throughput |
|-------|------|------------|
| 1 | 7.72 ms | 2.7 K elem/s |
| 8 | 10.3 ms | 16.3 K elem/s |
| 16 | 10.4 ms | 32.4 K elem/s |
| 64 | 10.7 ms | 126 K elem/s |
| 256 | 10.9 ms | 494 K elem/s |

### GPU Native Training (v0.3.0+)

> **Note:** Native GPU training via `train_step_gpu_native` (Criterion group: `gpu_native_training`, uses `GpuAdam` optimizer — all state stays on GPU). **Numbers not re-measured in 2026-06-27 run** (gpu_backward bench skipped to avoid conflicts with parallel test run).

| Batch | Native GPU | Hybrid GPU | CPU | Native Speedup vs Hybrid |
|-------|------------|------------|-----|--------------------------|
| 1 | 3.04 ms | 7.68 ms | 118 µs | 2.5x |
| 8 | 3.82 ms | 6.55 ms | 582 µs | 1.7x |
| 16 | 3.95 ms | 9.22 ms | 1.15 ms | 2.3x |
| 64 | 3.96 ms | 9.87 ms | 4.41 ms | 2.5x |
| 256 | 3.75 ms | 10.1 ms | 17.4 ms | 2.7x |

### GPU Train Options Impact (batch=64)

> **Note:** Measured via `bench_gpu_train_step_with_options` (Criterion group: `gpu_train_options`), which uses the hybrid `train_step_with_options` path. **Numbers not re-measured in 2026-06-27 run.**

| Option | Time | Overhead |
|--------|------|----------|
| No options | 8.40 ms | baseline |
| Grad clip (1.0) | 10.3 ms | +23% |
| Weight decay (0.01) | 10.2 ms | +21% |
| Both | 10.6 ms | +26% |

### GPU Softmax Performance

| Batch | Forward + Softmax | Forward Only | Softmax Overhead |
|-------|-------------------|--------------|------------------|
| 1 | 1.98 ms | 1.19 ms | +66% |
| 16 | 2.12 ms | 1.31 ms | +62% |
| 64 | 2.18 ms | 1.18 ms | +85% |
| 256 | 2.42 ms | 1.20 ms | +102% |

### GPU Limitations (wgpu 0.23)

- **No DeviceLost event:** wgpu 0.23 does not propagate `DeviceLost` errors. GPU crashes may appear as hangs.
- **Memory limits:** MAX_VRAM_ALLOC = 2GB per buffer. Use `BatchTooLarge` error for early rejection.
- **Backend selection:** Use `WgpuOptions::compute()` for best compute performance settings.

### Native GPU Training (v0.3.0+)

ArKan supports **fully native GPU training** where forward pass, backward pass, and optimizer updates all run on GPU without CPU↔GPU weight transfers.

**API Usage:**
```rust
use arkan::gpu::{GpuAdam, GpuAdamConfig};

// Create network and optimizer
let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)?;
let layer_sizes = gpu_network.layer_param_sizes();
let mut optimizer = GpuAdam::new(
    backend.device_arc(), backend.queue_arc(),
    &layer_sizes, GpuAdamConfig::with_lr(0.001),
);

// Native GPU training - no CPU transfers!
let loss = gpu_network.train_step_gpu_native(
    &input, &target, batch_size, &mut workspace, &mut optimizer
)?;
```

**Performance Comparison:**

| Method | Batch=64 | Notes |
|--------|----------|-------|
| CPU train_step | 4.77 ms | Baseline |
| Hybrid GPU (old) | ~10 ms | Forward GPU, optimizer CPU, sync overhead |
| **Native GPU** | **~2-3 ms** | Full GPU pipeline, no transfers |

**Benefits:**
- ✅ **2-5x faster** than hybrid approach for large batches
- ✅ No CPU↔GPU weight synchronization overhead
- ✅ Gradients stay on GPU between backward and optimizer steps
- ✅ Supports both Adam and SGD optimizers

**When to use:**
- Large batch training (batch ≥ 64)
- Repeated training iterations (gradients reused on GPU)
- When GPU VRAM is available

---

## 🎯 Single-Sample Latency (Real-Time Poker)

Critical for MCTS/CFR solvers where thousands of single inferences per second are required.

### Latency Distribution (poker config, batch=1)

| Percentile | Latency |
|------------|---------|
| Min | 14.3 µs |
| **P50 (median)** | **15.0 µs** |
| P90 | 15.1 µs |
| P99 | 18.7 µs |
| P999 | 48.8 µs |
| Max | 279.3 µs |

**Throughput at P50:** ~66,000 inferences/second

### forward_single vs forward_batch(1)

> **Note:** Criterion group: `single_sample_latency` (bench: `latency`). Bench functions: `forward_single` and `forward_batch_1`.

| Method | Time | Notes |
|--------|------|-------|
| `forward_single` | ~14.6 µs | Optimized single-sample path |
| `forward_batch(1)` | 26.7 µs | Batch overhead visible |

**Recommendation:** Use `forward_single` for real-time play (~1.8x faster), `forward_batch` for training.

---

## 🔄 Training Performance

### Backward Pass Overhead (batch=64)

> **Note:** Criterion groups: `forward_only`, `forward_training`, `full_train_step` (bench: `backward`). The `backward_overhead_batch64` group measures these side-by-side with bench functions `1_forward_inference`, `2_forward_training`, and `3_full_train_step`.

| Operation | Time | Overhead vs Forward |
|-----------|------|---------------------|
| forward_only | 1.70 ms | baseline |
| forward_training | 1.70 ms | ~0% (buffer prep is free) |
| **full_train_step** | **4.48 ms** | **+163%** |

**Analysis:** Backward pass takes roughly 2.6x the forward pass time, which is typical for gradient computation. Reused ArKan execution storage avoids storage growth on warmed paths; parallel runtime allocations depend on the call context described above.

### Training Options Impact (batch=64)

> **Note:** Criterion group: `train_options_batch64` (bench: `optimizer`). Bench functions: `no_options`, `grad_clip_1.0`, `weight_decay_0.01`, `clip_and_decay`. Uses `train_step_with_options` API.

| Option | Time | Overhead |
|--------|------|----------|
| No options | 4.48 ms | baseline |
| Gradient clipping (1.0) | 4.50 ms | +0.4% (noise) |
| Weight decay (0.01) | 4.47 ms | -0.2% (noise) |
| Both | 4.50 ms | +0.4% (noise) |

**Conclusion:** Training options have negligible performance impact.

---

## 📐 Architecture Scaling

### Latency (batch=1)

| Architecture | Params | Memory | Latency | Throughput |
|--------------|--------|--------|---------|------------|
| tiny `[3,10,1]` | 331 | 1.3 KB | **387 ns** | 7.8 M elem/s |
| medium `[10,64,64,10]` | 43K | 168.5 KB | 23.5 µs | 426 K elem/s |
| **poker `[21,64,64,24]`** | **56K** | **218.6 KB** | **14.6 µs** | **1.4 M elem/s** |
| large `[32,128,128,128,32]` | 328K | 1.28 MB | 139 µs | 230 K elem/s |
| wide `[21,256,24]` | 92K | 361 KB | 41.1 µs | 512 K elem/s |
| deep `[21,32,32,32,32,32,24]` | 44K | 174 KB | 22.7 µs | 926 K elem/s |

### Throughput (batch=64)

| Architecture | Forward | Train Step |
|--------------|---------|------------|
| tiny | 21.1 µs (9.1 M elem/s) | 56.7 µs (3.4 M elem/s) |
| medium | 1.38 ms (464 K elem/s) | 3.49 ms (184 K elem/s) |
| **poker** | **1.70 ms (790 K elem/s)** | **4.48 ms (300 K elem/s)** |
| large | 9.12 ms (225 K elem/s) | 24.7 ms (83 K elem/s) |
| wide | 2.63 ms (512 K elem/s) | 7.17 ms (188 K elem/s) |
| deep | 1.49 ms (903 K elem/s) | 3.88 ms (347 K elem/s) |

**Insight:** Deep narrow networks (5x32) are faster than wide shallow ones (1x256) for the same parameter count.

---

## 📏 Spline Configuration Analysis

### Spline Order Impact (grid=5, batch=64)

| Order | Name | Basis Size | Time | Throughput | Params |
|-------|------|------------|------|------------|--------|
| 1 | linear | 6 | 1.02 ms | **1.32 M elem/s** | 42K |
| 2 | quadratic | 7 | 1.39 ms | 967 K elem/s | 49K |
| **3** | **cubic** | **8** | **1.70 ms** | **790 K elem/s** | **56K** |
| 4 | quartic | 9 | 2.10 ms | 640 K elem/s | 63K |
| 5 | quintic | 10 | 2.42 ms | 555 K elem/s | 70K |

**Trade-off:** Each order increase adds ~350 µs latency but improves function smoothness.

### Grid Size Impact (order=3 cubic, batch=64)

| Grid | Basis Size | Time | Throughput | Params |
|------|------------|------|------------|--------|
| 3 | 6 | 1.72 ms | 781 K elem/s | 42K |
| **5** | **8** | **1.70 ms** | **790 K elem/s** | **56K** |
| 8 | 11 | 1.75 ms | 767 K elem/s | 77K |
| 12 | 15 | 1.79 ms | 751 K elem/s | 105K |
| 16 | 19 | 1.80 ms | 746 K elem/s | 133K |

**Insight:** Grid size has minimal impact on forward speed due to local spline evaluation (only `order+1` basis functions computed). Choose grid size based on required expressiveness.

### Recommended Configurations

| Use Case | Grid | Order | Params | Latency (batch=1) |
|----------|------|-------|--------|-------------------|
| **Fast inference** | 3 | 2 | 35K | ~12 µs |
| **Balanced (default)** | 5 | 3 | 56K | ~15 µs |
| **High accuracy** | 8 | 3 | 77K | ~16 µs |
| **Smooth functions** | 5 | 5 | 70K | ~20 µs |

---

## 💾 Memory Analysis

### Network Memory

| Component | Size (poker config) |
|-----------|---------------------|
| Weights | 218.6 KB |
| Workspace (batch=64) | ~100 KB |
| **Total inference** | **~320 KB** |

### Optimizer Memory Overhead

| Optimizer | Additional Memory |
|-----------|-------------------|
| Raw SGD (no state) | 0 KB |
| SGD + momentum | 218.6 KB (+100%) |
| Adam | 437.2 KB (+200%) |

### Workspace Reuse

| Mode | Time (batch=64) |
|------|-----------------|
| With reuse | 1.70 ms |
| Without reuse (alloc each time) | 1.70 ms |

**Result:** Workspace allocation is fast (~0% overhead), but reuse is still recommended for hot paths.

---

## 🏎️ Memory Bandwidth (CPU)

### Achieved Bandwidth

| Batch | Est. Memory | Time | Bandwidth |
|-------|-------------|------|-----------|
| 1 | 222 KB | 15.0 µs | 14.1 GB/s |
| 16 | 269 KB | 427 µs | 616 MB/s |
| 64 | 418 KB | 1.70 ms | 240 MB/s |
| 256 | 1012 KB | 6.82 ms | 145 MB/s |
| 1024 | 3391 KB | 25.4 ms | 130 MB/s |

**Analysis:** Small batches achieve higher bandwidth due to cache locality. Large batches are compute-bound rather than memory-bound.

---

## ⚡ ArKan CPU vs PyTorch CPU Comparison

> **Measured:** 2026-06-27 via `cargo bench --bench forward --bench backward` (Criterion 100 samples, median)
> and `scripts/bench_competitors.py` (median of 50 repeats, 10 warmup).  
> **Config:** ArKan poker preset `[21, 64, 64, 24]`, grid=5, order=3.  
> **Baseline:** "faithful-PyTorch" = pure B-spline without residual term (same math as ArKan).  
> **Estimates:** the faithful-PyTorch column at `[21,64,64,24]` is **extrapolated**
> from measured `[16,64,64,8]` numbers, not measured at this shape. Rows carrying
> "(est.)" are therefore an approximation, and the speedup ratios inherit that.  
> **Caveat:** efficient-kan adds SiLU base residual (different formulation). FastKAN uses RBF (different math entirely). Neither is an apples-to-apples comparison; see the per-implementation notes in each table.

### Forward Pass (Inference) — ArKan CPU vs faithful-PyTorch (same math)

| Batch | ArKan (Rust) | faithful-PyTorch | **Speedup** | Note |
|-------|-------------|-----------------|-------------|------|
| **1** | **26.8 µs** | ~1.5 ms (est.) | **~56x** | Zero-alloc single-sample dominance |
| 64 | **1.694 ms** | ~2.9 ms (est.) | **~1.7x** | ArKan ahead |
| 256 | **6.687 ms** | ~4.4 ms (est.) | **0.66x** | PyTorch BLAS wins at large batch |

*Estimates for faithful-PyTorch at [21,64,64,24] extrapolated from measured [16,64,64,8] numbers (see comparison.md).*

### Full Train Step (Forward + Backward + Optimizer) — ArKan CPU

ArKan uses internal SGD. Python competitors use Adam (slightly heavier).

| Batch | ArKan train_step (SGD) | Note |
|-------|----------------------|------|
| 1 | **0.102 ms** | |
| 64 | **4.488 ms** | |
| 256 | **18.02 ms** | |

### Backward Pass (Estimated from bench data)

| Batch | ArKan bwd est. | Note |
|-------|---------------|------|
| 1 | ~0.075 ms | full_step(0.102) - forward(0.027) |
| 64 | ~2.79 ms | full_step(4.49) - forward(1.69) |
| 256 | ~11.3 ms | full_step(18.0) - forward(6.69) |

### Key Takeaways (Updated)

1. **Low-latency dominance:** ArKan is ~44-56x faster than faithful-PyTorch for batch=1 (same math)
2. **Mid-batch win:** ArKan is ~1.7x faster at batch=64
3. **Large batch regression:** At batch=256+, PyTorch BLAS overtakes ArKan CPU — ArKan GPU compensates here
4. **Storage reuse benefit:** Warmed execution avoids new ArKan storage allocations; this alone does not guarantee a latency distribution or exclude parallel runtime allocations.
5. **GPU compensates:** ArKan GPU provides 4.8x speedup at batch=256 over ArKan CPU, recovering the large-batch gap
6. **FastKAN comparison:** FastKAN (RBF) is faster than ArKan CPU at all batch sizes but uses different math — the comparison is not apples-to-apples

---

## 🖥️ Comprehensive GPU Benchmarks

> **Requirements:**
> - `gpu` feature enabled
> - Compatible GPU (Vulkan, DX12, or Metal)
> - Set `ARKAN_GPU_BENCH=1` environment variable (CI-safe: skips when not set)

**Tested on:** NVIDIA GeForce RTX 4070 SUPER (Vulkan)

### GPU vs CPU Forward Pass Scaling

The crossover point where GPU becomes faster than CPU depends on batch size:

| Batch | CPU | GPU | Winner | Speedup |
|-------|-----|-----|--------|---------|
| 1 | 15.0 µs | 254 µs | CPU | 17x CPU |
| 8 | 200 µs | 264 µs | CPU | 1.3x CPU |
| 16 | 381 µs | 272 µs | GPU | 1.4x GPU |
| 32 | 762 µs | 282 µs | GPU | 2.7x GPU |
| 64 | 1.70 ms | 1.18 ms | GPU | 1.4x GPU |
| 256 | 6.38 ms | 1.40 ms | GPU | 4.6x GPU |
| 1024 | 25.4 ms | 1.85 ms | GPU | 13.7x GPU |

**Key Insight:** GPU crossover point is around batch size 32-64. For single-sample real-time inference (poker MCTS), CPU is better. For batch training, GPU excels.

### GPU Architecture Scaling

#### Latency (batch=1, GPU)

| Architecture | Params | GPU Latency | CPU Latency | GPU Overhead |
|--------------|--------|-------------|-------------|--------------|
| tiny `[3,10,1]` | 331 | 195 µs | 387 ns | 504x |
| medium `[10,64,64,10]` | 43K | 240 µs | 23.5 µs | 10.2x |
| **poker `[21,64,64,24]`** | **56K** | **254 µs** | **14.6 µs** | **17.4x** |
| large `[32,128,128,128,32]` | 328K | 350 µs | 139 µs | 2.5x |
| wide `[21,256,24]` | 92K | 230 µs | 41.1 µs | 5.6x |
| deep `[21,32,32,32,32,32,24]` | 44K | 290 µs | 22.7 µs | 12.8x |

**CPU wins for all architectures at batch=1** due to GPU dispatch overhead.

#### Throughput (batch=64, GPU)

| Architecture | GPU Time | CPU Time | GPU Speedup |
|--------------|----------|----------|-------------|
| tiny | 190 µs | 21.1 µs | 0.11x |
| medium | 250 µs | 1.38 ms | 5.5x |
| **poker** | **1.18 ms** | **1.70 ms** | **1.4x** |
| large | 400 µs | 9.12 ms | 22.8x |
| wide | 245 µs | 2.63 ms | 10.7x |
| deep | 300 µs | 1.49 ms | 5.0x |

### GPU Spline Configuration Impact

#### Spline Order (grid=5, batch=64)

| Order | Name | GPU Time | CPU Time | GPU Speedup |
|-------|------|----------|----------|-------------|
| 1 | linear | 1.02 ms | 1.02 ms | 1.0x |
| 2 | quadratic | 1.10 ms | 1.39 ms | 1.3x |
| **3** | **cubic** | **1.18 ms** | **1.70 ms** | **1.4x** |
| 4 | quartic | 1.30 ms | 2.10 ms | 1.6x |
| 5 | quintic | 1.42 ms | 2.42 ms | 1.7x |

**GPU advantage increases with spline order** due to more parallelizable B-spline computation.

#### Grid Size (order=3, batch=64)

| Grid | Basis Size | GPU Time | CPU Time | GPU Speedup |
|------|------------|----------|----------|-------------|
| 3 | 6 | 1.12 ms | 1.72 ms | 1.5x |
| **5** | **8** | **1.18 ms** | **1.70 ms** | **1.4x** |
| 8 | 11 | 1.26 ms | 1.75 ms | 1.4x |
| 12 | 15 | 1.38 ms | 1.79 ms | 1.3x |
| 16 | 19 | 1.50 ms | 1.80 ms | 1.2x |

### GPU Train Step Performance

> **Note:** "Native" = `train_step_gpu_native` with `GpuAdam` (Criterion group: `gpu_native_training`). "Hybrid" = `train_step_mse`/`train_step_sgd` with CPU Adam/SGD (groups: `gpu_train_step_adam`, `gpu_train_step_sgd`). All numbers need re-measurement.

| Optimizer | Batch=64 | Batch=256 | Notes |
|-----------|----------|-----------|-------|
| Adam (native) | 3.96 ms | 4.50 ms | `train_step_gpu_native` + `GpuAdam`, no CPU transfers |
| SGD (native) | 3.04 ms | 3.65 ms | `train_step_gpu_native` estimate (SGD variant) |
| Adam (hybrid) | 9.87 ms | 10.6 ms | `train_step_mse` — forward GPU → backward + Adam CPU |
| SGD (hybrid) | 7.12 ms | 8.10 ms | `train_step_sgd` — forward GPU → backward + SGD CPU |

**Native GPU training is 2.5x faster than hybrid approach!**

### GPU Latency Distribution (batch=1)

| Percentile | GPU | CPU |
|------------|-----|-----|
| Min | 220 µs | 14.3 µs |
| P50 | 254 µs | 15.0 µs |
| P90 | 275 µs | 15.1 µs |
| P99 | 310 µs | 18.7 µs |
| P999 | 380 µs | 48.8 µs |
| Max | 520 µs | 279.3 µs |

**CPU has lower and more consistent latency** for single samples due to GPU dispatch overhead.

### GPU Memory Bandwidth

| Batch | Est. GPU Memory | Time | Bandwidth |
|-------|-----------------|------|-----------|
| 1 | ~250 KB | 254 µs | 0.98 GB/s |
| 64 | ~500 KB | 1.18 ms | 0.42 GB/s |
| 256 | ~1.2 MB | 1.40 ms | 0.86 GB/s |
| 1024 | ~4 MB | 1.85 ms | 2.2 GB/s |

### CPU vs GPU Training (batch=64)

| Configuration | CPU | GPU | Winner |
|---------------|-----|-----|--------|
| Forward only | 1.70 ms | 1.18 ms | GPU 1.4x |
| Train (Adam native) | 4.48 ms | 3.96 ms | GPU 1.1x |
| Train (SGD native) | 4.48 ms | 3.04 ms | GPU 1.5x |
| Train (Adam hybrid) | 4.48 ms | 9.87 ms | CPU 2.2x |

**Note:** Native GPU training now beats CPU! Hybrid mode (forward GPU → backward CPU) is slower due to sync overhead.

### GPU vs PyTorch GPU Comparison

**Re-measured 2026-06-27** via `scripts/bench_pytorch_gpu.py` (median of 50 repeats, 10 warmup). ArKan GPU uses wgpu/Vulkan, NOT CUDA. Python competitors use PyTorch CUDA. Config mismatch: Python benches use [16,64,64,8]; ArKan GPU uses [21,64,64,24] (slightly larger).

| Implementation | Forward batch=64 | Forward batch=256 | Forward batch=1024 | Math | Notes |
|----------------|-----------------|------------------|--------------------|------|-------|
| **FastKAN (PyTorch CUDA)** | **0.790 ms** | **0.960 ms** | **0.867 ms** | RBF (not B-spline) | Fastest: avoids B-spline compute entirely |
| **ArKan GPU (wgpu/Vulkan)** | **1.296 ms** | **1.421 ms** | **1.436 ms** | Pure B-spline | *Config [21,64,64,24] — slightly larger than Python configs* |
| efficient-kan (PyTorch CUDA) | 2.242 ms | 2.309 ms | 2.330 ms | B-spline + SiLU base | Slowest B-spline implementation |
| faithful-PyTorch (CUDA) | 4.277 ms | 5.388 ms | 4.195 ms | Pure B-spline | Python B-spline — no kernel optimization |

**Assessment (honest):**  
- FastKAN wins by using RBF instead of B-splines — this is a math trade-off, not a pure speed optimization.  
- ArKan GPU is the **fastest pure B-spline GPU implementation** in this comparison, ~1.7x faster than efficient-kan CUDA.  
- ArKan GPU (wgpu) is ~1.6-1.9x slower than FastKAN (CUDA RBF) — this is the cost of true B-splines on GPU.  
- faithful-PyTorch CUDA is ~3x slower than ArKan GPU, confirming the value of the custom WGSL shader path.

### Latency Percentiles (batch=1, GPU forward)

> Note: The numbers below are from a prior measurement run. Re-measured Criterion median for batch=1 GPU forward is **1.154 ms** (wgpu). The percentile distribution analysis requires the `ARKAN_GPU_BENCH=1` bench separately. Python competitor batch=1 CUDA results from 2026-06-27 bench: efficient-kan 1.907ms, FastKAN 0.551ms, faithful-PyTorch 3.044ms (median, 50 repeats).

| Implementation | batch=1 median | Math | Backend |
|----------------|---------------|------|---------|
| **FastKAN** | **0.551 ms** | RBF | PyTorch CUDA |
| efficient-kan | 1.907 ms | B-spline + residual | PyTorch CUDA |
| **ArKan (wgpu)** | **~1.154 ms** | Pure B-spline | wgpu Vulkan |
| faithful-PyTorch | 3.044 ms | Pure B-spline | PyTorch CUDA |

**Key insight:** At batch=1, GPU dispatch overhead dominates for all implementations. ArKan CPU (26.8 µs) is ~40x faster than any GPU implementation for single-sample inference.

To run PyTorch GPU comparison:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/Blealtan/efficient-kan.git
pip install git+https://github.com/ZiyaoLi/fast-kan.git
python scripts/bench_pytorch_gpu.py
```

---

## 🎮 Poker Solver Workload

Simulating real poker solver usage patterns:

| Scenario | Time | Notes |
|----------|------|-------|
| Pure inference | 14.6 µs | Just forward_single |
| + Light processing | 14.6 µs | + sum outputs |
| + Softmax | 14.6 µs | + softmax on strategy |

**Conclusion:** Post-processing overhead is negligible compared to network forward pass.

---

## 📋 How to Run Benchmarks

```bash
# All CPU benchmarks (no feature flags required)
cargo bench

# Specific CPU benchmark suites (all declared harness=false in Cargo.toml)
cargo bench --bench forward       # Groups: forward_batch, train_step, try_overhead, workspace_creation
cargo bench --bench backward      # Groups: forward_only, forward_training, full_train_step, backward_overhead_batch64
cargo bench --bench scaling       # Architecture scaling (batch=1 and batch=64)
cargo bench --bench spline_config # Groups: spline_order_batch64, grid_size_batch64
cargo bench --bench memory        # Memory bandwidth analysis
cargo bench --bench optimizer     # Groups: raw_train_step, train_options_batch64, optimizer_init, learning_rates_batch64
cargo bench --bench latency       # Groups: single_sample_latency, latency_distribution
cargo bench --bench baked         # Groups: baked_batch1_small, baked_batch1_medium

# Optional feature flags for CPU benches
# NOTE: there is no `simd` feature. SIMD via `wide` is unconditional; passing
# `--features simd` was a no-op before 0.4.0 and is now a hard error.
cargo bench --features parallel   # Enable Rayon parallel paths

# GPU benchmarks (require gpu feature + ARKAN_GPU_BENCH=1 env var; skipped silently otherwise)
# Windows PowerShell:
$env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_forward --features gpu
$env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_backward --features gpu

# Linux/macOS:
ARKAN_GPU_BENCH=1 cargo bench --bench gpu_forward --features gpu
ARKAN_GPU_BENCH=1 cargo bench --bench gpu_backward --features gpu

# PyTorch comparison (CPU)
python scripts/bench_pytorch.py        # Forward only
python scripts/bench_pytorch_train.py  # Full training

# PyTorch GPU comparison (requires CUDA)
python scripts/bench_pytorch_gpu.py    # GPU KAN implementations
```

### Running GPU Tests

```bash
# All GPU integration tests (ignored by default, require physical GPU)
cargo test --features gpu -- --ignored

# GPU benchmarks as a quick smoke check
$env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_forward --features gpu -- --noplot
```

### CI Smoke Test

```bash
# Quick verification (CPU + optional GPU)
cargo test                                           # CPU tests (no feature flags)
cargo test --features gpu -- --ignored               # GPU tests (if GPU available)
cargo bench --bench forward -- --noplot              # Quick CPU benchmark
```

---

## Baked (int8) Inference

> **Date:** 2026-07-26 (re-measured after the order-4/5 basis fix, `576fbc7`)
> **Measured by:** `cargo test --release --test baked_parity -- --nocapture` and `cargo bench --bench baked`
> **Method:** Random KanNetwork (untrained, random weights), `grid_range = (-1, 1)`,
> baked with 256 calibration inputs in `[-0.9, 0.9]`, tested against 2000 inputs in the
> same range, against the `KanNetwork::forward_single` f32 baseline.
> **Source files:** `tests/baked_parity.rs`, `benches/baked.rs`
>
> **Any order-4 or order-5 baked figure published before `576fbc7` is void.** The
> fixed-point basis was numerically wrong at those orders — max absolute basis error
> 0.208 (order 4) and 0.775 (order 5), end-to-end NRMSE 13.98% and 89.30%. Orders 2 and
> 3 were unaffected and are bit-identical across that fix.

### What baked is for

Size. **Not** latency — baked is 1.4–2.1× *slower* than f32 at batch=1, measured
below. Per-output accuracy used to be the other disqualifier and no longer is:
the ≥1σ worst case is 0.8–7.9%, down from 34.6–53.9%, after three fixed-point
defects were fixed (see below).

### Accuracy

The int8 path uses i64 accumulators, per-output-channel i8 weights
(`s_w[j] = 127 / max|w[j,*,*]|`) and u16 B-spline basis values (Q0.15). Two
metrics, because one of them hides the problem:

- **NRMSE** (aggregate): `||baked - f32||₂ / ||f32||₂` over all outputs × 2000 test
  inputs. The standard quantization quality metric — and flattering here.
- **Worst-case on significant outputs**: `max per-element |baked - f32| / |f32|`
  restricted to outputs where `|f32| > τ · σ_j` (per-output std over the test
  set). This is the number that decides whether you can use baked.

#### Per-config, order 3

| Config | Architecture | NRMSE | worst @0.1σ | @0.5σ | @1.0σ |
|--------|-------------|-------|-------------|-------|-------|
| single-layer | 4→2 | **0.34%** | 9.2% | 2.3% | 1.2% |
| small (1-hidden) | 4→[8]→2 | **0.17%** | 0.8% | 0.8% | 0.8% |
| medium (2-hidden) | 8→[16,8]→4 | **0.57%** | 47.0% | 10.5% | 5.0% |

#### All orders, 2-hidden (`baked_parity_all_orders`, 8→[16,8]→4, seed 4242)

| Order | NRMSE | worst @0.1σ | @0.5σ | @1.0σ |
|-------|-------|-------------|-------|-------|
| 2 | 0.49% | 61.3% | 14.2% | **7.9%** |
| 3 | 0.59% | 3.1% | 3.1% | **3.1%** |
| 4 | 0.37% | 16.5% | 4.6% | **3.4%** |
| 5 | 0.50% | 54.1% | 12.2% | **6.4%** |

At the 0.1σ cut the figure blows up past 100% because a few-LSB absolute error
divided by a near-noise reference explodes; that part *is* an artifact. The ≥1σ
column is not, and `baked_parity` now **gates** it at 15%.

#### What changed: three fixed-point defects, all in `src/baked.rs`

Before these, the same suite read 0.60–2.65% NRMSE with a **34.6–53.9%** worst
case on ≥1σ outputs, and the honest advice was "use baked for ranking, never for
reading an output as a quantity". Each was measured, not guessed.

1. **The ACT_TARGET clip.** Every requantized activation saturated at ±2^28, and
   the exit scale is `s_act_last = 2^28 / p99.9`, so `|output[j]| <= p99.9` held
   by construction — no input could make the model return a larger magnitude. On
   a calibrated 4→2 net, 2 of the 2000 *calibration* samples already sat above
   the ceiling (worst f32 −0.520944 vs baked −0.471458, 9.50% from clipping
   alone). The stated justification — that saturating protects the next layer's
   z scale — does not survive checking: `q_z` is clamped to the grid range in
   the inter-layer step, which is the same clamp the f32 path applies to `z`.
   The clip was a *second*, tighter saturation the f32 path does not have, so it
   could only add error, on hidden layers as much as on the output layer.
   Removing it on hidden layers alone moved the ≥1σ worst case from
   25.2/51.8/33.7/29.5% to 7.9/3.1/3.4/6.4% for orders 2/3/4/5.
   The percentile still sets `s_act`; only the clip is gone.
2. **`norm_a_fixed` carried 3 bits.** The inter-layer scale was
   `A_FIXED[i] = round(2^32 / (s_act_prev · std_i))` consumed with a hardcoded
   `>> 16`, which on freshly-built 4×8×2, 8×16×4, 16×32×8 and 32×64×16 nets
   landed on the integers **8, 7, 8, 9** — a systematic 5.6–7.1% error on the z
   scale at every hop. A per-layer shift now puts it in [2^29, 2^30) (measured
   5.4e8–1.0e9 at shifts 42–43). This is what moved NRMSE by 2–4×; it moved the
   ≥1σ tail much less than the audit predicted (53.9% → 51.8% on order 3).
3. **`A_FIXED == 0` was reachable and silent.** `round(16 · p99.9 / std)` is 0
   whenever a layer's outputs have `p99.9 < std/32`; scaling one layer's weights
   by 1e-3 is enough. The next layer's inputs then collapse to a constant —
   verified: the model returned the same `(0.12060832, 0.3845482)` for every
   input while f32 varied. Now unreachable by construction.

Remaining tail: the ≥1σ worst case is 0.8–7.9%, and the surviving outliers are
absolute-error events (one sample's error 20–35× the mean absolute error), not a
systematic scale error. Int16 weights would be the next lever, at the cost of
half the size win.

### Model Size (compression)

Per-channel metadata (M0[j] + shift[j] per output channel) adds a small overhead vs
the old per-layer scalar. The trade-off is well worth it for the accuracy improvement.

| Config | f32 weight bytes | baked bytes | Compression ratio |
|--------|-----------------|-------------|-------------------|
| small 4→[8]→2 | 1,576 B | 728 B | **2.16×** |
| medium 8→[16,8]→4 | 9,328 B | 3,140 B | **2.97×** |

The compression comes from replacing f32 weights (4 bytes each) with i8 (1 byte each),
plus storing i64 biases and int32 normalization constants. For larger networks the ratio
will approach 4× as bias/metadata overhead becomes relatively smaller.

**This is the only thing baking currently buys you.**

### Batch=1 Latency (deployment scenario)

Criterion medians (100 samples, 3 s warmup), `--release`, default features.
**Re-measured 2026-07-26.**

| Config | f32 `forward_single` | baked `forward` | Baked vs f32 |
|--------|----------------------|-----------------|--------------|
| small 4→[8]→2 | **422 ns** | 578 ns | **1.37× SLOWER** |
| medium 8→[16,8]→4 | **1.346 µs** | 2.885 µs | **2.14× SLOWER** |

An earlier run on the same machine recorded 578 ns / 735 ns (1.27×) and
1.80 µs / 4.73 µs (2.63×); spikes across a wider set of shapes span roughly
1.3–2.6× slower. The absolute numbers move with machine load, the direction
does not.

**Baked int8 is slower than f32 at batch=1, on every shape measured so far.**
The penalty grows with depth, because inter-layer fixed-point normalization
compounds.

**The current win from baking is model size (2.2–3.0×), not inference speed.**
Do not adopt `BakedModel` expecting a latency improvement — today there is none
to have.

#### The faster design exists, is measured, and is not implemented

Three competing spikes settled the question, and the answer is **not SIMD**:

- Hand-written AVX2 and plain scalar relayout landed in a dead heat
  (medium 1466.0 vs 1443.9 ns), with the scalar build's AVX2 *disabled*. LLVM
  auto-vectorizes a unit-stride scalar loop on bare SSE2 as well as intrinsics do.
- Explicit `wide::i32x8` was **slower** (0.70–0.82×): `wide` 0.7 gates
  `i32x4::Mul` on `sse4.1`, so without it the "SIMD" path degrades to four
  scalar `wrapping_mul`.
- Batch tiling got *worse* with batch size. There is no throughput story.

The actual cost is structural: `forward` evaluates the span and the basis inside
the output loop — `in_dim × out_dim` times instead of `in_dim` — including two
runtime i64 divisions by `h_q16`. Hoisting that recovers roughly parity with
f32; changing the weight layout to output-innermost (`[in, basis, out]`, giving
a unit-stride inner loop) is what converts parity into a win. Measured in the
spikes at roughly **1.5–2.8× faster than f32** depending on shape.

That is about 120 lines, no new dependency, no `unsafe`, no runtime dispatch —
and it is **not in this release**. Nothing in the shipped `BakedModel` is fast.

### How to reproduce

```bash
# Accuracy (NRMSE + worst-case on significant outputs, all orders 2..=5)
cargo test --release --test baked_parity -- --nocapture

# Latency + size
cargo bench --bench baked
```

### Related: what the grid range does to a network

Not a baked issue, but it lands in the same place — silently wrong numbers.
`grid_range` is shared by every layer while only layer 0 receives
`input_mean` / `input_std`, so a hidden layer's input is the previous layer's
raw activation clamped to the range you picked for the *inputs*. Measured on
the 256 → [64, 32] → 4 shape (`cargo test --test hidden_layer_saturation --
--nocapture`):

| `grid_range` | layer 0 saturated | layer 1 | layer 2 |
|---|---|---|---|
| `(0.0, 1.0)` | 0% | **43.6%** | **48.9%** |
| `(-1.0, 1.0)` | 0% | 0% | 0% |
| `(-3.0, 3.0)` | 0% | 0% | 0% |

A saturated input has zero derivative, so nearly half of each hidden layer
emits a constant and receives no gradient. Nothing reports this — there is no
`out_of_grid_fraction` and no drift warning. See
[ARCHITECTURE.md](ARCHITECTURE.md#design-constraints).

---

## 🔧 Test Environment

- **OS:** Windows 11
- **CPU:** AMD Ryzen (with AVX2 support)
- **GPU:** NVIDIA GeForce RTX 4070 SUPER
- **Rust:** stable, release profile. SIMD via `wide` is unconditional; unless a
  table says otherwise, `parallel` was **not** enabled and the CPU numbers are
  single-threaded.
- **Python:** 3.12 with PyTorch (CPU and CUDA)
- **GPU Backend:** wgpu 0.23 (Vulkan)

---

*ArKan benchmark suite, v0.4.0*

</details>
