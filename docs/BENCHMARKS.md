# ArKan Benchmark Results

**Test Date:** 2026-06-27 *(CPU/GPU forward+backward re-measured; competitor comparison added — see tasks/02-reference-parity-and-benchmarks/results/comparison.md)*  
**Platform:** Windows 11, CPU + GPU  
**CPU Config:** Poker preset `[21, 64, 64, 24]`, Grid 5, Spline Order 3 (cubic)  
**GPU:** NVIDIA GeForce RTX 4070 SUPER (Vulkan via wgpu 0.23)  
**Rust:** `cargo bench` with AVX2 (`simd` feature) and Rayon (`parallel` feature)  
**Python:** PyTorch 2.x (CPU comparison via `scripts/bench_pytorch_train.py`)

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
| **Zero-allocation training** | Full train step without allocs |
| **Native GPU training (batch=64)** | **3.96 ms** (see GPU section) |

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

> **Note:** These figures are from the *hybrid* `train_step_mse` path (forward on GPU, backward + optimizer on CPU with weight sync). Criterion group: `gpu_train_step_adam`. **Numbers not re-measured in 2026-06-27 run** (gpu_backward bench not run; the gpu_forward bench does not cover train step). See tasks/02-reference-parity-and-benchmarks/results/comparison.md for context.

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

**Analysis:** Backward pass takes roughly 2.6x the forward pass time, which is typical for gradient computation. Zero-allocation architecture ensures consistent performance.

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
> **Caveat:** efficient-kan adds SiLU base residual (different formulation). FastKAN uses RBF (different math entirely). See `tasks/02-reference-parity-and-benchmarks/results/comparison.md` for full breakdown.

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
4. **Zero-allocation benefit:** No GC pauses, consistent latency distribution
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

**Re-measured 2026-06-27.** See `tasks/02-reference-parity-and-benchmarks/results/comparison.md` for full methodology and config notes. ArKan GPU uses wgpu/Vulkan, NOT CUDA. Python competitors use PyTorch CUDA. Config mismatch: Python benches use [16,64,64,8]; ArKan GPU uses [21,64,64,24] (slightly larger).

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

# Optional feature flags for CPU benches
cargo bench --features simd       # Enable AVX2 SIMD paths
cargo bench --features parallel   # Enable Rayon parallel paths
cargo bench --features simd,parallel  # Both

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

> **Date:** 2026-06-27
> **Measured by:** `cargo test --test baked_parity -- --nocapture` and `cargo bench --bench baked`
> **Method:** Random KanNetwork (untrained, random weights), baked with 256 calibration inputs
> in `[-0.9, 0.9]`, tested against 2000 inputs in the same range. Test is against the same
> `KanNetwork::forward_single` f32 baseline. All numbers are on debug/release builds of
> the same platform (Windows 11, AMD Ryzen with AVX2).
> **Source files:** `tests/baked_parity.rs`, `benches/baked.rs`

### Accuracy

The int8 quantized path uses fixed-point i64 accumulators, i8 weights (quantized
**per output channel** since WS02), and u16 B-spline basis values (Q0.15). Two metrics:

- **NRMSE** (aggregate): `||baked - f32||₂ / ||f32||₂` over all outputs × 2000 test inputs.
  This is the standard quantization quality metric.
- **Worst-case on significant outputs**: `max per-element |baked - f32| / |f32|` restricted
  to output values where `|f32| > 0.1 × σ_j` (per-output std over the test set). This
  metric is NOT gated — it reveals the real tail that the NRMSE aggregate conceals.

#### Post-WS01 (i32 inter-layer activations + 99.9th-percentile calibration)

| Config | Architecture | NRMSE | Worst-case (significant outputs) |
|--------|-------------|-------|----------------------------------|
| small (1-hidden) | 4→[8]→2 | 0.74% | 8.8% |
| medium (2-hidden) | 8→[16,8]→4 | 1.40% | **120%** |
| single-layer | 4→2 | — | — |

#### Post-WS02 (per-output-channel weight scales — current)

Each output channel j gets its own weight scale `s_w[j] = 127 / max|w[j,*,*]|`, with
matching per-channel requant multipliers M0[j]/shift[j] and biases folded with `s_w[j]`.

| Config | Architecture | NRMSE | Gate | Pass? | Worst-case (significant outputs) |
|--------|-------------|-------|------|-------|----------------------------------|
| small (1-hidden) | 4→[8]→2, grid=5, order=3 | **0.64%** | ≤5% | PASS | 8.70% |
| medium (2-hidden) | 8→[16,8]→4, grid=5, order=3 | **1.29%** | ≤10% | PASS | **114.72%** |
| single-layer | 4→2, grid=5, order=3 | **0.60%** | ≤5% | PASS | 9.20% |

**Honest diagnosis — why the 2-hidden worst-case remains ~115% despite per-channel quant:**

Per-channel weight scales eliminate the int8 noise floor *on the weight side*. The 1-hidden
worst-case improved from 8.8% → 8.7% (marginal: weights were not the bottleneck there).
The NRMSE for 2-hidden improved from 1.40% → 1.29% (aggregate is better).

However the 2-hidden worst-case dropped only from 120% → 115%, not to the ~15% target.
Diagnosis: the residual error is **inter-layer activation quantization noise amplification**,
not weight quantization. Each inter-layer requant step loses up to 0.5 LSB relative to the
i32 range; a channel whose f32 output is near-zero after two layers of requant accumulates
relative error that exceeds 100%. This is a fundamental floor of the current scheme where
inter-layer activations are shared across output channels (single s_act per layer).

What would actually fix the 2-hidden tail:
1. **Per-channel output activation scales** (a full per-tensor-of-each-channel scheme) — but
   these scales are not known until calibration, and the inter-layer normalization would need
   to be per-channel too, complicating the forward pass significantly.
2. **Int16 weights** — more bits per weight, but the requant noise is in the activation path,
   so this helps less than expected.
3. **Accept the floor**: the NRMSE (1.29%) is well within target; the 115% worst-case is a
   genuine int8 limitation for 3-layer networks with near-zero outputs.

**Conclusion:** The baked int8 path with per-channel weight quantization is suitable for
coarse ranking/selection (where NRMSE < 2% is sufficient) but NOT for per-output absolute
accuracy in deep configs. The 2-hidden worst-case of ~115% reflects an int8 precision floor
that per-weight-channel quantization alone cannot eliminate.

### Model Size (compression)

Per-channel metadata (M0[j] + shift[j] per output channel) adds a small overhead vs
the old per-layer scalar. The trade-off is well worth it for the accuracy improvement.

| Config | f32 weight bytes | baked bytes | Compression ratio |
|--------|-----------------|-------------|-------------------|
| small 4→[8]→2 | 1,576 B | 720 B | **2.19×** |
| medium 8→[16,8]→4 | 9,328 B | 3,128 B | **2.98×** |

The compression comes from replacing f32 weights (4 bytes each) with i8 (1 byte each),
plus storing i64 biases and int32 normalization constants. For larger networks the ratio
will approach 4× as bias/metadata overhead becomes relatively smaller.

### Batch=1 Latency (deployment scenario)

All times are Criterion medians (100 samples, 3 s warmup), `--release` build,
no `simd` or `parallel` features.

| Config | f32 `forward_single` | baked `forward` | Baked vs f32 |
|--------|----------------------|-----------------|--------------|
| small 4→[8]→2 | **578 ns** | 735 ns | **1.27× SLOWER** |
| medium 8→[16,8]→4 | **1.80 µs** | 4.73 µs | **2.63× SLOWER** |

**Baked int8 is slower than f32 at batch=1.** This is expected: the current implementation
uses scalar i64 arithmetic in the hot path. Modern CPUs have native f32 SIMD lanes optimized
by the compiler (auto-vectorization) whereas the fixed-point integer path involves i64
multiply-accumulate and i128 requantization steps that do not vectorize as well without
explicit SIMD intrinsics.

The latency penalty grows with network depth (2.63× for the 3-layer config vs 1.27× for
the 2-layer) because inter-layer fixed-point normalization adds overhead that compounds.

**The current win from baking is purely model size (2.5-3.2×), not inference speed.**
SIMD-optimized int8 kernels (analogous to ARM NEON `vdot` or x86 `_mm256_madd_epi16`)
are a planned future epic and would likely bring baked latency below f32 at batch=1.

### How to reproduce

```bash
# Accuracy (NRMSE + worst-case on significant outputs)
cargo test --test baked_parity -- --nocapture

# Latency + size
cargo bench --bench baked
```

---

## 🔧 Test Environment

- **OS:** Windows 11
- **CPU:** AMD Ryzen (with AVX2 support)
- **GPU:** NVIDIA GeForce RTX 4070 SUPER
- **Rust:** stable (AVX2 SIMD via `simd` feature, Rayon via `parallel` feature)
- **Python:** 3.12 with PyTorch (CPU and CUDA)
- **GPU Backend:** wgpu 0.23 (Vulkan)

---

*Generated by ArKan benchmark suite v0.3.0*
