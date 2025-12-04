# ArKan Benchmark Results

**Test Date:** December 4, 2025  
**Platform:** Windows, CPU + GPU  
**CPU Config:** Poker preset `[21, 64, 64, 24]`, Grid 5, Spline Order 3 (cubic)  
**GPU:** NVIDIA GeForce RTX 4070 SUPER (Vulkan via wgpu)  
**Rust:** `cargo bench` with AVX2/Rayon enabled

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Single inference latency (P50)** | **30.5 µs** |
| **Single inference throughput** | **~33,000 inferences/sec** |
| **vs PyTorch (batch=1)** | **32x faster** |
| **Memory footprint** | 218.6 KB (weights only) |
| **Zero-allocation training** | ✅ Full train step without allocs |

---

## 🖥️ GPU Backend Performance

> **Note:** GPU benchmarks require the `gpu` feature flag and a compatible GPU.
> Run with:
> ```bash
> # Windows PowerShell
> $env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_forward --bench gpu_backward --features gpu
>
> # Linux/macOS
> ARKAN_GPU_BENCH=1 cargo bench --bench gpu_forward --bench gpu_backward --features gpu
> ```

### GPU vs CPU Forward Pass

**GPU:** NVIDIA GeForce RTX 4070 SUPER (Vulkan)

| Batch | CPU | GPU | Speedup | Notes |
|-------|-----|-----|---------|-------|
| 1 | 30.5 µs | 294 µs | 0.1x | CPU wins (GPU dispatch overhead) |
| 8 | 246 µs | 311 µs | 0.8x | Near crossover |
| 16 | 492 µs | 310 µs | 1.6x | GPU starts winning |
| 64 | 1.95 ms | 314 µs | 6.2x | GPU wins decisively |
| 256 | 7.29 ms | 541 µs | 13.5x | GPU advantage grows |
| 1024 | 29.2 ms | 1.63 ms | 17.9x | Best GPU efficiency |

**Key Insight:** GPU crossover point is around batch size 8-16. For single-sample latency-critical applications (e.g., real-time MCTS), CPU is preferred.

### GPU Train Step (Adam optimizer)

| Batch | Time | Throughput |
|-------|------|------------|
| 1 | 15.2 ms | 1.4 K elem/s |
| 8 | 9.4 ms | 17.8 K elem/s |
| 16 | 8.4 ms | 39.9 K elem/s |
| 64 | 9.6 ms | 139.8 K elem/s |
| 256 | 11.1 ms | 485.8 K elem/s |

### GPU Train Step (SGD optimizer)

| Batch | Time | Throughput |
|-------|------|------------|
| 1 | 11.8 ms | 1.8 K elem/s |
| 8 | 10.9 ms | 15.4 K elem/s |
| 16 | 10.7 ms | 31.4 K elem/s |
| 64 | 11.7 ms | 114.5 K elem/s |
| 256 | 12.4 ms | 432.5 K elem/s |

### GPU Train Options Impact (batch=64)

| Option | Time | Overhead |
|--------|------|----------|
| No options | 10.7 ms | baseline |
| Grad clip (1.0) | 12.3 ms | +15% |
| Weight decay (0.01) | 12.6 ms | +18% |
| Both | 11.8 ms | +10% |

### GPU Softmax Performance

| Batch | Forward + Softmax | Forward Only | Softmax Overhead |
|-------|-------------------|--------------|------------------|
| 1 | 346 µs | 289 µs | +20% |
| 64 | 337 µs | 306 µs | +10% |
| 256 | 594 µs | 507 µs | +17% |
| 1024 | 1.69 ms | 1.62 ms | +4% |

### GPU Limitations (wgpu 0.23)

- **No DeviceLost event:** wgpu 0.23 does not propagate `DeviceLost` errors. GPU crashes may appear as hangs.
- **Memory limits:** MAX_VRAM_ALLOC = 2GB per buffer. Use `BatchTooLarge` error for early rejection.
- **Backend selection:** Use `WgpuOptions::compute()` for best compute performance settings.

### Native GPU Training (v0.3.0+)

ArKan supports **fully native GPU training** where forward pass, backward pass, and optimizer updates all run on GPU without CPU↔GPU weight transfers.

**API Usage:**
```rust
use arkan::gpu::{GpuAdam, GpuAdamConfig, GpuSgd, GpuSgdConfig};

// Create network and optimizer
let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)?;
let layer_sizes = gpu_network.layer_param_sizes();
let mut optimizer = GpuAdam::new(device, queue, &layer_sizes, GpuAdamConfig::with_lr(0.001));

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
| Min | 29.7 µs |
| **P50 (median)** | **30.5 µs** |
| P90 | 31.6 µs |
| P99 | 38.2 µs |
| P999 | 52.0 µs |
| Max | 78.4 µs |

**Throughput at P50:** ~33,000 inferences/second

### forward_single vs forward_batch(1)

| Method | Time | Notes |
|--------|------|-------|
| `forward_single` | ~15 µs | Optimized single-sample path |
| `forward_batch(1)` | 30.5 µs | Batch overhead visible |

**Recommendation:** Use `forward_single` for real-time play (~2x faster), `forward_batch` for training.

---

## 🔄 Training Performance

### Backward Pass Overhead (batch=64)

| Operation | Time | Overhead vs Forward |
|-----------|------|---------------------|
| forward_only | 1.95 ms | baseline |
| forward_training | 1.95 ms | ~0% (buffer prep is free) |
| **full_train_step** | **4.77 ms** | **+145%** |

**Analysis:** Backward pass takes roughly 2.4x the forward pass time, which is typical for gradient computation. Zero-allocation architecture ensures consistent performance.

### Training Options Impact (batch=64)

| Option | Time | Overhead |
|--------|------|----------|
| No options | 4.77 ms | baseline |
| Gradient clipping (1.0) | 4.59 ms | -3.8% (noise) |
| Weight decay (0.01) | 4.59 ms | -3.8% (noise) |
| Both | 4.67 ms | -2.1% (noise) |

**Conclusion:** Training options have negligible performance impact.

---

## 📐 Architecture Scaling

### Latency (batch=1)

| Architecture | Params | Memory | Latency | Throughput |
|--------------|--------|--------|---------|------------|
| tiny `[3,10,1]` | 331 | 1.3 KB | **428 ns** | 7.0 M elem/s |
| medium `[10,64,64,10]` | 43K | 168.5 KB | 24.3 µs | 411 K elem/s |
| **poker `[21,64,64,24]`** | **56K** | **218.6 KB** | **30.5 µs** | **688 K elem/s** |
| large `[32,128,128,128,32]` | 328K | 1.28 MB | 164 µs | 195 K elem/s |
| wide `[21,256,24]` | 92K | 361 KB | 49.3 µs | 426 K elem/s |
| deep `[21,32,32,32,32,32,24]` | 44K | 174 KB | 24.5 µs | 857 K elem/s |

### Throughput (batch=64)

| Architecture | Forward | Train Step |
|--------------|---------|------------|
| tiny | 23.4 µs (8.2 M elem/s) | 61.5 µs (3.1 M elem/s) |
| medium | 1.47 ms (434 K elem/s) | 3.89 ms (165 K elem/s) |
| **poker** | **1.95 ms (689 K elem/s)** | **4.95 ms (271 K elem/s)** |
| large | 10.9 ms (188 K elem/s) | 27.6 ms (74 K elem/s) |
| wide | 3.18 ms (422 K elem/s) | 7.78 ms (173 K elem/s) |
| deep | 1.63 ms (826 K elem/s) | 4.11 ms (327 K elem/s) |

**Insight:** Deep narrow networks (5x32) are faster than wide shallow ones (1x256) for the same parameter count.

---

## 📏 Spline Configuration Analysis

### Spline Order Impact (grid=5, batch=64)

| Order | Name | Basis Size | Time | Throughput | Params |
|-------|------|------------|------|------------|--------|
| 1 | linear | 6 | 1.18 ms | **1.14 M elem/s** | 42K |
| 2 | quadratic | 7 | 1.62 ms | 828 K elem/s | 49K |
| **3** | **cubic** | **8** | **2.05 ms** | **656 K elem/s** | **56K** |
| 4 | quartic | 9 | 2.49 ms | 539 K elem/s | 63K |
| 5 | quintic | 10 | 2.83 ms | 475 K elem/s | 70K |

**Trade-off:** Each order increase adds ~400 µs latency but improves function smoothness.

### Grid Size Impact (order=3 cubic, batch=64)

| Grid | Basis Size | Time | Throughput | Params |
|------|------------|------|------------|--------|
| 3 | 6 | 2.02 ms | 665 K elem/s | 42K |
| **5** | **8** | **2.01 ms** | **669 K elem/s** | **56K** |
| 8 | 11 | 2.12 ms | 633 K elem/s | 77K |
| 12 | 15 | 2.15 ms | 624 K elem/s | 105K |
| 16 | 19 | 2.11 ms | 636 K elem/s | 133K |

**Insight:** Grid size has minimal impact on forward speed due to local spline evaluation (only `order+1` basis functions computed). Choose grid size based on required expressiveness.

### Recommended Configurations

| Use Case | Grid | Order | Params | Latency (batch=1) |
|----------|------|-------|--------|-------------------|
| **Fast inference** | 3 | 2 | 35K | ~25 µs |
| **Balanced (default)** | 5 | 3 | 56K | ~28 µs |
| **High accuracy** | 8 | 3 | 77K | ~29 µs |
| **Smooth functions** | 5 | 5 | 70K | ~35 µs |

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
| With reuse | 1.83 ms |
| Without reuse (alloc each time) | 1.83 ms |

**Result:** Workspace allocation is fast (~0% overhead), but reuse is still recommended for hot paths.

---

## 🏎️ Memory Bandwidth (CPU)

### Achieved Bandwidth

| Batch | Est. Memory | Time | Bandwidth |
|-------|-------------|------|-----------|
| 1 | 222 KB | 30.5 µs | 6.9 GB/s |
| 16 | 269 KB | 453 µs | 580 MB/s |
| 64 | 418 KB | 1.95 ms | 209 MB/s |
| 256 | 1012 KB | 7.29 ms | 136 MB/s |
| 1024 | 3391 KB | 29.2 ms | 113 MB/s |

**Analysis:** Small batches achieve higher bandwidth due to cache locality. Large batches are compute-bound rather than memory-bound.

---

## ⚡ ArKan CPU vs PyTorch CPU Comparison

### Forward Pass (Inference)

| Batch | ArKan | PyTorch | **Speedup** |
|-------|-------|---------|-------------|
| **1** | **30.5 µs** | 990 µs | **32x** |
| 16 | 454 µs | 1.67 ms | **3.7x** |
| 64 | 1.95 ms | 3.27 ms | **1.7x** |
| 256 | 7.29 ms | 9.65 ms | **1.3x** |

### Training Step (Forward + Backward + SGD)

| Batch | ArKan | PyTorch | Speedup |
|-------|-------|---------|--------|
| 1 | 106 µs | 3.17 ms | **30x** |
| 16 | 1.19 ms | 3.48 ms | **2.9x** |
| 64 | 4.77 ms | 5.99 ms | **1.3x** |
| 256 | 19.5 ms | 19.9 ms | **1.02x** |

### Key Takeaways

1. **Low-latency dominance:** ArKan is 30-32x faster for single-sample inference
2. **Training competitive:** ArKan maintains advantage across all batch sizes (1.02x-30x)
3. **Zero-allocation benefit:** Consistent performance without GC pauses or jitter
4. **Large batch improvement:** After optimization, ArKan now beats PyTorch even at batch=256

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
| 1 | 30.5 µs | 294 µs | CPU | 9.6x CPU |
| 8 | 246 µs | 311 µs | CPU | 1.3x CPU |
| 16 | 492 µs | 310 µs | GPU | 1.6x GPU |
| 64 | 1.95 ms | 314 µs | GPU | 6.2x GPU |
| 256 | 7.29 ms | 541 µs | GPU | 13.5x GPU |
| 1024 | 29.2 ms | 1.63 ms | GPU | 17.9x GPU |

**Key Insight:** GPU crossover point is around batch size 16. For single-sample real-time inference (poker MCTS), CPU is better. For batch training, GPU excels.

### GPU Architecture Scaling

#### Latency (batch=1, GPU)

| Architecture | Params | GPU Latency | CPU Latency | GPU Overhead |
|--------------|--------|-------------|-------------|--------------|
| tiny `[3,10,1]` | 331 | 225 µs | 428 ns | 526x |
| medium `[10,64,64,10]` | 43K | 276 µs | 24.3 µs | 11.4x |
| **poker `[21,64,64,24]`** | **56K** | **294 µs** | **30.5 µs** | **9.6x** |
| large `[32,128,128,128,32]` | 328K | 388 µs | 164 µs | 2.4x |
| wide `[21,256,24]` | 92K | 260 µs | 49.3 µs | 5.3x |
| deep `[21,32,32,32,32,32,24]` | 44K | 326 µs | 24.5 µs | 13.3x |

**CPU wins for all architectures at batch=1** due to GPU dispatch overhead.

#### Throughput (batch=64, GPU)

| Architecture | GPU Time | CPU Time | GPU Speedup |
|--------------|----------|----------|-------------|
| tiny | 222 µs | 23.4 µs | 0.11x |
| medium | 286 µs | 1.47 ms | 5.1x |
| **poker** | **314 µs** | **1.95 ms** | **6.2x** |
| large | 453 µs | 10.9 ms | 24.1x |
| wide | 278 µs | 3.18 ms | 11.4x |
| deep | 344 µs | 1.63 ms | 4.7x |

### GPU Spline Configuration Impact

#### Spline Order (grid=5, batch=64)

| Order | Name | GPU Time | CPU Time | GPU Speedup |
|-------|------|----------|----------|-------------|
| 1 | linear | 306 µs | 1.18 ms | 3.9x |
| 2 | quadratic | 309 µs | 1.62 ms | 5.2x |
| **3** | **cubic** | **314 µs** | **2.05 ms** | **6.5x** |
| 4 | quartic | 352 µs | 2.49 ms | 7.1x |
| 5 | quintic | 381 µs | 2.83 ms | 7.4x |

**GPU advantage increases with spline order** due to more parallelizable B-spline computation.

#### Grid Size (order=3, batch=64)

| Grid | Basis Size | GPU Time | CPU Time | GPU Speedup |
|------|------------|----------|----------|-------------|
| 3 | 6 | 299 µs | 2.02 ms | 6.8x |
| **5** | **8** | **314 µs** | **2.01 ms** | **6.4x** |
| 8 | 11 | 337 µs | 2.12 ms | 6.3x |
| 12 | 15 | 386 µs | 2.15 ms | 5.6x |
| 16 | 19 | 434 µs | 2.11 ms | 4.9x |

### GPU Train Step Performance

| Optimizer | Batch=64 | Batch=256 | Notes |
|-----------|----------|-----------|-------|
| Adam | 9.6 ms | 11.1 ms | Full Adam with moment update |
| SGD | 11.7 ms | 12.4 ms | Including weight sync |
| + Grad Clipping | +1.5 ms | +2 ms | Small overhead |
| + Weight Decay | +1.9 ms | +2 ms | Small overhead |

### GPU Latency Distribution (batch=1)

| Percentile | GPU | CPU |
|------------|-----|-----|
| Min | 252 µs | 29.7 µs |
| P50 | 282 µs | 30.5 µs |
| P90 | 305 µs | 31.6 µs |
| P99 | 352 µs | 38.2 µs |
| P999 | 432 µs | 52.0 µs |
| Max | 587 µs | 78.4 µs |

**CPU has lower and more consistent latency** for single samples due to GPU dispatch overhead.

### GPU Memory Bandwidth

| Batch | Est. GPU Memory | Time | Bandwidth |
|-------|-----------------|------|-----------|
| 1 | ~250 KB | 294 µs | 0.85 GB/s |
| 64 | ~500 KB | 314 µs | 1.6 GB/s |
| 256 | ~1.2 MB | 541 µs | 2.2 GB/s |
| 1024 | ~4 MB | 1.63 ms | 2.5 GB/s |

### CPU vs GPU Training (batch=64)

| Configuration | CPU | GPU | Winner |
|---------------|-----|-----|--------|
| Forward only | 1.95 ms | 314 µs | GPU 6.2x |
| Train (Adam) | 4.77 ms | 9.6 ms | CPU 2.0x |
| Train (SGD) | 4.50 ms | 11.7 ms | CPU 2.6x |

**Note:** GPU training is currently slower than CPU due to weight sync overhead between GPU and CPU optimizers. Future versions will implement GPU-native optimizers.

### GPU vs PyTorch GPU Comparison

Compare ArKan GPU with PyTorch-based KAN implementations (CUDA):

| Implementation | Forward (batch=64) | Train Step | Notes |
|----------------|-------------------|------------|-------|
| **ArKan GPU (wgpu)** | **314 µs** | **9.6 ms** | WebGPU (Vulkan/DX12/Metal) |
| efficient-kan (PyTorch CUDA) | ~1.5 ms | ~5 ms | Native B-spline |
| fast-kan (PyTorch CUDA) | ~0.5 ms | ~3 ms | RBF approximation |
| ArKan-style (PyTorch CUDA) | ~2 ms | ~8 ms | Custom B-spline |

**Note:** PyTorch CUDA has highly optimized kernels. ArKan wgpu targets cross-platform compatibility.

To run PyTorch GPU comparison:
```bash
pip install torch efficient-kan
pip install git+https://github.com/ZiyaoLi/fast-kan.git
python scripts/bench_pytorch_gpu.py
```

---

## 🎮 Poker Solver Workload

Simulating real poker solver usage patterns:

| Scenario | Time | Notes |
|----------|------|-------|
| Pure inference | 14.4 µs | Just forward_single |
| + Light processing | 14.4 µs | + sum outputs |
| + Softmax | 14.4 µs | + softmax on strategy |

**Conclusion:** Post-processing overhead is negligible compared to network forward pass.

---

## 📋 How to Run Benchmarks

```bash
# All CPU benchmarks
cargo bench

# Specific CPU benchmark suites
cargo bench --bench forward      # Original forward/train benchmarks
cargo bench --bench backward     # Backward pass analysis
cargo bench --bench scaling      # Architecture scaling
cargo bench --bench spline_config # Spline order/grid impact
cargo bench --bench memory       # Memory bandwidth analysis
cargo bench --bench optimizer    # Training options overhead
cargo bench --bench latency      # Single-sample latency distribution

# GPU benchmarks (require ARKAN_GPU_BENCH=1 env var for CI-safety)
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
# All GPU parity tests (ignored by default)
cargo test --features gpu --test gpu_parity -- --ignored

# Specific GPU test
cargo test --features gpu --test gpu_parity test_forward_single_parity -- --ignored
```

### CI Smoke Test

```bash
# Quick verification script (CPU + optional GPU)
cargo test                                    # CPU tests
cargo test --features gpu -- --ignored        # GPU tests (if GPU available)
cargo bench --bench forward -- --noplot       # Quick CPU benchmark
```

---

## 🔧 Test Environment

- **OS:** Windows
- **Rust:** stable (with AVX2 support)
- **Python:** 3.12 with PyTorch (CPU and CUDA)
- **GPU Backend:** wgpu 0.23 (Vulkan/DX12)

---

*Generated by ArKan benchmark suite v0.2.0*
