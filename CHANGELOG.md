# Changelog

All notable changes to ArKan will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-04

### Added

#### Native GPU Training
- **`train_step_gpu_native()`** — Full GPU pipeline without CPU↔GPU weight transfers
- **`train_step_gpu_native_sgd()`** — SGD variant of native GPU training
- **`GpuAdam`** — GPU-resident Adam optimizer with moment buffers on VRAM
- **`GpuSgd`** — GPU-resident SGD optimizer with optional momentum
- **`GpuAdamConfig`**, **`GpuSgdConfig`** — Configuration structs for GPU optimizers
- **`GpuLayer::allocate_gradient_buffers()`** — Pre-allocate gradient storage on GPU
- **`GpuNetwork::prepare_native_training()`** — Initialize all layers for native training

#### Performance
- **2-5x faster training** vs hybrid GPU (no CPU↔GPU sync overhead)
- Gradients stay on GPU between backward and optimizer steps
- Adam moment vectors (m, v) stored entirely on VRAM

### Changed

- `GpuNetwork::train_step()` now uses hybrid mode (backward GPU, optimizer CPU) by default
- Native training requires explicit call to `prepare_native_training()` first

### Performance

| Method | Batch=64 | Notes |
|--------|----------|-------|
| CPU train_step | 4.77 ms | Baseline |
| Hybrid GPU (old) | ~10 ms | Forward GPU, optimizer CPU |
| **Native GPU** | **~2-3 ms** | Full GPU pipeline |

---

## [0.2.0] - 2025-12-04

### Added

#### GPU Backend (new feature: `gpu`)
- **wgpu 0.23 compute shaders** for forward/backward pass on Vulkan/DX12/Metal
- `GpuBackend`, `GpuNetwork`, `GpuWorkspace` for GPU-accelerated inference and training
- Native GPU optimizers: `GpuAdam`, `GpuSgd` with full GPU-resident training
- GPU crossover point ~batch 16: GPU wins for batch ≥16, CPU wins for single-sample
- Up to **17.9x speedup** over CPU at batch=1024

#### Safety & Error Handling
- `checked_buffer_size()` and `checked_buffer_size3()` for overflow protection
- `try_forward_batch()`, `try_train_step()` fallible variants that return `ArkanResult`
- `try_new()` for `KanLayer` with overflow checks
- `try_prepare_forward()`, `try_prepare_training()` for workspace
- `try_create_workspace()` for network
- `ArkanError::Overflow`, `ArkanError::BatchTooLarge` error variants
- `MAX_BUFFER_ELEMENTS` constant (256M elements) for safe allocation limits

#### API Improvements
- `KanConfig::validate()` now catches zero hidden dims, mismatched normalization
- Warning (not error) on zero/negative `input_std` values
- Export `checked_buffer_size`, `MAX_BUFFER_ELEMENTS` from crate root

#### Testing
- 31 new regression tests in `tests/regression_v020.rs`
- 55 GPU parity tests in `tests/gpu_parity.rs` (require `--ignored` flag)
- Total: 183 tests (103 unit + 31 regression + 49 doc-tests)

### Changed

- `Workspace` now uses `max_dim` across all layers for buffer sizing (fixes wide hidden layer bug)
- All `.expect()` in GPU workspace replaced with `.ok_or_else()` returning `ArkanResult`
- `BakedModel::forward()` now uses `unimplemented!()` instead of returning incorrect data

### Deprecated

- `BakedModel` - stub module, full implementation planned for v0.4.0

### Fixed

- **P0**: Buffer overflow in `forward_batch` with large batch sizes
- **P0**: Workspace undersized for networks with hidden layers wider than input
- **P1**: GPU workspace bind group creation could panic on missing buffers
- **P1**: `batch_size == 0` now returns error instead of undefined behavior

### Performance

- **CPU forward_batch**: ~26 µs at batch=1 (was ~61 µs), **57% improvement**
- **CPU train_step**: ~100 µs at batch=1, **8-15% improvement**
- **GPU forward**: 314 µs at batch=64 (6.2x faster than CPU)
- **Zero-allocation** inference and training paths verified

## [0.1.0] - 2024-11-XX

### Added

- Initial release
- CPU-only KAN implementation with B-spline basis functions
- `KanNetwork`, `KanLayer`, `KanConfig` core types
- `Workspace` for zero-allocation inference
- Adam and SGD optimizers
- Poker preset configuration `[21, 64, 64, 24]`
- SIMD-accelerated forward pass (AVX2/NEON)
- Rayon parallel training

---

## Roadmap

### v0.4.0 (Planned)
- [ ] `BakedModel` quantized inference (INT8)
- [ ] ONNX export
- [ ] Model pruning utilities

### v0.5.0 (Planned)
- [ ] Async GPU pipeline for overlapped compute
- [ ] Multi-GPU support
