# Changelog

История изменений в аудите функциональности ArKan.

---

## 2025-12-07

### PyTorch cross-entropy parity tests
- ✅ **8 новых тестов** в `src/loss.rs`:
  - `test_cross_entropy_pytorch_perfect_prediction`
  - `test_cross_entropy_pytorch_confident_wrong`
  - `test_cross_entropy_pytorch_uncertain`
  - `test_cross_entropy_pytorch_multiclass`
  - `test_cross_entropy_pytorch_soft_targets`
  - `test_cross_entropy_gradient_direction`
  - `test_cross_entropy_with_mask`
  - `test_cross_entropy_numerical_stability`
- ✅ Loss Functions оценка: ⭐⭐⭐⭐ → ⭐⭐⭐⭐⭐

### Serialization tests extension
- ✅ **10 новых тестов** в `tests/coverage_tests.rs`:
  - Multi-size: minimal (2→1), deep (5 layers), wide (531KB)
  - Corrupted data: 6 invalid JSON, 5 truncated bincode, bit flips
  - Structure preservation, size scaling
- ✅ Serialization оценка: ⭐⭐⭐⭐ → ⭐⭐⭐⭐⭐
- ⚠️ **Осталось:** Versioning моделей

---

## 2025-12-06

### GPU Training tests and fixes
- ✅ **10 тестов** в `tests/gpu_training_parity.rs`:
  - Gradient clipping effect
  - Native training stability (1000 steps)
  - Adam convergence
  - Hybrid vs Native parity
- ✅ **Hybrid Adam bug fix:** `unpad_weights()` обрезает GPU padding
- ✅ GPU Training оценка: ⭐⭐⭐⭐⭐

### Memory Management tests
- ✅ **19 тестов** в `tests/memory_management.rs`:
  - Async download (5 тестов)
  - Large tensors (до 500MB)
  - Alignment (3 теста)
  - Stress tests (3 теста)
- ✅ GpuTensor оценка: ⭐⭐⭐⭐⭐

### Optimizer tests
- ✅ **9 тестов** в `tests/optimizer_correctness.rs`:
  - Adam formula numerical
  - Bias correction factors
  - Custom betas
  - GPU Adam momentum parity
- ✅ Optimizers оценка: ⭐⭐⭐⭐⭐

### VramLimit configuration
- ✅ **VramLimit enum:** Bytes, Gigabytes, Percent, Unlimited
- ✅ **Методы:** `with_max_vram(gb)`, `with_max_vram_percent()`
- ⚠️ **Ограничение:** NVIDIA возвращает `u64::MAX` для max_buffer_size

### Spline edge cases
- ✅ **18 тестов** в `tests/spline_edge_cases.rs`:
  - grid_size: 2, 32, 64
  - spline_order: 5, 6
  - extreme x: 1e-30, 1e30
  - MAX_GRID_SIZE = 64

### SIMD and numerical correctness
- ✅ **19 тестов** в `tests/forward_correctness.rs`:
  - 170 SIMD комбинаций
  - Wide layers до 1024
- ✅ CPU Forward оценка: ⭐⭐⭐⭐⭐

### Parallel backward
- ✅ **11 тестов** в `tests/backward_correctness.rs`:
  - Sequential vs parallel parity
  - Wide layers до 1024
  - Spline orders 2-6
- ✅ CPU Backward оценка: ⭐⭐⭐⭐⭐

### Training options
- ✅ **11 тестов** в `tests/training_options.rs`:
  - Gradient clipping effect
  - Weight decay
  - lr=0 edge case
  - Large batch до 4096

### Async forward
- ✅ **`forward_batch_async()`** с GpuForwardHandle
- ✅ **4 теста:** parity, try_recv, multiple submits

---

## 2025-12-05

### Loss Functions extension
- ✅ **KAN-specific regularization:**
  - `l1_sparsity_loss`, `entropy_regularization`, `smoothness_penalty`
  - `kan_combined_loss`
- ✅ **Physics-informed:** `pde_residual_loss`, `r_squared`
- ✅ **34 unit теста**

### Bug fixes
- ✅ **Serialization knots bug** — Custom Deserialize для KanLayer
- ✅ **Gradient check** — Multi-epsilon метод, 95% pass rate

### Initial coverage tests
- ✅ `forward_batch_parallel` parity
- ✅ GPU forward parity
- ✅ GPU training convergence
- ✅ Multi-layer gradient check
- ✅ Serialization roundtrip

---

## 2025-01-20

### Optimizer Module v2.0
- ✅ **LBFGS полная реализация:**
  - Strong Wolfe line search
  - Backtracking fallback
  - Two-loop recursion
- ✅ **SGD Nesterov momentum**
- ✅ **ParamGroup структура**
- ✅ **Workspace zero_grad()**

### Optimizer Module v2.1
- ✅ **trait Optimizer** — Unified API
- ✅ **Thread Safety** — Send + Sync
- ✅ **Versioning** — bump_version()
- ✅ **SafetyConfig** — NaN handling, AMP placeholder

### PyTorch reference tests
- ✅ **15 тестов** в `tests/pytorch_reference.rs`:
  - Adam, AdamW, SGD, Nesterov, LBFGS

### GPU Backward bug fix
- ✅ **BUG FIX:** `compute_input_grad = layer_idx > 0` → `= true`
- ✅ **11 тестов** в `tests/gpu_backward_parity.rs`
- ✅ GPU Backward оценка: ⭐⭐⭐⭐⭐

---

## Initial Audit (2025-12-05)

Первоначальный аудит функционала проекта ArKan.
