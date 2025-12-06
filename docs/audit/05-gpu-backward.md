# 5. GPU Backward Pass

**Оценка:** ⭐⭐⭐⭐ (4/5)

---

## 5.1 `GpuNetwork::backward_batch`

| Аспект | Задумано | Реально |
|--------|----------|--------|
| Назначение | GPU backward pass | 🟢 Работает |
| Compute shaders | Backward pipeline | 🟢 |
| Gradient buffers | GPU-resident | 🟢 |
| Chain rule | Layer-by-layer backprop | 🟢 |

---

## 5.2 Тесты (`tests/gpu_backward_parity.rs`) ✨ v0.3.1

| Тест | Что проверяет | Оценка |
|------|---------------|--------|
| `test_backward_parity` | GPU grad == CPU grad | 🟢 Parity |
| `test_forward_training_parity` | Training mode parity | 🟢 Parity |
| `test_gpu_cpu_weight_gradient_parity_single_layer` | Weight grad (single layer) | 🟢 Прямое сравнение |
| `test_gpu_cpu_weight_gradient_parity_multi_layer` | Weight grad (3 layers) | 🟢 Multi-layer |
| `test_gpu_bias_gradient_isolated` | grad_bias[j] = Σ_b grad_output[b,j] | 🟢 Математическая |
| `test_gpu_cpu_input_gradient_parity` | Input gradient (dL/dx) | 🟢 Chain rule |
| `test_gpu_backward_batch_size_variations` | Batch 1, 7, 16, 64, 128 | 🟢 Edge cases |
| `test_gpu_numerical_gradient_check` | Central differences | 🟢 Золотой стандарт |
| `test_gpu_gradient_accumulation` | Каждый backward свежий | 🟢 Isolation |
| `test_gpu_backward_spline_order_variations` | Orders 2, 3, 4, 5 | 🟢 Config coverage |
| `test_gpu_backward_spline_order_2_regression` | Order=2 input grads non-zero | 🟢 Regression |
| `test_gpu_backward_wide_layer` | 32→256, batch=64 | 🟢 Wide layer |
| `test_gpu_backward_zero_grad_output` | Zero grad → zero output | 🟢 Edge case |

---

## 5.3 Gradient Computation

| Аспект | Задумано | Реально |
|--------|----------|--------|
| Weight gradients | dL/dW | 🟢 |
| Bias gradients | dL/db | 🟢 |
| Input gradients | dL/dx (for chain) | 🟢 **FIXED v0.3.1** |
| Spline derivatives | dB/dx in shader | 🟢 |

**BUG FIX v0.3.1:** Input gradients для single-layer сетей возвращались нулевыми.  
**Причина:** `compute_input_grad = layer_idx > 0`  
**Исправление:** `compute_input_grad = true` для всех слоёв

---

## 5.4 Parity with CPU

| Аспект | Задумано | Реально |
|--------|----------|--------|
| Output match | GPU == CPU | 🟢 EPSILON=1e-4 |
| Training convergence | Same behavior | 🟢 Оба сходятся |

**Тесты:**
| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_train_step_parity` | `tests/gpu_parity.rs` | Full train step | 🟢 Parity |
| `test_gpu_training_convergence` | `tests/coverage_tests.rs` | Оба сходятся | 🟢 E2E |

---

## 5.5 Выводы

| Аспект | Статус |
|--------|--------|
| Gradient parity | 🟢 11 тестов прямого сравнения |
| Training convergence | 🟢 E2E test |
| Numerical gradient check | 🟢 92% pass (f32 precision) |
| Batch size variations | 🟢 1, 7, 16, 64, 128 |
| Spline orders | 🟢 2, 3, 4, 5 |

**Оценка честности тестов:** ⭐⭐⭐⭐⭐ (5/5)
- ✅ Прямое сравнение градиентов GPU vs CPU
- ✅ Numerical gradient check — золотой стандарт
- ✅ Изолированный тест bias градиентов
- ✅ Input gradient тест — chain rule verification

---

## 5.6 Мертвые зоны

| Область | Риск | Причина |
|---------|------|----------|
| ~~Прямое сравнение grad GPU vs CPU~~ | ~~🔴~~ | ✅ Покрыто |
| ~~Bias gradients на GPU~~ | ~~🔴~~ | ✅ Изолированный тест |
| ~~Input gradients (dL/dx)~~ | ~~🟡~~ | ✅ Исправлено v0.3.1 |
| ~~Gradient accumulation~~ | ~~🟡~~ | ✅ Покрыто |
| ~~Backward с разными batch~~ | ~~🟡~~ | ✅ 5 размеров |
| ~~Numerical gradient check~~ | ~~🔴~~ | ✅ Central differences |
