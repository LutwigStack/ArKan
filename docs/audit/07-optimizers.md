# 7. Optimizers

**Оценка:** ⭐⭐⭐⭐ (4/5)

---

## 7.1 `Adam` (CPU) — v2.1

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Momentum (β1, β2) | ✓ | 🟢 |
| Bias correction | ✓ | 🟢 |
| Weight decay | ✓ | 🟢 |
| Gradient clipping | В TrainOptions | 🟢 |
| Thread Safety | Send + Sync | 🟢 |
| Versioning | bump_version() | 🟢 |
| NaN Handling | fail_on_nan / skip_step_on_nan | 🟢 |

**Тесты CPU Adam:**
| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_adam_state_creation` | `src/optimizer.rs` | Создание momentum буферов | 🟢 Базовый |
| `test_adam_update` | `src/optimizer.rs` | Вес уменьшается при +grad | 🟢 Функциональный |
| `test_adam_formula_numerical` | `tests/optimizer_correctness.rs` | Ручной reference | 🟢 Математический |
| `test_adam_bias_correction_factors` | `tests/optimizer_correctness.rs` | (1-β^t) корректно | 🟢 Математический |
| `test_adam_convergence_quadratic` | `tests/optimizer_correctness.rs` | Сходимость на f(x)=x² | 🟢 Convergence |
| `test_adam_weight_decay_formula` | `tests/optimizer_correctness.rs` | AdamW decoupled | 🟢 Математический |
| `test_adam_custom_betas` | `tests/optimizer_correctness.rs` | β1=0.5, β2=0.9999 | 🟢 Конфигурации |
| `test_adam_momentum_accumulation` | `tests/optimizer_correctness.rs` | m, v накапливают | 🟢 Состояние |

**PyTorch Reference Tests:**
| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_pytorch_adam_default_quadratic` | `tests/pytorch_reference.rs` | vs PyTorch (tol=1e-5) | 🟢 PyTorch parity |
| `test_pytorch_adam_with_weight_decay` | `tests/pytorch_reference.rs` | L2 decay | 🟢 PyTorch parity |
| `test_pytorch_adamw_decoupled_weight_decay` | `tests/pytorch_reference.rs` | AdamW formula | 🟢 PyTorch parity |
| `test_pytorch_adam_custom_betas` | `tests/pytorch_reference.rs` | β1=0.5, β2=0.9999 | 🟢 PyTorch parity |

---

## 7.2 `SGD` (CPU) — v2.0

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Momentum | ✓ | 🟢 |
| Weight decay | ✓ | 🟢 |
| Nesterov momentum | Look-ahead | 🟢 **РЕАЛИЗОВАНО** |
| Thread Safety | Send + Sync | 🟢 |

**Тесты SGD:**
| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_sgd_new_api` | `src/optimizer.rs` | SGDConfig::with_momentum() | 🟢 API |
| `test_sgd_nesterov` | `src/optimizer.rs` | Nesterov formula | 🟢 Algorithm |
| `test_sgd_nesterov_vs_standard` | `src/optimizer.rs` | Nesterov more aggressive | 🟢 Comparison |

**PyTorch Reference Tests:**
| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_pytorch_sgd_no_momentum` | `tests/pytorch_reference.rs` | SGD basic | 🟢 PyTorch parity |
| `test_pytorch_sgd_with_momentum` | `tests/pytorch_reference.rs` | v = μ*v + g | 🟢 PyTorch parity |
| `test_pytorch_sgd_nesterov` | `tests/pytorch_reference.rs` | θ -= lr*(μ*v + g) | 🟢 PyTorch parity |
| `test_pytorch_sgd_with_weight_decay` | `tests/pytorch_reference.rs` | L2 in gradient | 🟢 PyTorch parity |

---

## 7.3 `LBFGS` — v2.0

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Two-loop recursion | L-BFGS algorithm | 🟢 **РЕАЛИЗОВАНО** |
| Strong Wolfe line search | C1=1e-4, C2=0.9 | 🟢 **РЕАЛИЗОВАНО** |
| Backtracking fallback | Armijo, ρ=0.5 | 🟢 **РЕАЛИЗОВАНО** |
| NoLineSearch | Fixed step | 🟢 **РЕАЛИЗОВАНО** |
| Rollback | Restore on failed | 🟢 **РЕАЛИЗОВАНО** |

**Тесты LBFGS:**
| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_lbfgs_creation` | `src/optimizer.rs` | LBFGSConfig defaults | 🟢 API |
| `test_lbfgs_two_loop_recursion` | `src/optimizer.rs` | Steepest descent | 🟢 Algorithm |
| `test_lbfgs_pack_unpack` | `src/optimizer.rs` | flatten/restore roundtrip | 🟢 Utility |
| `test_pytorch_lbfgs_quadratic_convergence` | `tests/pytorch_reference.rs` | Quadratic loss | 🟢 Convergence |

---

## 7.4 `GpuAdam`

| Аспект | Задумано | Реально |
|--------|----------|---------|
| GPU compute | ✓ | 🟢 |
| Momentum states | GPU buffers | 🟢 |
| Bias correction | ✓ | 🟢 |
| Gradient clipping | ✓ | 🟢 |

**Тесты GpuAdam:**
| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_gpu_adam_vs_cpu_adam_single_step` | `tests/optimizer_correctness.rs` | Hybrid vs Native | 🟢 Parity |
| `test_gpu_adam_momentum_parity` | `tests/optimizer_correctness.rs` | 10 steps parity | 🟢 Parity |
| `test_gpu_adam_custom_betas` | `tests/optimizer_correctness.rs` | Custom configs | 🟢 Конфигурации |

---

## 7.5 LR Schedulers

| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_step_lr` | `src/optimizer.rs` | StepLR decay | 🟢 Функциональный |
| `test_cosine_lr` | `src/optimizer.rs` | CosineAnnealing | 🟢 Функциональный |

---

## 7.6 Выводы

| Аспект | Статус |
|--------|--------|
| CPU Adam | 🟢 Полное — numerical, bias correction, custom betas |
| CPU SGD | 🟢 Полное — momentum, Nesterov, weight decay |
| LBFGS | 🟢 Полное — two-loop, Strong Wolfe, backtracking |
| GPU Adam | 🟢 Полное — hybrid/native parity |
| Schedulers | 🟢 Базовое |

**Оценка честности тестов:** ⭐⭐⭐⭐⭐ (5/5)
- ✅ PyTorch reference тесты
- ✅ Numerical formula verification
- ✅ Bias correction factors checked
- ✅ GPU Adam momentum parity

---

## 7.7 Мертвые зоны

| Область | Риск | Причина |
|---------|------|----------|
| ~~GpuAdam momentum parity~~ | ~~🟡~~ | ✅ **ИСПРАВЛЕНО** |
| ~~Bias correction формула~~ | ~~🟡~~ | ✅ **ИСПРАВЛЕНО** |
| ~~β1, β2 нестандартные~~ | ~~🟡~~ | ✅ **ИСПРАВЛЕНО** |
| ~~LBFGS line search~~ | ~~🔴~~ | ✅ **ИСПРАВЛЕНО v2.0** |
| ~~Nesterov momentum~~ | ~~🟡~~ | ✅ **ИСПРАВЛЕНО v2.0** |
| LBFGS Rosenbrock test | 🟡 Средний | TODO |
