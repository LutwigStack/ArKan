# 15. Test Coverage Summary

---

## Integration Tests (`tests/`)

| Файл | Назначение | Статус |
|------|------------|--------|
| `gpu_parity.rs` | GPU == CPU output | 🟢 |
| `gpu_training_parity.rs` | GPU training parity | 🟢 |
| `gpu_backward_parity.rs` | GPU gradient parity | 🟢 |
| `gpu_memory_safety.rs` | GPU memory safety | 🟢 |
| `gradient_check.rs` | Numerical vs Analytical | 🟢 |
| `spline_parity.rs` | ArKan == SciPy | 🟢 |
| `spline_edge_cases.rs` | B-Spline edge cases | 🟢 |
| `spline_derivative_debug.rs` | Derivative accuracy | 🟢 |
| `forward_correctness.rs` | SIMD + численная | 🟢 |
| `backward_correctness.rs` | Parallel backward | 🟢 |
| `training_options.rs` | TrainOptions effects | 🟢 |
| `optimizer_correctness.rs` | Adam numerical | 🟢 |
| `pytorch_reference.rs` | PyTorch parity | 🟢 |
| `memory_management.rs` | GPU memory | 🟢 |
| `coverage_tests.rs` | Новое покрытие | 🟢 |
| `regression_v020.rs` | Overflow protection | 🟢 |
| `debug_span.rs` | Span edge cases | 🟢 |

---

## Unit Tests (in `src/`)

| Модуль | Тестов | Покрытие |
|--------|--------|----------|
| `spline.rs` | 4 | 🟢 |
| `optimizer.rs` | 20+ | 🟢 |
| `network.rs` | 14 | 🟢 |
| `layer.rs` | 8+ | 🟢 |
| `buffer.rs` | 10+ | 🟢 |
| `config.rs` | 7+ | 🟢 |
| `loss.rs` | 40+ | 🟢 |
| `baked.rs` | 2 | 🟡 |

---

## Coverage Status

| Область | Статус |
|---------|--------|
| B-Spline computation | 🟢 Полное (scipy parity) |
| CPU forward | 🟢 Полное (170 SIMD комбинаций) |
| CPU backward | 🟢 Через gradient check |
| CPU training | 🟢 Convergence + options |
| GPU forward | 🟢 Parity + async |
| GPU backward | 🟢 Parity + gradient check |
| GPU training | 🟢 Native + Hybrid (10 тестов) |
| Optimizers | 🟢 PyTorch parity |
| Memory | 🟢 Async + large tensors |
| Serialization | 🟢 Multi-size + corrupted |
| Loss Functions | 🟢 PyTorch parity |

---

## Gradient Check Notes

**95% pass rate** — это теоретический максимум для f32.

Неудавшиеся 5% имеют |grad| < 4×10⁻⁵, что ниже минимального детектируемого градиента |grad|_min ≈ 6×10⁻⁵.

См. комментарий в `tests/coverage_tests.rs::test_gradient_check_deep_network`.
