# 2. CPU Backward Pass

**Оценка:** ⭐⭐⭐⭐⭐ (5/5)

---

## 2.1 `KanLayer::backward` (Sequential)

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | Вычисление градиентов | 🟢 Работает |
| Параллелизм | Последовательный | 🟢 Для малых batch |
| Gradient accumulation | Накопление по batch | 🟢 Работает |
| Chain rule | dL/dW через backprop | 🟢 Работает |

---

## 2.2 `KanLayer::backward_parallel` (Parallel) ✨ v0.3.0

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | Параллельное вычисление градиентов | 🟢 Работает |
| Алгоритм | Thread-local gradients + reduce | 🟢 Работает |
| Автовыбор | `batch >= multithreading_threshold` → parallel | 🟢 Интегрировано |
| Memory overhead | O(threads × params) | 🟢 Приемлемо |
| Parity с sequential | До 5e-5 разница | 🟢 Протестировано |

---

## 2.3 Тесты через gradient check

| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_gradient_check_simple_network` | `tests/gradient_check.rs` | Numerical vs Ana, простая сеть | 🟢 Базовый |
| `test_gradient_check_single_hidden` | `tests/gradient_check.rs` | 1 hidden layer | 🟢 Базовый |
| `test_gradient_check_multi_layer` | `tests/gradient_check.rs` | 3 hidden layers | 🟢 Полный |
| `test_gradient_check_deep_network` | `tests/coverage_tests.rs` | 4 layers, 95% pass | 🟢 Регрессионный |
| `test_gradcheck_single_layer` | `src/network.rs` | Маленькая сеть | 🟢 Базовый |
| `test_gradient_zero_at_optimum` | `tests/gradient_check.rs` | grad≈0 при target==output | 🟢 Математический |
| `test_gradient_descent_direction` | `tests/gradient_check.rs` | grad указывает на убывание | 🟢 Математический |

---

## 2.4 Тесты `backward_parallel` (parity)

| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_backward_vs_parallel_parity_small_batch` | `tests/backward_correctness.rs` | batch=16 | 🟢 Базовый |
| `test_backward_vs_parallel_parity_large_batch` | `tests/backward_correctness.rs` | batch=256 | 🟢 Масштабируемость |
| `test_backward_parallel_wide_layer_1024` | `tests/backward_correctness.rs` | Wide output (32→1024) | 🟢 Wide layer |
| `test_backward_parallel_wide_input_1024` | `tests/backward_correctness.rs` | Wide input (1024→16) | 🟢 Wide layer |
| `test_backward_parallel_spline_orders` | `tests/backward_correctness.rs` | Orders 2,3,4,5,6 | 🟢 Config coverage |
| `test_backward_parallel_batch_size_1` | `tests/backward_correctness.rs` | batch=1 | 🟢 Edge case |
| `test_backward_parallel_zero_grad_output` | `tests/backward_correctness.rs` | Zero grad → zero result | 🟢 Edge case |
| `test_backward_parallel_sparse_grad_output` | `tests/backward_correctness.rs` | Masked/sparse gradients | 🟢 Masking |
| `test_backward_parallel_deterministic` | `tests/backward_correctness.rs` | Determinism check | 🟢 Reproducibility |
| `test_network_train_step_uses_parallel` | `tests/backward_correctness.rs` | Network integration | 🟢 Integration |
| `test_network_train_step_uses_sequential` | `tests/backward_correctness.rs` | Network integration | 🟢 Integration |

---

## 2.5 Тесты по spline order

| Тест | Файл | Что проверяет | Оценка |
|------|------|---------------|--------|
| `test_gradient_check_spline_order_2` | `tests/gradient_check.rs` | order=2 градиенты | 🟢 Config |
| `test_gradient_check_spline_order_3` | `tests/gradient_check.rs` | order=3 градиенты | 🟢 Config |
| `test_gradient_check_spline_order_4` | `tests/gradient_check.rs` | order=4 градиенты | 🟢 Config |

---

## 2.6 Выводы

| Аспект | Статус |
|--------|--------|
| Gradient correctness | 🟢 Численная проверка |
| Multi-layer flow | 🟢 До 4 слоёв |
| Spline orders | 🟢 2, 3, 4, 5, 6 |
| Sequential/Parallel parity | 🟢 11 тестов, до 5e-5 |
| Wide layers (1024) | 🟢 Протестировано |
| Network integration | 🟢 Auto-select по threshold |

**Оценка честности тестов:** ⭐⭐⭐⭐⭐ (5/5)
- ✅ Numerical gradient check — ловит большинство багов
- ✅ Parity тесты sequential vs parallel — 11 тестов
- ✅ Wide layer coverage до 1024 нейронов
- ✅ Spline orders 2-6 покрыты
- ✅ Edge cases: batch=1, zero grad, sparse grad

---

## 2.7 Мертвые зоны

| Область | Риск | Причина |
|---------|------|----------|
| ~~Параллелизм backward~~ | ~~🔴~~ | ✅ Реализовано `backward_parallel` |
| ~~Wide layers~~ | ~~🟡~~ | ✅ Покрыто до 1024 |
| Bias gradients напрямую | 🟡 Средний | Проверяется через parity |
| Градиенты |grad|<4e-5 | 🟡 Средний | Ниже f32 precision |
| Очень глубокие сети (>5 слоёв) | 🟡 Средний | Тесты до 4 слоёв |
