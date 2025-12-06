# 16. Action Items & Improvements

---

## High Priority 🔴

| Задача | Статус | Описание |
|--------|--------|----------|
| ~~Gradient clipping в GpuAdam~~ | ✅ ИСПРАВЛЕНО | `train_step_gpu_native_with_options` |
| ~~Hybrid Adam bug~~ | ✅ ИСПРАВЛЕНО | `unpad_weights` обрезает padding |
| ~~cross_entropy без теста~~ | ✅ ИСПРАВЛЕНО | 8 PyTorch parity тестов |
| ~~SIMD пути не изолированы~~ | ✅ ИСПРАВЛЕНО | 170 комбинаций |

---

## Medium Priority 🟡

| Задача | Статус | Описание |
|--------|--------|----------|
| ~~Lock-free ReplayBuffer~~ | ✅ Done | ShardedReplayBuffer с 16 shards |
| ~~LBFGS Rosenbrock test~~ | ✅ Done | PyTorch reference comparison (2 теста) |
| Model versioning | TODO | Backward compatibility |
| BakedModel serialization test | TODO | to_bytes/from_bytes roundtrip |

---

## Low Priority 🟢

| Задача | Статус | Описание |
|--------|--------|----------|
| DQN automated tests | TODO | Bellman equation check |
| GPU loss functions | TODO | Loss вычисляется на CPU |
| Memory leak detection | TODO | valgrind/miri для GPU сложно |

---

## Known Performance Issues

### CPU
1. **`forward_batch` последовательный** — использовать `forward_batch_parallel`

### GPU
1. **Sync после каждого step** — можно sync реже

---

## Planned Improvements

| Приоритет | Задача | Сложность | Статус |
|-----------|--------|-----------|--------|
| ~~🔴 HIGH~~ | ~~Gradient clipping~~ | ~~Medium~~ | ✅ Done |
| ~~🔴 HIGH~~ | ~~Hybrid Adam bug~~ | ~~Medium~~ | ✅ Done |
| ~~🔴 HIGH~~ | ~~LBFGS line search~~ | ~~Hard~~ | ✅ Done |
| ~~🔴 HIGH~~ | ~~Nesterov momentum~~ | ~~Easy~~ | ✅ Done |
| ~~🔴 HIGH~~ | ~~Async download~~ | ~~Medium~~ | ✅ Done |
| ~~🟡 MED~~ | ~~Lock-free ReplayBuffer~~ | ~~Medium~~ | ✅ Done |
| ~~🟡 MED~~ | ~~LBFGS Rosenbrock test~~ | ~~Easy~~ | ✅ Done |
| 🟢 LOW | Model versioning | Easy | TODO |

---

## Completed ✅

- ~~Serialization knots bug~~ — Custom Deserialize для KanLayer
- ~~forward_batch_parallel~~ — Добавлен
- ~~GPU backward parity~~ — 11 тестов
- ~~gradient_check 95%~~ — Теоретический максимум f32
- ~~Async GPU pipeline~~ — forward_batch_async
- ~~GpuAdam momentum parity~~ — 9 тестов
- ~~Hybrid Adam bug~~ — unpad_weights()
- ~~Lock-free ReplayBuffer~~ — ShardedReplayBuffer (2025-12-07)
- ~~LBFGS Rosenbrock test~~ — PyTorch parity тесты (2025-12-07)
- ~~PyTorch cross_entropy parity~~ — 8 тестов
- ~~Serialization multi-size~~ — 10 тестов
