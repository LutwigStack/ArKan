# 14. Examples

**Оценка:** ⭐⭐⭐ (3/5)

---

## 14.1 basic.rs — Basic Inference

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | Демонстрация базового inference | 🟢 Работает |
| Workspace | Pre-allocated zero-alloc | 🟢 |
| Single inference | ~30µs latency | 🟢 |

**Что демонстрирует:**
- KanConfig создание
- KanNetwork::new()
- Workspace allocation
- forward_single и forward_batch

---

## 14.2 training.rs — Training Example

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | Демонстрация обучения | 🟢 Работает |
| Adam optimizer | AdamConfig, Adam::new() | 🟢 |
| TrainOptions | Gradient clipping, weight decay | 🟢 |
| Loss tracking | MSE отслеживание | 🟢 |

**Что демонстрирует:**
- Network configuration
- Training loop с Adam
- Gradient clipping и weight decay
- Early stopping pattern

---

## 14.3 gpu_forward.rs — GPU Inference

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | Демонстрация GPU inference | 🟢 Работает |
| WgpuBackend | GPU initialization | 🟢 |
| GpuNetwork | from_cpu() conversion | 🟢 |
| Parity check | CPU == GPU | 🟢 |

**Что демонстрирует:**
- GPU backend initialization
- CPU→GPU network conversion
- Batch forward на GPU
- CPU/GPU parity verification

---

## 14.4 sinusoid/ — Sin(x) Approximation

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | Proof-of-concept обучения | 🟢 Работает |
| Задача | sin(x) approximation | 🟢 MSE = 6e-7 |
| Сложность | Простая 1D функция | 🟢 |

**Что демонстрирует:**
- Training on synthetic data
- Cosine annealing LR schedule
- Seed selection for reproducibility
- Validation metrics (MSE, MAE, max_error)

**Результат:** MSE < 1e-5 достигается за ~10k epochs

---

## 14.5 mnist/ — Image Classification

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | Реальная задача классификации | 🟢 Работает |
| Dataset | 60k train, 10k test | 🟢 |
| Accuracy | > 90% | 🟢 92.76% |
| GPU support | --gpu flag | 🟢 |

**Что демонстрирует:**
- MNIST data loading и normalization
- One-hot encoding
- Softmax classification
- Mini-batch training
- CPU и GPU training modes
- Accuracy evaluation

**Результат:** 92.76% test accuracy

---

## 14.6 game2048/ — DQN Reinforcement Learning

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Назначение | RL с KAN | 🟢 Работает |
| Алгоритм | DQN (Double DQN optional) | 🟢 |
| Параллелизм | 32 parallel envs | 🟢 |
| Performance | 40-50 ep/s | 🟢 |

### 14.6.1 Experience Collection

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Parallel envs | rayon | 🟢 32 среды |
| Thread-local agents | Избежать lock | 🟢 `thread_local!` |
| Zero-alloc states | Fixed arrays | 🟢 `[f32; 256]` |

### 14.6.2 `compute_targets`

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Parallel forward | ✓ | 🟢 `forward_batch_parallel` |
| Policy network | batch forward | 🟢 |
| Target network | batch forward | 🟢 |

**История:** Изначально 11-15 ep/s, после оптимизации 40-50 ep/s.

### 14.6.3 `ReplayBuffer`

| Аспект | Задумано | Реально |
|--------|----------|---------|
| Ring buffer | Circular overwrite | 🟢 |
| sample_batch_into | Pre-allocated | 🟢 |
| Lock contention | RwLock | 🟡 Всё ещё есть |

**TODO:** Lock-free sampling или sharded buffer.

---

## 14.7 Выводы

| Пример | Статус | Тесты |
|--------|--------|-------|
| basic.rs | 🟢 | Нет автотестов |
| training.rs | 🟢 | Нет автотестов |
| gpu_forward.rs | 🟢 | Нет автотестов |
| sinusoid/ | 🟢 | Convergence test |
| mnist/ | 🟢 | Accuracy check |
| game2048/ | 🟢 | Manual only |

**Оценка честности тестов:** ⭐⭐⭐ (3/5)
- ✅ sinusoid и mnist показывают convergence
- ⚠️ Нет автоматических тестов для standalone examples
- ⚠️ game2048 только manual testing
- ❌ Нет CI для примеров

---

## 14.8 Мертвые зоны

| Область | Риск | Причина |
|---------|------|----------|
| Examples compilation | 🟡 Средний | Нет CI проверки что примеры компилируются |
| DQN target Q-value | 🔴 Высокий | Нет теста Bellman equation |
| ReplayBuffer sampling | 🔴 Высокий | Нет теста что sampling fair |

---

## 14.9 Место для оптимизации

| Область | Тип | Сложность | Описание |
|---------|-----|-----------|----------|
| CI для examples | 🧹 Clean | 🟢 Низкая | GitHub Actions для проверки компиляции |
| game2048 PPO | 🔧 Feature | 🟡 Средняя | PPO вместо DQN для лучшего sample efficiency |
| Lock-free ReplayBuffer | 🚀 Perf | 🟡 Средняя | Sharded buffer для parallel sampling |
| CIFAR-10 example | 🔧 Feature | 🟡 Средняя | Более сложный vision benchmark |
| Jupyter notebooks | 🔧 Feature | 🟢 Низкая | Интерактивные туториалы |
| Epsilon decay | 🟡 Средний | Не тестируется exploration |
| Terminal state handling | 🟡 Средний | Q(terminal) должен быть 0 |
