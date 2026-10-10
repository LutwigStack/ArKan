# **ArKan**

![crates.io](https://img.shields.io/crates/v/arkan.svg)
![docs.rs](https://docs.rs/arkan/badge.svg)
![ci](https://github.com/LutwigStack/ArKan/actions/workflows/ci.yml/badge.svg)

<a name="arkan-ru"></a>   **ArKan** — это высокопроизводительная реализация сетей Колмогорова-Арнольда (KAN) на Rust, оптимизированная для задач с критическими требованиями к задержкам (Low Latency Inference).

Библиотека создавалась специально для интеграции в игровые солверы (например, Poker AI / MCTS), где требуется выполнять тысячи одиночных инференсов в секунду без оверхеда, свойственного большим ML-фреймворкам.

## **Теория: Что такое KAN?**

В отличие от классических многослойных перцептронов (MLP), где функции активации зафиксированы на узлах (нейронах), а обучаются линейные веса, в **Kolmogorov-Arnold Networks (KAN)** всё наоборот:

* **Узлы** выполняют простое суммирование.  
* **Ребра (связи)** содержат обучаемые нелинейные функции активации.

### **Математическая модель**

В основе лежит теорема представления Колмогорова-Арнольда. Для слоя с `N_in` входами и `N_out` выходами преобразование выглядит так:

```text
x[l+1, j] = Σᵢ φ[l,j,i](x[l, i])      где i = 1..N_in
```

Где `φ[l,j,i]` — это обучаемая 1D-функция, которая связывает `i`-й нейрон входного слоя с `j`-м нейроном выходного.

### **Реализация в ArKan (B-Splines)**

В данной библиотеке функции `φ` параметризуются с помощью **B-сплайнов** (Basis splines). Это позволяет менять форму функции активации локально, сохраняя гладкость.

Уравнение для конкретного веса в ArKan:

```text
φ(x) = Σᵢ cᵢ · Bᵢ(x)      где i = 1..(G+p)
```

* `Bᵢ(x)` — базисные функции сплайна.
* `cᵢ` — обучаемые коэффициенты.
* `G` — размер сетки (grid size).
* `p` — порядок сплайна (spline order).

## **Ключевые возможности**

* **Zero-Allocation Inference:** Последовательный `forward` проход выполняется на предвыделенном буфере (`Workspace`) без аллокаций при повторном использовании.
* **Reusable Training Storage:** Полный training step (forward + backward + SGD/Adam) повторно использует память ArKan при прогретом Workspace. Параллельные вызовы из внешнего потока могут выделять блоки очереди планировщика Rayon; подробнее ниже.
* **SIMD-Optimized B-Splines:** Вычисление базисных функций B-сплайнов векторизовано через крейт `wide` (256-битный `f32x8`, т.е. AVX2 на x86-64). SIMD не является feature-флагом — векторизация включена всегда.  
* **Cache-Friendly Layout:** Веса хранятся в формате `[Output][Input][Basis]` для последовательного доступа к памяти и минимизации промахов кэша.  
* **Standalone:** Сборка по умолчанию тянет только `wide`, `rand` и `thiserror`. Никаких `torch` или `burn`, идеально для встраивания.  
* **Baked (int8) Inference:** `BakedModel` — рабочий путь квантизованного инференса (per-channel int8 веса + int16 базис). 2.2–3.0× меньше памяти. **Медленнее** f32-пути при batch=1, и худший случай ошибки на отдельном выходе большой — годится для ранжирования/argmax, но не для абсолютных значений. Читайте раздел ниже до того, как это внедрять.
* **GPU-ускорение (wgpu):** Опциональный GPU бэкенд с WGSL compute шейдерами для параллельного forward/backward.

## **Cargo features**

> **Установка.** Крейт пока не опубликован на crates.io — `cargo add arkan` вернёт
> `could not be found in registry index`. До публикации подключайте через git:
> `arkan = { git = "https://github.com/LutwigStack/ArKan" }`.
> Версии в примерах ниже — то, чем станет ближайший релиз.

```toml
[dependencies]
arkan = "0.4"                                    # только wide + rand + thiserror
arkan = { version = "0.4", features = ["serde"] } # + сериализация
```

| Флаг | Что включает | По умолчанию | Доп. зависимости |
|---|---|---|---|
| — | Инференс и обучение на CPU, SIMD B-сплайны, `BakedModel` (int8) | ✅ | `wide`, `rand`, `thiserror` |
| `parallel` | `KanLayer::backward_parallel`, `KanNetwork::forward_batch_parallel` и авто-параллельная ветка backward-прохода в `train_step` при `batch >= multithreading_threshold` | ❌ | `rayon` |
| `serde` | `to_bytes()` / `from_bytes()` для `KanNetwork` и `BakedModel` | ❌ | `serde`, `bincode` |
| `gpu` | GPU бэкенд на `wgpu` (Vulkan/DX12/Metal/WebGPU) | ❌ | `wgpu`, `bytemuck`, `pollster`, `log` |

SIMD — **не** флаг: векторизация B-сплайнов через `wide` работает всегда.
Без `parallel` методы `*_parallel` не существуют, а backward-проход всегда
однопоточный: градиенты те же (паритет проверяется в
`tests/backward_correctness.rs`), просто одно ядро.

### Минимальная версия Rust (MSRV)

`rust-version` в `Cargo.toml` — **1.73**. Это сборка по умолчанию и с `serde`;
именно её пинует CI. Опциональные фичи требуют больше — не из-за нашего кода,
а из-за их зависимостей:

| Сборка | MSRV | Чем задан |
|---|---|---|
| по умолчанию, `serde` | **1.73** | наш `div_ceil` (`int_roundings`, стабилен с 1.73) |
| `parallel` | **1.80** | `rayon-core` |
| `gpu` | **1.85** | `indexmap` (через `wgpu` 23 → `naga`) |

Числа получены прогоном тулчейнов, а не на глаз: 1.72 падает, 1.73 собирается.
`Cargo.lock` не коммитится, поэтому нижняя граница ползёт вверх сама собой
вместе с релизами зависимостей; за этим следит джоба `msrv` в CI.

## **Что библиотека НЕ умеет**

Прочитайте до внедрения. Всё измерено, всё воспроизводится тестами из репозитория.

**`grid_range` общий для всех слоёв, но нормализуется только слой 0.**
`input_mean` / `input_std` получает только первый слой; скрытые слои создаются с
единичной нормализацией, и статистика нигде не обновляется. То есть вход скрытого
слоя — это **сырая активация** предыдущего слоя, зажатая в тот же `grid_range`,
который вы выбрали для входов. Ничто не удерживает выход KAN-слоя внутри его
собственного grid range.

Выбор `grid_range` по диапазону входов тихо убивает скрытые слои. Замер на форме
`examples/game2048` (256 → [64, 32] → 4, one-hot входы,
`cargo test --test hidden_layer_saturation -- --nocapture`):

| `grid_range` | слой 0 | слой 1 | слой 2 |
|---|---|---|---|
| `(0.0, 1.0)` | 0% | **43.6%** | **48.9%** |
| `(-1.0, 1.0)` | 0% | 0% | 0% |
| `(-3.0, 3.0)` | 0% | 0% | 0% |

У насыщенного входа нулевая производная, поэтому почти половина каждого скрытого
слоя выдавала константу и не получала градиента. Берите симметричный диапазон,
рассчитанный на **активации**, а не на входы.

**Насыщение молчаливое.** Нет `out_of_grid_fraction`, нет предупреждения о дрейфе,
нет настраиваемой экстраполяции, нет перекалибровки сетки. Дрейф распределения
выглядит как «модель просто плохая», а не как диагностика.

**`BakedModel` медленнее f32 при batch=1** (в 1.4–2.1 раза) и имеет большой
худший случай ошибки на отдельном выходе (34–54% на выходах ≥1σ у сети с двумя
скрытыми слоями). Подробности — в разделе ниже.

**GPU поддерживает порядки сплайна 2–5**, CPU — 2–7, `BakedModel` — 2–5
(вне диапазона `from_network` **паникует**).

## **Baked (int8) Inference**

`BakedModel` is a quantized, fixed-point inference-only representation of a trained
`KanNetwork`. It uses **per-channel int8 weights**, **int16 B-spline basis** (Q0.15),
and **i32 inter-layer activations** — no f32 in the hot path.

### Usage

```rust
use arkan::{BakedModel, KanNetwork, KanConfig};

fn main() -> arkan::ArkanResult<()> {
    // 1. Train a KanNetwork as usual.
    let config = KanConfig::preset();
    let network = KanNetwork::new(config.clone());
    // ... train ...

    // 2. Collect a calibration set (flat: n_samples * input_dim f32 values).
    //    Use REAL representative inputs: the 99.9th percentile of the activations
    //    they produce sets the baked activation scale.
    //    256–1024 samples is typical. Empty calibration uses the uncalibrated heuristic.
    let n_samples = 256;
    let calibration: Vec<f32> = (0..n_samples * config.input_dim)
        .map(|i| (i % 17) as f32 / 17.0 - 0.5) // stand-in for your real data
        .collect();

    // 3. Bake.
    let baked = BakedModel::try_from_network(&network, Some(&calibration))?;

    // 4. Allocate scratch once and reuse it for fixed-point inference.
    let mut scratch = baked.create_workspace();
    let input = vec![0.5f32; config.input_dim];
    let mut output = vec![0.0f32; config.output_dim];
    baked.forward_with_workspace(&input, &mut output, &mut scratch);

    // 5. Check size.
    println!("Baked model: {} bytes", baked.size_bytes());
    Ok(())
}
```

`try_from_network` rejects unsupported orders, invalid topology/normalization,
nonfinite calibration and unrepresentable fixed-point values. The legacy
`from_network` wrapper panics on those errors. `BakedWorkspace` supports repeated
inference without allocating; `forward` is the allocating convenience wrapper.

### Serialization (requires `serde` feature)

```rust,ignore
// Serialize — prepends 12-byte magic + 4-byte version, then bincode body.
let bytes: Vec<u8> = baked.to_bytes()?;

// Deserialize — validates magic and version before parsing; returns Err on mismatch.
let baked2 = BakedModel::from_bytes(&bytes)?;
```

### Tradeoffs (honest)

The figures below are the historical 2026-07-26 measurements. Current workspace
and calibration changes require fresh measurements before drawing latency conclusions.

All measured 2026-07-26 — `cargo test --release --test baked_parity -- --nocapture`
and `cargo bench --bench baked`. Random-init networks, `grid_range = (-1, 1)`,
256 calibration and 2000 test samples in `[-0.9, 0.9]`.

| Property | Value |
|---|---|
| Weight compression | **2.2–3.0×** vs f32 parameters |
| NRMSE vs f32, 1-hidden (4→[8]→2) | 0.17% |
| NRMSE vs f32, 2-hidden (8→[16,8]→4) | 0.57% (order 3); 0.37–0.59% across orders 2–5 |
| **Worst-case error on outputs ≥1σ, 1-hidden** | 0.8% |
| **Worst-case error on outputs ≥1σ, 2-hidden** | **3.1–7.9%** (gated at 15%) |
| **Latency vs f32 at batch=1** | **1.4× slower** (4→[8]→2), **2.1× slower** (8→[16,8]→4) |
| Suitable for | Ranking, argmax, classification, and — now — reading outputs |
| Spline orders supported | 2–5 only. `from_network` **panics** outside that range |

Three things this table is saying that are easy to skim past:

1. **Baked latency needs a fresh measurement.** The historical table measured
   the allocating convenience method. Current benchmarks use reusable workspaces;
   the archived latency ratios do not describe that path.
2. **The ≥1σ tail used to be 34–54% and is now 3–8%.** Three fixed-point defects
   caused it, all now fixed: activations saturated at the calibration set's
   99.9th percentile on *every* layer (so the model could never return a larger
   magnitude, and hidden layers got a saturation the f32 path does not have);
   the inter-layer scale `norm_a_fixed` carried three bits, landing on the
   integers 7–9; and it rounded to **0** — silently collapsing the next layer's
   inputs to a constant — whenever a layer's `p99.9` fell below `std/32`.
   `tests/baked_parity.rs` now gates the ≥1σ worst case, not just NRMSE.
3. **Calibration is not optional.** Without it the activation scale falls back to
   a coarse heuristic and accuracy degrades badly. Passing an **empty** slice is
   worse and silent: `from_network` does not error, and the resulting model
   measures **100% NRMSE** against f32 (pinned in `tests/readme_snippets.rs`).

Full numbers, including per-order tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#baked-int8-inference).
See `examples/baked_inference.rs` for a complete runnable demonstration.

---

## **GPU Backend (Опционально)**

ArKan включает опциональный GPU бэкенд на основе `wgpu` для WebGPU/Vulkan/Metal/DX12 ускорения.

### **Установка**

```toml
[dependencies]
arkan = { version = "0.4", features = ["gpu"] }
```

### **Использование**

```rust,ignore
use arkan::{KanConfig, KanNetwork};
use arkan::gpu::{WgpuBackend, WgpuOptions, GpuNetwork};
use arkan::optimizer::{Adam, AdamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Инициализация GPU бэкенда
    let backend = WgpuBackend::init(WgpuOptions::default())?;
    println!("GPU: {}", backend.adapter_info().name);

    // Создание CPU сети
    let config = KanConfig::preset();
    let mut cpu_network = KanNetwork::new(config.clone());

    // Создание GPU сети из CPU сети
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)?;
    let mut workspace = gpu_network.create_workspace(64)?;

    // Forward инференс
    let input = vec![0.5f32; config.input_dim];
    let output = gpu_network.forward_single(&input, &mut workspace)?;

    // Обучение с Adam оптимизатором
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));
    let target = vec![1.0f32; config.output_dim];

    let loss = gpu_network.train_step_mse(
        &input, &target, 1,
        &mut workspace, &mut optimizer, &mut cpu_network
    )?;

    println!("Loss: {}", loss);
    Ok(())
}
```

### **GPU возможности**

| Функция | Статус |
|---------|--------|
| Forward инференс | ✅ |
| Forward training (сохранение активаций) | ✅ |
| Backward pass | ✅ (GPU шейдеры) |
| Adam/SGD оптимизатор | ✅ |
| Синхронизация весов CPU↔GPU | ✅ |
| Многослойные сети | ✅ |
| Batch обработка | ✅ |
| train_step_with_options | ✅ |
| Gradient clipping | ✅ |
| Weight decay | ✅ |

### **Ограничения GPU (wgpu 0.23)**

- **Порядок сплайна:** GPU шейдеры поддерживают только порядки 2–5 (`MIN_GPU_SPLINE_ORDER=2`, `MAX_GPU_SPLINE_ORDER=5`). CPU поддерживает 2–7.
- **Нет пробрасывания DeviceLost:** wgpu 0.23 не предоставляет ошибки `DeviceLost`. Падение GPU может выглядеть как зависание вместо корректной ошибки.
- **Лимит памяти:** По умолчанию `MAX_VRAM_ALLOC = 2GB` на буфер. Настраивается через `WgpuOptions`. Для больших тензоров рекомендуется ~30% от реального VRAM (например, 3GB для RTX 4070 SUPER с 12GB).
- **Vec4 выравнивание:** Веса дополняются до границы vec4 (4 элемента) для эффективности шейдеров.
- **Автоматического CPU fallback НЕТ:** если GPU недоступен, `WgpuBackend::init` возвращает ошибку `AdapterNotFound`. Переключение на `KanNetwork` — задача вызывающего кода.

### **Запуск GPU тестов и бенчмарков**

```bash
# GPU parity тесты
cargo test --features gpu --test gpu_parity -- --ignored

# GPU бенчмарки (Windows PowerShell)
$env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_forward --features gpu

# GPU бенчмарки (Linux/macOS)
ARKAN_GPU_BENCH=1 cargo bench --bench gpu_forward --features gpu
```

### **GPU производительность vs PyTorch CUDA**

Старые сравнительные результаты недействительны до повторных замеров с одинаковыми
конфигурациями и математикой. Методика и архив: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## **Бенчмарки (CPU)**

Старые CPU/PyTorch сравнения недействительны до согласованного повторного замера.
Новые времена и коэффициенты ускорения здесь не заявлены. См.
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Reusable Training Storage:** training loop (forward + backward + optimizer step) повторно использует память ArKan при прогретом Workspace. Проверяется `tests/allocation_budget.rs` — counting `GlobalAlloc` на forward_batch, forward_single, train_step и train_step_with_optimizer (Adam и SGD). При включённом `parallel` вызовы из внешнего потока могут регулярно выделять блоки очереди планировщика Rayon. Нулевое число аллокаций наблюдалось для повторных вызовов внутри одного охватывающего их прогретого пула Rayon; это не гарантия для произвольного пула или контекста вызова.

## **Сравнение с аналогами (Prior Art)**

ArKan занимает нишу **специализированного высокопроизводительного инференса**.

| Крейт | Назначение | Отличие ArKan |
| :---- | :---- | :---- |
| [`burn-efficient-kan`](https://crates.io/crates/burn-efficient-kan) | Часть экосистемы [Burn](https://burn.dev). Отлично подходит для обучения на GPU. | ArKan — легковесная библиотека с опциональным GPU через wgpu. Минимальные зависимости в базовой конфигурации. |
| [`fekan`](https://crates.io/crates/fekan) | Богатый функционал (CLI, dataset loaders). General-purpose библиотека. | ArKan изначально спроектирован под SIMD (AVX2), параллелизм и GPU-ускорение. |
| [`rusty_kan`](https://crates.io/crates/rusty_kan) | Базовая реализация, образовательный проект. | ArKan фокусируется на production-ready оптимизациях: workspace, батчинг, GPU. |

## Быстрый старт

Установка из crates.io:

```toml
[dependencies]
arkan = "0.4"
```

Пример использования (смотрите также `examples/basic.rs` и `examples/training.rs`):
```rust,no_run
use arkan::{KanConfig, KanNetwork};

fn main() {
    // 1. Конфигурация (Poker Solver preset)
    let config = KanConfig::preset();

    // 2. Инициализация сети
    let network = KanNetwork::new(config.clone());

    // 3. Создание Workspace (аллокация памяти один раз)
    let mut workspace = network.create_workspace(64); // Max batch size = 64

    // 4. Данные
    let inputs = vec![0.0f32; 64 * config.input_dim];
    let mut outputs = vec![0.0f32; 64 * config.output_dim];

    // 5. Инференс (Zero allocations here!)
    network.forward_batch(&inputs, &mut outputs, &mut workspace);

    println!("Inference done. Output[0]: {}", outputs[0]);
}
```

## **Архитектура**

Реализации разделены внутри одного crate на `model`, `math`, `memory`, `cpu`,
`training`, `optimizer`, `loss`, `baked`, `gpu` и внутренний `format`.
Старые пути `config`, `network`, `layer`, `spline`, `buffer` сохраняют те же типы
через reexport. Проверенная топология неизменяема; `try_parameters_mut` изменяет
только коэффициенты и bias. `ForwardPass` удерживает модель и Workspace до backward.

Карта модулей, правила нормализации, clipping, совместимости форматов и будущего
разделения crates: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## **Лицензия**

Распространяется под двойной лицензией **MIT** и **Apache-2.0**.

<a name="arkan-en"></a>

# **ArKan (English Version)**

**ArKan** is a high-performance implementation of Kolmogorov-Arnold Networks (KAN) in Rust, optimized for tasks with critical latency requirements (Low Latency Inference).

The library was created specifically for integration into game solvers (e.g., Poker AI / MCTS), where thousands of single inferences per second are required without the overhead typical of large ML frameworks.

## **Theory: What is KAN?**

Unlike classical Multi-Layer Perceptrons (MLP), where activation functions are fixed on nodes (neurons) and linear weights are learned, in **Kolmogorov-Arnold Networks (KAN)**, it's the opposite:

* **Nodes** perform simple summation.  
* **Edges** contain learnable non-linear activation functions.

### **Mathematical Model**

Based on the Kolmogorov-Arnold representation theorem. For a layer with `N_in` inputs and `N_out` outputs, the transformation looks like this:

```text
x[l+1, j] = Σᵢ φ[l,j,i](x[l, i])      where i = 1..N_in
```

Where `φ[l,j,i]` is a learnable 1D function connecting the `i`-th input neuron to the `j`-th output neuron.

### **Implementation in ArKan (B-Splines)**

In this library, `φ` functions are parameterized using **B-Splines**. This allows modifying the shape of the activation function locally while maintaining smoothness.

Equation for a specific weight in ArKan:

```text
φ(x) = Σᵢ cᵢ · Bᵢ(x)      where i = 1..(G+p)
```

* `Bᵢ(x)` — B-spline basis functions.
* `cᵢ` — learnable coefficients.
* `G` — grid size.
* `p` — spline order.

## **Key Features**

* **Zero-Allocation Inference:** Serial `forward` runs on a pre-allocated buffer (`Workspace`) without allocations on reuse.
* **Reusable Training Storage:** The full training step (forward + backward + SGD/Adam) reuses ArKan storage on a warmed-up Workspace. Parallel calls from external threads can allocate Rayon scheduling-queue blocks; see below.
* **SIMD-Optimized B-Splines:** B-spline basis evaluation is vectorized via the `wide` crate (256-bit `f32x8`, i.e. AVX2 on x86-64). SIMD is not a feature flag — vectorization is always on.  
* **Cache-Friendly Layout:** Weights are stored in `[Output][Input][Basis]` format for sequential memory access and minimal cache misses.  
* **Standalone:** A default build pulls only `wide`, `rand` and `thiserror`. No `torch` or `burn` bloat, ideal for embedding.  
* **Baked (int8) Inference:** `BakedModel` is a working quantized inference path (per-channel int8 weights + int16 basis). 2.2–3.0× smaller than f32. It is **slower** than the f32 path at batch=1 and its per-output tail error is large — suitable for ranking/argmax, not for absolute values. Read [Baked (int8) Inference](#baked-int8-inference) before adopting it.
* **GPU Acceleration (wgpu):** Optional GPU backend with WGSL compute shaders for parallel forward/backward passes.

## **Cargo features**

> **Installation.** Not published on crates.io yet — `cargo add arkan` currently
> returns `could not be found in registry index`. Until it is, depend on it via git:
> `arkan = { git = "https://github.com/LutwigStack/ArKan" }`.
> The versions in the snippets below are what the next release will be.

```toml
[dependencies]
arkan = "0.4"                                     # wide + rand + thiserror only
arkan = { version = "0.4", features = ["serde"] } # + serialization
```

| Flag | What it turns on | Default | Extra deps |
|---|---|---|---|
| — | CPU inference and training, SIMD B-splines, `BakedModel` (int8) | ✅ | `wide`, `rand`, `thiserror` |
| `parallel` | `KanLayer::backward_parallel`, `KanNetwork::forward_batch_parallel`, and the automatic parallel branch of `train_step`'s backward pass for `batch >= multithreading_threshold` | ❌ | `rayon` |
| `serde` | `to_bytes()` / `from_bytes()` for `KanNetwork` and `BakedModel` | ❌ | `serde`, `bincode` |
| `gpu` | `wgpu` GPU backend (Vulkan/DX12/Metal/WebGPU) | ❌ | `wgpu`, `bytemuck`, `pollster`, `log` |

SIMD is **not** a feature flag: B-spline vectorization via `wide` is always on.
Without `parallel` the `*_parallel` methods do not exist and the backward pass is
always single-threaded — identical gradients (parity asserted in
`tests/backward_correctness.rs`), just one core.

### Minimum Supported Rust Version (MSRV)

`rust-version` in `Cargo.toml` is **1.73** — the default build and the `serde`
build, and the toolchain CI pins. The optional features need more, not because of
our code but because of theirs:

| Build | MSRV | Set by |
|---|---|---|
| default, `serde` | **1.73** | our own `div_ceil` (`int_roundings`, stable since 1.73) |
| `parallel` | **1.80** | `rayon-core` |
| `gpu` | **1.85** | `indexmap`, via `wgpu` 23 → `naga` |

These were measured by running the toolchains, not guessed: 1.72 fails, 1.73
builds. No `Cargo.lock` is committed, so dependency resolution always picks the
newest compatible versions and this floor drifts upward on its own as they
release. The `msrv` CI job is what tells us when it moved.

## **What this library does NOT do**

Read this before adopting. Everything here is measured and reproducible from
tests in this repository.

**`grid_range` is shared by every layer; hidden normalization starts as identity.**
`input_mean` / `input_std` are applied to the first layer only; hidden layers are
built with identity normalization, including equal-width hidden layers. Explicit
per-layer normalization updates are supported; training does not estimate running
statistics. With the default hidden statistics, a
hidden layer's input is the previous layer's **raw activation**, clamped to the
same `grid_range` you chose for the inputs. Nothing bounds a KAN layer's output
to its own grid range.

Picking `grid_range` from the range of the *inputs* silently kills the hidden
layers. Measured on the `examples/game2048` shape (256 → [64, 32] → 4, one-hot
inputs, `cargo test --test hidden_layer_saturation -- --nocapture`):

| `grid_range` | layer 0 | layer 1 | layer 2 |
|---|---|---|---|
| `(0.0, 1.0)` | 0% | **43.6%** | **48.9%** |
| `(-1.0, 1.0)` | 0% | 0% | 0% |
| `(-3.0, 3.0)` | 0% | 0% | 0% |

A saturated input has zero derivative, so nearly half of each hidden layer
emitted a constant and received no gradient. Use a symmetric range sized for the
**activations**, not for the inputs.

**Saturation is silent.** There is no `out_of_grid_fraction`, no drift warning,
no configurable extrapolation and no grid recalibration. Distribution drift looks
like "the model just isn't very good", not like a diagnostic.

**`BakedModel` is slower than f32 at batch=1** (1.4–2.1×) and has a large
per-output tail (34–54% worst case on outputs ≥1σ for a 2-hidden net). See
[Baked (int8) Inference](#baked-int8-inference) above and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#baked-int8-inference).

**Spline order support differs by backend:** CPU 2–7, GPU 2–5, `BakedModel` 2–5
(`from_network` **panics** outside that range, even though `KanConfig::validate()`
accepts up to 7).

## **GPU Backend (Optional)**

ArKan includes an optional GPU backend using `wgpu` for WebGPU/Vulkan/Metal/DX12 acceleration.

### **Installation**

```toml
[dependencies]
arkan = { version = "0.4", features = ["gpu"] }
```

### **Usage**

```rust,ignore
use arkan::{KanConfig, KanNetwork};
use arkan::gpu::{WgpuBackend, WgpuOptions, GpuNetwork};
use arkan::optimizer::{Adam, AdamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())?;
    println!("GPU: {}", backend.adapter_info().name);

    // Create CPU network
    let config = KanConfig::preset();
    let mut cpu_network = KanNetwork::new(config.clone());

    // Create GPU network from CPU network
    let mut gpu_network = GpuNetwork::from_cpu(&backend, &cpu_network)?;
    let mut workspace = gpu_network.create_workspace(64)?;

    // Forward inference
    let input = vec![0.5f32; config.input_dim];
    let output = gpu_network.forward_single(&input, &mut workspace)?;

    // Training with Adam optimizer
    let mut optimizer = Adam::new(&cpu_network, AdamConfig::with_lr(0.001));
    let target = vec![1.0f32; config.output_dim];

    let loss = gpu_network.train_step_mse(
        &input, &target, 1, 
        &mut workspace, &mut optimizer, &mut cpu_network
    )?;

    println!("Loss: {}", loss);
    Ok(())
}
```

### **GPU Features**

| Feature | Status |
|---------|--------|
| Forward inference | ✅ |
| Forward training (saves activations) | ✅ |
| Backward pass | ✅ (GPU shaders) |
| Adam/SGD optimizer | ✅ |
| Weight sync CPU↔GPU | ✅ |
| Multi-layer networks | ✅ |
| Batch processing | ✅ |
| train_step_with_options | ✅ |
| Gradient clipping | ✅ |
| Weight decay | ✅ |

### **Weight Synchronization**

```rust,ignore
// Sync weights from CPU to GPU (after loading a model)
gpu_network.sync_weights_cpu_to_gpu(&cpu_network)?;

// Sync weights from GPU to CPU (for saving/export)
gpu_network.sync_weights_gpu_to_cpu(&mut cpu_network)?;
```

### **Training with Options**

```rust,ignore
use arkan::TrainOptions;

let opts = TrainOptions {
    max_grad_norm: Some(1.0),  // Gradient clipping
    weight_decay: 0.01,         // AdamW-style weight decay
};

let loss = gpu_network.train_step_with_options(
    &input, &target, None, batch_size,
    &mut workspace, &mut optimizer, &mut cpu_network,
    &opts
)?;
```

### **GPU Limitations (wgpu 0.23)**

- **Spline order:** GPU shaders support orders 2–5 only (`MIN_GPU_SPLINE_ORDER=2`, `MAX_GPU_SPLINE_ORDER=5`). CPU supports 2–7.
- **No DeviceLost propagation:** wgpu 0.23 does not expose `DeviceLost` errors. GPU crashes may appear as hangs instead of proper errors.
- **Memory limits:** Default `MAX_VRAM_ALLOC = 2GB` per buffer. Configurable via `WgpuOptions`. For large tensors, use ~30% of your actual VRAM (e.g., 3GB for RTX 4070 SUPER with 12GB).
- **Vec4 alignment:** Weights are padded to vec4 (4-element) boundaries for shader efficiency.
- **No automatic CPU fallback:** if no GPU is available, `WgpuBackend::init` returns `AdapterNotFound`. Falling back to `KanNetwork` is the caller's job.

### **Choosing Backend**

```rust,ignore
// High-performance GPU (default, 2GB limit)
let backend = WgpuBackend::init(WgpuOptions::default())?;

// Compute-optimized (unlimited VRAM)
let backend = WgpuBackend::init(WgpuOptions::compute())?;

// Custom VRAM limit in GB (recommended for known hardware)
let backend = WgpuBackend::init(WgpuOptions::with_max_vram(3))?; // 3GB

// Percentage of device max (works on AMD/Intel, not useful for NVIDIA)
let backend = WgpuBackend::init(WgpuOptions::with_max_vram_percent(30))?;

// No VRAM limit (use device max)
let backend = WgpuBackend::init(WgpuOptions::unlimited_vram())?;

// Low-memory/integrated GPU
let backend = WgpuBackend::init(WgpuOptions::low_memory())?;

// Force specific adapter
let opts = WgpuOptions {
    force_adapter_name: Some("NVIDIA".to_string()),
    ..Default::default()
};
let backend = WgpuBackend::init(opts)?;
```

### **Running GPU Tests and Benchmarks**

```bash
# GPU parity tests
cargo test --features gpu --test gpu_parity -- --ignored

# GPU benchmarks (Windows PowerShell)
$env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_forward --features gpu

# GPU benchmarks (Linux/macOS)
ARKAN_GPU_BENCH=1 cargo bench --bench gpu_forward --features gpu
ARKAN_GPU_BENCH=1 cargo bench --bench gpu_backward --features gpu
```

### **GPU Performance vs PyTorch CUDA**

Historical comparison ratios are invalid pending a matched rerun with consistent
model shapes and mathematics. Methodology and archived results:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## **Benchmarks (CPU)**

Historical CPU/PyTorch comparisons are invalid pending a matched rerun. No current
timings or speedup ratios are claimed here. See
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

**Reusable Training Storage:** the training loop (forward + backward + optimizer step) reuses ArKan storage on a warmed-up Workspace. Checked by `tests/allocation_budget.rs` — a counting `GlobalAlloc` over forward_batch, forward_single, train_step and train_step_with_optimizer (both Adam and SGD). With `parallel`, external-thread calls can allocate recurring Rayon scheduling-queue blocks. Repeated calls inside one enclosing, warmed Rayon pool observed zero allocations; arbitrary pools and call contexts are not guaranteed to allocate nothing.

## **Comparison with Analogues (Prior Art)**

ArKan occupies the niche of **specialized high-performance inference**.

| Crate | Purpose | Difference from ArKan |
| :---- | :---- | :---- |
| [`burn-efficient-kan`](https://crates.io/crates/burn-efficient-kan) | Part of the [Burn](https://burn.dev) ecosystem. | ArKan is lightweight with optional GPU via wgpu. Minimal dependencies in base config. |
| [`fekan`](https://crates.io/crates/fekan) | Rich functionality, general-purpose library. | ArKan is designed with SIMD, parallelism, and GPU acceleration from the start. |
| [`rusty_kan`](https://crates.io/crates/rusty_kan) | Basic implementation, educational project. | ArKan focuses on production-ready optimizations: workspace, batching, GPU. |

## **Quick Start**

Install from crates.io:

```toml
[dependencies]
arkan = "0.4"
```

Usage Example (see also `examples/basic.rs` and `examples/training.rs`):
```rust,no_run
use arkan::{KanConfig, KanNetwork};

fn main() {
    // 1. Configuration (Poker Solver preset)
    let config = KanConfig::preset();

    // 2. Network initialization
    let network = KanNetwork::new(config.clone());

    // 3. Create Workspace (memory allocated once)
    let mut workspace = network.create_workspace(64); // Max batch size = 64

    // 4. Data preparation
    let inputs = vec![0.0f32; 64 * config.input_dim];
    let mut outputs = vec![0.0f32; 64 * config.output_dim];

    // 5. Inference (Zero allocations here!)
    network.forward_batch(&inputs, &mut outputs, &mut workspace);

    println!("Inference done. Output[0]: {}", outputs[0]);
}
```

## **Architecture**

One crate owns the implementations in `model`, `math`, `memory`, `cpu`, `training`,
`optimizer`, `loss`, `baked`, `gpu`, and the private `format` boundary. The legacy
`config`, `network`, `layer`, `spline`, and `buffer` imports remain reexports of
the same types and functions.

`KanNetwork` retains checked immutable topology. Legacy public structural fields
remain available for source compatibility, but incompatible edits are rejected
by checked execution, conversion, and synchronization. Use `try_parameters_mut`
for fixed-length weight/bias updates. `ForwardPass` borrows the model and workspace
through backward; its resulting `Gradients` can be passed directly to an optimizer.

### Custom loss and optimizer

```rust
use arkan::{KanConfig, KanNetwork, SGD, SGDConfig, Optimizer, masked_bce_with_logits};

fn main() -> arkan::ArkanResult<()> {
    let mut network = KanNetwork::new(KanConfig::preset());
    let mut workspace = network.create_workspace(1);
    let mut optimizer = SGD::new(&network, SGDConfig::with_lr(0.01));
    let input = vec![0.0; network.config.input_dim];
    let targets = vec![1.0; network.config.output_dim];
    let mut logits = vec![0.0; targets.len()];
    let pass = network.try_forward_for_backward(&input, &mut logits, &mut workspace)?;
    let (loss, derivative) = masked_bce_with_logits(&logits, &targets, None);
    let gradients = pass.backward(&derivative)?;
    optimizer.step(&mut network, gradients.weights, gradients.biases, Some(1.0))?;
    println!("Loss: {loss}");
    Ok(())
}
```

MSE trainers use the same backward loop; `masked_mse_into` fills a reusable output
gradient buffer. Standalone optimizers unscale AMP gradients before optional global
clipping. Disabled clipping skips the norm pass. Weight decay belongs to the
optimizer configuration for `train_step_with_optimizer`; direct SGD uses
`TrainOptions::weight_decay`. Hybrid GPU options retain their additional explicit
decay step; native GPU decay belongs to its optimizer configuration.

CPU and GPU parameter/normalization snapshots require explicit synchronization.
Native GPU updates stay on the GPU until copied back. Serialization uses stable
V1 network/legacy and V2 baked records, independent of runtime module placement.

Full ownership map, numerical contracts, compatibility limits and the later
`arkan-core` / `arkan-wgpu` extraction prerequisites:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## **License**

Distributed under a dual license **MIT** and **Apache-2.0**.

