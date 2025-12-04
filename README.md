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

В основе лежит теорема представления Колмогорова-Арнольда. Для слоя с $N\_{in}$ входами и $N\_{out}$ выходами преобразование выглядит так:

$$x\_{l+1, j} \= \\sum\_{i=1}^{N\_{in}} \\phi\_{l, j, i}(x\_{l, i})$$

Где $\\phi\_{l, j, i}$ — это обучаемая 1D-функция, которая связывает $i$-й нейрон входного слоя с $j$-м нейроном выходного.

### **Реализация в ArKan (B-Splines)**

В данной библиотеке функции $\\phi$ параметризуются с помощью **B-сплайнов** (Basis splines). Это позволяет менять форму функции активации локально, сохраняя гладкость.

Уравнение для конкретного веса в ArKan:

$$\\phi(x) \= \\sum\_{i=1}^{G+p} c\_i \\cdot B\_i(x)$$

* $B\_i(x)$ — базисные функции сплайна.  
* $c\_i$ — обучаемые коэффициенты.  
* $G$ — размер сетки (grid size).  
* $p$ — порядок сплайна (spline order).

## **Ключевые возможности**

* **Zero-Allocation Inference:** Весь `forward` проход выполняется на предвыделенном буфере (`Workspace`). Никаких аллокаций в горячем пути (Hot Path).  
* **Zero-Allocation Training:** Полный training step (forward + backward + SGD/Adam) также работает без аллокаций при прогретом Workspace.
* **SIMD-Optimized B-Splines:** Вычисление базисных функций B-сплайнов векторизовано (AVX2/AVX-512 через крейт `wide`).  
* **Cache-Friendly Layout:** Веса хранятся в формате `[Output][Input][Basis]` для последовательного доступа к памяти и минимизации промахов кэша.  
* **Standalone:** Минимальные зависимости (`rayon`, `wide`). Не тянет за собой `torch` или `burn`, идеально для встраивания.  
* **Quantization Ready:** Архитектура подготовлена для работы с квантованными весами (baked models) для дальнейшего ускорения.
* **GPU-ускорение (wgpu):** Опциональный GPU бэкенд с WGSL compute шейдерами для параллельного forward/backward.

## **GPU Backend (Опционально)**

ArKan включает опциональный GPU бэкенд на основе `wgpu` для WebGPU/Vulkan/Metal/DX12 ускорения.

### **Установка**

```toml
[dependencies]
arkan = { version = "0.3.0", features = ["gpu"] }
```

### **Использование**

```rust,ignore
use arkan::{KanConfig, KanNetwork};
use arkan::gpu::{WgpuBackend, WgpuOptions, GpuNetwork};
use arkan::optimizer::{Adam, AdamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Инициализация GPU бэкенда
    let backend = WgpuBackend::init(WgpuOptions::default())?;
    println!("GPU: {}", backend.adapter_name());

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

- **Нет пробрасывания DeviceLost:** wgpu 0.23 не предоставляет ошибки `DeviceLost`. Падение GPU может выглядеть как зависание вместо корректной ошибки.
- **Лимит памяти:** `MAX_VRAM_ALLOC = 2GB` на буфер. Превышение возвращает ошибку `BatchTooLarge`.
- **Vec4 выравнивание:** Веса дополняются до границы vec4 (4 элемента) для эффективности шейдеров.
- **CPU fallback:** Если GPU недоступен, инициализация бэкенда корректно завершается с ошибкой `AdapterNotFound`.

### **Запуск GPU тестов и бенчмарков**

```bash
# GPU parity тесты
cargo test --features gpu --test gpu_parity -- --ignored

# GPU бенчмарки (Windows PowerShell)
$env:ARKAN_GPU_BENCH="1"; cargo bench --bench gpu_forward --features gpu

# GPU бенчмарки (Linux/macOS)
ARKAN_GPU_BENCH=1 cargo bench --bench gpu_forward --features gpu
```

## **Бенчмарки (CPU)**

Сравнение ArKan (Rust) против оптимизированной векторизованной реализации на PyTorch (CPU).

**Тестовый стенд:**

* **Config:** Input 21, Output 24, Hidden \[64, 64\], Grid 5, Spline Order 3\.  
* **ArKan:** `cargo bench --bench forward` (AVX2/Rayon enabled).  
* **PyTorch:** Optimized vectorized implementation (без Python-циклов).

| Batch Size | ArKan (Time) | ArKan (Throughput) | PyTorch (Time) | PyTorch (Throughput) | Вывод |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | **30.5 µs** | **0.69 M elems/s** | 990.0 µs | 0.02 M elems/s | **Rust быстрее в 32x** (Low Latency) |
| 16 | 454.8 µs | 0.74 M elems/s | 1.67 ms | 0.20 M elems/s | Rust быстрее в 3.7x |
| 64 | 1.95 ms | 0.69 M elems/s | 3.27 ms | 0.41 M elems/s | Rust быстрее в 1.7x |
| 256 | 7.29 ms | 0.74 M elems/s | 9.65 ms | 0.56 M elems/s | Rust быстрее в 1.3x

### **Анализ производительности**

1. **Small Batch Dominance:** На единичных запросах (`batch=1`) ArKan **опережает** PyTorch за счет отсутствия оверхеда интерпретатора и абстракций. Это позволяет совершать \~33,000 инференсов в секунду против \~1,000 у PyTorch.  
2. **Mid-Batch Performance:** На средних батчах (16-64) ArKan сохраняет преимущество в 1.7x-3.7x, демонстрируя хорошую масштабируемость.  
3. **Throughput Scaling:** На больших батчах (256+) ArKan сохраняет преимущество 1.3x благодаря zero-allocation архитектуре и эффективному использованию кэша.
4. **Zero-Allocation Training:** Весь training loop (forward + backward + update) работает без аллокаций при прогретом Workspace.

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
arkan = "0.3.0"
```

Пример использования (смотрите также `examples/basic.rs` и `examples/training.rs`):
```rust,ignore
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

* **`KanLayer`**: Реализует слой KAN. Хранит сплайновые коэффициенты. Использует локальное окно `order+1` для вычислений, что позволяет эффективно использовать кэш CPU.  
* **`Workspace`**: Ключевая структура для производительности. Содержит выровненные (`AlignedBuffer`) буферы для промежуточных вычислений. Переиспользуется между вызовами.  
* **`spline`**: Модуль с реализацией алгоритма Cox-de Boor. Содержит SIMD-интринсики.

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

Based on the Kolmogorov-Arnold representation theorem. For a layer with $N\_{in}$ inputs and $N\_{out}$ outputs, the transformation looks like this:

$$x\_{l+1, j} \= \\sum\_{i=1}^{N\_{in}} \\phi\_{l, j, i}(x\_{l, i})$$

Where $\\phi\_{l, j, i}$ is a learnable 1D function connecting the $i$-th input neuron to the $j$-th output neuron.

### **Implementation in ArKan (B-Splines)**

In this library, $\\phi$ functions are parameterized using **B-Splines**. This allows modifying the shape of the activation function locally while maintaining smoothness.

Equation for a specific weight in ArKan:

$$\\phi(x) \= \\sum\_{i=1}^{G+p} c\_i \\cdot B\_i(x)$$

* $B\_i(x)$ — B-spline basis functions.  
* $c\_i$ — learnable coefficients.  
* $G$ — grid size.  
* $p$ — spline order.

## **Key Features**

* **Zero-Allocation Inference:** The entire `forward` pass runs on a pre-allocated buffer (`Workspace`). No allocations in the Hot Path.  
* **Zero-Allocation Training:** The full training step (forward + backward + SGD/Adam) also runs without allocations on a warmed-up Workspace.
* **SIMD-Optimized B-Splines:** B-spline basis evaluation is vectorized (AVX2/AVX-512 via `wide` crate).  
* **Cache-Friendly Layout:** Weights are stored in `[Output][Input][Basis]` format for sequential memory access and minimal cache misses.  
* **Standalone:** Minimal dependencies (`rayon`, `wide`). No `torch` or `burn` bloat, ideal for embedding.  
* **Quantization Ready:** Architecture is ready for quantized weights (baked models) for further acceleration.
* **GPU Acceleration (wgpu):** Optional GPU backend with WGSL compute shaders for parallel forward/backward passes.

## **GPU Backend (Optional)**

ArKan includes an optional GPU backend using `wgpu` for WebGPU/Vulkan/Metal/DX12 acceleration.

### **Installation**

```toml
[dependencies]
arkan = { version = "0.3.0", features = ["gpu"] }
```

### **Usage**

```rust,ignore
use arkan::{KanConfig, KanNetwork};
use arkan::gpu::{WgpuBackend, WgpuOptions, GpuNetwork};
use arkan::optimizer::{Adam, AdamConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU backend
    let backend = WgpuBackend::init(WgpuOptions::default())?;
    println!("GPU: {}", backend.adapter_name());

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

- **No DeviceLost propagation:** wgpu 0.23 does not expose `DeviceLost` errors. GPU crashes may appear as hangs instead of proper errors.
- **Memory limits:** `MAX_VRAM_ALLOC = 2GB` per buffer. Exceeding this returns `BatchTooLarge` error.
- **Vec4 alignment:** Weights are padded to vec4 (4-element) boundaries for shader efficiency.
- **CPU fallback:** If GPU is unavailable, the backend initialization fails gracefully with `AdapterNotFound`.

### **Choosing Backend**

```rust,ignore
// High-performance GPU (default)
let backend = WgpuBackend::init(WgpuOptions::default())?;

// Compute-optimized (larger buffers)
let backend = WgpuBackend::init(WgpuOptions::compute())?;

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

## **Benchmarks (CPU)**

Comparison of ArKan (Rust) vs. optimized vectorized PyTorch implementation (CPU).

**Test Setup:**

* **Config:** Input 21, Output 24, Hidden \[64, 64\], Grid 5, Spline Order 3\.  
* **ArKan:** `cargo bench --bench forward` (AVX2/Rayon enabled).  
* **PyTorch:** Optimized vectorized implementation (no Python loops).

| Batch Size | ArKan (Time) | ArKan (Throughput) | PyTorch (Time) | PyTorch (Throughput) | Conclusion |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | **30.5 µs** | **0.69 M elems/s** | 990.0 µs | 0.02 M elems/s | **Rust is 32x faster** (Low Latency) |
| 16 | 454.8 µs | 0.74 M elems/s | 1.67 ms | 0.20 M elems/s | Rust is 3.7x faster |
| 64 | 1.95 ms | 0.69 M elems/s | 3.27 ms | 0.41 M elems/s | Rust is 1.7x faster |
| 256 | 7.29 ms | 0.74 M elems/s | 9.65 ms | 0.56 M elems/s | Rust is 1.3x faster

### **Performance Analysis**

1. **Small Batch Dominance:** On single requests (`batch=1`), ArKan **outperforms** PyTorch due to the lack of interpreter overhead and abstractions. This allows for \~33,000 inferences per second vs \~1,000 for PyTorch.  
2. **Mid-Batch Performance:** On medium batches (16-64), ArKan maintains a 1.7x-3.7x advantage, showing good scalability.  
3. **Throughput Scaling:** On large batches (256+), ArKan maintains 1.3x advantage due to zero-allocation architecture and efficient cache utilization.
4. **Zero-Allocation Training:** The entire training loop (forward + backward + update) runs without allocations on a warmed-up Workspace.

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
arkan = "0.3.0"
```

Usage Example (see also `examples/basic.rs` and `examples/training.rs`):
```rust,ignore
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

* **`KanLayer`**: Implements the KAN layer. Stores spline coefficients. Uses a local window `order+1` for calculations, allowing efficient CPU cache usage.  
* **`Workspace`**: Key structure for performance. Contains aligned (`AlignedBuffer`) buffers for intermediate calculations. Reused between calls.  
* **`spline`**: Module with the Cox-de Boor algorithm implementation. Contains SIMD intrinsics.

## **License**

Distributed under a dual license **MIT** and **Apache-2.0**.

