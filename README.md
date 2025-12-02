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
* **SIMD-Optimized B-Splines:** Вычисление базисных функций B-сплайнов векторизовано (AVX2/AVX-512 через крейт `wide`).  
* **Cache-Friendly Layout:** Веса хранятся в формате `[Output][Input][Basis]` для последовательного доступа к памяти и минимизации промахов кэша.  
* **Standalone:** Минимальные зависимости (`rayon`, `wide`). Не тянет за собой `torch` или `burn`, идеально для встраивания.  
* **Quantization Ready:** Архитектура подготовлена для работы с квантованными весами (baked models) для дальнейшего ускорения.

## **Бенчмарки (CPU)**

Сравнение ArKan (Rust) против оптимизированной векторизованной реализации на PyTorch (CPU).

**Тестовый стенд:**

* **Config:** Input 21, Output 24, Hidden \[64, 64\], Grid 5, Spline Order 3\.  
* **ArKan:** `cargo bench --bench forward` (AVX2/Rayon enabled).  
* **PyTorch:** Optimized vectorized implementation (без Python-циклов).

| Batch Size | ArKan (Time) | ArKan (Throughput) | PyTorch (Time) | PyTorch (Throughput) | Вывод |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | **30.1 µs** | **0.69 M elems/s** | 950.0 µs | 0.02 M elems/s | **Rust быстрее в 31x** (Low Latency) |
| 16 | 0.98 ms | 0.34 M elems/s | 1.85 ms | 0.18 M elems/s | Rust быстрее в 1.9x |
| 64 | 3.93 ms | 0.34 M elems/s | 2.78 ms | 0.48 M elems/s | PyTorch вырывается вперед (AVX/MKL) |
| 256 | 15.7 ms | 0.34 M elems/s | 9.32 ms | 0.58 M elems/s | PyTorch эффективнее на массовых данных |

### **Анализ производительности**

1. **Small Batch Dominance:** На единичных запросах (`batch=1`) ArKan **уничтожает** PyTorch за счет отсутствия оверхеда интерпретатора и абстракций. Это позволяет совершать \~33,000 инференсов в секунду против \~1,000 у PyTorch.  
2. **Throughput Scaling:** На больших батчах PyTorch выигрывает за счет высокооптимизированных BLAS-библиотек (MKL/OpenBLAS), которые лучше утилизируют кэш на огромных матрицах. Однако цель ArKan — предсказуемая задержка (Latency) на малых батчах, критичная для real-time систем.

## **Сравнение с аналогами (Prior Art)**

ArKan занимает нишу **специализированного высокопроизводительного инференса**.

| Крейт | Назначение | Отличие ArKan |
| :---- | :---- | :---- |
| [`burn-efficient-kan`](https://crates.io/crates/burn-efficient-kan) | Часть экосистемы [Burn](https://burn.dev). Отлично подходит для обучения на GPU. | ArKan — легковесная библиотека без тяжелых зависимостей (Burn/Torch/WGPU). Идеальна для встраивания. |
| [`fekan`](https://crates.io/crates/fekan) | Богатый функционал (CLI, dataset loaders). General-purpose библиотека. | ArKan изначально спроектирован под SIMD (AVX2) и параллелизм, тогда как в `fekan` это пока в планах. |
| [`rusty_kan`](https://crates.io/crates/rusty_kan) | Базовая реализация, образовательный проект. | ArKan фокусируется на production-ready оптимизациях: workspace, батчинг, многопоточность. |

## Быстрый старт

Установка из crates.io:

```toml
[dependencies]
arkan = "0.1"
```
Пример использования (смотрите также `examples/basic.rs`):
```
use arkan::{KanConfig, KanNetwork};

fn main() {
    // 1. Конфигурация (Poker Solver preset)
    let config = KanConfig::default_poker();

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
* **SIMD-Optimized B-Splines:** B-spline basis evaluation is vectorized (AVX2/AVX-512 via `wide` crate).  
* **Cache-Friendly Layout:** Weights are stored in `[Output][Input][Basis]` format for sequential memory access and minimal cache misses.  
* **Standalone:** Minimal dependencies (`rayon`, `wide`). No `torch` or `burn` bloat, ideal for embedding.  
* **Quantization Ready:** Architecture is ready for quantized weights (baked models) for further acceleration.

## **Benchmarks (CPU)**

Comparison of ArKan (Rust) vs. optimized vectorized PyTorch implementation (CPU).

**Test Setup:**

* **Config:** Input 21, Output 24, Hidden \[64, 64\], Grid 5, Spline Order 3\.  
* **ArKan:** `cargo bench --bench forward` (AVX2/Rayon enabled).  
* **PyTorch:** Optimized vectorized implementation (no Python loops).

| Batch Size | ArKan (Time) | ArKan (Throughput) | PyTorch (Time) | PyTorch (Throughput) | Conclusion |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | **30.1 µs** | **0.69 M elems/s** | 950.0 µs | 0.02 M elems/s | **Rust is 31x faster** (Low Latency) |
| 16 | 0.98 ms | 0.34 M elems/s | 1.85 ms | 0.18 M elems/s | Rust is 1.9x faster |
| 64 | 3.93 ms | 0.34 M elems/s | 2.78 ms | 0.48 M elems/s | PyTorch pulls ahead (AVX/MKL) |
| 256 | 15.7 ms | 0.34 M elems/s | 9.32 ms | 0.58 M elems/s | PyTorch is more efficient on mass data |

### **Performance Analysis**

1. **Small Batch Dominance:** On single requests (`batch=1`), ArKan **destroys** PyTorch due to the lack of interpreter overhead and abstractions. This allows for \~33,000 inferences per second vs \~1,000 for PyTorch.  
2. **Throughput Scaling:** On very large batches, PyTorch wins due to highly optimized BLAS libraries (MKL), but ArKan maintains predictable execution time without GC pauses (Stop-the-world). The goal of ArKan is low latency.

## **Comparison with Analogues (Prior Art)**

ArKan occupies the niche of **specialized high-performance inference**.

| Crate | Purpose | Difference from ArKan |
| :---- | :---- | :---- |
| [`burn-efficient-kan`](https://crates.io/crates/burn-efficient-kan) | Part of the [Burn](https://burn.dev) ecosystem. | ArKan is a lightweight library without heavy framework dependencies. Ideal for embedding. |
| [`fekan`](https://crates.io/crates/fekan) | Rich functionality, general-purpose library. | ArKan is designed with SIMD/Parallelism as core features, whereas `fekan` plans to add them later. |
| [`rusty_kan`](https://crates.io/crates/rusty_kan) | Basic implementation, educational project. | ArKan focuses on production-ready optimizations: workspace, batching, multithreading. |

## **Quick Start**

Install from crates.io:

```toml
[dependencies]
arkan = "0.1"
```

Usage Example (see also `examples/basic.rs`):
```
use arkan::{KanConfig, KanNetwork};

fn main() {
    // 1. Configuration (Poker Solver preset)
    let config = KanConfig::default_poker();

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

