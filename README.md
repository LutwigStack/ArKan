# ArKan

![status](https://img.shields.io/badge/status-internal-blue)
![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)
![rustc](https://img.shields.io/badge/rust-1.75%2B-orange)

Высокопроизводительная реализация KAN (Kolmogorov-Arnold Network) для покерного солвера: выравнивание под SIMD, предвыделенный workspace без аллокаций в хот-пате, заготовка под квантованные (baked) модели.

## Возможности
- Выкладка весов в row-major `[out][in][basis]` для кеш-дружественного доступа.
- Предвыделенный workspace для батчевого `forward` без аллокаций.
- B-spline базис с глобальными и локальными весами (grid + window).
- Настраиваемый `simd_width` (4/8/16) для выравнивания.
- Статусы/магик-байты для сохранения spline и baked моделей.

## Быстрый старт
Пока нет публикации на crates.io, можно подключить из репозитория:
```bash
cargo add --git https://github.com/<owner>/ArKan arkan
```
Пример использования:
```rust
use arkan::{KanConfig, KanNetwork};

let config = KanConfig::default_poker();
let network = KanNetwork::new(config.clone());
let mut workspace = network.create_workspace(64);

let inputs = vec![0.0f32; 64 * config.input_dim];
let mut outputs = vec![0.0f32; 64 * config.output_dim];
network.forward_batch(&inputs, &mut outputs, &mut workspace);
```

## Архитектура
- `KanLayer`: сплайновые коэффициенты с локальным окном `order+1`, выравненные под SIMD.
- `KanNetwork`: батчевый forward с нулевыми аллокациями через `Workspace`.
- `Workspace`: выровненные буферы (`AlignedBuffer`) + индексы гридов и precomputed basis.
- `spline`: Cox-de Boor для B-spline, SIMD-нормализация пачек.
- `baked`: магик-байты и заглушка под квантованные веса (f16).

## Бенчмарки
- Rust (Criterion): `cargo bench --bench forward`
- Python (CPU, PyTorch, векторизованный скрипт): `python scripts/bench_pytorch.py`

| Batch | ArKan (Rust) время | ArKan thrpt | PyTorch KAN время | PyTorch thrpt | Комментарий |
|-------|--------------------|-------------|-------------------|---------------|-------------|
| 1     | ~30 µs             | ~0.69 M/s   | ~0.95 ms          | ~0.02 M/s     | Rust быстрее ~30x |
| 16    | ~0.98 ms           | ~0.34 M/s   | ~1.85 ms          | ~0.18 M/s     | |
| 64    | ~3.93 ms           | ~0.34 M/s   | ~2.78 ms          | ~0.48 M/s     | PyTorch догоняет за счет einsum |
| 256   | ~15.7 ms           | ~0.34 M/s   | ~9.32 ms          | ~0.58 M/s     | На CPU PyTorch может быть быстрее для крупных батчей |

Примечание: Python-бенч требует `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Для честного сравнения с CUDA/torch_kan можно добавить GPU-замеры.

## Roadmap перед релизом
- [x] Структуры слоёв/сети, workspace, оптимизаторы, лоссы.
- [x] Пройти clippy + тесты.
- [x] Добавить бенчмарки (Criterion).
- [ ] Таблица сравнения Rust vs Python (добавить результаты PyTorch).
- [ ] Настроить CI (test + clippy) и бейджи GitHub Actions.
- [ ] Опубликовать на crates.io и docs.rs, обновить бейджи и `cargo add arkan`.

## Лицензия
MIT OR Apache-2.0, на выбор пользователя.
