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
- Python (CPU, PyTorch, наивная KAN): `python scripts/bench_pytorch.py`

| Batch | ArKan (Rust) время | ArKan thrpt | PyTorch KAN время | PyTorch thrpt | Комментарий |
|-------|--------------------|-------------|-------------------|---------------|-------------|
| 1     | ~30 µs             | ~0.69 M/s   | ~32.9 ms          | ~0.00064 M/s  | Rust быстрее ~1000x |
| 16    | ~0.98 ms           | ~0.34 M/s   | ~540 ms           | ~0.00062 M/s  | |
| 64    | ~3.93 ms           | ~0.34 M/s   | ~2238 ms          | ~0.00060 M/s  | |
| 256   | ~15.7 ms           | ~0.34 M/s   | ~8778 ms          | ~0.00061 M/s  | |

Примечание: Python-бенч требует `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Roadmap перед релизом
- [x] Структуры слоёв/сети, workspace, оптимизаторы, лоссы.
- [x] Пройти clippy + тесты.
- [x] Добавить бенчмарки (Criterion).
- [ ] Таблица сравнения Rust vs Python (добавить результаты PyTorch).
- [ ] Настроить CI (test + clippy) и бейджи GitHub Actions.
- [ ] Опубликовать на crates.io и docs.rs, обновить бейджи и `cargo add arkan`.

## Лицензия
MIT OR Apache-2.0, на выбор пользователя.
