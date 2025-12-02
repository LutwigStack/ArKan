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

## Roadmap перед релизом
- [x] Структуры слоёв/сети, workspace, оптимизаторы, лоссы.
- [x] Пройти clippy + тесты.
- [ ] Добавить бенчмарки и таблицу Rust vs Python.
- [ ] Настроить CI (test + clippy) и бейджи GitHub Actions.
- [ ] Опубликовать на crates.io и docs.rs, обновить бейджи и `cargo add arkan`.

## Лицензия
MIT OR Apache-2.0, на выбор пользователя.
