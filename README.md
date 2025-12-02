# ArKan

Высокопроизводительная реализация KAN (Kolmogorov-Arnold Network) для покерного солвера: выравнивание под SIMD, нулевая аллокация в хот-пате, заготовка под квантованные (baked) модели.

## Возможности
- Размещение весов в row-major `[out][in][basis]` для кеш-дружественного доступа.
- Предвыделенный workspace для батчевого `forward` без аллокаций в цикле.
- B-spline базис с глобальными и локальными весами (grid + window).
- Настраиваемый `simd_width` (4/8/16) для выравнивания под SIMD.

## Пример
```rust
use arkan::{KanConfig, KanNetwork};

let config = KanConfig::default_poker();
let network = KanNetwork::new(config.clone());
let mut workspace = network.create_workspace(64);

let inputs = vec![0.0f32; 64 * config.input_dim];
let mut outputs = vec![0.0f32; 64 * config.output_dim];
network.forward_batch(&inputs, &mut outputs, &mut workspace);
```

## Лицензия
MIT OR Apache-2.0, на выбор пользователя.
