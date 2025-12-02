# ArKan

High-performance Kolmogorov-Arnold Network (KAN) in Rust, focused on low-latency inference for poker solvers and other game/real-time workloads. SIMD-friendly layout, zero allocations on the hot path, ready for baked/quantized models.

## Why KAN here?
- Functions live on edges (learnable 1D splines), nodes just sum. This maps well to cache-friendly, vectorized code.
- We parametrize edge functions with B-splines (`grid_size + order` basis), keeping them smooth and local.
- Weight layout `[out][in][basis]` and preallocated workspace avoid allocator overhead.

## Key Features
- Zero-allocation forward: all buffers come from a preallocated `Workspace`.
- SIMD B-splines: vectorized basis evaluation (AVX2/AVX-512 via `wide`).
- Cache-friendly weights: row-major `[Output][Input][Basis]`.
- Minimal deps: mostly `rayon`, `wide`; no heavy frameworks.
- Quantization-ready: baked (f16/i8) path stubbed for production.

## Benchmarks (CPU)
- Rust: `cargo bench --bench forward` (AVX2/Rayon enabled)
- PyTorch (vectorized CPU script): `python scripts/bench_pytorch.py`

| Batch | ArKan time | ArKan thrpt | PyTorch time | PyTorch thrpt | Comment |
|-------|------------|-------------|--------------|---------------|---------|
| 1     | ~30 µs     | ~0.69 M/s   | ~0.95 ms     | ~0.02 M/s     | Rust ~30x faster (latency) |
| 16    | ~0.98 ms   | ~0.34 M/s   | ~1.85 ms     | ~0.18 M/s     | Rust ~1.9x faster |
| 64    | ~3.93 ms   | ~0.34 M/s   | ~2.78 ms     | ~0.48 M/s     | PyTorch faster (MKL/BLAS) |
| 256   | ~15.7 ms   | ~0.34 M/s   | ~9.32 ms     | ~0.58 M/s     | PyTorch wins on large batches |

Notes: PyTorch benchmark uses a vectorized Cox–de Boor implementation on CPU (no Python loops). Install with `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Quick Start
Until crates.io release, use git:
```toml
[dependencies]
arkan = { git = "https://github.com/LutwigStack/ArKan" }
```
Example:
```rust
use arkan::{KanConfig, KanNetwork};

fn main() {
    let config = KanConfig::default_poker();
    let network = KanNetwork::new(config.clone());
    let mut workspace = network.create_workspace(64);

    let inputs = vec![0.0f32; 64 * config.input_dim];
    let mut outputs = vec![0.0f32; 64 * config.output_dim];
    network.forward_batch(&inputs, &mut outputs, &mut workspace);
    println!("out0 = {}", outputs[0]);
}
```

## Architecture
- `KanLayer`: spline weights with local window `order+1`, aligned for SIMD.
- `KanNetwork`: batch forward with zero allocations via `Workspace`.
- `Workspace`: aligned buffers (`AlignedBuffer`) plus grid indices/basis caches, reused across calls.
- `spline`: Cox–de Boor basis, SIMD normalization.
- `baked`: magic bytes and stub for quantized weights (f16/i8).

## Roadmap
- [x] Core layers/network, workspace, optimizers, losses
<|start_of_focus|>
- [x] Zero-allocation forward; SIMD basis
- [x] Benchmarks: Criterion (Rust) + vectorized PyTorch CPU
- [ ] CI (clippy + test) and badge
- [ ] Publish to crates.io / docs.rs and add crates.io/docs.rs badges
- [ ] Baked/quantized export path

## License
Dual-licensed under MIT or Apache-2.0.
