# Changelog

All notable changes to ArKan will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Unreleased

Pre-1.0, so a minor bump carries the breaking changes below.

### Breaking

#### Feature flags

Four of the six declared features gated nothing at all — `grep -rn 'feature =
"X"' src/` returned zero hits for `simd`, `parallel`, `nightly` and
`quantization`. The declared set now matches what the code actually does.

- **Removed `simd`.** It was a **no-op**: B-spline vectorization goes through the
  `wide` crate unconditionally (`src/layer.rs`, `src/spline.rs`) and the flag
  gated nothing. If you passed `features = ["simd"]`, drop it — **nothing about
  your build changes**, the SIMD paths were and remain always on.
- **Removed `nightly`.** No-op, zero uses. Stable Rust only.
- **Removed `quantization` and the `half` dependency.** No-op: `half::` was never
  referenced anywhere in the crate. `BakedModel` has its own int8/int16
  fixed-point scheme and never used `f16`. If you passed
  `features = ["quantization"]`, drop it — baked inference is available in a
  default build.
- **`parallel` is now real** and `rayon` is an optional dependency behind it
  (`parallel = ["dep:rayon"]`). Everything that used rayon is now
  `#[cfg(feature = "parallel")]`:
  - `KanLayer::backward_parallel` — **does not exist** without the feature.
  - `KanNetwork::forward_batch_parallel` — **does not exist** without the feature.
  - the automatic parallel branch of the backward pass inside
    `KanNetwork::train_step` (and the rest of the `train_step` / `try_train_step`
    family).

  Without `parallel`, `KanConfig::multithreading_threshold` is ignored and every
  batch size takes the sequential `KanLayer::backward` path. Gradients are
  unchanged (parity is asserted in `tests/backward_correctness.rs`); large
  batches simply run on one core instead of the thread pool. **If you call
  `backward_parallel` or `forward_batch_parallel`, or train with large batches
  and want multi-core, add `features = ["parallel"]`.**

  Rationale: ArKan's niche is embeddable, low-latency, batch=1 inference. Those
  users should not have to pull `rayon` → `rayon-core` → `crossbeam-*` +
  `either`. A default build's normal dependency edges are now exactly `rand`,
  `thiserror` and `wide`.

#### Behaviour

- `KanLayer::backward` / `backward_parallel` now report `grad_input = 0` for
  inputs where normalization saturates against `grid_range`, instead of scaling
  the spline derivative by `1 / std` as if the clamp were not there. This is the
  mathematically correct gradient, but the numbers you get out of `backward`
  change for any saturated input.
- `BakedModel::from_network` now **panics** with a descriptive message for a
  layer whose `spline_order` is outside `2..=5`, instead of reading out of
  bounds. Orders 4 and 5 also produce different (correct) values now — the
  fixed-point basis Q scales were wrong.

### Added

#### BakedModel — working int8 quantized inference path

- **`BakedModel`** is now a fully functional quantized inference path (was a
  non-functional stub that panicked on `forward()`). It is **no longer
  deprecated**.
- **Per-channel int8 weights** — each output neuron has its own scale
  `s_w[j] = 127 / max|w[j,*,*]|`, maximizing range utilization per channel.
- **int16 basis** — B-spline bases stored in Q0.15 (`u16`); accumulator `i64`.
- **i32 inter-layer activations** — 28-bit target range (`2^28`) reduces
  inter-layer error amplification vs the old i16 scheme.
- **Percentile calibration** — `BakedModel::from_network(net, Some(&calib))`
  uses the 99.9th-percentile of activation magnitudes to set requantization
  scales; outliers saturate instead of wasting dynamic range.
- **Magic + versioned serialization** (`serde` feature) — `to_bytes()` prepends
  `MAGIC_BAKED` (`b"KAN_BAKED_v1"`) and a `u32` format version before the
  bincode body. `from_bytes()` validates both before deserializing and returns a
  descriptive `Err` (not a panic) on mismatch or truncation.
- **`examples/baked_inference.rs`** — runnable end-to-end example: build,
  train briefly, calibrate, bake, compare outputs, print size ratio.

#### Accuracy — measured, including the tail

`cargo test --test baked_parity -- --nocapture`, random-init networks,
256 calibration samples and 2000 test samples in `[-0.9, 0.9]`, `grid_range =
(-1, 1)`. "Worst-case" is `max |baked − f32| / |f32|` restricted to outputs
above the given multiple of the per-output std.

| Architecture | order | NRMSE | worst @0.1σ | @0.5σ | @1.0σ |
|---|---|---|---|---|---|
| 4→2 | 3 | 0.60% | 9.2% | 8.3% | 8.3% |
| 4→[8]→2 | 3 | 0.64% | 8.7% | 8.7% | 8.7% |
| 8→[16,8]→4 | 3 | 1.29% | 114.7% | 27.1% | 18.7% |
| 8→[16,8]→4 (seed 4242) | 2 | 2.65% | 474.5% | 62.3% | **34.6%** |
| 8→[16,8]→4 (seed 4242) | 3 | 1.71% | 53.9% | 53.9% | **53.9%** |
| 8→[16,8]→4 (seed 4242) | 4 | 2.26% | 149.5% | 63.1% | **50.0%** |
| 8→[16,8]→4 (seed 4242) | 5 | 2.28% | 326.1% | 78.3% | **35.1%** |

The aggregate NRMSE (0.6–2.7%) is good; the **per-output tail is not**. On a
2-hidden net the worst-case error on decision-relevant outputs (≥1σ) is
**34.6–53.9%**. Baked is therefore suitable for **ranking / argmax /
classification** and unsuitable for per-output absolute accuracy. See
`docs/BENCHMARKS.md` for the two identified, still-unfixed causes.

**Baked is slower than f32 at batch=1** (1.4× on 4→[8]→2, 2.1× on
8→[16,8]→4, re-measured 2026-07-26). Its win today is size (2.2–3.0×), not
speed.

### Fixed

- **Order-4 and order-5 baked B-spline basis were numerically wrong**
  (`576fbc7`). `eval_basis_fixed` summed numerator terms at inconsistent
  fixed-point Q scales: order 4 (Q64) left `12*t3` and `4*t3` at Q48; order 5
  (Q80) was short by one to two factors of 65536 in nine terms. Max absolute
  basis error vs the f32 reference dropped from **0.208 → 0.000117** (order 4)
  and **0.775 → 0.000143** (order 5), over all 65536 `t_q16` values. End-to-end
  NRMSE on a 4→[8]→2 net went **13.98% → 0.98%** (order 4) and **89.30% →
  1.21%** (order 5). Order 3 is bit-identical.
  **Any published baked accuracy number for orders 4 or 5 predating this commit
  was measured on broken arithmetic and is void.**
- **Two panics reachable through `BakedModel`** (`8f0c44d`).
  `BakedModel::from_network` now panics with a descriptive message for a layer
  whose `spline_order` is outside `2..=5`, instead of reading out of bounds —
  `KanConfig::validate()` accepts up to `MAX_SPLINE_ORDER = 7`, but baked only
  implements 2..=5. And a `grid_range` narrower than ~1.5e-5 (e.g. the
  validate-approved `(0.0, 0.000005)`) no longer panics with `min > max`;
  `q_rmax` is floored at `q_rmin`. Deleting the fallback arm also drops baked
  support for `spline_order = 1`.
- **`grad_input` was non-zero where the clamp had saturated** — forward and
  backward disagreed (`186fd75`, CPU and GPU). Forward computes
  `z = clamp((x − mean)/std, grid_min, grid_max)`, so `dz/dx` is exactly 0
  outside the range; backward scaled the spline derivative by `1/std`
  unconditionally. Hidden layers are where this bites, because their inputs are
  the previous layer's raw activations and routinely leave the grid (50% of them
  in `tests/clamp_gradient_parity.rs`'s fixture). Finite differences on layer-0
  weights showed gaps up to **1.6e-2 against true gradients of ~5e-4 — roughly
  30× too large, several with the wrong sign**. GPU `grad_input` on saturated
  inputs went from 5.7e-1 to 0. Forward now records the clamp in the high bit of
  the stored span index (`SPAN_CLAMPED_FLAG`) and backward reads it; no new
  buffer, no signature change.
- **`examples/game2048` starved both of its hidden layers** (`f614b50`). It used
  `grid_range(0.0, 1.0)` reasoning "one-hot values are 0 or 1" — true of the
  inputs, false of everything downstream. Measured on the shipped config
  (256 → [64, 32] → 4): layer 0 0% saturated, **layer 1 43.6%, layer 2 48.9%**.
  Now `(-1.0, 1.0)`, which measures 0% on all three.
  `tests/hidden_layer_saturation.rs` pins both directions.
- **`benches/optimizer.rs` did not compile** (`f06296e`) —
  `SGD::new(&network, 0.001, 0.9, 0.0)` against the real
  `SGD::new(&KanNetwork, SGDConfig)`. `docs/BENCHMARKS.md` was citing numbers
  from a bench that could not be built. CI now type-checks benches
  (`--all-targets`).
- Two deny-by-default `clippy::erasing_op` errors in
  `tests/forward_correctness.rs` (`0317a1b`), the ~70-warning clippy backlog
  (`a175247`) and `cargo fmt` across the repo (`029ad82`).
- `clippy::incompatible_msrv` false positives on `Option::is_none_or` in the
  `gpu` module, which can never build at 1.73 anyway (`876099b`).

### Changed

- `BakedModel` re-export in `lib.rs` is now a clean `pub use` (no longer
  wrapped in `#[allow(deprecated)]`).
- `MAGIC_SPLINE` constant removed from `lib.rs` — no spline file format exists
  in this library; the constant was dead code.
- `package.description` no longer says "for poker solver". That was a leftover
  from where this library started; it is a general-purpose KAN crate.
- `examples/game2048` now depends on `arkan` with `features = ["gpu", "parallel"]`
  because it calls `forward_batch_parallel`.
- **Test coverage that would have caught the order-4/5 defect** (`1ab2a5d`).
  The basis sweep used `step_by(64)`, visiting 1.56% of the domain — now
  exhaustive. Every baked test, bench and example hard-coded `order = 3`, so
  orders 4 and 5 had zero end-to-end coverage; `baked_parity_all_orders` now
  covers 2..=5.
- **Removed `examples/comprehensive_test.rs` and
  `examples/gpu_comprehensive_test.rs`** (`445a020`, 549 lines). Both counted
  failures into local integers and then returned `Ok(())` unconditionally, so
  they exited 0 whether they passed or not. Every check they made is already
  covered by `tests/`.
- `.claude/` added to `.gitignore` (`fa26d28`).

#### Declared MSRV

`rust-version = "1.73"` is now set. Determined by bisecting real toolchains
against a checkout with no `Cargo.lock`: 1.72 fails on our own `div_ceil`
(`int_roundings`, stable since 1.73), 1.73 builds the library with default
features and with `serde`.

There is no single MSRV, because the optional features' dependencies set their
own floors. The README carries the full table:

| Build | MSRV | Set by |
|---|---|---|
| default, `serde` | 1.73 | our own `div_ceil` |
| `parallel` | 1.80 | `rayon-core` |
| `gpu` | 1.85 | `indexmap`, via `wgpu` 23 → `naga` |

No `Cargo.lock` is committed, so resolution always picks the newest compatible
dependencies and these floors drift upward as those crates release.

#### Documentation

- **README, BENCHMARKS, ARCHITECTURE and this file reconciled with the code.**
  The English README still described `BakedModel` as a deprecated stub whose
  `forward()` panics; the GPU snippet called `backend.adapter_name()`, which does
  not exist (`adapter_info()` does), in both language halves; the baked
  calibration snippet was an empty `vec![]`, which silently bakes a model at 100%
  NRMSE; the GPU-vs-PyTorch table predated the 2026-06-27 re-measurement;
  `docs/ARCHITECTURE.md` documented the weight layout with input and output
  swapped and listed three features that no longer exist; `docs/BENCHMARKS.md`
  told contributors to pass `--features simd` (now a hard error) and linked four
  times into `tasks/`, which is gitignored and package-excluded.
- **`tests/readme_snippets.rs`** — every README and BENCHMARKS code snippet,
  copied verbatim. The CPU ones run; the `gpu` ones are type-checked by
  `cargo clippy --all-targets --features gpu` in CI. A README example that stops
  compiling now fails a job.
- Four test files cited `FUNCTIONALITY_AUDIT.md` — a gitignored, package-excluded
  file — from source that ships in the published crate. Reworded.
- `cargo doc` is now clean under `-D warnings` for default, `serde`, `gpu` and
  `--all-features`. 72 unresolved intra-doc links are gone: most were math and
  index notation (`M0[j]`, `grad_weights[j,i,k]`) that rustdoc parsed as links
  and now render as code, but 14 were genuinely broken references — including
  `crate::KanNetwork::forward`, a method that does not exist (the f32 CPU path
  is `forward_single` / `forward_batch`).

#### CI

`cargo fmt`, `cargo clippy --all-targets`, tests, doctests and `cargo doc` now
run across a feature matrix (`--no-default-features`, `serde`, `parallel`,
`serde,parallel`, `gpu`, `--all-features`), plus `cargo package` and a pinned
1.73 MSRV job. Benches are type-checked (`--all-targets`) and `gpu` test code is
compile-checked (`--no-run`, since GPU tests need an adapter no runner has).

### Known limitations (not fixed in 0.4.0)

Documented rather than hidden. None of these are regressions; they are the state
of the library as shipped.

- **`BakedModel` is slower than `KanNetwork` at batch=1** (1.4–2.1× on the two
  shipped bench configs). A design that reverses this — the win is a weight
  **layout** change to output-innermost plus hoisting the basis evaluation out
  of the `j` loop, *not* SIMD — has been prototyped and measured at roughly
  1.5–2.8× **faster** than f32 depending on shape. It is **not implemented**.
- **`BakedModel` cannot return an output magnitude above the calibration set's
  99.9th percentile.** `ACT_CLAMP` (2^28) is applied to the output layer's
  activations too, and the exit scale is `2^28 / p99.9`, so the clamp becomes a
  hard ceiling on the dequantized output. Unfixed.
- **`norm_a_fixed` carries only ~3 bits.** The inter-layer scale
  `A_FIXED[i] = round(2^32 / (s_act_prev · std_i))` lands on small integers
  (measured 5–10 on the parity fixtures), so rounding it costs a systematic
  2.6–6.7% per inter-layer hop. Together with the ceiling above, this is the
  likely origin of the ≥1σ tail. Unfixed.
- **Only layer 0 receives `input_mean` / `input_std`.** Every hidden layer is
  built with identity normalization and there is no running-statistics update,
  so a hidden layer's input is the previous layer's *raw* activation, clamped to
  the shared `grid_range`. Nothing bounds a KAN layer's output to its own grid
  range. Choose `grid_range` for the activations, not for the inputs.
- **Saturation is silent.** There is no `out_of_grid_fraction`, no drift
  warning, no configurable extrapolation and no grid recalibration. Distribution
  drift shows up as unexplained accuracy loss, not as a diagnostic.

---

## [0.3.0] - 2025-12-06

### Added

#### Native GPU Training
- **`train_step_gpu_native()`** — Full GPU pipeline without CPU↔GPU weight transfers
- **`train_step_gpu_native_sgd()`** — SGD variant of native GPU training
- **`GpuAdam`** — GPU-resident Adam optimizer with moment buffers on VRAM
- **`GpuSgd`** — GPU-resident SGD optimizer with optional momentum
- **`GpuAdamConfig`**, **`GpuSgdConfig`** — Configuration structs for GPU optimizers
- **`GpuLayer::allocate_gradient_buffers()`** — Pre-allocate gradient storage on GPU
- **`GpuNetwork::prepare_native_training()`** — Initialize all layers for native training
- **`VramLimit` enum** (`Bytes`, `Gigabytes`, `Percent`, `Unlimited`) with `with_max_vram()` / `with_max_vram_percent()` helpers
- **`forward_batch_async()`** returning `GpuForwardHandle` for async GPU inference

#### Optimizer Module v2.0 / v2.1
- **`StandaloneLBFGS`** with two-loop recursion and Strong Wolfe / backtracking line search
- **`SGD` Nesterov momentum**, **`ParamGroup`** structure, **`Workspace::zero_grad()`**
- **`trait Optimizer`** — unified API; `Send + Sync` thread safety; `bump_version()`
- **`SafetyConfig`** (`safety` field on `AdamConfig`/`SGDConfig`) — NaN handling, AMP placeholder

#### Loss Functions
- KAN-specific regularization: `l1_sparsity_loss`, `entropy_regularization`, `smoothness_penalty`, `kan_combined_loss`
- Physics-informed losses: `pde_residual_loss`, `r_squared`

#### Reinforcement Learning Utilities
- **`ShardedReplayBuffer`** — 16-shard replay buffer for reduced lock contention (DQN use-case)

#### CI / Integration Tests
- GitHub Actions workflow: `build`, `examples`, `gpu-build`, `docs` jobs
- `tests/examples_integration.rs` — 12 integration tests covering inference, training, config, workspace
- `examples/game2048` DQN unit tests: Bellman equation, terminal state, selective update, shard fairness

#### Performance
- **2-5x faster training** vs hybrid GPU (no CPU↔GPU sync overhead)
- Gradients stay on GPU between backward and optimizer steps
- Adam moment vectors (m, v) stored entirely on VRAM
- **`forward_batch_parallel()`** — multi-core CPU inference via Rayon

### Changed

- `GpuNetwork::train_step()` now uses hybrid mode (backward GPU, optimizer CPU) by default
- Native training requires explicit call to `prepare_native_training()` first

### Fixed

- **GPU input gradient bug** — `compute_input_grad` now true for all layers (was false for layer 0)
- **Serialization knots bug** — custom `Deserialize` for `KanLayer` recomputes knots after load

### Performance

| Method | Batch=64 | Notes |
|--------|----------|-------|
| CPU train_step | 4.77 ms | Baseline |
| Hybrid GPU (old) | ~10 ms | Forward GPU, optimizer CPU |
| **Native GPU** | **~2-3 ms** | Full GPU pipeline |

---

## [0.2.0] - 2025-12-05

### Added

#### GPU Backend (new feature: `gpu`)
- **wgpu 0.23 compute shaders** for forward/backward pass on Vulkan/DX12/Metal
- `GpuBackend`, `GpuNetwork`, `GpuWorkspace` for GPU-accelerated inference and training
- Native GPU optimizers: `GpuAdam`, `GpuSgd` with full GPU-resident training
- GPU crossover point ~batch 16: GPU wins for batch ≥16, CPU wins for single-sample
- Up to **17.9x speedup** over CPU at batch=1024

#### Safety & Error Handling
- `checked_buffer_size()` and `checked_buffer_size3()` for overflow protection
- `try_forward_batch()`, `try_train_step()` fallible variants that return `ArkanResult`
- `try_new()` for `KanLayer` with overflow checks
- `try_prepare_forward()`, `try_prepare_training()` for workspace
- `try_create_workspace()` for network
- `ArkanError::Overflow`, `ArkanError::BatchTooLarge` error variants
- `MAX_BUFFER_ELEMENTS` constant (256M elements) for safe allocation limits

#### API Improvements
- `KanConfig::validate()` now catches zero hidden dims, mismatched normalization
- Warning (not error) on zero/negative `input_std` values
- Export `checked_buffer_size`, `MAX_BUFFER_ELEMENTS` from crate root

#### Testing
- 31 new regression tests in `tests/regression_v020.rs`
- 55 GPU parity tests in `tests/gpu_parity.rs` (require `--ignored` flag)
- Total: 183 tests (103 unit + 31 regression + 49 doc-tests)

### Changed

- `Workspace` now uses `max_dim` across all layers for buffer sizing (fixes wide hidden layer bug)
- All `.expect()` in GPU workspace replaced with `.ok_or_else()` returning `ArkanResult`
- `BakedModel::forward()` now uses `unimplemented!()` instead of returning incorrect data

### Deprecated

- `BakedModel` - stub module, full implementation planned for v0.4.0

### Fixed

- **P0**: Buffer overflow in `forward_batch` with large batch sizes
- **P0**: Workspace undersized for networks with hidden layers wider than input
- **P1**: GPU workspace bind group creation could panic on missing buffers
- **P1**: `batch_size == 0` now returns error instead of undefined behavior

### Performance

- **CPU forward_batch**: ~26 µs at batch=1 (was ~61 µs), **57% improvement**
- **CPU train_step**: ~100 µs at batch=1, **8-15% improvement**
- **GPU forward**: 314 µs at batch=64 (6.2x faster than CPU)
- **Zero-allocation** inference and training paths verified

## [0.1.0] - 2024-11-XX

### Added

- Initial release
- CPU-only KAN implementation with B-spline basis functions
- `KanNetwork`, `KanLayer`, `KanConfig` core types
- `Workspace` for zero-allocation inference
- Adam and SGD optimizers
- Poker preset configuration `[21, 64, 64, 24]`
- SIMD-accelerated forward pass (AVX2/NEON)
- Rayon parallel training

---

## Roadmap

Nothing below is implemented. Items are listed only where a concrete design or
measurement exists; see "Known limitations" under 0.4.0 for what they would fix.

### Next
- [ ] `BakedModel` fast path — output-innermost weight layout + hoisted basis
      evaluation. Prototyped and measured at ~1.5–2.8× faster than f32 at
      batch=1; not implemented.
- [ ] Grid-domain observability — `out_of_grid_fraction`, drift warning,
      configurable extrapolation, grid recalibration. Saturation is silent today.
- [ ] Close the baked ≥1σ tail — stop applying `ACT_CLAMP` to the output layer,
      and give `norm_a_fixed` more than ~3 bits.

### Unscheduled
- [ ] ONNX export
- [ ] Model pruning utilities
- [ ] Async GPU pipeline for overlapped compute
- [ ] Multi-GPU support
