# ArKan Documentation

Technical documentation for the ArKan KAN (Kolmogorov-Arnold Network) library.

## Contents

- [**ARCHITECTURE.md**](ARCHITECTURE.md) — System architecture, module structure, and design decisions, including the [constraints that will bite you first](ARCHITECTURE.md#design-constraints)
- [**BENCHMARKS.md**](BENCHMARKS.md) — Performance benchmarks, CPU vs GPU comparisons, and the [baked int8 accuracy and latency numbers](BENCHMARKS.md#baked-int8-inference)

## Quick Links

- [Main README](../README.md) — Getting started, installation, basic usage
- [API Documentation](https://docs.rs/arkan) — Generated Rust docs
- [Examples](../examples/) — Working examples (sinusoid, MNIST, 2048)

## GPU Backend

ArKan includes an optional GPU backend using wgpu, behind `features = ["gpu"]`. See [ARCHITECTURE.md](ARCHITECTURE.md#gpu-backend-srcgpu) for details on:

- Hybrid training (GPU forward/backward + CPU optimizer)
- Native GPU training with GpuAdam
- Memory management and workspace allocation

## Contributing

When adding new documentation:
1. Place technical docs in this `docs/` folder
2. Update this README with links
3. Keep the main README focused on getting started
