"""
PyTorch KAN training benchmark for comparison with ArKan.

This script benchmarks:
1. Forward pass (inference)
2. Backward pass (gradient computation)
3. Full training step (forward + backward + optimizer step)
"""

import time
import statistics
from dataclasses import dataclass

import torch
from bench_reference import find_span, compute_basis_vectorized, forward_layer


@dataclass
class KanConfig:
    input_dim: int = 21
    output_dim: int = 24
    hidden_dims: tuple[int, ...] = (64, 64)
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple[float, float] = (-3.0, 3.0)


def compute_knots(cfg: KanConfig) -> torch.Tensor:
    t_min, t_max = cfg.grid_range
    n_knots = cfg.grid_size + 2 * cfg.spline_order + 1
    h = (t_max - t_min) / cfg.grid_size
    return torch.tensor(
        [t_min + (i - cfg.spline_order) * h for i in range(n_knots)],
        dtype=torch.float32,
    )


def make_network(cfg: KanConfig) -> list[dict]:
    generator = torch.Generator(device="cpu").manual_seed(42)
    layers = []
    dims = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    global_basis = cfg.grid_size + cfg.spline_order
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        weights = torch.randn(out_dim, in_dim, global_basis, generator=generator, dtype=torch.float32, requires_grad=True)
        bias = torch.zeros(out_dim, dtype=torch.float32, requires_grad=True)
        layers.append({"in": in_dim, "out": out_dim, "weights": weights, "bias": bias})
    return layers


def forward_vectorized(
    network: list[dict], cfg: KanConfig, inputs: torch.Tensor, knots: torch.Tensor
) -> torch.Tensor:
    x = inputs
    for layer in network:
        x = forward_layer(x, layer['weights'], layer['bias'], knots, cfg.grid_size, cfg.spline_order, layer.get('mean', 0.0), layer.get('std', 1.0))
    return x


def bench_forward(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
    torch.manual_seed(42)
    knots = compute_knots(cfg)
    network = make_network(cfg)
    for layer in network:
        layer["weights"].requires_grad_(False)
        layer["bias"].requires_grad_(False)
    inputs = torch.rand(batch, cfg.input_dim)

    # Warmup
    forward_vectorized(network, cfg, inputs, knots)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward_vectorized(network, cfg, inputs, knots)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0


def bench_backward(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
    """Benchmark backward pass only (gradient computation)."""
    torch.manual_seed(42)
    knots = compute_knots(cfg)
    inputs = torch.rand(batch, cfg.input_dim, requires_grad=False)
    targets = torch.rand(batch, cfg.output_dim)

    times = []
    for _ in range(repeats):
        # Create fresh network for each iteration to avoid in-place modification issues
        torch.manual_seed(42)
        network = make_network(cfg)
        
        outputs = forward_vectorized(network, cfg, inputs, knots)
        loss = ((outputs - targets) ** 2).mean()

        t0 = time.perf_counter()
        loss.backward()
        times.append(time.perf_counter() - t0)

    return statistics.median(times) * 1000.0


def bench_train_step(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
    """Benchmark full training step (forward + backward + SGD update)."""
    torch.manual_seed(42)
    knots = compute_knots(cfg)
    inputs = torch.rand(batch, cfg.input_dim, requires_grad=False)
    targets = torch.rand(batch, cfg.output_dim)
    lr = 0.001

    times = []
    for _ in range(repeats):
        # Create fresh network for each iteration
        torch.manual_seed(42)
        network = make_network(cfg)
        
        t0 = time.perf_counter()

        # Forward
        outputs = forward_vectorized(network, cfg, inputs, knots)
        loss = ((outputs - targets) ** 2).mean()

        # Backward
        loss.backward()

        # SGD update
        with torch.no_grad():
            for layer in network:
                layer["weights"] -= lr * layer["weights"].grad
                layer["bias"] -= lr * layer["bias"].grad

        times.append(time.perf_counter() - t0)

    return statistics.median(times) * 1000.0


def main():
    cfg = KanConfig()

    print("=" * 70)
    print("PyTorch KAN Benchmarks (CPU)")
    print(f"Config: Input {cfg.input_dim}, Output {cfg.output_dim}, Hidden {cfg.hidden_dims}")
    print(f"Spline: grid_size={cfg.grid_size}, order={cfg.spline_order}")
    print("=" * 70)

    print("\n--- Forward Pass (Inference) ---")
    for batch in (1, 16, 64, 256):
        t_ms = bench_forward(batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"batch={batch:3d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")

    print("\n--- Backward Pass (Gradient only) ---")
    for batch in (1, 16, 64, 256):
        t_ms = bench_backward(batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"batch={batch:3d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")

    print("\n--- Full Train Step (Forward + Backward + SGD) ---")
    for batch in (1, 16, 64, 256):
        t_ms = bench_train_step(batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"batch={batch:3d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")


if __name__ == "__main__":
    main()
