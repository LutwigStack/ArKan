import math
import time
from dataclasses import dataclass

import torch


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
    return torch.tensor([t_min + (i - cfg.spline_order) * h for i in range(n_knots)], dtype=torch.float32)


def find_span(x: torch.Tensor, knots: torch.Tensor, order: int, grid_size: int) -> torch.Tensor:
    n = grid_size + order
    span = torch.clamp((x * grid_size).floor().to(torch.long), min=order, max=n - 1)
    return span


def compute_basis(x: torch.Tensor, span: torch.Tensor, knots: torch.Tensor, order: int) -> torch.Tensor:
    # Cox–de Boor for a single scalar x; result shape [order+1]
    basis = torch.zeros(order + 1, dtype=torch.float32)
    left = torch.zeros(order + 1, dtype=torch.float32)
    right = torch.zeros(order + 1, dtype=torch.float32)
    basis[0] = 1.0
    for j in range(1, order + 1):
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = basis[r] / denom if denom.abs() > 1e-6 else 0.0
            basis[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        basis[j] = saved
    return basis


def make_network(cfg: KanConfig) -> list[dict]:
    layers = []
    dims = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    global_basis = cfg.grid_size + cfg.spline_order
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        weights = torch.randn(out_dim, in_dim, global_basis, dtype=torch.float32)
        bias = torch.zeros(out_dim, dtype=torch.float32)
        layers.append({"in": in_dim, "out": out_dim, "weights": weights, "bias": bias})
    return layers


def forward(network: list[dict], cfg: KanConfig, inputs: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    batch = inputs.shape[0]
    x = inputs
    for layer in network:
        out = torch.zeros(batch, layer["out"], dtype=torch.float32)
        for b in range(batch):
            for i in range(layer["in"]):
                span = find_span(x[b, i], knots, cfg.spline_order, cfg.grid_size)
                basis = compute_basis(x[b, i], span, knots, cfg.spline_order)
                start = span - cfg.spline_order
                # accumulate
                out[b] += torch.sum(
                    layer["weights"][:, i, start : start + cfg.spline_order + 1]
                    * basis[None, :],
                    dim=1,
                )
        out += layer["bias"]
        x = out
    return x


def bench(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
    knots = compute_knots(cfg)
    network = make_network(cfg)
    inputs = torch.rand(batch, cfg.input_dim)
    # Warmup
    forward(network, cfg, inputs, knots)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward(network, cfg, inputs, knots)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000.0  # ms


def main():
    cfg = KanConfig()
    for batch in (1, 16, 64, 256):
        t_ms = bench(batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"batch={batch:3d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")


if __name__ == "__main__":
    main()
