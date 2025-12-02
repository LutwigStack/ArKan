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
    return torch.tensor(
        [t_min + (i - cfg.spline_order) * h for i in range(n_knots)],
        dtype=torch.float32,
    )


def find_span(x: torch.Tensor, order: int, grid_size: int) -> torch.Tensor:
    # x: [B, I]
    n = grid_size + order
    return torch.clamp((x * grid_size).floor().to(torch.long), min=order, max=n - 1)


def compute_basis_vectorized(
    x: torch.Tensor, span: torch.Tensor, knots: torch.Tensor, order: int
) -> torch.Tensor:
    # x: [B, I], span: [B, I] -> basis: [B, I, order+1]
    batch, in_dim = x.shape
    basis = torch.zeros(batch, in_dim, order + 1, dtype=torch.float32)
    left = torch.zeros(order + 1, batch, in_dim, dtype=torch.float32)
    right = torch.zeros(order + 1, batch, in_dim, dtype=torch.float32)

    basis[:, :, 0] = 1.0
    for j in range(1, order + 1):
        # gather knot differences for all elements
        idx_left = span + 1 - j
        idx_right = span + j
        left[j] = x - knots[idx_left]
        right[j] = knots[idx_right] - x

        saved = torch.zeros(batch, in_dim, dtype=torch.float32)
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = torch.where(denom.abs() > 1e-6, basis[:, :, r] / denom, 0.0)
            basis[:, :, r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        basis[:, :, j] = saved
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


def forward_vectorized(
    network: list[dict], cfg: KanConfig, inputs: torch.Tensor, knots: torch.Tensor
) -> torch.Tensor:
    x = inputs  # [B, In]
    for layer in network:
        span = find_span(x, cfg.spline_order, cfg.grid_size)  # [B, In]
        basis = compute_basis_vectorized(x, span, knots, cfg.spline_order)  # [B, In, p+1]

        start = span - cfg.spline_order  # [B, In]
        basis_idx = start.unsqueeze(-1) + torch.arange(cfg.spline_order + 1)
        # weights: [O, I, basis]
        w = layer["weights"].unsqueeze(0)  # [1, O, I, basis]
        w = w.expand(basis_idx.shape[0], -1, -1, -1)  # [B, O, I, basis]
        idx = basis_idx.unsqueeze(1).clamp(min=0)  # [B, 1, I, p+1]
        idx = idx.expand(-1, layer["out"], -1, -1)  # [B, O, I, p+1]
        w_chunks = torch.gather(w, dim=3, index=idx)

        # Multiply and sum: basis [B,1,I,p+1] -> output [B,O]
        b_exp = basis.unsqueeze(1)
        out = (w_chunks * b_exp).sum(dim=3).sum(dim=2)  # sum over basis, then inputs
        out = out + layer["bias"]
        x = out
    return x


def bench(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
    knots = compute_knots(cfg)
    network = make_network(cfg)
    inputs = torch.rand(batch, cfg.input_dim)

    # Warmup
    forward_vectorized(network, cfg, inputs, knots)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        forward_vectorized(network, cfg, inputs, knots)
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
