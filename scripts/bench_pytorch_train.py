"""
PyTorch KAN training benchmark for comparison with ArKan.

This script benchmarks:
1. Forward pass (inference)
2. Backward pass (gradient computation)
3. Full training step (forward + backward + optimizer step)
"""

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
    n = grid_size + order
    return torch.clamp((x * grid_size).floor().to(torch.long), min=order, max=n - 1)


def compute_basis_vectorized(
    x: torch.Tensor, span: torch.Tensor, knots: torch.Tensor, order: int
) -> torch.Tensor:
    batch, in_dim = x.shape
    device = x.device
    dtype = x.dtype
    
    # Use functional approach to avoid in-place modifications
    basis_list = [torch.ones(batch, in_dim, dtype=dtype, device=device)]
    
    for j in range(1, order + 1):
        basis_list.append(torch.zeros(batch, in_dim, dtype=dtype, device=device))
    
    basis = torch.stack(basis_list, dim=2)  # [batch, in_dim, order+1]
    
    for j in range(1, order + 1):
        saved = torch.zeros(batch, in_dim, dtype=dtype, device=device)
        for r in range(j):
            idx_right = span + r + 1
            idx_left = span + 1 - j + r
            
            right_val = knots[idx_right] - x
            left_val = x - knots[idx_left]
            
            denom = right_val + left_val
            mask = denom.abs() > 1e-6
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            temp = (basis[:, :, r] / safe_denom) * mask.float()
            
            # Create new tensor instead of in-place modification
            new_basis_r = saved + right_val * temp
            saved = left_val * temp
            
            # Update basis using index_copy or clone
            basis = basis.clone()
            basis[:, :, r] = new_basis_r
        
        basis = basis.clone()
        basis[:, :, j] = saved
    
    return basis


def make_network(cfg: KanConfig) -> list[dict]:
    layers = []
    dims = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    global_basis = cfg.grid_size + cfg.spline_order
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        weights = torch.randn(out_dim, in_dim, global_basis, dtype=torch.float32, requires_grad=True)
        bias = torch.zeros(out_dim, dtype=torch.float32, requires_grad=True)
        layers.append({"in": in_dim, "out": out_dim, "weights": weights, "bias": bias})
    return layers


def forward_vectorized(
    network: list[dict], cfg: KanConfig, inputs: torch.Tensor, knots: torch.Tensor
) -> torch.Tensor:
    x = inputs
    for layer in network:
        span = find_span(x, cfg.spline_order, cfg.grid_size)
        basis = compute_basis_vectorized(x, span, knots, cfg.spline_order)

        start = span - cfg.spline_order
        basis_idx = start.unsqueeze(-1) + torch.arange(cfg.spline_order + 1)
        w = layer["weights"].unsqueeze(0)
        w = w.expand(basis_idx.shape[0], -1, -1, -1)
        idx = basis_idx.unsqueeze(1).clamp(min=0)
        idx = idx.expand(-1, layer["out"], -1, -1)
        w_chunks = torch.gather(w, dim=3, index=idx)

        b_exp = basis.unsqueeze(1)
        out = (w_chunks * b_exp).sum(dim=3).sum(dim=2)
        out = out + layer["bias"]
        x = out
    return x


def bench_forward(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
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
    return min(times) * 1000.0


def bench_backward(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
    """Benchmark backward pass only (gradient computation)."""
    knots = compute_knots(cfg)
    inputs = torch.rand(batch, cfg.input_dim, requires_grad=False)
    targets = torch.rand(batch, cfg.output_dim)

    times = []
    for _ in range(repeats):
        # Create fresh network for each iteration to avoid in-place modification issues
        network = make_network(cfg)
        
        outputs = forward_vectorized(network, cfg, inputs, knots)
        loss = ((outputs - targets) ** 2).mean()

        t0 = time.perf_counter()
        loss.backward()
        times.append(time.perf_counter() - t0)

    return min(times) * 1000.0


def bench_train_step(batch: int, cfg: KanConfig, repeats: int = 5) -> float:
    """Benchmark full training step (forward + backward + SGD update)."""
    knots = compute_knots(cfg)
    inputs = torch.rand(batch, cfg.input_dim, requires_grad=False)
    targets = torch.rand(batch, cfg.output_dim)
    lr = 0.001

    times = []
    for _ in range(repeats):
        # Create fresh network for each iteration
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

    return min(times) * 1000.0


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
