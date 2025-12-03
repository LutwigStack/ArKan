#!/usr/bin/env python3
"""
GPU PyTorch KAN benchmarks for comparison with ArKan GPU backend.

Benchmarks three KAN implementations on GPU:
1. efficient-kan (B-spline, PyTorch native)
2. fast-kan (RBF approximation, faster)
3. ArKan-style custom B-spline (for direct comparison)

Install dependencies:
    pip install torch efficient-kan
    pip install git+https://github.com/ZiyaoLi/fast-kan.git

Run:
    python scripts/bench_pytorch_gpu.py
"""

import time
import sys
from dataclasses import dataclass
from typing import Optional

import torch

# Check CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. This benchmark requires a CUDA-capable GPU.")
    print("For CPU benchmarks, use: python scripts/bench_pytorch_train.py")
    sys.exit(1)

DEVICE = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

# Try importing KAN implementations
HAVE_EFFICIENT_KAN = False
HAVE_FAST_KAN = False

try:
    from efficient_kan import KAN as EfficientKAN
    HAVE_EFFICIENT_KAN = True
    print("✓ efficient-kan available")
except ImportError:
    print("✗ efficient-kan not installed (pip install efficient-kan)")

try:
    from fastkan import FastKAN
    HAVE_FAST_KAN = True
    print("✓ fast-kan available")
except ImportError:
    print("✗ fast-kan not installed (pip install git+https://github.com/ZiyaoLi/fast-kan.git)")


@dataclass
class KanConfig:
    """Configuration matching ArKan preset."""
    input_dim: int = 21
    output_dim: int = 24
    hidden_dims: tuple[int, ...] = (64, 64)
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple[float, float] = (-3.0, 3.0)


def torch_sync():
    """Synchronize CUDA for accurate timing."""
    torch.cuda.synchronize()


def compute_knots(cfg: KanConfig, device: torch.device) -> torch.Tensor:
    """Compute knot vector for B-spline."""
    t_min, t_max = cfg.grid_range
    n_knots = cfg.grid_size + 2 * cfg.spline_order + 1
    h = (t_max - t_min) / cfg.grid_size
    return torch.tensor(
        [t_min + (i - cfg.spline_order) * h for i in range(n_knots)],
        dtype=torch.float32,
        device=device,
    )


def find_span(x: torch.Tensor, order: int, grid_size: int) -> torch.Tensor:
    """Find knot span index."""
    n = grid_size + order
    return torch.clamp((x * grid_size).floor().to(torch.long), min=order, max=n - 1)


def compute_basis_vectorized_gpu(
    x: torch.Tensor, span: torch.Tensor, knots: torch.Tensor, order: int
) -> torch.Tensor:
    """Compute B-spline basis on GPU."""
    batch, in_dim = x.shape
    device = x.device
    dtype = x.dtype
    
    basis = torch.zeros(batch, in_dim, order + 1, dtype=dtype, device=device)
    basis[:, :, 0] = 1.0
    
    left = torch.zeros(order + 1, batch, in_dim, dtype=dtype, device=device)
    right = torch.zeros(order + 1, batch, in_dim, dtype=dtype, device=device)
    
    for j in range(1, order + 1):
        idx_left = span + 1 - j
        idx_right = span + j
        left[j] = x - knots[idx_left]
        right[j] = knots[idx_right] - x
        
        saved = torch.zeros(batch, in_dim, dtype=dtype, device=device)
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            mask = denom.abs() > 1e-6
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            temp = (basis[:, :, r] / safe_denom) * mask
            basis[:, :, r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        basis[:, :, j] = saved
    
    return basis


def make_network_gpu(cfg: KanConfig, device: torch.device) -> list[dict]:
    """Create ArKan-style network on GPU."""
    layers = []
    dims = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    global_basis = cfg.grid_size + cfg.spline_order
    
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        weights = torch.randn(out_dim, in_dim, global_basis, dtype=torch.float32, device=device)
        bias = torch.zeros(out_dim, dtype=torch.float32, device=device)
        layers.append({"in": in_dim, "out": out_dim, "weights": weights, "bias": bias})
    
    return layers


def forward_arkan_style_gpu(
    network: list[dict], cfg: KanConfig, inputs: torch.Tensor, knots: torch.Tensor
) -> torch.Tensor:
    """Forward pass (ArKan-style B-spline) on GPU."""
    x = inputs
    for layer in network:
        span = find_span(x, cfg.spline_order, cfg.grid_size)
        basis = compute_basis_vectorized_gpu(x, span, knots, cfg.spline_order)
        
        start = span - cfg.spline_order
        basis_idx = start.unsqueeze(-1) + torch.arange(cfg.spline_order + 1, device=x.device)
        
        w = layer["weights"].unsqueeze(0).expand(basis_idx.shape[0], -1, -1, -1)
        idx = basis_idx.unsqueeze(1).clamp(min=0).expand(-1, layer["out"], -1, -1)
        w_chunks = torch.gather(w, dim=3, index=idx)
        
        b_exp = basis.unsqueeze(1)
        out = (w_chunks * b_exp).sum(dim=3).sum(dim=2)
        out = out + layer["bias"]
        x = out
    
    return x


# ============================================================================
# Benchmark Functions
# ============================================================================

def bench_forward_gpu(
    name: str,
    forward_fn,
    batch: int,
    cfg: KanConfig,
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    """Benchmark forward pass on GPU."""
    inputs = torch.rand(batch, cfg.input_dim, device=DEVICE)
    
    # Warmup
    for _ in range(warmup):
        _ = forward_fn(inputs)
        torch_sync()
    
    # Measure
    times = []
    for _ in range(repeats):
        torch_sync()
        t0 = time.perf_counter()
        _ = forward_fn(inputs)
        torch_sync()
        times.append(time.perf_counter() - t0)
    
    return min(times) * 1000.0  # ms


def bench_train_step_gpu(
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: int,
    cfg: KanConfig,
    warmup: int = 5,
    repeats: int = 50,
) -> float:
    """Benchmark full train step (forward + backward + optimizer) on GPU."""
    inputs = torch.rand(batch, cfg.input_dim, device=DEVICE)
    targets = torch.rand(batch, cfg.output_dim, device=DEVICE)
    loss_fn = torch.nn.MSELoss()
    
    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        torch_sync()
    
    # Measure
    times = []
    for _ in range(repeats):
        optimizer.zero_grad()
        torch_sync()
        t0 = time.perf_counter()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        torch_sync()
        times.append(time.perf_counter() - t0)
    
    return min(times) * 1000.0  # ms


def bench_arkan_style_forward(batch: int, cfg: KanConfig, repeats: int = 100) -> float:
    """Benchmark ArKan-style B-spline forward on GPU."""
    knots = compute_knots(cfg, DEVICE)
    network = make_network_gpu(cfg, DEVICE)
    inputs = torch.rand(batch, cfg.input_dim, device=DEVICE)
    
    # Warmup
    for _ in range(10):
        forward_arkan_style_gpu(network, cfg, inputs, knots)
        torch_sync()
    
    times = []
    for _ in range(repeats):
        torch_sync()
        t0 = time.perf_counter()
        forward_arkan_style_gpu(network, cfg, inputs, knots)
        torch_sync()
        times.append(time.perf_counter() - t0)
    
    return min(times) * 1000.0  # ms


def run_efficient_kan_benchmarks(cfg: KanConfig):
    """Run benchmarks for efficient-kan."""
    print("\n--- Efficient-KAN (GPU) ---")
    
    layers = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    model = EfficientKAN(
        layers,
        grid_size=cfg.grid_size,
        spline_order=cfg.spline_order,
        scale_noise=0.1,
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nForward pass:")
    for batch in (1, 16, 64, 256, 1024):
        t_ms = bench_forward_gpu("efficient-kan", lambda x: model(x), batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"  batch={batch:4d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")
    
    print("\nTrain step (Adam):")
    for batch in (1, 16, 64, 256):
        t_ms = bench_train_step_gpu("efficient-kan", model, optimizer, batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"  batch={batch:4d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")


def run_fast_kan_benchmarks(cfg: KanConfig):
    """Run benchmarks for fast-kan."""
    print("\n--- FastKAN (GPU, RBF) ---")
    
    layers = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    model = FastKAN(
        layers,
        num_grids=cfg.grid_size,
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nForward pass:")
    for batch in (1, 16, 64, 256, 1024):
        t_ms = bench_forward_gpu("fast-kan", lambda x: model(x), batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"  batch={batch:4d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")
    
    print("\nTrain step (Adam):")
    for batch in (1, 16, 64, 256):
        t_ms = bench_train_step_gpu("fast-kan", model, optimizer, batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"  batch={batch:4d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")


def run_arkan_style_benchmarks(cfg: KanConfig):
    """Run benchmarks for ArKan-style B-spline implementation."""
    print("\n--- ArKan-style B-spline (PyTorch GPU) ---")
    print("(Custom B-spline implementation matching ArKan algorithm)")
    
    print("\nForward pass:")
    for batch in (1, 16, 64, 256, 1024):
        t_ms = bench_arkan_style_forward(batch, cfg)
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"  batch={batch:4d}: {t_ms:8.3f} ms  thrpt={thrpt/1e6:6.2f} M elems/s")


def run_latency_percentiles(cfg: KanConfig):
    """Run latency percentile analysis (batch=1)."""
    print("\n--- Latency Percentiles (batch=1, forward only) ---")
    
    # Test all available implementations
    results = {}
    
    # ArKan-style
    knots = compute_knots(cfg, DEVICE)
    network = make_network_gpu(cfg, DEVICE)
    inputs = torch.rand(1, cfg.input_dim, device=DEVICE)
    
    iterations = 5000
    latencies = []
    
    # Warmup
    for _ in range(500):
        forward_arkan_style_gpu(network, cfg, inputs, knots)
        torch_sync()
    
    for _ in range(iterations):
        torch_sync()
        t0 = time.perf_counter()
        forward_arkan_style_gpu(network, cfg, inputs, knots)
        torch_sync()
        latencies.append((time.perf_counter() - t0) * 1e6)  # microseconds
    
    latencies.sort()
    results["arkan-style"] = latencies
    
    if HAVE_EFFICIENT_KAN:
        layers = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
        model = EfficientKAN(layers, grid_size=cfg.grid_size, spline_order=cfg.spline_order).to(DEVICE)
        model.eval()
        
        latencies = []
        with torch.no_grad():
            for _ in range(500):
                model(inputs)
                torch_sync()
            
            for _ in range(iterations):
                torch_sync()
                t0 = time.perf_counter()
                model(inputs)
                torch_sync()
                latencies.append((time.perf_counter() - t0) * 1e6)
        
        latencies.sort()
        results["efficient-kan"] = latencies
    
    if HAVE_FAST_KAN:
        layers = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
        model = FastKAN(layers, num_grids=cfg.grid_size).to(DEVICE)
        model.eval()
        
        latencies = []
        with torch.no_grad():
            for _ in range(500):
                model(inputs)
                torch_sync()
            
            for _ in range(iterations):
                torch_sync()
                t0 = time.perf_counter()
                model(inputs)
                torch_sync()
                latencies.append((time.perf_counter() - t0) * 1e6)
        
        latencies.sort()
        results["fast-kan"] = latencies
    
    # Print results
    print(f"\nLatency Distribution ({iterations} samples):")
    print("-" * 70)
    print(f"{'Implementation':<20} {'Min':>8} {'P50':>8} {'P90':>8} {'P99':>8} {'Max':>8}")
    print("-" * 70)
    
    for name, lats in results.items():
        n = len(lats)
        p50 = lats[n // 2]
        p90 = lats[n * 90 // 100]
        p99 = lats[n * 99 // 100]
        print(f"{name:<20} {lats[0]:7.1f}µs {p50:7.1f}µs {p90:7.1f}µs {p99:7.1f}µs {lats[-1]:7.1f}µs")


def run_architecture_scaling(cfg: KanConfig):
    """Benchmark different network architectures."""
    print("\n--- Architecture Scaling (batch=64, forward) ---")
    
    architectures = [
        ("tiny_3_10_1", [3, 10, 1]),
        ("medium_10_64_64_10", [10, 64, 64, 10]),
        ("poker_21_64_64_24", [21, 64, 64, 24]),
        ("large_32_128x3_32", [32, 128, 128, 128, 32]),
        ("wide_21_256_24", [21, 256, 24]),
        ("deep_21_32x5_24", [21, 32, 32, 32, 32, 32, 24]),
    ]
    
    print("-" * 80)
    print(f"{'Architecture':<25} {'efficient-kan':>15} {'fast-kan':>15} {'arkan-style':>15}")
    print("-" * 80)
    
    batch = 64
    
    for arch_name, layers in architectures:
        results = {}
        input_dim = layers[0]
        output_dim = layers[-1]
        hidden = layers[1:-1]
        
        arch_cfg = KanConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(hidden),
            grid_size=cfg.grid_size,
            spline_order=cfg.spline_order,
        )
        
        # ArKan-style
        knots = compute_knots(arch_cfg, DEVICE)
        network = make_network_gpu(arch_cfg, DEVICE)
        inputs = torch.rand(batch, input_dim, device=DEVICE)
        
        # Warmup
        for _ in range(10):
            forward_arkan_style_gpu(network, arch_cfg, inputs, knots)
            torch_sync()
        
        times = []
        for _ in range(50):
            torch_sync()
            t0 = time.perf_counter()
            forward_arkan_style_gpu(network, arch_cfg, inputs, knots)
            torch_sync()
            times.append(time.perf_counter() - t0)
        results["arkan-style"] = min(times) * 1000
        
        if HAVE_EFFICIENT_KAN:
            model = EfficientKAN(layers, grid_size=cfg.grid_size, spline_order=cfg.spline_order).to(DEVICE)
            model.eval()
            
            with torch.no_grad():
                for _ in range(10):
                    model(inputs)
                    torch_sync()
                
                times = []
                for _ in range(50):
                    torch_sync()
                    t0 = time.perf_counter()
                    model(inputs)
                    torch_sync()
                    times.append(time.perf_counter() - t0)
            results["efficient-kan"] = min(times) * 1000
        
        if HAVE_FAST_KAN:
            model = FastKAN(layers, num_grids=cfg.grid_size).to(DEVICE)
            model.eval()
            
            with torch.no_grad():
                for _ in range(10):
                    model(inputs)
                    torch_sync()
                
                times = []
                for _ in range(50):
                    torch_sync()
                    t0 = time.perf_counter()
                    model(inputs)
                    torch_sync()
                    times.append(time.perf_counter() - t0)
            results["fast-kan"] = min(times) * 1000
        
        # Print row
        efficient = f"{results.get('efficient-kan', 0):>12.3f} ms" if HAVE_EFFICIENT_KAN else "N/A"
        fast = f"{results.get('fast-kan', 0):>12.3f} ms" if HAVE_FAST_KAN else "N/A"
        arkan = f"{results.get('arkan-style', 0):>12.3f} ms"
        print(f"{arch_name:<25} {efficient:>15} {fast:>15} {arkan:>15}")


def run_spline_config_analysis(cfg: KanConfig):
    """Benchmark different spline configurations."""
    print("\n--- Spline Order Impact (grid=5, batch=64) ---")
    
    if not HAVE_EFFICIENT_KAN:
        print("Requires efficient-kan for accurate spline order comparison")
        return
    
    batch = 64
    layers = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    inputs = torch.rand(batch, cfg.input_dim, device=DEVICE)
    
    print("-" * 60)
    print(f"{'Order':<10} {'Name':<12} {'Time (ms)':>12} {'Throughput':>15}")
    print("-" * 60)
    
    for order, name in [(1, "linear"), (2, "quadratic"), (3, "cubic"), (4, "quartic"), (5, "quintic")]:
        model = EfficientKAN(layers, grid_size=5, spline_order=order).to(DEVICE)
        model.eval()
        
        with torch.no_grad():
            for _ in range(10):
                model(inputs)
                torch_sync()
            
            times = []
            for _ in range(50):
                torch_sync()
                t0 = time.perf_counter()
                model(inputs)
                torch_sync()
                times.append(time.perf_counter() - t0)
        
        t_ms = min(times) * 1000
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"{order:<10} {name:<12} {t_ms:>12.3f} {thrpt/1e6:>12.2f} M/s")
    
    print("\n--- Grid Size Impact (order=3, batch=64) ---")
    print("-" * 60)
    print(f"{'Grid':<10} {'Basis Size':<12} {'Time (ms)':>12} {'Throughput':>15}")
    print("-" * 60)
    
    for grid in [3, 5, 8, 12, 16]:
        model = EfficientKAN(layers, grid_size=grid, spline_order=3).to(DEVICE)
        model.eval()
        
        with torch.no_grad():
            for _ in range(10):
                model(inputs)
                torch_sync()
            
            times = []
            for _ in range(50):
                torch_sync()
                t0 = time.perf_counter()
                model(inputs)
                torch_sync()
                times.append(time.perf_counter() - t0)
        
        t_ms = min(times) * 1000
        basis_size = grid + 3  # order + grid
        elems = batch * cfg.input_dim
        thrpt = elems / (t_ms / 1000.0)
        print(f"{grid:<10} {basis_size:<12} {t_ms:>12.3f} {thrpt/1e6:>12.2f} M/s")


def run_memory_analysis(cfg: KanConfig):
    """Analyze GPU memory usage."""
    print("\n--- GPU Memory Analysis ---")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    base_mem = torch.cuda.memory_allocated()
    
    layers = [cfg.input_dim, *cfg.hidden_dims, cfg.output_dim]
    
    print(f"\nConfig: {layers}, grid={cfg.grid_size}, order={cfg.spline_order}")
    print("-" * 50)
    
    if HAVE_EFFICIENT_KAN:
        torch.cuda.empty_cache()
        model = EfficientKAN(layers, grid_size=cfg.grid_size, spline_order=cfg.spline_order).to(DEVICE)
        model_mem = torch.cuda.memory_allocated() - base_mem
        
        inputs = torch.rand(64, cfg.input_dim, device=DEVICE)
        _ = model(inputs)
        torch_sync()
        
        peak_mem = torch.cuda.max_memory_allocated() - base_mem
        print(f"efficient-kan:")
        print(f"  Model memory:     {model_mem / 1024:.1f} KB")
        print(f"  Peak (batch=64):  {peak_mem / 1024:.1f} KB")
        
        del model
        torch.cuda.empty_cache()
    
    if HAVE_FAST_KAN:
        torch.cuda.reset_peak_memory_stats()
        model = FastKAN(layers, num_grids=cfg.grid_size).to(DEVICE)
        model_mem = torch.cuda.memory_allocated() - base_mem
        
        inputs = torch.rand(64, cfg.input_dim, device=DEVICE)
        _ = model(inputs)
        torch_sync()
        
        peak_mem = torch.cuda.max_memory_allocated() - base_mem
        print(f"\nfast-kan:")
        print(f"  Model memory:     {model_mem / 1024:.1f} KB")
        print(f"  Peak (batch=64):  {peak_mem / 1024:.1f} KB")


def main():
    cfg = KanConfig()
    
    print("=" * 70)
    print("PyTorch KAN GPU Benchmarks")
    print(f"Config: Input {cfg.input_dim}, Output {cfg.output_dim}, Hidden {cfg.hidden_dims}")
    print(f"Spline: grid_size={cfg.grid_size}, order={cfg.spline_order}")
    print("=" * 70)
    
    # Run benchmarks
    run_arkan_style_benchmarks(cfg)
    
    if HAVE_EFFICIENT_KAN:
        run_efficient_kan_benchmarks(cfg)
    
    if HAVE_FAST_KAN:
        run_fast_kan_benchmarks(cfg)
    
    run_latency_percentiles(cfg)
    run_architecture_scaling(cfg)
    
    if HAVE_EFFICIENT_KAN:
        run_spline_config_analysis(cfg)
    
    run_memory_analysis(cfg)
    
    print("\n" + "=" * 70)
    print("Compare with ArKan GPU:")
    print("  cargo bench --bench gpu_forward --bench gpu_backward --features gpu -- --gpu")
    print("=" * 70)


if __name__ == "__main__":
    main()
