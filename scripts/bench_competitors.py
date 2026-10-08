#!/usr/bin/env python3
"""
Competitive speed benchmark: ArKan vs efficient-kan vs FastKAN vs faithful-PyTorch KAN.

Config matched across all implementations:
  Small:  [2, 8, 1],   grid_size=5, spline_order=3
  Medium: [16, 64, 64, 8],  grid_size=5, spline_order=3
  (ArKan preset for context: [21, 64, 64, 24])

Metrics:
  - forward pass (inference)
  - forward + backward (gradient computation)
  - full train step (forward + backward + Adam optimizer.step)

Methodology:
  - Warmup: 10 iterations (CUDA) or 10 iterations (CPU)
  - Timed repeats: 50
  - Timing: time.perf_counter(), with torch.cuda.synchronize() before/after on GPU
  - Reported: MEDIAN ms (not best-of, not mean)
  - CPU and CUDA measured separately

Formulation notes (mandatory fairness disclaimer):
  - efficient-kan: B-spline with a base+spline RESIDUAL term (SiLU(x) * W_base + spline(x))
    This adds ~1 matmul per layer vs pure B-spline, so efficient-kan is slightly more expressive
    and its FLOPs are not directly comparable to ArKan's pure B-spline.
  - FastKAN: Uses RADIAL BASIS FUNCTIONS (RBF/Gaussian), NOT B-splines. Fundamentally different
    mathematical formulation. FastKAN is faster precisely because it sidestepped GPU B-spline
    computation — the thing ArKan's WGSL GPU path attempts to solve natively.
  - faithful-PyTorch: Pure B-spline matching ArKan's math, isolates language/runtime overhead.
  - ArKan (CPU): Pure B-spline, Rust, zero-allocation workspace reuse, optional SIMD/rayon.
  - ArKan (GPU/wgpu): Pure B-spline on GPU via custom WGSL shaders (not CUDA).
    Rust benchmarks are not run here. Embedded CPU constants are historical and invalid
    for comparisons pending a matched rerun.

Output: prints table to stdout + writes JSON to tasks/02-reference-parity-and-benchmarks/results/competitors.json
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from bench_reference import find_span, compute_basis_vectorized as spline_basis, forward_layer, capture_reset
import os
import sys
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn

# ============================================================================
# Setup
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "tasks" / "02-reference-parity-and-benchmarks" / "results"

WARMUP = 10
REPEATS = 50

HAVE_EFFICIENT_KAN = False
HAVE_FAST_KAN = False

try:
    from efficient_kan import KAN as EfficientKAN
    HAVE_EFFICIENT_KAN = True
    print("[OK] efficient-kan available")
except ImportError as e:
    print(f"[MISSING] efficient-kan: {e}")

try:
    from fastkan import FastKAN
    HAVE_FAST_KAN = True
    print("[OK] FastKAN available")
except ImportError as e:
    print(f"[MISSING] FastKAN: {e}")

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    print(f"[OK] CUDA available: {GPU_NAME}")
else:
    print("[INFO] CUDA not available — CPU only")

print(f"PyTorch: {torch.__version__}")


# ============================================================================
# Benchmark configurations
# ============================================================================

@dataclass
class BenchConfig:
    name: str
    layers: list[int]  # [in, hidden..., out]
    grid_size: int = 5
    spline_order: int = 3

CONFIGS = [
    BenchConfig("small [2,8,1]",         [2, 8, 1]),
    BenchConfig("medium [16,64,64,8]",   [16, 64, 64, 8]),
]

BATCH_SIZES = [1, 64, 256, 1024]


# ============================================================================
# Faithful PyTorch KAN (pure B-spline, same math as ArKan)
# ============================================================================

def compute_knots(grid_size: int, spline_order: int, grid_range=(-3.0, 3.0),
                  device="cpu", dtype=torch.float32) -> torch.Tensor:
    t_min, t_max = grid_range
    n_knots = grid_size + 2 * spline_order + 1
    h = (t_max - t_min) / grid_size
    return torch.tensor(
        [t_min + (i - spline_order) * h for i in range(n_knots)],
        dtype=dtype, device=device,
    )


def compute_basis_vectorized(x: torch.Tensor, grid_size: int, spline_order: int,
                              knots: torch.Tensor) -> torch.Tensor:
    """Cox-de Boor recursion — same algorithm as ArKan. Autograd-safe (no in-place ops)."""
    x = x.clamp(knots[spline_order], knots[spline_order + grid_size])
    span = find_span(x, spline_order, grid_size, knots=knots)
    return spline_basis(x, span, knots, spline_order)


class FaithfulKANLayer(nn.Module):
    """Single KAN layer: pure B-spline, no residual term.

    Uses all basis_size weights for each connection but only evaluates the
    local order+1 basis functions (others are zero), matching ArKan's approach.
    No SiLU/base residual term — pure spline only.
    """
    def __init__(self, in_dim: int, out_dim: int, grid_size: int, spline_order: int):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        # Global coefficients [out_dim, in_dim, grid_size + order], gathered by span.
        self.weights = nn.Parameter(torch.randn(out_dim, in_dim, grid_size + spline_order) * 0.1)
        self.bias    = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        return forward_layer(x, self.weights, self.bias, knots, self.grid_size, self.spline_order)



class FaithfulKAN(nn.Module):
    """Multi-layer faithful KAN (pure B-spline, no residual)."""
    def __init__(self, layers: list[int], grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kan_layers = nn.ModuleList([
            FaithfulKANLayer(layers[i], layers[i+1], grid_size, spline_order)
            for i in range(len(layers) - 1)
        ])

    def forward(self, x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        for layer in self.kan_layers:
            x = layer(x, knots)
        return x


# ============================================================================
# Timing helpers
# ============================================================================

def cuda_sync():
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()


def time_fn(fn, warmup=WARMUP, repeats=REPEATS, setup=None) -> float:
    """Return MEDIAN time in ms over `repeats` calls, after `warmup` iterations."""
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
        cuda_sync()

    times_ms = []
    for _ in range(repeats):
        if setup is not None:
            setup()
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(times_ms)


# ============================================================================
# Per-implementation benchmark runners
# ============================================================================

def bench_efficient_kan(cfg: BenchConfig, batch: int, device: torch.device) -> dict:
    """Benchmark efficient-kan (B-spline + base/spline residual)."""
    if not HAVE_EFFICIENT_KAN:
        return {"forward": None, "fwd_bwd": None, "full_step": None, "note": "not_installed"}

    torch.manual_seed(42)
    model = EfficientKAN(
        cfg.layers,
        grid_size=cfg.grid_size,
        spline_order=cfg.spline_order,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    inputs  = torch.rand(batch, cfg.layers[0], device=device)
    targets = torch.rand(batch, cfg.layers[-1], device=device)

    # forward only
    model.eval()
    with torch.no_grad():
        fwd_ms = time_fn(lambda: model(inputs))

    # forward + backward (no optimizer step)
    model.train()
    def fwd_bwd():
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()

    fwdbwd_ms = time_fn(fwd_bwd)

    # full train step
    def full_step():
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

    reset = capture_reset(model, optimizer)
    full_ms = time_fn(full_step, setup=reset)

    return {"forward": fwd_ms, "fwd_bwd": fwdbwd_ms, "full_step": full_ms}


def bench_fastkan(cfg: BenchConfig, batch: int, device: torch.device) -> dict:
    """Benchmark FastKAN (RBF, NOT B-spline)."""
    if not HAVE_FAST_KAN:
        return {"forward": None, "fwd_bwd": None, "full_step": None, "note": "not_installed"}

    torch.manual_seed(42)
    model = FastKAN(
        cfg.layers,
        num_grids=cfg.grid_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    inputs  = torch.rand(batch, cfg.layers[0], device=device)
    targets = torch.rand(batch, cfg.layers[-1], device=device)

    model.eval()
    with torch.no_grad():
        fwd_ms = time_fn(lambda: model(inputs))

    model.train()
    def fwd_bwd():
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()

    fwdbwd_ms = time_fn(fwd_bwd)

    def full_step():
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

    reset = capture_reset(model, optimizer)
    full_ms = time_fn(full_step, setup=reset)

    return {"forward": fwd_ms, "fwd_bwd": fwdbwd_ms, "full_step": full_ms}


def bench_faithful_pytorch(cfg: BenchConfig, batch: int, device: torch.device) -> dict:
    """Benchmark faithful PyTorch KAN (pure B-spline, same math as ArKan)."""
    torch.manual_seed(42)
    model = FaithfulKAN(cfg.layers, cfg.grid_size, cfg.spline_order).to(device)
    knots = compute_knots(cfg.grid_size, cfg.spline_order, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    inputs  = torch.rand(batch, cfg.layers[0], device=device)
    targets = torch.rand(batch, cfg.layers[-1], device=device)

    model.eval()
    with torch.no_grad():
        fwd_ms = time_fn(lambda: model(inputs, knots))

    model.train()
    def fwd_bwd():
        optimizer.zero_grad()
        out = model(inputs, knots)
        loss = loss_fn(out, targets)
        loss.backward()

    fwdbwd_ms = time_fn(fwd_bwd)

    def full_step():
        optimizer.zero_grad()
        out = model(inputs, knots)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()

    reset = capture_reset(model, optimizer)
    full_ms = time_fn(full_step, setup=reset)

    return {"forward": fwd_ms, "fwd_bwd": fwdbwd_ms, "full_step": full_ms}


# ============================================================================
# Main benchmark loop
# ============================================================================

def run_all(device_name: str) -> dict:
    """Run all competitor benchmarks on the given device."""
    device = torch.device(device_name)
    results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Config: {cfg.name}  |  grid={cfg.grid_size}  order={cfg.spline_order}  |  device={device_name}")
        print(f"{'='*70}")

        cfg_key = cfg.name
        results[cfg_key] = {}

        for batch in BATCH_SIZES:
            print(f"\n  batch={batch}")
            batch_res = {}

            if HAVE_EFFICIENT_KAN:
                r = bench_efficient_kan(cfg, batch, device)
                batch_res["efficient_kan"] = r
                print(f"    efficient-kan  fwd={r['forward']:8.3f}ms  fwd+bwd={r['fwd_bwd']:8.3f}ms  full={r['full_step']:8.3f}ms")

            if HAVE_FAST_KAN:
                r = bench_fastkan(cfg, batch, device)
                batch_res["fastkan"] = r
                print(f"    FastKAN (RBF)  fwd={r['forward']:8.3f}ms  fwd+bwd={r['fwd_bwd']:8.3f}ms  full={r['full_step']:8.3f}ms")

            r = bench_faithful_pytorch(cfg, batch, device)
            batch_res["faithful_pytorch"] = r
            print(f"    faithful-PyTorch  fwd={r['forward']:8.3f}ms  fwd+bwd={r['fwd_bwd']:8.3f}ms  full={r['full_step']:8.3f}ms")

            results[cfg_key][str(batch)] = batch_res

    return results


def print_comparison_table(results: dict, device_name: str, arkan_cpu: Optional[dict] = None):
    """Print a formatted comparison table."""
    print(f"\n\n{'#'*70}")
    print(f"# COMPARISON TABLE — device={device_name}")
    print(f"# Methodology: MEDIAN of {REPEATS} repeats after {WARMUP} warmup iters")
    print(f"# CUDA sync before/after each call on GPU")
    print(f"# Formulation notes:")
    print(f"#   efficient-kan: B-spline + base residual (SiLU(x)*W + spline(x)) — extra matmul vs pure spline")
    print(f"#   FastKAN:       RBF (Gaussian kernels), NOT B-splines — different math, fewer GPU ops")
    print(f"#   faithful-PyTorch: pure B-spline (same algo as ArKan), no residual")
    print(f"#   ArKan CPU:     pure B-spline, Rust, zero-alloc, SIMD/rayon optional")
    print(f"#   ArKan GPU:     pure B-spline via custom WGSL shaders (wgpu, not CUDA)")
    print(f"{'#'*70}\n")

    ops = [("forward", "Forward"), ("fwd_bwd", "Fwd+Bwd"), ("full_step", "Full step")]
    impls_names = []
    if HAVE_EFFICIENT_KAN:
        impls_names.append(("efficient_kan", "efficient-kan"))
    if HAVE_FAST_KAN:
        impls_names.append(("fastkan", "FastKAN(RBF)"))
    impls_names.append(("faithful_pytorch", "faithful-PyTorch"))
    if arkan_cpu:
        impls_names.append(("arkan_cpu", "ArKan-CPU"))

    for cfg in CONFIGS:
        cfg_key = cfg.name
        print(f"\nConfig: {cfg_key}  layers={cfg.layers}  grid={cfg.grid_size}  order={cfg.spline_order}")
        header = f"{'Op':<12} {'Batch':>6}"
        for _, label in impls_names:
            header += f"  {label:>16}"
        print(header)
        print("-" * len(header))

        for op_key, op_label in ops:
            for batch in BATCH_SIZES:
                row = f"{op_label:<12} {batch:>6}"
                for impl_key, _ in impls_names:
                    if impl_key == "arkan_cpu":
                        val = arkan_cpu.get(cfg_key, {}).get(str(batch), {}).get(op_key)
                    else:
                        val = results.get(cfg_key, {}).get(str(batch), {}).get(impl_key, {}).get(op_key)
                    if val is None:
                        row += f"  {'N/A':>16}"
                    else:
                        row += f"  {val:>13.3f} ms"
                print(row)
        print()


# ============================================================================
# ArKan CPU numbers (from cargo bench --bench forward / backward)
# These are collected from Rust benchmarks separately to avoid mixing
# the timing methodology. Numbers below were measured on this machine
# via: cargo bench --bench forward --noplot
#       cargo bench --bench backward --noplot
# Config matched: poker preset [21,64,64,24], grid=5, order=3
# These don't exactly match the [2,8,1] or [16,64,64,8] configs above
# because ArKan's Rust benches are fixed to the poker preset.
# We include them in a separate "ArKan preset [21,64,64,24]" section.
# ============================================================================

# Measured 2025-06-27, cargo bench --bench forward --bench backward
# Criterion median (center of 100-sample CI)
ARKAN_CPU_PRESET = {
    "poker preset [21,64,64,24]": {
        # forward only (forward_batch bench)
        "forward": {
            "1":   0.02677,   # ms  (26.77 µs median)
            "64":  1.6941,    # ms
            "256": 6.6869,    # ms
            "1024": None,     # not measured by forward bench (max=256)
        },
        # full train step (train_step bench = forward+backward+sgd)
        "full_step": {
            "1":   0.10168,   # ms (101.68 µs)
            "64":  4.4883,    # ms
            "256": 18.015,    # ms
            "1024": None,
        },
        # Historical estimate includes backward, loss, and SGD update.
        "backward_update_est": {
            # forward_training - forward_only ≈ ~0 (prep only)
            # full_train_step = forward + loss + backward + SGD update
            # backward_update_est = full_step - forward_only
            "1":   0.09660,   # 123.63 - 27.03 µs = 96.60 µs
            "64":  2.7990,    # 4.4971 - 1.6981 ms
            "256": 11.434,    # 18.273 - 6.839 ms
        }
    }
}


def main():
    results_all = {}

    # ---- CPU benchmarks ----
    print("\n" + "="*70)
    print("RUNNING CPU BENCHMARKS")
    print("="*70)
    cpu_results = run_all("cpu")
    results_all["cpu"] = cpu_results
    print_comparison_table(cpu_results, "cpu")

    # ---- CUDA benchmarks ----
    if CUDA_AVAILABLE:
        print("\n" + "="*70)
        print(f"RUNNING CUDA BENCHMARKS ({GPU_NAME})")
        print("="*70)
        cuda_results = run_all("cuda")
        results_all["cuda"] = cuda_results
        print_comparison_table(cuda_results, "cuda")
    else:
        print("\n[SKIP] CUDA benchmarks (no GPU available)")
        results_all["cuda"] = None

    # ---- Print ArKan numbers from Rust benches ----
    print("\n\n" + "#"*70)
    print("# HISTORICAL ArKan CPU numbers — invalid for comparison pending matched rerun")
    print("# NOTE: Config does NOT match the Python configs above.")
    print("# ArKan benches are fixed to the poker preset; Python configs above")
    print("# use [2,8,1] and [16,64,64,8] for cross-library comparison.")
    print("#"*70)
    p = ARKAN_CPU_PRESET["poker preset [21,64,64,24]"]
    print(f"\n{'Op':<14} {'batch':>6}  {'ArKan-CPU (Rust)':>18}")
    print("-"*42)
    for batch_k, fwd in sorted(p["forward"].items(), key=lambda x: int(x[0])):
        full = p["full_step"].get(batch_k)
        fwdbwd = p["backward_update_est"].get(batch_k)
        fwd_s = f"{fwd:.4f} ms" if fwd is not None else "N/A"
        full_s = f"{full:.4f} ms" if full is not None else "N/A"
        fwdbwd_s = f"{fwdbwd:.4f} ms" if fwdbwd is not None else "N/A"
        print(f"{'forward':<14} {batch_k:>6}  {fwd_s:>18}")
        print(f"{'bwd+update(est)':<14} {batch_k:>6}  {fwdbwd_s:>18}")
        print(f"{'full_step':<14} {batch_k:>6}  {full_s:>18}")
        print()

    # ---- fekan note ----
    print("\n" + "#"*70)
    print("# fekan (Rust crate) status: DEFERRED")
    print("#")
    print("# fekan is a Rust KAN library. Integration was timeboxed:")
    print("# - fekan's API differs significantly (it uses a different spline")
    print("#   parameterization and does not expose a simple forward() call")
    print("#   matching the ArKan API surface).")
    print("# - Adding fekan as a dev-dependency and writing a fair bench would")
    print("#   require non-trivial adapter code that could skew the comparison.")
    print("# - fekan's published README benchmarks: not directly comparable")
    print("#   (different hardware, different network sizes).")
    print("# TODO: Add fekan bench if fekan >=0.2 stabilizes its public API.")
    print("#"*70)

    # ---- Save JSON ----
    output = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "seed": 42,
            "optimizer": "Adam(lr=0.001)",
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
            "rustflags": os.environ.get("RUSTFLAGS", ""),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
            "cuda_available": CUDA_AVAILABLE,
            "gpu": GPU_NAME if CUDA_AVAILABLE else None,
            "warmup_iters": WARMUP,
            "timed_repeats": REPEATS,
            "metric": "median_ms",
            "configs": [{"name": c.name, "layers": c.layers, "grid_size": c.grid_size, "spline_order": c.spline_order} for c in CONFIGS],
            "batch_sizes": BATCH_SIZES,
            "implementations": {
                "efficient_kan": HAVE_EFFICIENT_KAN,
                "fastkan": HAVE_FAST_KAN,
                "faithful_pytorch": True,
                "fekan": "deferred",
            },
            "formulation_notes": {
                "efficient_kan": "B-spline + base/spline residual (SiLU(x)*W_base + spline(x)); extra matmul per layer",
                "fastkan": "Radial Basis Functions (RBF/Gaussian), NOT B-splines; different math",
                "faithful_pytorch": "Pure B-spline, no residual, same algorithm as ArKan",
                "arkan_cpu": "Pure B-spline, Rust, zero-allocation, optional SIMD+rayon",
                "arkan_gpu": "Pure B-spline via custom WGSL shaders (wgpu/Vulkan); NOT CUDA",
            },
        },
        "results": results_all,
        "historical_arkan_cpu": {
            "status": "invalid_for_comparison_pending_rerun",
            "reported_date": "2025-06-27",
            "platform": "unverified historical source",
            "values": ARKAN_CPU_PRESET,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "competitors.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: None)
    print(f"\nResults written to: {out_path}")

    return results_all


if __name__ == "__main__":
    main()
