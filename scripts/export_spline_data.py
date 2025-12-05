#!/usr/bin/env python3
"""
Export B-spline basis function values from SciPy for parity testing with ArKan.

This script generates reference values using scipy.interpolate.BSpline
that can be compared against ArKan's compute_basis() implementation.

The knot generation logic matches ArKan's compute_knots() exactly:
- For grid_size G and order k: n_knots = G + 2*k + 1
- Knots are uniformly spaced: t_i = t_min + (i - k) * h
- where h = (t_max - t_min) / G

Output format (JSON):
{
    "config": { "grid": 5, "order": 3, "range": [-1.0, 1.0] },
    "test_cases": [
        { "x": 0.5, "span": 5, "basis": [0.1, 0.4, 0.4, 0.1], "basis_sum": 1.0 },
        ...
    ]
}

Usage:
    python scripts/export_spline_data.py
    python scripts/export_spline_data.py --output data/spline_test.json
"""

import json
import argparse
import numpy as np
from scipy.interpolate import BSpline


def compute_knots_arkan(grid_size: int, order: int, grid_range: tuple) -> np.ndarray:
    """
    Generate knots using ArKan's formula.
    
    ArKan uses uniform knots:
        n_knots = grid_size + 2 * order + 1
        knots[i] = t_min + (i - order) * h
    where h = (t_max - t_min) / grid_size
    """
    t_min, t_max = grid_range
    n_knots = grid_size + 2 * order + 1
    h = (t_max - t_min) / grid_size
    
    knots = np.array([t_min + (i - order) * h for i in range(n_knots)], dtype=np.float64)
    return knots


def find_span_arkan(x: float, knots: np.ndarray, order: int, grid_size: int) -> int:
    """
    Find knot span using ArKan's logic.
    
    Returns index i such that knots[i] <= x < knots[i+1],
    clamped to valid range [order, order + grid_size - 1].
    """
    t_min = knots[order]
    t_max = knots[order + grid_size]
    step = (t_max - t_min) / grid_size
    x_clamped = np.clip(x, t_min, t_max)
    
    raw_idx = int(np.floor((x_clamped - t_min) / step))
    
    # Clamp to valid interval
    max_interval = grid_size - 1
    idx = max(0, min(raw_idx, max_interval))
    
    return idx + order


def compute_basis_scipy(x: float, knots: np.ndarray, order: int, span: int) -> np.ndarray:
    """
    Compute B-spline basis functions using SciPy.
    
    For a given x and span, computes the (order+1) non-vanishing basis functions:
    B_{span-order}(x), B_{span-order+1}(x), ..., B_{span}(x)
    
    Uses scipy.interpolate.BSpline with unit coefficient vectors.
    """
    n_basis = order + 1
    basis_values = np.zeros(n_basis, dtype=np.float64)
    
    # The active basis functions are B_{span-order} through B_{span}
    # In SciPy's numbering, basis function i has support [knots[i], knots[i+order+1])
    
    n_splines = len(knots) - order - 1  # Total number of basis functions
    
    for local_idx in range(n_basis):
        global_idx = span - order + local_idx
        
        if 0 <= global_idx < n_splines:
            # Create coefficient vector with 1 at position global_idx
            coeffs = np.zeros(n_splines, dtype=np.float64)
            coeffs[global_idx] = 1.0
            
            # Create B-spline and evaluate
            try:
                spline = BSpline(knots, coeffs, order)
                basis_values[local_idx] = spline(x)
            except Exception:
                basis_values[local_idx] = 0.0
    
    return basis_values


def generate_test_cases(grid_size: int, order: int, grid_range: tuple, 
                        num_random: int = 20) -> list:
    """
    Generate test cases for a specific configuration.
    
    Includes:
    - Boundary cases (exactly at grid points)
    - Interior points (between grid points)
    - Edge cases (at the limits of the range)
    """
    t_min, t_max = grid_range
    knots = compute_knots_arkan(grid_size, order, grid_range)
    
    test_cases = []
    
    # Grid points (exactly at knots within the active range)
    h = (t_max - t_min) / grid_size
    for i in range(grid_size + 1):
        x = t_min + i * h
        # Slightly offset right boundary to stay in valid range
        if i == grid_size:
            x = x - 1e-9
        
        span = find_span_arkan(x, knots, order, grid_size)
        basis = compute_basis_scipy(x, knots, order, span)
        
        test_cases.append({
            "x": float(x),
            "span": int(span),
            "basis": [float(b) for b in basis],
            "basis_sum": float(np.sum(basis)),
            "type": "grid_point"
        })
    
    # Random interior points
    np.random.seed(42)
    for _ in range(num_random):
        x = np.random.uniform(t_min, t_max - 1e-9)
        span = find_span_arkan(x, knots, order, grid_size)
        basis = compute_basis_scipy(x, knots, order, span)
        
        test_cases.append({
            "x": float(x),
            "span": int(span),
            "basis": [float(b) for b in basis],
            "basis_sum": float(np.sum(basis)),
            "type": "random"
        })
    
    # Midpoints between knots
    for i in range(grid_size):
        x = t_min + (i + 0.5) * h
        span = find_span_arkan(x, knots, order, grid_size)
        basis = compute_basis_scipy(x, knots, order, span)
        
        test_cases.append({
            "x": float(x),
            "span": int(span),
            "basis": [float(b) for b in basis],
            "basis_sum": float(np.sum(basis)),
            "type": "midpoint"
        })
    
    return test_cases, knots.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Export B-spline basis values from SciPy for ArKan parity testing"
    )
    parser.add_argument(
        "--output", "-o",
        default="tests/spline_test_data.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()
    
    # Test configurations matching common ArKan usage
    configs = [
        {"grid": 3, "order": 2, "range": [-1.0, 1.0]},
        {"grid": 5, "order": 3, "range": [-1.0, 1.0]},
        {"grid": 8, "order": 3, "range": [-2.0, 2.0]},
        {"grid": 5, "order": 4, "range": [-1.0, 1.0]},
        {"grid": 4, "order": 2, "range": [0.0, 1.0]},
    ]
    
    output_data = {
        "version": "1.0",
        "description": "B-spline basis function reference values from SciPy",
        "generator": "scipy.interpolate.BSpline",
        "configurations": []
    }
    
    for cfg in configs:
        grid_size = cfg["grid"]
        order = cfg["order"]
        grid_range = tuple(cfg["range"])
        
        test_cases, knots = generate_test_cases(grid_size, order, grid_range)
        
        config_data = {
            "config": {
                "grid_size": grid_size,
                "order": order,
                "range": list(grid_range)
            },
            "knots": knots,
            "num_basis_functions": grid_size + order,
            "test_cases": test_cases
        }
        
        output_data["configurations"].append(config_data)
        
        # Verify partition of unity
        errors = [tc for tc in test_cases if abs(tc["basis_sum"] - 1.0) > 1e-10]
        if errors:
            print(f"WARNING: Partition of unity violated for grid={grid_size}, order={order}")
            for e in errors[:3]:
                print(f"  x={e['x']:.6f}, sum={e['basis_sum']:.10f}")
        else:
            print(f"✓ Config grid={grid_size}, order={order}: {len(test_cases)} test cases, partition of unity OK")
    
    # Write output
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nExported to {args.output}")
    print(f"Total configurations: {len(configs)}")
    total_cases = sum(len(cfg["test_cases"]) for cfg in output_data["configurations"])
    print(f"Total test cases: {total_cases}")


if __name__ == "__main__":
    main()
