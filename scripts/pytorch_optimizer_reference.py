#!/usr/bin/env python3
"""
Generate reference values from PyTorch optimizers for ArKan verification.

This script produces expected parameter trajectories that Rust tests can compare against.
We use simple optimization problems with known solutions:
1. Quadratic function: f(x) = Σ x_i²  (minimum at origin)
2. Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²  (minimum at (1,1))

Run: python pytorch_optimizer_reference.py > pytorch_reference_values.txt
"""

import torch
import json
from typing import List, Dict, Any

def quadratic_loss(params: torch.Tensor) -> torch.Tensor:
    """Simple quadratic: f(x) = Σ x_i²"""
    return (params ** 2).sum()

def rosenbrock_loss(params: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"""
    x, y = params[0], params[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def run_adam_steps(
    init_params: List[float],
    lr: float,
    betas: tuple,
    eps: float,
    weight_decay: float,
    n_steps: int,
    loss_fn,
) -> Dict[str, Any]:
    """Run Adam optimization and record trajectory."""
    params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam(
        [params], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    
    trajectory = [params.detach().tolist()]
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(params)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        trajectory.append(params.detach().tolist())
    
    return {
        "optimizer": "Adam",
        "config": {
            "lr": lr,
            "betas": list(betas),
            "eps": eps,
            "weight_decay": weight_decay,
        },
        "init_params": init_params,
        "n_steps": n_steps,
        "trajectory": trajectory,
        "losses": losses,
        "final_params": params.detach().tolist(),
    }

def run_adamw_steps(
    init_params: List[float],
    lr: float,
    betas: tuple,
    eps: float,
    weight_decay: float,
    n_steps: int,
    loss_fn,
) -> Dict[str, Any]:
    """Run AdamW (decoupled weight decay) optimization and record trajectory."""
    params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [params], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    
    trajectory = [params.detach().tolist()]
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(params)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        trajectory.append(params.detach().tolist())
    
    return {
        "optimizer": "AdamW",
        "config": {
            "lr": lr,
            "betas": list(betas),
            "eps": eps,
            "weight_decay": weight_decay,
        },
        "init_params": init_params,
        "n_steps": n_steps,
        "trajectory": trajectory,
        "losses": losses,
        "final_params": params.detach().tolist(),
    }

def run_sgd_steps(
    init_params: List[float],
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    n_steps: int,
    loss_fn,
) -> Dict[str, Any]:
    """Run SGD optimization and record trajectory."""
    params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD(
        [params], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
    )
    
    trajectory = [params.detach().tolist()]
    losses = []
    
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(params)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        trajectory.append(params.detach().tolist())
    
    return {
        "optimizer": "SGD",
        "config": {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        },
        "init_params": init_params,
        "n_steps": n_steps,
        "trajectory": trajectory,
        "losses": losses,
        "final_params": params.detach().tolist(),
    }

def run_lbfgs_steps(
    init_params: List[float],
    lr: float,
    max_iter: int,
    history_size: int,
    line_search_fn: str,
    n_steps: int,
    loss_fn,
) -> Dict[str, Any]:
    """Run L-BFGS optimization and record trajectory."""
    params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [params],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn=line_search_fn if line_search_fn else None,
    )
    
    trajectory = [params.detach().tolist()]
    losses = []
    
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        return loss
    
    for step in range(n_steps):
        loss = optimizer.step(closure)
        losses.append(loss.item())
        trajectory.append(params.detach().tolist())
    
    return {
        "optimizer": "LBFGS",
        "config": {
            "lr": lr,
            "max_iter": max_iter,
            "history_size": history_size,
            "line_search_fn": line_search_fn,
        },
        "init_params": init_params,
        "n_steps": n_steps,
        "trajectory": trajectory,
        "losses": losses,
        "final_params": params.detach().tolist(),
    }

def main():
    results = []
    
    # ============================================================
    # 1. Adam on quadratic function
    # ============================================================
    
    # Test 1a: Default Adam on quadratic
    results.append(run_adam_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # Test 1b: Adam with weight decay (L2 regularization style)
    results.append(run_adam_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # Test 1c: Adam with custom betas
    results.append(run_adam_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.001,
        betas=(0.5, 0.9999),
        eps=1e-8,
        weight_decay=0.0,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # ============================================================
    # 2. AdamW (decoupled weight decay) on quadratic
    # ============================================================
    
    # Test 2a: AdamW with decoupled weight decay
    results.append(run_adamw_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # ============================================================
    # 3. SGD with momentum on quadratic
    # ============================================================
    
    # Test 3a: SGD without momentum
    results.append(run_sgd_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # Test 3b: SGD with momentum
    results.append(run_sgd_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0,
        nesterov=False,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # Test 3c: SGD with Nesterov momentum
    results.append(run_sgd_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0,
        nesterov=True,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # Test 3d: SGD with weight decay
    results.append(run_sgd_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.01,
        nesterov=False,
        n_steps=10,
        loss_fn=quadratic_loss,
    ))
    
    # ============================================================
    # 4. L-BFGS on Rosenbrock function
    # ============================================================
    
    # Test 4a: L-BFGS with strong_wolfe line search
    results.append(run_lbfgs_steps(
        init_params=[-1.0, 1.0],
        lr=1.0,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
        n_steps=20,
        loss_fn=rosenbrock_loss,
    ))
    
    # Test 4b: L-BFGS on quadratic (should converge very fast)
    results.append(run_lbfgs_steps(
        init_params=[1.0, 2.0, 3.0, 4.0],
        lr=1.0,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
        n_steps=5,
        loss_fn=quadratic_loss,
    ))
    
    # ============================================================
    # Output JSON
    # ============================================================
    
    print(json.dumps(results, indent=2))
    
    # Also print summary for human readability
    print("\n" + "=" * 60, file=__import__('sys').stderr)
    print("SUMMARY", file=__import__('sys').stderr)
    print("=" * 60, file=__import__('sys').stderr)
    
    for i, r in enumerate(results):
        print(f"\n{i+1}. {r['optimizer']} ({r['config']})", file=__import__('sys').stderr)
        print(f"   Init: {r['init_params']}", file=__import__('sys').stderr)
        print(f"   Final: {r['final_params']}", file=__import__('sys').stderr)
        print(f"   Loss: {r['losses'][0]:.6f} → {r['losses'][-1]:.6f}", file=__import__('sys').stderr)

if __name__ == "__main__":
    main()
