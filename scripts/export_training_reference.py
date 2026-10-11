"""
Export training reference data for the ArKan differential test harness.

Supports multiple configs (optimizer, spline order, loss, depth).
Each config emits one JSON file to tests/reference_data/.

Usage:
  python scripts/export_training_reference.py            # export all configs
  python scripts/export_training_reference.py sgd_order3 # single config

KEY DESIGN CHOICES (matching ArKan exactly):
  - grid_range = (-1.0, 1.0), data in (-0.9, 0.9): stays inside knot span
  - input_mean = 0, input_std = 1 → ArKan normalization is identity
  - Pure B-spline forward: phi(x) = sum_k c_k * B_k(x), NO base/residual term
  - Weight layout: weights[out][in][global_basis_k] flat in ArKan as
    (out * in_dim + in) * global_basis_size + global_basis_k

FIND_SPAN EQUIVALENCE (grid_range=(-1,1)):
  ArKan:  t_min=knots[order], step=(t_max-t_min)/G
          ratio=(x-t_min)/step+eps, span=clamp(floor(ratio),0,G-1)+order
  PyTorch: shared bench_reference clamps layer inputs before span and basis.
"""

import json
import os
import sys

import torch
import numpy as np
from bench_reference import forward_layer

# ---------------------------------------------------------------------------
# Default values — individual configs override these
# ---------------------------------------------------------------------------
SEED = 42
GRID_RANGE = (-1.0, 1.0)
N_SAMPLES = 32
N_STEPS = 50
LR = 0.001

# ---------------------------------------------------------------------------
# Configs to export: (name, overrides dict)
# ---------------------------------------------------------------------------
ALL_CONFIGS = {
    "sgd_order3": {
        "input_dim": 2, "hidden_dims": [8], "output_dim": 1,
        "grid_size": 5, "spline_order": 3,
        "optimizer_type": "sgd", "lr": 0.001, "momentum": 0.0,
        "loss_type": "mse",
    },
    "adam_order3": {
        "input_dim": 2, "hidden_dims": [8], "output_dim": 1,
        "grid_size": 5, "spline_order": 3,
        "optimizer_type": "adam", "lr": 0.001,
        "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.0,
        "loss_type": "mse",
    },
    "sgd_order4": {
        "input_dim": 2, "hidden_dims": [8], "output_dim": 1,
        "grid_size": 5, "spline_order": 4,
        "optimizer_type": "sgd", "lr": 0.001, "momentum": 0.0,
        "loss_type": "mse",
    },
    "bce_order3": {
        "input_dim": 2, "hidden_dims": [8], "output_dim": 1,
        "grid_size": 5, "spline_order": 3,
        "optimizer_type": "sgd", "lr": 0.001, "momentum": 0.0,
        "loss_type": "bce",
    },
    "multilayer": {
        "input_dim": 2, "hidden_dims": [8, 8], "output_dim": 1,
        "grid_size": 5, "spline_order": 3,
        "optimizer_type": "sgd", "lr": 0.001, "momentum": 0.0,
        "loss_type": "mse",
    },
}

# ---------------------------------------------------------------------------
# Spline math — faithful ArKan reimplementation
# ---------------------------------------------------------------------------

def compute_knots(grid_size: int, order: int, grid_range: tuple) -> torch.Tensor:
    """Uniform knot vector — identical to ArKan's compute_knots (float32)."""
    t_min, t_max = grid_range
    n_knots = grid_size + 2 * order + 1
    h = (t_max - t_min) / grid_size
    return torch.tensor(
        [t_min + (i - order) * h for i in range(n_knots)],
        dtype=torch.float32,
    )


def forward_one_layer(
    x: torch.Tensor,
    weights: torch.Tensor,   # [out_dim, in_dim, global_basis]
    bias: torch.Tensor,      # [out_dim]
    knots: torch.Tensor,
    order: int,
    grid_size: int,
) -> torch.Tensor:
    """Forward pass for one KAN layer."""
    return forward_layer(x, weights, bias, knots, grid_size, order)


def network_forward(
    layers: list,
    knots: torch.Tensor,
    x: torch.Tensor,
    order: int,
    grid_size: int,
) -> torch.Tensor:
    """Multi-layer forward pass."""
    cur = x
    for layer in layers:
        cur = forward_one_layer(cur, layer["weights"], layer["bias"], knots, order, grid_size)
    return cur


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def compute_loss(preds: torch.Tensor, targets: torch.Tensor, loss_type: str):
    """Return (scalar_loss, grad_output) matching ArKan's loss implementations."""
    if loss_type == "mse":
        loss = ((preds - targets) ** 2).mean()
    elif loss_type == "bce":
        # PyTorch binary_cross_entropy_with_logits matches ArKan masked_bce_with_logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            preds, targets, reduction="mean"
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss


# ---------------------------------------------------------------------------
# Main export function for a single config
# ---------------------------------------------------------------------------

def export_config(name: str, cfg: dict, out_dir: str):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dtype = torch.float32

    input_dim = cfg["input_dim"]
    hidden_dims = cfg["hidden_dims"]
    output_dim = cfg["output_dim"]
    grid_size = cfg["grid_size"]
    order = cfg["spline_order"]
    optimizer_type = cfg["optimizer_type"]
    lr = cfg.get("lr", LR)
    loss_type = cfg.get("loss_type", "mse")

    knots = compute_knots(grid_size, order, GRID_RANGE)
    global_basis = grid_size + order

    # Build architecture
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for in_d, out_d in zip(dims[:-1], dims[1:]):
        scale = (2.0 / (in_d + out_d)) ** 0.5
        weights = (torch.rand(out_d, in_d, global_basis, dtype=dtype) - 0.5) * scale * 0.1
        weights.requires_grad_(True)
        bias = torch.zeros(out_d, dtype=dtype, requires_grad=True)
        layers.append({"weights": weights, "bias": bias, "in": in_d, "out": out_d})

    # Dataset: inputs in (-0.9, 0.9) to stay inside grid_range=(-1, 1)
    inputs_np = np.random.uniform(-0.9, 0.9, size=(N_SAMPLES, input_dim)).astype(np.float32)
    inputs = torch.tensor(inputs_np, dtype=dtype)

    if loss_type == "mse":
        targets_np = (
            np.sin(np.pi * inputs_np[:, 0]) * inputs_np[:, 1] + 0.5 * inputs_np[:, 0]
        ).reshape(-1, 1).astype(np.float32)
        # Repeat for multi-output if needed
        if output_dim > 1:
            targets_np = np.tile(targets_np, (1, output_dim))
    elif loss_type == "bce":
        # Binary targets in {0, 1} based on sign of first feature
        targets_np = (inputs_np[:, 0] > 0).astype(np.float32).reshape(-1, 1)
        if output_dim > 1:
            targets_np = np.tile(targets_np, (1, output_dim))
    targets = torch.tensor(targets_np, dtype=dtype)

    # Capture init weights
    init_weights_per_layer = []
    for layer in layers:
        flat_w = layer["weights"].detach().clone().numpy().flatten().tolist()
        flat_b = layer["bias"].detach().clone().numpy().tolist()
        init_weights_per_layer.append({"weights": flat_w, "bias": flat_b})

    # -----------------------------------------------------------------------
    # Step 0: forward + gradient
    # -----------------------------------------------------------------------
    for layer in layers:
        if layer["weights"].grad is not None:
            layer["weights"].grad.zero_()
        if layer["bias"].grad is not None:
            layer["bias"].grad.zero_()

    preds = network_forward(layers, knots, inputs, order, grid_size)
    loss0 = compute_loss(preds, targets, loss_type)
    loss0.backward()

    step0_forward = preds.detach().numpy().flatten().tolist()
    step0_grads = []
    for layer in layers:
        gw = layer["weights"].grad.detach().clone().numpy().flatten().tolist()
        gb = layer["bias"].grad.detach().clone().numpy().tolist()
        step0_grads.append({"weights": gw, "bias": gb})

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    loss_trajectory = []

    # Fresh leaf tensors for training
    train_layers = []
    for layer in layers:
        w = layer["weights"].detach().clone().to(dtype).requires_grad_(True)
        b = layer["bias"].detach().clone().to(dtype).requires_grad_(True)
        train_layers.append({"weights": w, "bias": b})

    # Build optimizer
    params = []
    for layer in train_layers:
        params += [layer["weights"], layer["bias"]]

    if optimizer_type == "sgd":
        momentum = cfg.get("momentum", 0.0)
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_type == "adam":
        beta1 = cfg.get("beta1", 0.9)
        beta2 = cfg.get("beta2", 0.999)
        eps = cfg.get("epsilon", 1e-8)
        wd = cfg.get("weight_decay", 0.0)
        # ponytail: checked-in Adam uses zero decay; nonzero decay needs AdamW for ArKan parity.
        optimizer = torch.optim.Adam(
            params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd
        )
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    for step in range(N_STEPS):
        optimizer.zero_grad()
        preds = network_forward(train_layers, knots, inputs, order, grid_size)
        loss = compute_loss(preds, targets, loss_type)
        loss_trajectory.append(float(loss.item()))
        loss.backward()
        optimizer.step()

    # Capture final weights
    final_weights_per_layer = []
    for layer in train_layers:
        flat_w = layer["weights"].detach().clone().numpy().flatten().tolist()
        flat_b = layer["bias"].detach().clone().numpy().tolist()
        final_weights_per_layer.append({"weights": flat_w, "bias": flat_b})

    # -----------------------------------------------------------------------
    # Assemble JSON
    # -----------------------------------------------------------------------
    optimizer_info = {
        "type": optimizer_type,
        "lr": lr,
    }
    if optimizer_type == "sgd":
        optimizer_info["momentum"] = cfg.get("momentum", 0.0)
    elif optimizer_type == "adam":
        optimizer_info["beta1"] = cfg.get("beta1", 0.9)
        optimizer_info["beta2"] = cfg.get("beta2", 0.999)
        optimizer_info["epsilon"] = cfg.get("epsilon", 1e-8)
        optimizer_info["weight_decay"] = cfg.get("weight_decay", 0.0)

    ref = {
        "config": {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "grid_size": grid_size,
            "spline_order": order,
            "grid_range": list(GRID_RANGE),
            "global_basis_size": global_basis,
            "dims": dims,
        },
        "optimizer": optimizer_info,
        "loss_type": loss_type,
        "dataset": {
            "inputs": inputs.numpy().flatten().tolist(),
            "targets": targets.numpy().flatten().tolist(),
            "n_samples": N_SAMPLES,
        },
        "init_weights": init_weights_per_layer,
        "step0_forward": step0_forward,
        "step0_grads": step0_grads,
        "loss_trajectory": loss_trajectory,
        "final_weights": final_weights_per_layer,
    }

    filename = f"training_ref_{name}.json"
    out_path = os.path.join(out_dir, filename)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ref, f, indent=2)

    print(f"[{name}] Exported to {os.path.abspath(out_path)}")
    print(f"  arch={dims}, order={order}, optimizer={optimizer_type}, loss={loss_type}")
    print(f"  step0_loss={loss_trajectory[0]:.6f}, final_loss={loss_trajectory[-1]:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "tests", "reference_data")

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(ALL_CONFIGS.keys())

    for name in names:
        if name not in ALL_CONFIGS:
            print(f"Unknown config '{name}'. Available: {list(ALL_CONFIGS.keys())}")
            sys.exit(1)
        export_config(name, ALL_CONFIGS[name], out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
