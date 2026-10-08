"""Pure B-spline tensor operations matching ArKan's coefficient layout."""
import torch


def find_span(x, order, grid_size, grid_range=(-3.0, 3.0), knots=None):
    lower, upper = grid_range if knots is None else (knots[order], knots[order + grid_size])
    ratio = (x.clamp(lower, upper) - lower) / ((upper - lower) / grid_size)
    return (ratio + 1e-6).floor().long().clamp(0, grid_size - 1) + order


def compute_basis_vectorized(x, span, knots, order):
    # Functional recursion keeps input gradients valid on CPU and CUDA.
    left = [None] + [x - knots[span + 1 - j] for j in range(1, order + 1)]
    right = [None] + [knots[span + j] - x for j in range(1, order + 1)]
    basis = [torch.ones_like(x)]
    for j in range(1, order + 1):
        saved = torch.zeros_like(x)
        columns = []
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            valid = denom > 0.0
            temp = basis[r] / torch.where(valid, denom, torch.ones_like(denom)) * valid
            columns.append(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
        basis = columns + [saved]
    return torch.stack(basis, dim=-1)


def forward_layer(x, weights, bias, knots, grid_size, order, mean=0.0, std=1.0):
    x = ((x - mean) / std).clamp(knots[order], knots[order + grid_size])
    span = find_span(x, order, grid_size, knots=knots)
    basis = compute_basis_vectorized(x, span, knots, order)
    indices = span.unsqueeze(-1) - order + torch.arange(order + 1, device=x.device)
    expanded = weights.unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    indices = indices.unsqueeze(1).expand(-1, weights.shape[0], -1, -1)
    selected = torch.gather(expanded, dim=3, index=indices)
    return (selected * basis.unsqueeze(1)).sum(dim=3).sum(dim=2) + bias


def capture_reset(model, optimizer):
    """Prepare reusable optimizer buffers and restore the same initial step outside timing."""
    import copy
    model_state = copy.deepcopy(model.state_dict())
    # PyTorch initializes Adam/momentum buffers lazily. Allocate them before timing.
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.zero_()
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    def reset():
        model.load_state_dict(model_state)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        optimizer.zero_grad(set_to_none=True)
    reset()
    return reset
