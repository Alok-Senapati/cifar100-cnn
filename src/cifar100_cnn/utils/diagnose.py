"""Diagnostic utilities for neural network gradients and activations."""

from __future__ import annotations

import torch.nn as nn


def compute_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Compute the L2 norm of gradients for each model parameter and overall.

    This helper traverses all named parameters in the model that have accumulated
    gradients (after `loss.backward()`), computes their Euclidean (L2) norm,
    and calculates the total aggregate gradient norm.

    Args:
        model: PyTorch neural network model with computed parameter gradients.

    Returns:
        A dictionary mapping parameter names (e.g. `'grad_norm/head.weight'`)
        and `'grad_norm/total'` to their corresponding float L2 norm values.
    """
    norms: dict[str, float] = {}
    total_norm_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = float(param.grad.detach().data.norm(2).item())
            norms[f"grad_norm/{name}"] = param_norm
            total_norm_sq += param_norm**2

    norms["grad_norm/total"] = total_norm_sq**0.5
    return norms
