"""Diagnostic utilities for neural network gradients and activations."""

from __future__ import annotations

import torch.nn as nn


def compute_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Compute per-parameter and aggregate L2 gradient norms.

    Parameters without gradients are skipped. The aggregate is the square root
    of the sum of squared per-parameter norms, and is zero when no gradients exist.

    Args:
        model: Model whose parameters may have accumulated gradients.

    Returns:
        A mapping from parameter names and the aggregate key grad_norm/total
        to their respective L2 norms.
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
