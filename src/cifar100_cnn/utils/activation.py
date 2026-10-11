"""Capture a bounded set of intermediate tensors for model inspection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["LayerActivation", "capture_layer_activations"]


@dataclass(frozen=True)
class LayerActivation:
    """A detached activation snapshot produced by one model layer.

    Feature-map tensors are limited to a configurable number of channels so an
    interactive inspection does not retain a full model graph or large tensors.
    """

    name: str
    layer_type: str
    output_shape: tuple[int, ...]
    values: Tensor


def capture_layer_activations(
    model: nn.Module,
    image: Tensor,
    max_channels: int = 16,
) -> tuple[list[LayerActivation], Tensor]:
    """Run one image through ``model`` and capture display-sized layer outputs.

    Leaf modules are captured in execution order. Residual blocks are captured
    as well, because their summed output is created by tensor addition rather
    than a dedicated PyTorch module.

    Args:
        model: An evaluation-mode image classifier.
        image: One normalized CHW image matching the model's input dimensions.
        max_channels: Maximum feature channels retained for each activation.

    Returns:
        Layer snapshots in forward execution order and the model's output logits.

    Raises:
        TypeError: If the model or image has an unsupported type.
        ValueError: If the image is not a single CHW image or max_channels is invalid.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 3:
        raise ValueError("image must be one CHW image with shape (channels, height, width).")
    if type(max_channels) is not int or max_channels <= 0:
        raise ValueError("max_channels must be a positive integer.")

    records: list[LayerActivation] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def save_output(name: str, module: nn.Module, output: object) -> None:
        if not isinstance(output, Tensor) or output.ndim not in (2, 4):
            return
        detached = output.detach().to(device="cpu", dtype=torch.float32)
        shape = tuple(int(dimension) for dimension in detached.shape)
        detached = detached[:, :max_channels]
        records.append(
            LayerActivation(
                name=name or "model",
                layer_type=type(module).__name__,
                output_shape=shape,
                values=detached.contiguous(),
            )
        )

    try:
        for name, module in model.named_modules():
            is_leaf = not any(module.children())
            is_residual_block = type(module).__name__ == "ResidualBlock"
            if not name or not (is_leaf or is_residual_block):
                continue
            handles.append(
                module.register_forward_hook(
                    lambda current_module, _inputs, output, layer_name=name: save_output(
                        layer_name, current_module, output
                    )
                )
            )
        with torch.inference_mode():
            logits = model(image.unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()

    if not isinstance(logits, Tensor):
        raise TypeError("model must return a tensor of class logits.")
    return records, logits.detach().cpu()
