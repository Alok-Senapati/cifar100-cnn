"""Checkpoint loading helpers for saved PyTorch classification models.

Checkpoints written by cifar100_cnn.model.trainer.train contain both
the model import path and constructor arguments, allowing this module to
recreate the architecture before restoring its state dictionary.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import torch
import torch.nn as nn


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
    eval_mode: bool = True,
) -> nn.Module:
    """Dynamically reconstruct a neural network model and restore its trained weights.

    This function inspects the metadata stored inside the checkpoint file,
    dynamically imports the model's originating class, reconstructs the
    architecture with the exact initialization arguments, loads the trained
    parameters (state dict), and moves the model to the target device.

    Args:
        checkpoint_path: Path to the saved PyTorch checkpoint file (`.pt` or `.pth`).
        device: Target device (e.g. `'cpu'`, `'cuda'`, or `torch.device`) to place the model on.
            Defaults to CPU if not specified.
        eval_mode: When True, switches the model to evaluation mode (`model.eval()`).
            Defaults to True.

    Returns:
        The instantiated `torch.nn.Module` with restored weights.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint file is corrupted or missing required keys.
        ImportError: If the model module or class cannot be dynamically imported.
        TypeError: If the model constructor rejects the saved initialization arguments.
        RuntimeError: If loading the state dictionary into the model fails.

    Notes:
        The checkpoint is deserialized with weights_only=False because it
        contains metadata and optimizer state in addition to tensor weights.
        Only load checkpoints from a trusted source, since PyTorch deserialization
        can execute code embedded in a pickle payload.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    target_device = torch.device(device) if device is not None else torch.device("cpu")

    try:
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    except Exception as exc:
        raise ValueError(f"Failed to deserialize checkpoint file '{path}': {exc}") from exc

    if not isinstance(checkpoint, dict):
        obj_type = type(checkpoint).__name__
        raise ValueError(f"Invalid checkpoint in '{path}'. Expected dict, got {obj_type}.")

    if "model_meta" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError(
            f"Checkpoint at '{path}' is missing required keys ('model_meta' or 'model_state')."
        )

    model_meta = checkpoint["model_meta"]
    if not isinstance(model_meta, dict):
        raise ValueError(f"Invalid model metadata in '{path}'. Expected a dictionary.")
    module_path = model_meta.get("module_path")
    class_name = model_meta.get("class_name")
    init_args = model_meta.get("init_args", {})
    if not isinstance(init_args, dict):
        raise ValueError(f"Invalid constructor arguments in '{path}'. Expected a dictionary.")

    if not module_path or not class_name:
        raise ValueError(
            f"Incomplete metadata in '{path}': module_path={module_path}, class_name={class_name}"
        )

    try:
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Could not load class '{class_name}' from module '{module_path}': {exc}"
        ) from exc

    try:
        model_instance: nn.Module = model_class(**init_args)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate '{class_name}' with saved arguments {init_args}: {exc}"
        ) from exc

    try:
        model_instance.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load state dictionary into '{class_name}': {exc}") from exc

    model_instance.to(target_device)
    if eval_mode:
        model_instance.eval()

    return model_instance
