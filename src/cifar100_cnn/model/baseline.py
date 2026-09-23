"""Baseline Convolutional Neural Network for CIFAR-100 image classification."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import torch
import torch.nn as nn

__all__ = ["BaselineCNN"]


class BaselineCNN(nn.Module):
    """Convolutional Neural Network baseline model for image classification.

    The architecture consists of a configurable sequence of convolutional blocks
    followed by a fully connected classification head.

    Each convolutional block comprises:
        1. 2D Convolution (3x3 kernel, stride 1, padding 1)
        2. ReLU activation
        3. 2D Max Pooling (2x2 kernel, stride 2)

    The classification head flattens the extracted feature map and applies:
        1. Flatten layer
        2. Fully connected linear layer (feature_dims -> fc_hidden)
        3. ReLU activation
        4. Fully connected linear layer (fc_hidden -> n_classes)

    Attributes:
        in_dims: Dimensions of input images as (channels, height, width).
        conv_channels: Tuple of output channel sizes for each conv block.
        fc_hidden: Number of hidden units in the first fully connected layer.
        n_classes: Number of target output classes.
        conv_layers: ModuleDict of sequentially arranged convolutional blocks.
        classifier_layer: Sequential module of the classification head.
    """

    def __init__(
        self,
        in_dims: tuple[int, int, int] | Sequence[int],
        conv_channels: Sequence[int],
        fc_hidden: int,
        n_classes: int = 100,
    ) -> None:
        """Initialize the BaselineCNN model.

        Args:
            in_dims: Dimensions of input images as (channels, height, width).
            conv_channels: Sequence of output channels for each conv block.
            fc_hidden: Number of hidden units in the first fully connected layer.
            n_classes: Number of output classification targets. Defaults to 100.

        Raises:
            TypeError: If arguments are of incorrect types.
            ValueError: If dimensions, channels, or hidden sizes are non-positive,
                if conv_channels is empty, or if spatial dimensions collapse to
                zero during repeated 2x2 max pooling.
        """
        super().__init__()

        # Validate the input channel and spatial dimensions.
        if not isinstance(in_dims, (tuple, list)):
            raise TypeError(
                f"in_dims must be a tuple or list of 3 integers, got {type(in_dims).__name__}."
            )
        if len(in_dims) != 3:
            raise ValueError(
                f"in_dims must have exactly 3 elements (channels, height, width), "
                f"got {len(in_dims)}."
            )
        for dim, name in zip(in_dims, ("channels", "height", "width"), strict=True):
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise ValueError(f"in_dims {name} must be a positive integer, got {dim}.")
        self.in_dims: tuple[int, int, int] = (in_dims[0], in_dims[1], in_dims[2])

        # Validate the output channels for each convolutional block.
        if not isinstance(conv_channels, (list, tuple)):
            raise TypeError(
                "conv_channels must be a list or tuple of integers, "
                f"got {type(conv_channels).__name__}."
            )
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one channel specification.")
        for idx, ch in enumerate(conv_channels):
            if not isinstance(ch, int) or isinstance(ch, bool) or ch <= 0:
                raise ValueError(f"conv_channels[{idx}] must be a positive integer, got {ch}.")
        self.conv_channels: tuple[int, ...] = tuple(conv_channels)

        # Validate the classifier hidden-layer width.
        if not isinstance(fc_hidden, int) or isinstance(fc_hidden, bool) or fc_hidden <= 0:
            raise ValueError(f"fc_hidden must be a positive integer, got {fc_hidden}.")
        self.fc_hidden = fc_hidden

        # Validate the number of output classes.
        if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes <= 0:
            raise ValueError(f"n_classes must be a positive integer, got {n_classes}.")
        self.n_classes = n_classes

        # Build each convolutional block and track its output dimensions.
        layer_in_dims = self.in_dims
        conv_layers_map: dict[str, nn.Sequential] = {}
        for idx, channels in enumerate(conv_channels):
            current_h, current_w = layer_in_dims[1], layer_in_dims[2]
            # Each pooling layer needs at least two pixels in both spatial dimensions.
            if current_h < 2 or current_w < 2:
                raise ValueError(
                    f"Input spatial dimensions ({current_h}, {current_w}) at conv block {idx} "
                    f"are too small for 2x2 max-pooling. Reduce the number of convolutional layers "
                    f"or use larger input dimensions."
                )

            conv_layers_map[f"conv_{idx}"] = nn.Sequential(
                nn.Conv2d(layer_in_dims[0], channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            # Pooling halves each spatial dimension using floor division.
            layer_in_dims = (channels, layer_in_dims[1] // 2, layer_in_dims[2] // 2)

        # The classifier receives every value in the final feature map.
        feature_dims = int(reduce(lambda i, j: i * j, layer_in_dims))
        self.conv_layers = nn.ModuleDict(conv_layers_map)

        # Flatten the final feature map and map it to class logits.
        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dims, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through convolutional layers and classification head.

        Args:
            x: Input batch tensor of shape (batch_size, channels, height, width).

        Returns:
            Output logits tensor of shape (batch_size, n_classes).

        Raises:
            TypeError: If input x is not a torch.Tensor.
            ValueError: If input x does not have 4 dimensions or if its channels
                and spatial dimensions do not match self.in_dims.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (batch_size, channels, height, width), "
                f"got {x.ndim}D tensor with shape {tuple(x.shape)}."
            )
        if tuple(x.shape[1:]) != self.in_dims:
            c, h, w = self.in_dims
            raise ValueError(
                f"Expected input tensor with shape (*, {c}, {h}, {w}), got shape {tuple(x.shape)}."
            )

        conv_layer_out = x
        for layer in self.conv_layers.values():
            conv_layer_out = layer(conv_layer_out)
        return self.classifier_layer(conv_layer_out)
