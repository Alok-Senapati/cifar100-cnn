"""Plain CNN stages for controlled residual-learning ablations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["PlainBlock", "PlainStageCNN"]


class PlainBlock(nn.Module):
    """Two-convolution block with no identity or projection shortcut.

    It matches the convolution, normalization, and activation depth of a
    residual block. The first convolution can downsample at a stage transition.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Create the block's two convolutional transformations.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output channels for both convolutions.
            stride: Stride for the first convolution; must be 1 or 2.

        Raises:
            TypeError: If a channel count or stride is not an integer.
            ValueError: If channel counts are non-positive or stride is unsupported.
        """
        super().__init__()
        if any(type(value) is not int for value in (in_channels, out_channels, stride)):
            raise TypeError("channel counts and stride must be integers.")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2 for this CIFAR block.")

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolution, normalization, and activation operations."""
        return self.layers(x)


class PlainStageCNN(nn.Module):
    """CIFAR classifier with ResNet-like stages and no skip connections.

    The stem, stage widths, blocks per stage, downsampling, global average
    pooling, and classifier match SmallResNet. PlainBlock replaces residual
    addition with a second ReLU, making this model a control for residual-
    learning studies.
    """

    def __init__(
        self,
        in_dims: tuple[int, int, int] | Sequence[int],
        n_classes: int = 100,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: int = 2,
        initialize_weights: bool = True,
    ) -> None:
        """Build the stem, plain stages, global pool, and classifier.

        Args:
            in_dims: Input dimensions in (channels, height, width) order.
            n_classes: Number of output classes.
            stage_channels: Output channel count for every stage.
            blocks_per_stage: Number of two-convolution blocks in each stage.
            initialize_weights: Apply Kaiming, unit BatchNorm, and Xavier
                initialization when true.

        Raises:
            TypeError: If dimensions, channels, counts, or the initialization
                flag have invalid types.
            ValueError: If dimensions, channel counts, class count, or block
                count are non-positive, or no stage widths are supplied.
        """
        super().__init__()
        if not isinstance(in_dims, (tuple, list)) or len(in_dims) != 3:
            raise ValueError("in_dims must contain (channels, height, width).")
        if any(type(value) is not int for value in in_dims):
            raise TypeError("in_dims values must be integers.")
        if any(value <= 0 for value in in_dims):
            raise ValueError("in_dims values must be positive.")
        if type(n_classes) is not int or type(blocks_per_stage) is not int:
            raise TypeError("n_classes and blocks_per_stage must be integers.")
        if type(initialize_weights) is not bool:
            raise TypeError("initialize_weights must be a bool.")
        if n_classes <= 0:
            raise ValueError("n_classes must be positive.")
        if blocks_per_stage <= 0:
            raise ValueError("blocks_per_stage must be positive.")
        if not isinstance(stage_channels, (tuple, list)):
            raise TypeError("stage_channels must be a tuple or list of integers.")
        if any(type(channel) is not int for channel in stage_channels):
            raise TypeError("stage_channels must contain integers.")
        if not stage_channels or any(channel <= 0 for channel in stage_channels):
            raise ValueError("stage_channels must contain positive channel counts.")

        self.in_dims = tuple(in_dims)
        self.n_classes = n_classes
        self.stage_channels = tuple(stage_channels)
        self.blocks_per_stage = blocks_per_stage
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_dims[0], self.stage_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        current_channels = self.stage_channels[0]
        for stage_index, out_channels in enumerate(self.stage_channels):
            first_stride = 1 if stage_index == 0 else 2
            blocks = [PlainBlock(current_channels, out_channels, stride=first_stride)]
            current_channels = out_channels
            blocks.extend(
                PlainBlock(current_channels, out_channels) for _ in range(1, self.blocks_per_stage)
            )
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.stage_channels[-1], self.n_classes)
        if initialize_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize convolution, BatchNorm, and linear layers."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for an NCHW image batch matching the configured size.

        Raises:
            TypeError: If x is not a tensor.
            ValueError: If x is not an NCHW batch with the configured dimensions.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}.")
        if tuple(x.shape[1:]) != self.in_dims:
            raise ValueError(f"Expected input shape (N, *{self.in_dims}), got {tuple(x.shape)}.")

        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.global_pool(out)
        return self.classifier(torch.flatten(out, 1))
