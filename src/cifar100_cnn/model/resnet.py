"""Small CIFAR-style residual network implemented from first principles."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["ResidualBlock", "SmallResNet"]


class ResidualBlock(nn.Module):
    """Two-convolution residual block with an optional projection shortcut.

    The block learns a residual function ``F(x)`` and returns ``ReLU(F(x) + shortcut(x))``.
    A 1x1 projection is used only when spatial resolution or channel count changes.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Build a residual block, adding a projection when shape changes.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            stride: Spatial stride for the first convolution; must be 1 or 2.

        Raises:
            TypeError: If channel counts or stride are not integers.
            ValueError: If channel counts are not positive or stride is unsupported.
        """
        super().__init__()
        if any(type(value) is not int for value in (in_channels, out_channels, stride)):
            raise TypeError("channel counts and stride must be integers.")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2 for this CIFAR residual block.")

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            shortcut = nn.Identity()
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual transformation of ``x``."""
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """CIFAR residual classifier with configurable stages and global average pooling.

    The network avoids the ImageNet-style 7x7 stem and initial max pool because
    CIFAR images are only 32x32. Downsampling occurs with stride-2 convolutions
    at stage transitions.
    """

    def __init__(
        self,
        in_dims: tuple[int, int, int] | Sequence[int],
        n_classes: int = 100,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        blocks_per_stage: int = 2,
        initialize_weights: bool = True,
    ) -> None:
        """Build the stem, residual stages, global pool, and class head.

        Args:
            in_dims: Input image dimensions in (channels, height, width) order.
            n_classes: Number of output classes.
            stage_channels: Output channel count for each residual stage.
            blocks_per_stage: Number of residual blocks in every stage.
            initialize_weights: Apply Kaiming convolution and Xavier linear
                initialization, with unit BatchNorm scales and zero offsets.

        Raises:
            TypeError: If dimensions, stage widths, counts, or initialization
                options have invalid types.
            ValueError: If dimensions, class count, stage widths, or block count
                are not positive, or if no stages are provided.
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
        if any(type(ch) is not int for ch in stage_channels):
            raise TypeError("stage_channels must contain integers.")
        if not stage_channels or any(ch <= 0 for ch in stage_channels):
            raise ValueError("stage_channels must contain positive channel counts.")

        in_channels = int(in_dims[0])
        self.in_dims = tuple(int(v) for v in in_dims)
        self.n_classes = n_classes
        self.stage_channels = tuple(int(v) for v in stage_channels)
        self.blocks_per_stage = blocks_per_stage

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.stage_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.ReLU(),
        )

        stages: list[nn.Module] = []
        current_channels = self.stage_channels[0]
        for stage_index, out_channels in enumerate(self.stage_channels):
            blocks: list[nn.Module] = []
            first_stride = 1 if stage_index == 0 else 2
            blocks.append(ResidualBlock(current_channels, out_channels, stride=first_stride))
            current_channels = out_channels
            for _ in range(1, blocks_per_stage):
                blocks.append(ResidualBlock(current_channels, out_channels, stride=1))
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.stage_channels[-1], n_classes)

        if initialize_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize convolution, BatchNorm, and classifier parameters."""
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
        """Compute class logits for an NCHW batch matching ``in_dims``."""
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
        out = torch.flatten(out, 1)
        return self.classifier(out)
