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
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2 for this CIFAR residual block.")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual transformation of ``x``."""
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class SmallResNet(nn.Module):
    """CIFAR-100 residual classifier with four stages and global average pooling.

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
        super().__init__()
        if len(in_dims) != 3:
            raise ValueError("in_dims must contain (channels, height, width).")
        if n_classes <= 0:
            raise ValueError("n_classes must be positive.")
        if blocks_per_stage <= 0:
            raise ValueError("blocks_per_stage must be positive.")
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
            nn.ReLU(inplace=True),
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
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute CIFAR-100 class logits."""
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {tuple(x.shape)}.")
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)
