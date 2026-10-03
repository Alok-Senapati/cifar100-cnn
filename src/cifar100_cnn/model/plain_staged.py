"""Plain staged CNN used as a control for residual-learning ablations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["PlainBlock", "PlainStageCNN"]


class PlainBlock(nn.Module):
    """Two-convolution block matching residual-block depth without a shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
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
        """Apply the plain two-convolution transformation."""
        return self.layers(x)


class PlainStageCNN(nn.Module):
    """Residual-control network with matching stages but no skip connections."""

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
        if not stage_channels or any(ch <= 0 for ch in stage_channels):
            raise ValueError("stage_channels must contain positive channel counts.")
        if blocks_per_stage <= 0:
            raise ValueError("blocks_per_stage must be positive.")

        self.in_dims = tuple(int(v) for v in in_dims)
        self.stage_channels = tuple(int(v) for v in stage_channels)
        self.blocks_per_stage = blocks_per_stage
        self.n_classes = n_classes

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_dims[0], self.stage_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_channels[0]),
            nn.ReLU(inplace=True),
        )

        stages: list[nn.Module] = []
        current_channels = self.stage_channels[0]
        for stage_index, out_channels in enumerate(self.stage_channels):
            blocks: list[nn.Module] = []
            first_stride = 1 if stage_index == 0 else 2
            blocks.append(PlainBlock(current_channels, out_channels, stride=first_stride))
            current_channels = out_channels
            for _ in range(1, blocks_per_stage):
                blocks.append(PlainBlock(current_channels, out_channels))
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.stage_channels[-1], n_classes)

        if initialize_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
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
        """Compute class logits with the same stage layout as SmallResNet."""
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.global_pool(out)
        return self.classifier(torch.flatten(out, 1))
