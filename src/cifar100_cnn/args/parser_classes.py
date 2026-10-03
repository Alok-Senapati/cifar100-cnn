"""Configuration dataclasses used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseLineArgs:
    """Options used to configure baseline CIFAR-100 training."""

    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0
    training_name: str = "baseline-cnn"
    conv_channels: list[int] = field(default_factory=lambda: [64, 128])
    fc_hidden: int = 128
    optimizer: str = "adam"
    momentum: float = 0.9
    use_tensorboard: bool = True
    early_stop: bool = False
    patience: int = 20


@dataclass
class GAPClassifierArgs:
    """Options used to configure reduced-pooling GAP training on CIFAR-100."""

    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0
    training_name: str = "baseline-cnn"
    conv_channels: list[int] = field(default_factory=lambda: [64, 128])
    optimizer: str = "adam"
    momentum: float = 0.9
    use_tensorboard: bool = True
    early_stop: bool = False
    patience: int = 20
    initialize_weights: bool = True
    scheduler: Literal["none", "cosine", "plateau", "step"] = "none"
    min_lr: float = 1e-6
    lr_decay_factor: float = 0.5
    lr_step_size: int = 5
    use_augmentation: bool = False
    use_batchnorm: bool = False


@dataclass
class ResNetArgs:
    """Options for the from-scratch CIFAR residual-network experiment.

    The defaults intentionally match the established best training recipe so
    the architecture is the main experimental change.
    """

    epochs: int = 150
    lr: float = 1e-3
    batch_size: int = 512
    weight_decay: float = 1e-4
    training_name: str = "small-resnet"
    stage_channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    blocks_per_stage: int = 2
    optimizer: str = "adamw"
    momentum: float = 0.9
    use_tensorboard: bool = True
    early_stop: bool = False
    patience: int = 30
    initialize_weights: bool = True
    scheduler: Literal["none", "cosine", "plateau", "step"] = "step"
    min_lr: float = 1e-6
    lr_decay_factor: float = 0.5
    lr_step_size: int = 20
    use_augmentation: bool = True
