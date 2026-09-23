"""Configuration dataclasses used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseLineArgs:
    """Options used to configure baseline CIFAR-100 training.

    Attributes:
        epochs: Maximum number of training epochs.
        lr: Optimizer learning rate.
        batch_size: Number of examples in each training batch.
        weight_decay: Optimizer weight-decay coefficient.
        conv_channels: Output channel count for each convolutional block.
        fc_hidden: Number of hidden units in the classifier head.
        optimizer: Optimizer name accepted by the training helper.
        momentum: Momentum coefficient used by SGD.
        use_tensorboard: Whether to write training metrics to TensorBoard.
    """

    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0
    conv_channels: list[int] = field(default_factory=lambda: [64, 128])
    fc_hidden: int = 128
    optimizer: str = "adam"
    momentum: float = 0.9
    use_tensorboard: bool = True
