"""Configuration dataclasses used by training scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BaseLineArgs:
    """Options used to configure baseline CIFAR-100 training.

    Attributes:
        epochs: Maximum number of training epochs.
        lr: Optimizer learning rate.
        batch_size: Number of examples in each training batch.
        weight_decay: Optimizer weight-decay coefficient.
        training_name: Model training name.
        conv_channels: Output channel count for each convolutional block.
        fc_hidden: Number of hidden units in the classifier head.
        optimizer: Optimizer name accepted by the training helper.
        momentum: Momentum coefficient used by SGD.
        use_tensorboard: Whether to write training metrics to TensorBoard.
        early_stop: Enable or disable early stopping.
        patience: No of epochs for early stopping.
    """

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
    """Options used to configure reduced-pooling GAP training on CIFAR-100.

    Attributes:
        epochs: Maximum number of training epochs.
        lr: Optimizer learning rate.
        batch_size: Number of examples in each training batch.
        weight_decay: Optimizer weight-decay coefficient.
        training_name: Model training name.
        conv_channels: Output channel count for each convolutional block.
        optimizer: Optimizer name accepted by the training helper.
        momentum: Momentum coefficient used by SGD.
        use_tensorboard: Whether to write training metrics to TensorBoard.
        early_stop: Enable or disable early stopping.
        patience: No of epochs for early stopping.
        initialize_weights: Apply custom GAP model initialization instead of
            retaining PyTorch layer defaults.
        scheduler: Scheduler family to use for dynamic learning rate updates.
        min_lr: Learning rate floor for cosine and plateau scheduling; unused by step.
        lr_decay_factor: Multiplicative factor for plateau and step LR decay.
        lr_step_size: Epoch interval for StepLR decay.
        use_augmentation: Apply random crops and horizontal flips to training images only.
        use_batchnorm: Insert BatchNorm before each ReLU and disable convolution biases.
    """

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

    Defaults provide an augmented AdamW baseline with step decay. Treat these
    settings as a starting recipe and compare architecture or optimization
    changes with controlled runs.

    Attributes:
        epochs: Maximum number of training epochs.
        lr: Initial optimizer learning rate.
        batch_size: Number of examples in each training batch.
        weight_decay: Optimizer weight-decay coefficient.
        training_name: Name prefix for the output run directory.
        stage_channels: Output channel count for each residual stage.
        blocks_per_stage: Number of residual blocks in every stage.
        optimizer: Optimizer name accepted by the training helper.
        momentum: Momentum coefficient used by SGD.
        use_tensorboard: Whether to write metrics to TensorBoard.
        early_stop: Whether to stop when validation loss stalls.
        patience: Consecutive unimproved epochs allowed before early stopping.
        initialize_weights: Whether to apply the model's custom initialization.
        scheduler: Learning-rate scheduler family.
        min_lr: Learning-rate floor for cosine and plateau schedulers.
        lr_decay_factor: Multiplicative decay for step and plateau schedulers.
        lr_step_size: Epoch interval between StepLR updates.
        use_augmentation: Whether to augment training images.
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
