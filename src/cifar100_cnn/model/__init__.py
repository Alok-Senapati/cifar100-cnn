"""CNN architectures, checkpoint loading, training, and evaluation.

The baseline, residual, plain-staged, and reduced-pooling models are exported
alongside optimizer, scheduler, checkpoint, and training helpers.
"""

from cifar100_cnn.model.baseline import BaselineCNN
from cifar100_cnn.model.gap_classifier import ReducedPoolingGAPCNN
from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.plain_staged import PlainBlock, PlainStageCNN
from cifar100_cnn.model.reduced_pooling import BaselineReducedPoolingCNN
from cifar100_cnn.model.resnet import ResidualBlock, SmallResNet
from cifar100_cnn.model.trainer import evaluate, get_optimizer, get_scheduler, train

__all__ = [
    "BaselineCNN",
    "BaselineReducedPoolingCNN",
    "ReducedPoolingGAPCNN",
    "PlainBlock",
    "PlainStageCNN",
    "ResidualBlock",
    "SmallResNet",
    "evaluate",
    "get_optimizer",
    "get_scheduler",
    "load_model",
    "train",
]
