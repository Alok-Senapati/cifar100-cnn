"""Model architectures, checkpoint loading, training, and evaluation.

The baseline network, optimizer and scheduler factories, and training helpers
are re-exported here as the model package's public API.
"""

from cifar100_cnn.model.baseline import BaselineCNN
from cifar100_cnn.model.gap_classifier import ReducedPoolingGAPCNN
from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.reduced_pooling import BaselineReducedPoolingCNN
from cifar100_cnn.model.resnet import ResidualBlock, SmallResNet
from cifar100_cnn.model.trainer import evaluate, get_optimizer, get_scheduler, train

__all__ = [
    "BaselineCNN",
    "BaselineReducedPoolingCNN",
    "ReducedPoolingGAPCNN",
    "ResidualBlock",
    "SmallResNet",
    "evaluate",
    "get_optimizer",
    "get_scheduler",
    "load_model",
    "train",
]
