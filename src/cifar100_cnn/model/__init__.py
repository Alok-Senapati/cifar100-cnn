"""Model architectures, checkpoint loading, training, and evaluation."""

from cifar100_cnn.model.baseline import BaselineCNN
from cifar100_cnn.model.gap_classifier import ReducedPoolingGAPCNN
from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.plain_staged import PlainBlock, PlainStageCNN
from cifar100_cnn.model.resnet import ResidualBlock, SmallResNet
from cifar100_cnn.model.trainer import evaluate, get_optimizer, get_scheduler, train

__all__ = [
    "BaselineCNN",
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
