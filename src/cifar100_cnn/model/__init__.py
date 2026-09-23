"""Model architectures, checkpoint loading, training, and evaluation.

The baseline network and its standard training helpers are re-exported here as
the model package's public API.
"""

from cifar100_cnn.model.baseline import BaselineCNN
from cifar100_cnn.model.loader import load_model
from cifar100_cnn.model.trainer import evaluate, get_optimizer, train

__all__ = ["BaselineCNN", "evaluate", "get_optimizer", "load_model", "train"]
