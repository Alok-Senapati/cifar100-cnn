"""Command-line and runtime configuration types.

The package exposes dataclasses for baseline, GAP, and ResNet training options.
"""

from cifar100_cnn.args.parser_classes import BaseLineArgs, GAPClassifierArgs, ResNetArgs

__all__ = ["BaseLineArgs", "GAPClassifierArgs", "ResNetArgs"]
