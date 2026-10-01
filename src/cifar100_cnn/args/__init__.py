"""Command-line and runtime configuration types.

The package exposes dataclasses for baseline and GAP training options.
"""

from cifar100_cnn.args.parser_classes import BaseLineArgs, GAPClassifierArgs

__all__ = ["BaseLineArgs", "GAPClassifierArgs"]
