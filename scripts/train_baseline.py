"""Train and evaluate the baseline CNN on CIFAR-100."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from cifar100_cnn.args.parser_classes import BaseLineArgs
from cifar100_cnn.data.loader import get_cifar_dataset
from cifar100_cnn.model import BaselineCNN
from cifar100_cnn.model.trainer import evaluate, get_optimizer, train
from cifar100_cnn.utils.visualizer import visualize_confusion_matrix

RANDOM_SEED = 42
BASE_ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[1] / "artifacts" / "baseline_cnn"


def parse_arguments() -> BaseLineArgs:
    """Parse command-line options into the baseline training configuration."""
    parser = argparse.ArgumentParser(
        description="Train a baseline CNN for CIFAR-100 image classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight decay.")
    parser.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Output channel count for each convolutional block.",
    )
    parser.add_argument(
        "--fc-hidden",
        type=int,
        default=128,
        help="Number of hidden units in the classifier head.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default="adam",
        help="Optimizer to use.",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum used by SGD.")
    parser.add_argument(
        "--use-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable TensorBoard logging.",
    )
    return parser.parse_args(namespace=BaseLineArgs())


def main() -> None:
    """Train the model, evaluate the best checkpoint, and save run artifacts."""
    args = parse_arguments()
    torch.manual_seed(RANDOM_SEED)

    run_id = str(time.time_ns())
    artifacts_dir = BASE_ARTIFACT_DIRECTORY / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_cifar_dataset(train_batchsize=args.batch_size, eval_batchsize=256)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_init_args: dict[str, Any] = {
        "in_dims": dataset.img_size,
        "conv_channels": args.conv_channels,
        "fc_hidden": args.fc_hidden,
        "n_classes": len(dataset.classes),
    }
    model = BaselineCNN(**model_init_args).to(device)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay, args.momentum)
    criterion = nn.CrossEntropyLoss()

    writer: SummaryWriter | None = None
    try:
        if args.use_tensorboard:
            writer = SummaryWriter(log_dir=str(artifacts_dir / "tensorboard"))

        _, best_model = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epochs=args.epochs,
            train_loader=dataset.train_loader,
            val_loader=dataset.val_loader,
            device=device,
            model_init_args=model_init_args,
            output_path=artifacts_dir,
            use_tensorboard=args.use_tensorboard,
            writer=writer,
        )

        # Keep the numeric label IDs expected by scikit-learn and the plot helper.
        class_ids = list(range(len(dataset.classes)))
        test_results = evaluate(
            model=best_model,
            data_loader=dataset.test_loader,
            device=device,
            classes=class_ids,
        )
        visualize_confusion_matrix(
            test_results["y_true"],
            test_results["y_pred"],
            classes=class_ids,
            save_path=artifacts_dir / "confusion_matrix.png",
        )

        with (artifacts_dir / "training_args.json").open("w", encoding="utf-8") as output_file:
            json.dump(asdict(args), output_file, indent=2)
        with (artifacts_dir / "classification_report.json").open(
            "w", encoding="utf-8"
        ) as output_file:
            json.dump(test_results["report_dict"], output_file, indent=2)

        if writer is not None:
            writer.add_hparams(
                hparam_dict={
                    "model_type": "BaselineCNN",
                    "lr": args.lr,
                    "optimizer": args.optimizer,
                    "batch_size": args.batch_size,
                    "weight_decay": args.weight_decay,
                },
                metric_dict={
                    "eval/test_accuracy": float(test_results["report_dict"]["accuracy"]),
                },
            )
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
