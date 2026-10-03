"""Train the matched plain staged CNN used for residual ablation."""

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

from cifar100_cnn.args import ResNetArgs
from cifar100_cnn.data.loader import get_cifar_dataset
from cifar100_cnn.data.transforms import get_gpu_train_transform
from cifar100_cnn.model import evaluate, get_optimizer, get_scheduler, train
from cifar100_cnn.model.plain_staged import PlainStageCNN
from cifar100_cnn.utils.visualizer import visualize_confusion_matrix

RANDOM_SEED = 42
BASE_ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[1] / "artifacts" / "plain_staged"


def parse_arguments() -> ResNetArgs:
    """Use the residual experiment configuration so both runs share a recipe."""
    parser = argparse.ArgumentParser(
        description="Train the matched plain staged CIFAR-100 CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--training-name", type=str, default="plain-staged")
    parser.add_argument("--stage-channels", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--blocks-per-stage", type=int, default=2)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--use-tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--initialize-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine", "plateau"], default="step")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-step-size", type=int, default=20)
    parser.add_argument("--use-augmentation", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(namespace=ResNetArgs())


def main() -> None:
    args = parse_arguments()
    torch.manual_seed(RANDOM_SEED)
    artifacts_dir = BASE_ARTIFACT_DIRECTORY / f"{args.training_name}_{time.time_ns()}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_cifar_dataset(args.batch_size, 256, augment=args.use_augmentation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_init_args: dict[str, Any] = {
        "in_dims": dataset.img_size,
        "n_classes": len(dataset.classes),
        "stage_channels": args.stage_channels,
        "blocks_per_stage": args.blocks_per_stage,
        "initialize_weights": args.initialize_weights,
    }
    model = PlainStageCNN(**model_init_args).to(device)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay, args.momentum)
    scheduler = get_scheduler(
        optimizer,
        args.scheduler,
        args.min_lr,
        args.lr_decay_factor,
        args.lr_step_size,
    )
    transform = None
    if args.use_augmentation and device.type == "cuda":
        transform = get_gpu_train_transform(dataset.means, dataset.stds).to(device)

    writer = SummaryWriter(log_dir=str(artifacts_dir / "tensorboard")) if args.use_tensorboard else None
    try:
        _, best_model = train(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            epochs=args.epochs,
            train_loader=dataset.train_loader,
            val_loader=dataset.val_loader,
            device=device,
            model_init_args=model_init_args,
            lr_scheduler=scheduler,
            output_path=artifacts_dir,
            early_stopping=args.early_stop,
            patience=args.patience,
            use_tensorboard=args.use_tensorboard,
            writer=writer,
            transform=transform,
        )
        class_ids = list(range(len(dataset.classes)))
        results = evaluate(best_model, dataset.test_loader, device=device, classes=class_ids)
        visualize_confusion_matrix(
            results["y_true"], results["y_pred"], classes=class_ids,
            save_path=artifacts_dir / "confusion_matrix.png",
        )
        with (artifacts_dir / "training_args.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(args), f, indent=2)
        with (artifacts_dir / "classification_report.json").open("w", encoding="utf-8") as f:
            json.dump(results["report_dict"], f, indent=2)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
