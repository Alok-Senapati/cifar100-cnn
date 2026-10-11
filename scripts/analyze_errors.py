"""Run reproducible error analysis for a saved CIFAR-100 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cifar100_cnn.data.loader import get_cifar_dataset
from cifar100_cnn.model import evaluate, load_model
from cifar100_cnn.utils.error_analysis import per_class_recall, top_confusions


def parse_args() -> argparse.Namespace:
    """Parse checkpoint and reporting options."""
    parser = argparse.ArgumentParser(description="Analyze CIFAR-100 checkpoint errors.")
    parser.add_argument("checkpoint", type=Path, help="Path to best_model.pt")
    parser.add_argument("--top-k", type=int, default=15, help="Number of items to report")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults beside the checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate the checkpoint and save class-level error diagnostics."""
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = get_cifar_dataset(train_batchsize=512, eval_batchsize=256, augment=False)
    model = load_model(args.checkpoint, device=device, eval_mode=True)

    class_ids = list(range(len(dataset.classes)))
    results = evaluate(model, dataset.test_loader, device=device, classes=class_ids)

    confusions = top_confusions(results["y_true"], results["y_pred"], top_k=args.top_k)
    class_recall = per_class_recall(
        results["y_true"],
        results["y_pred"],
        n_classes=len(dataset.classes),
    )
    weakest = sorted(class_recall.items(), key=lambda item: item[1])[: args.top_k]

    payload = {
        "checkpoint": str(args.checkpoint),
        "test_accuracy": float(results["report_dict"]["accuracy"]),
        "top_confusions": [
            {
                "true_id": true_id,
                "true_class": dataset.classmap[true_id],
                "predicted_id": pred_id,
                "predicted_class": dataset.classmap[pred_id],
                "count": count,
            }
            for true_id, pred_id, count in confusions
        ],
        "weakest_classes_by_recall": [
            {
                "class_id": class_id,
                "class_name": dataset.classmap[class_id],
                "recall": recall,
            }
            for class_id, recall in weakest
        ],
    }

    output_path = args.output or args.checkpoint.with_name("error_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
