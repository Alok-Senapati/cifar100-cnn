"""Run reproducible error analysis for a saved CIFAR-100 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cifar100_cnn.data.loader import get_cifar_dataset
from cifar100_cnn.model import evaluate, load_model
from cifar100_cnn.utils.error_analysis import per_class_accuracy, top_confusions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CIFAR-100 checkpoint errors.")
    parser.add_argument("checkpoint", type=Path, help="Path to best_model.pt")
    parser.add_argument("--top-k", type=int, default=15, help="Number of confusion pairs to report")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults beside the checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_cifar_dataset(train_batchsize=512, eval_batchsize=256, augment=False)
    model = load_model(args.checkpoint, device=device, eval_mode=True)

    class_ids = list(range(len(dataset.classes)))
    results = evaluate(model, dataset.test_loader, device=device, classes=class_ids)

    confusions = top_confusions(results["y_true"], results["y_pred"], top_k=args.top_k)
    class_accuracy = per_class_accuracy(
        results["y_true"], results["y_pred"], n_classes=len(dataset.classes)
    )
    weakest = sorted(class_accuracy.items(), key=lambda item: item[1])[: args.top_k]

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
        "weakest_classes": [
            {
                "class_id": class_id,
                "class_name": dataset.classmap[class_id],
                "accuracy": accuracy,
            }
            for class_id, accuracy in weakest
        ],
    }

    output_path = args.output or args.checkpoint.with_name("error_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
