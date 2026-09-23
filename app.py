"""Streamlit dashboard for running CIFAR-100 inference with saved models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from torch import Tensor
from torchvision.datasets import CIFAR100

from cifar100_cnn.data.loader import BASE_DATA_DIR, IMAGE_TO_TENSOR, CIFARDataset, get_cifar_dataset
from cifar100_cnn.model.loader import load_model

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def discover_checkpoints(artifacts_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover saved checkpoints grouped by model type and run directory.

    The preferred layout is ``artifacts/<model_type>/<run_dir>/best_model.pt``.
    Direct run directories containing ``best_model.pt`` are also exposed under
    the ``legacy`` model type for compatibility with older training runs.
    """
    discovered: dict[str, dict[str, Path]] = {}
    if not artifacts_dir.is_dir():
        return discovered

    for model_type_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        direct_checkpoint = model_type_dir / "best_model.pt"
        if direct_checkpoint.is_file():
            discovered.setdefault("legacy", {})[model_type_dir.name] = direct_checkpoint
            continue

        checkpoints = {
            run_dir.name: checkpoint
            for run_dir in sorted(path for path in model_type_dir.iterdir() if path.is_dir())
            if (checkpoint := run_dir / "best_model.pt").is_file()
        }
        if checkpoints:
            discovered[model_type_dir.name] = checkpoints

    return discovered


@st.cache_resource(show_spinner=False)
def load_cifar_data() -> tuple[CIFARDataset, CIFAR100]:
    """Load normalized inference data and raw test images once per app process."""
    dataset = get_cifar_dataset(train_batchsize=1, eval_batchsize=1, num_workers=0)
    raw_test_dataset = CIFAR100(
        root=str(BASE_DATA_DIR),
        train=False,
        download=True,
        transform=IMAGE_TO_TENSOR,
    )
    return dataset, raw_test_dataset


@st.cache_resource(show_spinner=False)
def load_saved_model(checkpoint_path: str) -> torch.nn.Module:
    """Load a selected checkpoint on CPU and cache it until the path changes."""
    return load_model(checkpoint_path, device="cpu", eval_mode=True)


def tensor_to_image(image: Tensor) -> np.ndarray[Any, Any]:
    """Convert a CHW float tensor into an HWC array suitable for display."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected an RGB image tensor with shape (3, height, width).")
    return image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def predict(
    model: torch.nn.Module,
    image: Tensor,
) -> tuple[int, float, Tensor]:
    """Run one-image inference and return the class index, confidence, and probabilities."""
    with torch.inference_mode():
        probabilities = torch.softmax(model(image.unsqueeze(0)), dim=1)[0]
    class_index = int(probabilities.argmax().item())
    return class_index, float(probabilities[class_index].item()), probabilities


def main() -> None:
    """Render the model selector, test-image selector, and prediction result."""
    st.set_page_config(
        page_title="CIFAR-100 Model Lab",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #0b1220; }
        [data-testid="stSidebar"] { background: #111b2e; }
        .hero { padding: 1rem 0 1.5rem; }
        .eyebrow { color: #f5b942; font-size: .78rem; letter-spacing: .16em; font-weight: 700; }
        .hero h1 { color: #f8fafc; margin: .35rem 0 .3rem; }
        .hero p, .muted { color: #9caec8; }
        .result-card {
          background: #14233a; border: 1px solid #273a57;
          border-radius: 14px; padding: 1.1rem 1.25rem;
        }
        .result-card .label {
          color: #9caec8; font-size: .78rem; text-transform: uppercase;
          letter-spacing: .1em;
        }
        .result-card .prediction {
          color: #f8fafc; font-size: 1.8rem; font-weight: 700;
          margin-top: .3rem;
        }
        </style>
        <div class="hero">
          <div class="eyebrow">CIFAR-100 · MODEL LAB</div>
          <h1>Inspect a saved model</h1>
          <p>Choose a training run, select a test image, and examine the model's prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = discover_checkpoints(ARTIFACTS_DIR)
    if not models:
        st.warning(
            "No saved checkpoints were found. Run `scripts/train_baseline.py` first; "
            "models will appear under `artifacts/<model_type>/<run_id>/`."
        )
        return

    with st.sidebar:
        st.markdown("### Model selection")
        model_type = st.selectbox("Model type", list(models))
        run_names = list(models[model_type])
        run_name = st.selectbox("Training run", run_names)
        checkpoint_path = models[model_type][run_name]
        st.caption(f"Checkpoint: `{checkpoint_path.relative_to(PROJECT_ROOT)}`")

    try:
        dataset, raw_test_dataset = load_cifar_data()
        model = load_saved_model(str(checkpoint_path))
    except Exception as exc:
        st.error(f"Unable to load the selected model or CIFAR-100 data: {exc}")
        return

    image_index = st.selectbox(
        "Test image",
        range(len(raw_test_dataset)),
        format_func=lambda index: (
            f"{index:05d} · {raw_test_dataset.classes[raw_test_dataset.targets[index]]}"
        ),
    )
    raw_image, target = raw_test_dataset[image_index]
    normalized_image, _ = dataset.test_loader.dataset[image_index]

    try:
        predicted_index, confidence, probabilities = predict(model, normalized_image)
    except Exception as exc:
        st.error(f"Inference failed for the selected image: {exc}")
        return

    actual_class = raw_test_dataset.classes[target]
    predicted_class = dataset.classmap.get(predicted_index, str(predicted_index))

    image_column, result_column = st.columns([1.05, 1], gap="large")
    with image_column:
        st.image(tensor_to_image(raw_image), caption=f"Test image #{image_index}", width=420)
        st.caption(f"Ground truth · **{actual_class}**")

    with result_column:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="label">Predicted class</div>
              <div class="prediction">{predicted_class}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence, text=f"{predicted_class} · {confidence:.1%}")

        top_indices = probabilities.topk(k=min(5, probabilities.numel())).indices.tolist()
        st.markdown("#### Top predictions")
        for index in top_indices:
            label = dataset.classmap.get(int(index), str(index))
            st.write(f"**{label}** — {probabilities[index].item():.1%}")


if __name__ == "__main__":
    main()
