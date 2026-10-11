"""Streamlit dashboard for running CIFAR-100 inference with saved models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor, nn
from torchvision.datasets import CIFAR100

from cifar100_cnn.data.loader import BASE_DATA_DIR, IMAGE_TO_TENSOR, CIFARDataset, get_cifar_dataset
from cifar100_cnn.model.loader import load_model
from cifar100_cnn.utils.activation import LayerActivation, capture_layer_activations

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


def describe_layer(module: nn.Module) -> str:
    """Explain the selected module's role without assigning semantic concepts."""
    if isinstance(module, nn.Conv2d):
        return (
            "Learned spatial filters combine small neighborhoods across input channels. "
            "The response maps show where those filters activate for this image."
        )
    if isinstance(module, nn.BatchNorm2d):
        return "Learned per-channel scales and offsets normalize the convolution responses."
    if isinstance(module, nn.ReLU):
        return (
            "Keeps positive responses and sets negative responses to zero. "
            "It has no learned weights."
        )
    if isinstance(module, nn.MaxPool2d):
        return "Keeps the strongest response in each local window and reduces spatial resolution."
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return "Summarizes each feature channel across its spatial positions."
    if isinstance(module, nn.Linear):
        return "Learned weights combine input features into hidden values or class scores."
    if isinstance(module, nn.Flatten):
        return "Rearranges the feature map into a vector; it has no learned weights."
    if type(module).__name__ == "ResidualBlock":
        return (
            "Adds the transformed main path to its shortcut. The block carries the earlier "
            "signal forward while learning a residual update."
        )
    if isinstance(module, nn.Identity):
        return "Passes the input through unchanged as an identity shortcut."
    return "This operation transforms the feature representation before the next layer."


def architecture_sections(model: nn.Module) -> list[tuple[str, str, str]]:
    """Return user-facing stages mapped to real module paths in the model."""
    sections: list[tuple[str, str, str]] = []
    conv_layers = getattr(model, "conv_layers", None)
    if isinstance(conv_layers, nn.ModuleDict):
        for index, (name, block) in enumerate(conv_layers.items(), start=1):
            includes_pooling = any(isinstance(layer, nn.MaxPool2d) for layer in block.modules())
            resolution = (
                "downsamples the feature map" if includes_pooling else "keeps image resolution"
            )
            sections.append(
                (
                    f"Convolution block {index}",
                    f"conv_layers.{name}",
                    f"Convolutions learn local patterns and {resolution}.",
                )
            )
        if isinstance(getattr(model, "global_pool", None), nn.Module):
            sections.append(
                (
                    "Global average pooling",
                    "global_pool",
                    "Averages each feature channel over its spatial positions.",
                )
            )
        classifier_name = "classifier_layer"
        if isinstance(getattr(model, classifier_name, None), nn.Module):
            sections.append(
                (
                    "Classifier",
                    classifier_name,
                    "Combines the final features into CIFAR-100 class scores.",
                )
            )
        return sections

    if isinstance(getattr(model, "stem", None), nn.Module):
        sections.append(
            ("Stem", "stem", "Extracts the first feature maps while preserving image resolution.")
        )
        stages = getattr(model, "stages", None)
        if isinstance(stages, nn.ModuleList):
            for index, _stage in enumerate(stages):
                sections.append(
                    (
                        f"Residual stage {index + 1}",
                        f"stages.{index}",
                        "Refines features with residual blocks; later stages "
                        "reduce spatial resolution.",
                    )
                )
        sections.extend(
            [
                (
                    "Global average pooling",
                    "global_pool",
                    "Summarizes each channel across the remaining spatial positions.",
                ),
                ("Classifier", "classifier", "Maps the final features to CIFAR-100 class scores."),
            ]
        )
        return sections

    for name, module in model.named_children():
        sections.append((name.replace("_", " ").title(), name, describe_layer(module)))
    return sections


def plot_activation(activation: LayerActivation) -> Figure:
    """Create a compact feature-map grid or a dense-layer activation chart."""
    values = activation.values[0].numpy()
    if values.ndim == 1:
        figure, axis = plt.subplots(figsize=(8, 3))
        axis.bar(range(len(values)), values, color="#58c4b8")
        axis.set_xlabel("Unit")
        axis.set_ylabel("Activation")
        axis.set_title("First captured output units")
        figure.tight_layout()
        return figure

    channels = min(values.shape[0], 16)
    columns = min(4, channels)
    rows = (channels + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.2), squeeze=False)
    minimum = float(values[:channels].min())
    maximum = float(values[:channels].max())
    if minimum == maximum:
        maximum = minimum + 1e-6
    for channel, axis in enumerate(axes.flat):
        if channel < channels:
            axis.imshow(values[channel], cmap="magma", vmin=minimum, vmax=maximum)
            axis.set_title(f"Channel {channel}", fontsize=8)
        axis.set_axis_off()
    figure.tight_layout()
    return figure


def plot_convolution_filter(module: nn.Conv2d, output_channel: int) -> Figure:
    """Plot learned kernel slices for one convolution output channel."""
    input_channels = module.weight.shape[1]
    shown_channels = min(input_channels, 3)
    columns = shown_channels
    figure, axes = plt.subplots(
        1,
        columns,
        figsize=(columns * 2.4, 2.6),
        squeeze=False,
    )
    selected_weights = module.weight[output_channel, :shown_channels].detach().cpu().numpy()
    limit = float(abs(selected_weights).max()) or 1.0
    for channel, axis in enumerate(axes.flat):
        axis.imshow(
            selected_weights[channel],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(f"Input channel {channel}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Output filter {output_channel} · learned kernel weights")
    figure.tight_layout()
    return figure


def plot_linear_weights(module: nn.Linear) -> Figure:
    """Plot a readable crop of a learned dense-layer weight matrix."""
    weights = module.weight.detach().cpu().numpy()
    output_count = min(weights.shape[0], 24)
    input_count = min(weights.shape[1], 32)
    visible_weights = weights[:output_count, :input_count]
    limit = float(abs(visible_weights).max()) or 1.0
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.imshow(visible_weights, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xlabel(f"Input features (first {input_count})")
    axis.set_ylabel(f"Output units (first {output_count})")
    axis.set_title("Learned linear weights")
    figure.tight_layout()
    return figure


def plot_batchnorm_parameters(module: nn.BatchNorm2d) -> Figure | None:
    """Plot the learned scale and offset of an affine BatchNorm layer."""
    if module.weight is None or module.bias is None:
        return None
    scales = module.weight.detach().cpu().numpy()
    offsets = module.bias.detach().cpu().numpy()
    shown = min(len(scales), 24)
    figure, axis = plt.subplots(figsize=(8, 3))
    positions = np.arange(shown)
    axis.bar(positions - 0.2, scales[:shown], width=0.4, label="Scale")
    axis.bar(positions + 0.2, offsets[:shown], width=0.4, label="Offset")
    axis.set_xlabel(f"Channels (first {shown})")
    axis.set_title("Learned BatchNorm parameters")
    axis.legend()
    figure.tight_layout()
    return figure


def render_layer_explorer(
    model: torch.nn.Module,
    image: Tensor,
    classmap: dict[int, str] | None = None,
) -> None:
    """Render a clickable module tree and layer inspector for one image."""
    try:
        activations, logits = capture_layer_activations(model, image)
    except Exception as exc:
        st.error(f"Unable to capture model activations: {exc}")
        return

    if not activations:
        st.warning("No displayable layer outputs were captured for this model.")
        return

    st.markdown("### Click a layer to inspect it")
    st.caption(f"Input image tensor · {tuple(image.unsqueeze(0).shape)}")
    st.caption(
        "Feature maps show activation strength for this image. They are useful signals to inspect, "
        "but do not by themselves identify human concepts or explain a prediction."
    )
    st.caption(
        f"This model produced {len(activations)} inspectable outputs. "
        "Choose a stage below, then expand the detailed tree to inspect individual operations."
    )
    st.checkbox("Auto-play through the tree", key="explorer_autoplay")
    run_every = "900ms" if st.session_state.explorer_autoplay else None

    @st.fragment(run_every=run_every)
    def layer_controls() -> None:
        layer_count = len(activations)
        modules = dict(model.named_modules())
        selected_key = "explorer_layer_index"
        if selected_key not in st.session_state:
            st.session_state[selected_key] = 0
        st.session_state[selected_key] = min(
            max(0, st.session_state[selected_key]), layer_count - 1
        )
        selected_path_key = "explorer_selected_module_path"
        if selected_path_key not in st.session_state:
            st.session_state[selected_path_key] = activations[
                st.session_state[selected_key]
            ].name
        elif st.session_state[selected_path_key] not in modules:
            st.session_state[selected_path_key] = activations[
                st.session_state[selected_key]
            ].name
        if st.session_state.get("explorer_autoplay", False):
            st.session_state[selected_key] = (st.session_state[selected_key] + 1) % layer_count
            st.session_state[selected_path_key] = activations[
                st.session_state[selected_key]
            ].name

        activation_indices = {
            activation.name: index for index, activation in enumerate(activations)
        }

        def render_tree_node(
            name: str,
            module: nn.Module,
            depth: int,
            prefix: str,
            is_last: bool,
        ) -> None:
            """Render one expandable module branch and its selectable output node."""
            children = list(module.named_children())
            layer_index = activation_indices.get(name)
            expanded_key = f"explorer-tree-open:{name or 'root'}"
            if children and expanded_key not in st.session_state:
                st.session_state[expanded_key] = depth < 2
            expanded = bool(st.session_state.get(expanded_key, False))
            connector = "└── " if is_last else "├── "
            node_label = (
                f"{prefix}{connector}{name.rsplit('.', maxsplit=1)[-1]} "
                f"· {type(module).__name__}"
            )

            if children and layer_index is not None:
                node_column, toggle_column = st.columns([9, 1])
                if node_column.button(
                    f"● {node_label} · output",
                    key=f"explorer-tree-node:{name}",
                    type=(
                        "primary"
                        if layer_index == st.session_state[selected_key]
                        else "secondary"
                    ),
                    width="stretch",
                ):
                    st.session_state[selected_key] = layer_index
                    st.session_state[selected_path_key] = name
                if toggle_column.button(
                    "▾" if expanded else "▸",
                    key=f"explorer-tree-toggle:{name}",
                    help="Collapse branch" if expanded else "Expand branch",
                ):
                    st.session_state[expanded_key] = not expanded
            elif children:
                if st.button(
                    f"{'▾' if expanded else '▸'} {node_label}",
                    key=f"explorer-tree-toggle:{name}",
                    width="stretch",
                ):
                    st.session_state[expanded_key] = not expanded
            elif layer_index is not None:
                activation = activations[layer_index]
                if st.button(
                    f"{'●' if layer_index == st.session_state[selected_key] else '○'} {node_label}",
                    key=f"explorer-tree-node:{name}",
                    type=(
                        "primary"
                        if layer_index == st.session_state[selected_key]
                        else "secondary"
                    ),
                    width="stretch",
                ):
                    st.session_state[selected_key] = layer_index
                    st.session_state[selected_path_key] = name
                st.caption(f"{activation.output_shape}")
            else:
                st.caption(node_label)

            if expanded:
                child_prefix = prefix + ("    " if is_last else "│   ") + "    "
                for index, (child_name, child_module) in enumerate(children):
                    full_name = f"{name}.{child_name}" if name else child_name
                    render_tree_node(
                        full_name,
                        child_module,
                        depth + 1,
                        child_prefix,
                        index == len(children) - 1,
                    )

        flow_column, detail_column = st.columns([0.9, 1.5], gap="large")
        with flow_column:
            st.markdown("#### Interactive architecture")
            with st.container(height=660, border=True):
                st.markdown(f"**Input image** · {tuple(image.shape)}")
                for label, module_path, explanation in architecture_sections(model):
                    matching_indices = [
                        index
                        for index, item in enumerate(activations)
                        if item.name == module_path or item.name.startswith(f"{module_path}.")
                    ]
                    if not matching_indices:
                        continue
                    layer_index = matching_indices[-1]
                    activation = activations[layer_index]
                    st.markdown(
                        "<div style='text-align:center;color:#f5b942;font-size:1.15rem'>↓</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        f"{label}  ·  {activation.output_shape}",
                        key=f"explorer-flow:{module_path}",
                        type=(
                            "primary"
                            if st.session_state[selected_path_key] == module_path
                            else "secondary"
                        ),
                        width="stretch",
                    ):
                        st.session_state[selected_key] = layer_index
                        st.session_state[selected_path_key] = module_path
                    st.caption(explanation)
                st.markdown(
                    "<div style='text-align:center;color:#f5b942;font-size:1.15rem'>↓</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("**CIFAR-100 class scores**")

        activation = activations[st.session_state[selected_key]]
        selected_module = modules.get(st.session_state[selected_path_key])
        with detail_column:
            section_descriptions = {
                path: description for _label, path, description in architecture_sections(model)
            }
            is_group = st.session_state[selected_path_key] in section_descriptions and isinstance(
                selected_module, (nn.Sequential, nn.ModuleDict, nn.ModuleList)
            )
            st.markdown(f"### {st.session_state[selected_path_key]}")
            st.caption(type(selected_module).__name__ if selected_module else activation.layer_type)
            st.write(
                section_descriptions.get(
                    st.session_state[selected_path_key],
                    describe_layer(selected_module) if selected_module else describe_layer(model),
                )
            )
            st.code(f"Output shape: {activation.output_shape}", language=None)
            st.caption("At most 16 output channels are captured for this local view.")
            if selected_module is not None:
                parameter_count = sum(
                    parameter.numel() for parameter in selected_module.parameters()
                )
                parameter_label = (
                    "Parameters in this block" if is_group else "Parameters in this layer"
                )
                st.metric(parameter_label, f"{parameter_count:,}")
            if isinstance(selected_module, nn.Conv2d):
                output_channel = st.slider(
                    "Learned output filter",
                    min_value=0,
                    max_value=min(selected_module.out_channels, 16) - 1,
                    key=f"filter-{activation.name}",
                )
                figure = plot_convolution_filter(selected_module, output_channel)
                st.pyplot(figure, width="stretch")
                plt.close(figure)
                st.caption(
                    "These are learned kernel weights. Deeper filters mix many channels, "
                    "so a single slice is only one part of each filter."
                )
            elif isinstance(selected_module, nn.Linear):
                figure = plot_linear_weights(selected_module)
                st.pyplot(figure, width="stretch")
                plt.close(figure)
                st.caption("The matrix shows only the first 24 outputs and 32 inputs.")
            elif isinstance(selected_module, nn.BatchNorm2d):
                batchnorm_figure = plot_batchnorm_parameters(selected_module)
                if batchnorm_figure is None:
                    st.info("This BatchNorm layer has no affine learned parameters.")
                else:
                    st.pyplot(batchnorm_figure, width="stretch")
                    plt.close(batchnorm_figure)
            elif selected_module is not None and type(selected_module).__name__ == "ResidualBlock":
                main_path, shortcut_path = st.columns(2)
                main_path.info("Main path\n\nConv → BN → ReLU → Conv → BN")
                shortcut = selected_module.shortcut
                shortcut_description = (
                    "Identity" if isinstance(shortcut, nn.Identity) else "1×1 Conv → BN projection"
                )
                shortcut_path.info(f"Shortcut\n\n{shortcut_description}")
                st.caption("The two paths are added, then passed through ReLU.")
            else:
                if is_group:
                    st.info(
                        "This view shows the block output. Expand the module tree below "
                        "to inspect an individual layer's weights."
                    )
                else:
                    st.caption("This operation has no learned parameters of its own.")
            st.markdown("#### Response to this image")
            figure = plot_activation(activation)
            st.pyplot(figure, width="stretch")
            plt.close(figure)

        with st.expander("Browse individual layers", expanded=False):
            with st.container(height=540, border=True):
                st.markdown(f"**{model.__class__.__name__}**")
                children = list(model.named_children())
                for index, (name, module) in enumerate(children):
                    render_tree_node(name, module, 0, "", index == len(children) - 1)

    layer_controls()
    predicted_index = int(logits.softmax(dim=1).argmax(dim=1).item())
    predicted_class = (classmap or {}).get(predicted_index, str(predicted_index))
    st.caption(f"This forward pass predicts **{predicted_class}** (class {predicted_index}).")


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
          <h1>Explore a saved model</h1>
          <p>Choose a training run and image, then inspect predictions or follow
          activations through the network.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = discover_checkpoints(ARTIFACTS_DIR)
    if not models:
        st.warning(
            "No saved checkpoints were found. Run `scripts/train_baseline.py` first; "
            "models will appear under `artifacts/<model_type>/<training_name>_<run_id>/`."
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

    actual_class = raw_test_dataset.classes[target]
    view = st.radio("Explore", ["Prediction", "Layer explorer"], horizontal=True)

    if view == "Prediction":
        try:
            predicted_index, confidence, probabilities = predict(model, normalized_image)
        except Exception as exc:
            st.error(f"Inference failed for the selected image: {exc}")
            return

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
    else:
        image_column, summary_column = st.columns([1, 1], gap="large")
        with image_column:
            st.image(tensor_to_image(raw_image), caption=f"Test image #{image_index}", width=360)
            st.caption(f"Ground truth · **{actual_class}**")
        with summary_column:
            st.markdown(f"**Model family:** `{model.__class__.__name__}`")
            st.caption(f"Run · {run_name}")
            st.markdown("#### What you can inspect")
            st.write(
                "Convolution and pooling maps, residual block outputs, and classifier activations."
            )
        render_layer_explorer(model, normalized_image, dataset.classmap)


if __name__ == "__main__":
    main()
