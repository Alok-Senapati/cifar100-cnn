"""Serve the local neural-network explorer UI and inference API."""

from __future__ import annotations

import base64
import io
import json
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch import Tensor, nn

from cifar100_cnn.data.loader import (
    BASE_DATA_DIR,
    IMAGE_TO_TENSOR,
    compute_mean_and_std,
    get_train_val_split_indices,
)
from cifar100_cnn.model.loader import load_model
from cifar100_cnn.utils.activation import LayerActivation, capture_layer_activations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = PROJECT_ROOT / "portal_ui"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MAX_BODY_BYTES = 16_384
MAX_CACHE_ENTRIES = 2


def discover_checkpoints(artifacts_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover saved checkpoints grouped by model type and training run."""
    discovered: dict[str, dict[str, Path]] = {}
    if not artifacts_dir.is_dir():
        return discovered

    for model_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        direct_checkpoint = model_dir / "best_model.pt"
        if direct_checkpoint.is_file():
            discovered.setdefault("legacy", {})[model_dir.name] = direct_checkpoint
            continue

        runs = {
            run_dir.name: checkpoint
            for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir())
            if (checkpoint := run_dir / "best_model.pt").is_file()
        }
        if runs:
            discovered[model_dir.name] = runs
    return discovered


def architecture_sections(model: nn.Module) -> list[dict[str, str]]:
    """Describe the top-level forward path using paths from the real model."""
    sections: list[dict[str, str]] = []
    conv_layers = getattr(model, "conv_layers", None)
    if isinstance(conv_layers, nn.ModuleDict):
        for index, (name, block) in enumerate(conv_layers.items(), start=1):
            has_pool = any(isinstance(layer, nn.MaxPool2d) for layer in block.modules())
            resolution = "reduces spatial size" if has_pool else "preserves spatial size"
            sections.append(
                {
                    "label": f"Conv block {index}",
                    "path": f"conv_layers.{name}",
                    "kind": "Feature extraction",
                    "description": f"Learns local patterns and {resolution}.",
                }
            )
        if isinstance(getattr(model, "global_pool", None), nn.Module):
            sections.append(
                {
                    "label": "Global average pooling",
                    "path": "global_pool",
                    "kind": "Spatial summary",
                    "description": "Averages each feature channel over its spatial positions.",
                }
            )
        if isinstance(getattr(model, "classifier_layer", None), nn.Module):
            sections.append(
                {
                    "label": "Classifier",
                    "path": "classifier_layer",
                    "kind": "Class scores",
                    "description": "Combines the final features into CIFAR-100 class scores.",
                }
            )
        return sections

    if isinstance(getattr(model, "stem", None), nn.Module):
        sections.append(
            {
                "label": "Stem",
                "path": "stem",
                "kind": "Feature extraction",
                "description": "Creates the first feature maps while preserving image size.",
            }
        )
        stages = getattr(model, "stages", None)
        if isinstance(stages, nn.ModuleList):
            for index, _stage in enumerate(stages):
                sections.append(
                    {
                        "label": f"Residual stage {index + 1}",
                        "path": f"stages.{index}",
                        "kind": "Residual feature extraction",
                        "description": (
                            "Refines features with residual blocks; later stages reduce "
                            "spatial size."
                        ),
                    }
                )
        sections.extend(
            [
                {
                    "label": "Global average pooling",
                    "path": "global_pool",
                    "kind": "Spatial summary",
                    "description": "Summarizes each channel across the remaining image area.",
                },
                {
                    "label": "Classifier",
                    "path": "classifier",
                    "kind": "Class scores",
                    "description": "Maps the final features to CIFAR-100 class scores.",
                },
            ]
        )
        return sections

    for name, module in model.named_children():
        sections.append(
            {
                "label": name.replace("_", " ").title(),
                "path": name,
                "kind": type(module).__name__,
                "description": "Transforms the current feature representation.",
            }
        )
    return sections


def describe_module(module: nn.Module) -> str:
    """Return a short, architecture-aware explanation for a PyTorch layer."""
    if isinstance(module, nn.Conv2d):
        return "Learned filters combine local neighborhoods across input channels."
    if isinstance(module, nn.BatchNorm2d):
        return "Learned channel scales and offsets normalize convolution responses."
    if isinstance(module, nn.ReLU):
        return "Keeps positive responses and sets negative responses to zero."
    if isinstance(module, nn.MaxPool2d):
        return "Keeps the strongest local response while reducing spatial resolution."
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return "Averages each feature channel across its spatial positions."
    if isinstance(module, nn.Linear):
        return "Learned weights combine input features into hidden values or class scores."
    if isinstance(module, nn.Flatten):
        return "Rearranges the feature map into a vector for the classifier."
    if type(module).__name__ == "ResidualBlock":
        return "Adds the transformed main path to its shortcut before the next stage."
    if isinstance(module, nn.Identity):
        return "Passes its input unchanged along the shortcut path."
    return "Transforms the current representation before passing it forward."


def _colorized_map(values: Tensor, size: int = 144) -> str:
    """Encode one activation map as a small, self-contained heat-map PNG."""
    array = values.detach().cpu().numpy().astype(np.float32, copy=False)
    lower, upper = np.percentile(array, (1, 99))
    if upper <= lower:
        upper = lower + 1e-6
    scaled = np.clip((array - lower) / (upper - lower), 0, 1)
    stops = np.asarray(
        [
            [8, 7, 25],
            [78, 20, 105],
            [181, 54, 102],
            [244, 130, 57],
            [252, 235, 154],
        ],
        dtype=np.float32,
    )
    position = scaled * (len(stops) - 1)
    low = np.floor(position).astype(np.intp)
    high = np.minimum(low + 1, len(stops) - 1)
    fraction = (position - low)[..., None]
    rgb = (stops[low] * (1 - fraction) + stops[high] * fraction).astype(np.uint8)
    image = Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _module_tree(
    module: nn.Module,
    activations: list[LayerActivation],
    name: str = "",
) -> dict[str, Any]:
    """Build a JSON-safe hierarchy of modules and captured output shapes."""
    output = next((item for item in activations if item.name == name), None)
    return {
        "name": name or type(module).__name__,
        "path": name,
        "kind": type(module).__name__,
        "shape": list(output.output_shape) if output else None,
        "description": describe_module(module),
        "children": [
            _module_tree(child, activations, f"{name}.{child_name}" if name else child_name)
            for child_name, child in module.named_children()
        ],
    }


class ExplorerService:
    """Manage local CIFAR data, checkpoints, and bounded activation caches."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR, data_dir: Path = BASE_DATA_DIR):
        self.checkpoints = discover_checkpoints(artifacts_dir)
        self._checkpoint_ids = {
            f"{model_type}/{run_name}": path
            for model_type, runs in self.checkpoints.items()
            for run_name, path in runs.items()
        }
        self.data_dir = data_dir
        self._test_dataset: datasets.CIFAR100 | None = None
        self._classmap: dict[int, str] = {}
        self._normalization: transforms.Normalize | None = None
        self._models: OrderedDict[str, nn.Module] = OrderedDict()
        self._passes: OrderedDict[
            tuple[str, int], tuple[list[LayerActivation], Tensor]
        ] = OrderedDict()
        self._lock = RLock()

    def _load_data(self) -> None:
        """Load local CIFAR files only and reproduce the training normalization."""
        if self._test_dataset is not None:
            return
        try:
            train_dataset = datasets.CIFAR100(
                root=str(self.data_dir), train=True, download=False, transform=IMAGE_TO_TENSOR
            )
            self._test_dataset = datasets.CIFAR100(
                root=str(self.data_dir), train=False, download=False, transform=IMAGE_TO_TENSOR
            )
        except RuntimeError as exc:
            raise FileNotFoundError(
                f"CIFAR-100 is not installed under {self.data_dir}. "
                "The local explorer does not download datasets."
            ) from exc
        train_indices, _ = get_train_val_split_indices(train_dataset)
        means, stds = compute_mean_and_std(train_dataset, train_indices)
        self._normalization = transforms.Normalize(means.tolist(), stds.tolist())
        self._classmap = dict(enumerate(self._test_dataset.classes))

    def bootstrap(self) -> dict[str, Any]:
        """Return local runs and class names needed to initialize the interface."""
        with self._lock:
            self._load_data()
            assert self._test_dataset is not None
            return {
                "runs": [
                    {
                        "id": checkpoint_id,
                        "model_type": model_type,
                        "run_name": run_name,
                    }
                    for model_type, runs in sorted(self.checkpoints.items())
                    for run_name in sorted(runs)
                    for checkpoint_id in [f"{model_type}/{run_name}"]
                ],
                "classes": [
                    {"id": class_id, "name": name}
                    for class_id, name in sorted(self._classmap.items())
                ],
                "image_count": len(self._test_dataset),
            }

    def samples(self, class_id: int, limit: int = 48) -> list[int]:
        """Return local test image indices for one class."""
        with self._lock:
            self._load_data()
            assert self._test_dataset is not None
            return [
                index
                for index, target in enumerate(self._test_dataset.targets)
                if int(target) == class_id
            ][:limit]

    def image(self, image_index: int) -> tuple[bytes, int]:
        """Return one local CIFAR test image as PNG bytes and its target class."""
        with self._lock:
            self._load_data()
            assert self._test_dataset is not None
            if not 0 <= image_index < len(self._test_dataset):
                raise IndexError("image index is outside the CIFAR-100 test set.")
            image_array = self._test_dataset.data[image_index]
            target = int(self._test_dataset.targets[image_index])
        stream = io.BytesIO()
        Image.fromarray(image_array).save(stream, format="PNG")
        return stream.getvalue(), target

    def _model(self, checkpoint_id: str) -> nn.Module:
        """Load a known checkpoint into a small process-local LRU cache."""
        if checkpoint_id not in self._checkpoint_ids:
            raise KeyError("unknown model run")
        if checkpoint_id in self._models:
            self._models.move_to_end(checkpoint_id)
            return self._models[checkpoint_id]
        model = load_model(self._checkpoint_ids[checkpoint_id], device="cpu", eval_mode=True)
        self._models[checkpoint_id] = model
        while len(self._models) > MAX_CACHE_ENTRIES:
            self._models.popitem(last=False)
        return model

    def explain(
        self,
        checkpoint_id: str,
        image_index: int,
        selected_path: str = "",
        output_filter: int = 0,
    ) -> dict[str, Any]:
        """Run local inference and return the selected layer's visual data."""
        with self._lock:
            self._load_data()
            assert self._test_dataset is not None and self._normalization is not None
            if not 0 <= image_index < len(self._test_dataset):
                raise IndexError("image index is outside the CIFAR-100 test set.")
            model = self._model(checkpoint_id)
            cache_key = (checkpoint_id, image_index)
            if cache_key not in self._passes:
                raw_image, _target = self._test_dataset[image_index]
                normalized_image = self._normalization(raw_image)
                activations, logits = capture_layer_activations(model, normalized_image)
                self._passes[cache_key] = (activations, logits)
                while len(self._passes) > MAX_CACHE_ENTRIES:
                    self._passes.popitem(last=False)
            else:
                self._passes.move_to_end(cache_key)
            activations, logits = self._passes[cache_key]

            activation_index_by_name = {
                activation.name: index for index, activation in enumerate(activations)
            }
            sections = architecture_sections(model)
            if not selected_path:
                selected_path = activations[0].name
            matching = [
                index
                for index, activation in enumerate(activations)
                if activation.name == selected_path
                or activation.name.startswith(f"{selected_path}.")
            ]
            if selected_path not in activation_index_by_name and not matching:
                selected_path = activations[0].name
                matching = [0]
            selected_index = matching[-1] if matching else activation_index_by_name[selected_path]
            selected_activation = activations[selected_index]
            selected_module = dict(model.named_modules()).get(selected_path)
            if selected_module is None:
                selected_path = selected_activation.name
                selected_module = dict(model.named_modules())[selected_path]

            layer_summaries = [
                {
                    "name": activation.name,
                    "kind": activation.layer_type,
                    "shape": list(activation.output_shape),
                    "description": describe_module(dict(model.named_modules())[activation.name]),
                }
                for activation in activations
            ]
            section_data = []
            for section in sections:
                section_matches = [
                    index
                    for index, activation in enumerate(activations)
                    if activation.name == section["path"]
                    or activation.name.startswith(f"{section['path']}.")
                ]
                if section_matches:
                    section_data.append(
                        {**section, "activation_index": section_matches[-1]}
                    )

            values = selected_activation.values[0]
            maps = []
            if values.ndim == 3:
                for channel, feature_map in enumerate(values[:16]):
                    maps.append(
                        {
                            "channel": channel,
                            "mean": float(feature_map.mean().item()),
                            "image": _colorized_map(feature_map),
                        }
                    )
            normalized_logits = logits.softmax(dim=1)[0]
            top_predictions = [
                {
                    "class_id": int(index),
                    "name": self._classmap.get(int(index), str(int(index))),
                    "probability": float(normalized_logits[index].item()),
                }
                for index in normalized_logits.topk(5).indices
            ]
            filter_data: dict[str, Any] | None = None
            if isinstance(selected_module, nn.Conv2d):
                output_filter = min(max(0, output_filter), selected_module.out_channels - 1)
                kernel = selected_module.weight[
                    output_filter, : min(3, selected_module.in_channels)
                ]
                filter_data = {
                    "kind": "convolution",
                    "output_filter": output_filter,
                    "filter_count": selected_module.out_channels,
                    "kernels": kernel.detach().cpu().tolist(),
                }
            elif isinstance(selected_module, nn.Linear):
                filter_data = {
                    "kind": "linear",
                    "weights": selected_module.weight[:24, :32].detach().cpu().tolist(),
                }
            elif isinstance(selected_module, nn.BatchNorm2d):
                filter_data = {
                    "kind": "batchnorm",
                    "scales": selected_module.weight.detach().cpu()[:24].tolist()
                    if selected_module.weight is not None
                    else [],
                    "offsets": selected_module.bias.detach().cpu()[:24].tolist()
                    if selected_module.bias is not None
                    else [],
                }

            target = int(self._test_dataset.targets[image_index])
            return {
                "model": checkpoint_id,
                "image_index": image_index,
                "actual_class": self._classmap[target],
                "predictions": top_predictions,
                "sections": section_data,
                "layers": layer_summaries,
                "tree": _module_tree(model, activations),
                "selected": {
                    "path": selected_path,
                    "name": selected_path.rsplit(".", maxsplit=1)[-1],
                    "kind": type(selected_module).__name__,
                    "shape": list(selected_activation.output_shape),
                    "description": describe_module(selected_module),
                    "parameter_count": sum(
                        parameter.numel() for parameter in selected_module.parameters()
                    ),
                    "activation_maps": maps,
                    "parameters": filter_data,
                },
            }


def create_handler(service: ExplorerService) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to a single local explorer service."""

    class ExplorerHandler(BaseHTTPRequestHandler):
        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[local explorer] {format_string % args}")

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, value: object) -> None:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", body)

        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            route = parsed_url.path
            try:
                if route == "/api/bootstrap":
                    self._send_json(200, service.bootstrap())
                    return
                if route == "/api/samples":
                    query = parse_qs(parsed_url.query)
                    class_id = int(query.get("class_id", ["0"])[0])
                    self._send_json(200, {"indices": service.samples(class_id)})
                    return
                if route.startswith("/api/image/"):
                    image_index = int(route.removesuffix(".png").rsplit("/", maxsplit=1)[1])
                    image_data, target = service.image(image_index)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(image_data)))
                    self.send_header("X-Image-Class", str(target))
                    self.end_headers()
                    self.wfile.write(image_data)
                    return
                static_files = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/index.html": ("index.html", "text/html; charset=utf-8"),
                    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                }
                if route in static_files:
                    filename, content_type = static_files[route]
                    self._send_bytes(200, content_type, (UI_DIR / filename).read_bytes())
                    return
                self._send_json(404, {"error": "not found"})
            except (FileNotFoundError, RuntimeError) as exc:
                self._send_json(503, {"error": str(exc)})
            except (IndexError, ValueError, KeyError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/explain":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("request body size is invalid")
                payload = json.loads(self.rfile.read(length))
                response = service.explain(
                    checkpoint_id=str(payload["checkpoint_id"]),
                    image_index=int(payload["image_index"]),
                    selected_path=str(payload.get("selected_path", "")),
                    output_filter=int(payload.get("output_filter", 0)),
                )
                self._send_json(200, response)
            except (KeyError, IndexError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(503, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

    return ExplorerHandler


def serve_local_explorer(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the custom front end on loopback; no external service is contacted."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The explorer is local-only and must bind to loopback.")
    server = ThreadingHTTPServer((host, port), create_handler(ExplorerService()))
    print(f"Local CIFAR-100 explorer: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping local explorer.")
    finally:
        server.server_close()
