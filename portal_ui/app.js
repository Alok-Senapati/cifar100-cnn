"use strict";

const state = { bootstrap: null, model: "", image: 0, data: null, timer: null, busy: false };
const $ = (selector) => document.querySelector(selector);
const html = (value) => String(value ?? "").replace(/[&<>\"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
}[char]));

async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("is-visible");
  clearTimeout(toast.timeout);
  toast.timeout = setTimeout(() => element.classList.remove("is-visible"), 4500);
}

function setBusy(busy) {
  state.busy = busy;
  $("#replay-button").disabled = busy;
  $("#flow-summary").textContent = busy ? "Running a local forward pass…" : "Select a layer to inspect its response.";
}

async function loadBootstrap() {
  const data = await request("/api/bootstrap");
  state.bootstrap = data;
  if (!data.runs.length) throw new Error("No saved checkpoints found under artifacts/.");
  const families = [...new Set(data.runs.map((run) => run.model_type))];
  $("#model-type").innerHTML = families.map((family) => `<option value="${html(family)}">${html(family.replaceAll("_", " "))}</option>`).join("");
  $("#class-select").innerHTML = data.classes.map((item) => `<option value="${item.id}">${html(item.name)}</option>`).join("");
  $("#model-type").addEventListener("change", populateRuns);
  $("#model-run").addEventListener("change", () => { state.model = $("#model-run").value; explain(); });
  $("#class-select").addEventListener("change", loadSamples);
  $("#image-select").addEventListener("change", () => { state.image = Number($("#image-select").value); showImage(); explain(); });
  $("#replay-button").addEventListener("click", () => explain(""));
  $("#filter-index").addEventListener("input", () => { $("#filter-value").value = $("#filter-index").value; explain(state.data?.selected.path); });
  $("#auto-play").addEventListener("change", togglePlayback);
  populateRuns(false);
  await loadSamples();
}

function populateRuns(runInference = true) {
  const family = $("#model-type").value;
  const runs = state.bootstrap.runs.filter((run) => run.model_type === family);
  $("#model-run").innerHTML = runs.map((run) => `<option value="${html(run.id)}">${html(run.run_name)}</option>`).join("");
  state.model = runs[0]?.id || "";
  $("#checkpoint-caption").textContent = `${runs.length} local checkpoint${runs.length === 1 ? "" : "s"}`;
  $("#run-chip-name").textContent = runs[0]?.run_name || "No run";
  if (runInference) explain("");
}

async function loadSamples() {
  try {
    const classId = Number($("#class-select").value || 0);
    const result = await request(`/api/samples?class_id=${classId}`);
    $("#image-select").innerHTML = result.indices.map((index, n) => `<option value="${index}">Example ${n + 1} · #${index}</option>`).join("");
    state.image = Number(result.indices[0] || 0);
    showImage();
    await explain("");
  } catch (error) { toast(error.message); }
}

function showImage() {
  const image = $("#image-preview");
  image.onload = () => { image.classList.add("has-image"); $("#image-placeholder").hidden = true; };
  image.onerror = () => { image.classList.remove("has-image"); $("#image-placeholder").hidden = false; };
  image.src = `/api/image/${state.image}.png`;
  $("#image-caption").textContent = `Test image #${state.image} · local CIFAR-100`;
}

async function explain(path = state.data?.selected.path || "") {
  if (!state.model || state.busy) return;
  setBusy(true);
  try {
    const data = await request("/api/explain", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint_id: state.model, image_index: state.image, selected_path: path, output_filter: Number($("#filter-index").value || 0) }),
    });
    state.data = data;
    $("#run-chip-name").textContent = state.model.split("/").slice(1).join("/");
    renderFlow(data);
    renderTree(data.tree, $("#module-tree"));
    renderInspector(data);
    renderPredictions(data.predictions);
  } catch (error) { toast(error.message); }
  finally { setBusy(false); }
}

function renderFlow(data) {
  const nodes = [
    { label: "CIFAR image", path: "", kind: "Input", shape: "3 × 32 × 32" },
    ...data.layers.map((layer) => ({
      label: layer.name,
      path: layer.name,
      kind: layer.kind,
      shape: layer.shape.slice(1).join(" × "),
      description: layer.description,
    })),
    { label: "Class scores", path: "", kind: "100 classes", shape: "CIFAR-100" },
  ];
  $("#architecture-flow").innerHTML = nodes.map((node, index) => `${index ? '<span class="flow-arrow" aria-hidden="true">›</span>' : ""}<button type="button" class="flow-node ${node.path === data.selected.path ? "is-active" : ""}" ${node.path ? `data-path="${html(node.path)}"` : "disabled"} title="${html(node.description || node.label)}"><span class="flow-node-title">${html(node.label)}</span><span class="flow-node-type">${html(node.kind)}</span><span class="flow-shape">${html(node.shape)}</span></button>`).join("");
  $("#architecture-flow").querySelectorAll("[data-path]").forEach((button) => button.addEventListener("click", () => explain(button.dataset.path)));
  $("#flow-summary").textContent = `${data.layers.length} operations in forward order · select a layer to inspect it`;
}

function renderInspector(data) {
  const item = data.selected;
  $("#selected-title").textContent = item.path || item.name;
  $("#selected-kind").textContent = item.kind;
  $("#selected-description").textContent = item.description;
  $("#selected-shape").textContent = `[${item.shape.join(", ")}]`;
  $("#selected-parameters").textContent = item.parameter_count.toLocaleString();
  $("#actual-label").textContent = data.actual_class;
  $("#feature-map-grid").innerHTML = item.activation_maps.length ? item.activation_maps.map((map) => `<div class="feature-card"><img loading="lazy" src="${map.image}" alt="Activation channel ${map.channel}" /><span>Channel ${map.channel} · μ ${map.mean.toFixed(3)}</span></div>`).join("") : '<div class="empty-state">This operation has no spatial feature maps.</div>';
  $("#map-count").textContent = item.activation_maps.length ? `Showing ${item.activation_maps.length} channels` : "Vector output";
  renderWeights(item);
  const selected = item.path;
  $("#module-tree").querySelectorAll(".tree-select").forEach((button) => button.classList.toggle("is-selected", button.dataset.path === selected));
}

function renderTree(root, container) {
  function nodeMarkup(node, depth) {
    const children = node.children || [];
    return `<div class="tree-node" style="--depth:${Math.min(depth, 12)}"><div class="tree-row"><button class="tree-toggle" type="button" aria-label="Toggle children">${children.length ? "⌄" : "·"}</button><button class="tree-select" type="button" data-path="${html(node.path)}" title="${html(node.description)}">${html(node.name)}</button><span class="tree-output-shape">${node.shape ? html(`[${node.shape.join("×")}]`) : ""}</span></div>${children.length ? `<div class="tree-children">${children.map((child) => nodeMarkup(child, depth + 1)).join("")}</div>` : ""}</div>`;
  }
  container.innerHTML = nodeMarkup(root, 0);
  container.querySelectorAll(".tree-select").forEach((button) => button.addEventListener("click", () => explain(button.dataset.path)));
  container.querySelectorAll(".tree-toggle").forEach((button) => button.addEventListener("click", () => {
    const children = button.closest(".tree-row").nextElementSibling;
    if (children) { children.classList.toggle("tree-hidden"); button.textContent = children.classList.contains("tree-hidden") ? "›" : "⌄"; }
  }));
}

function renderWeights(item) {
  const parameters = item.parameters;
  const box = $("#weight-visualization");
  const slider = $("#filter-control");
  $("#weights-title").textContent = `What ${item.name} learned`;
  $("#weight-footnote").textContent = "";
  if (!parameters) {
    slider.hidden = true;
    $("#weights-description").textContent = "This operation has no directly learned parameters.";
    box.innerHTML = '<div class="weight-empty"><div class="weight-empty-mark">✳</div><span>Select a learned layer<br />to inspect its parameters</span></div>';
    return;
  }
  if (parameters.kind === "convolution") {
    slider.hidden = false;
    const range = $("#filter-index");
    range.max = Math.max(0, parameters.filter_count - 1);
    range.value = parameters.output_filter;
    $("#filter-value").value = parameters.output_filter;
    $("#weights-description").textContent = `${parameters.filter_count} learned output filters. Showing one slice across up to three input channels.`;
    box.innerHTML = `<div class="kernel-grid">${parameters.kernels.map((matrix, index) => `<div class="kernel-cell"><canvas data-matrix='${JSON.stringify(matrix)}'></canvas><span>Input ${index + 1}</span></div>`).join("")}</div>`;
    box.querySelectorAll("canvas").forEach((canvas) => drawMatrix(canvas, JSON.parse(canvas.dataset.matrix)));
  } else if (parameters.kind === "linear") {
    slider.hidden = true;
    $("#weights-description").textContent = "A cropped view of the learned dense weight matrix (rows × input features).";
    box.innerHTML = '<canvas class="weight-canvas"></canvas>';
    drawMatrix(box.querySelector("canvas"), parameters.weights);
    $("#weight-footnote").textContent = "Showing the first 24 outputs × 32 input features.";
  } else {
    slider.hidden = true;
    $("#weights-description").textContent = "Per-channel learned scale and offset used to normalize feature responses.";
    box.innerHTML = '<canvas class="weight-canvas"></canvas>';
    drawBatchNorm(box.querySelector("canvas"), parameters.scales, parameters.offsets);
  }
}

function drawMatrix(canvas, matrix) {
  const rows = matrix.length;
  const columns = matrix[0]?.length || 1;
  canvas.width = columns;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  const image = context.createImageData(columns, rows);
  const max = Math.max(1e-9, ...matrix.flat().map((value) => Math.abs(value)));
  matrix.flat().forEach((value, index) => {
    const t = Math.min(1, Math.abs(value) / max);
    const color = value < 0 ? [88, 120, 233] : [237, 127, 79];
    image.data.set([...color.map((channel) => Math.round(25 + t * (channel - 25))), 255], index * 4);
  });
  context.putImageData(image, 0, 0);
}

function drawBatchNorm(canvas, scales, offsets) {
  const width = Math.max(scales.length, 1);
  canvas.width = width;
  canvas.height = 100;
  const context = canvas.getContext("2d");
  context.fillStyle = "#0b1321";
  context.fillRect(0, 0, width, 100);
  const max = Math.max(1, ...scales.map(Math.abs), ...offsets.map(Math.abs));
  scales.forEach((scale, index) => {
    const x = index + .5;
    context.fillStyle = "#77dfbc";
    context.fillRect(x, 50 - scale / max * 42, 1, scale / max * 42);
    context.fillStyle = "#bb91ef";
    context.fillRect(x, 50, 1, offsets[index] / max * 38);
  });
}

function renderPredictions(predictions) {
  const best = predictions[0];
  $("#confidence-chip").textContent = best ? `${(best.probability * 100).toFixed(1)}% top score` : "—";
  $("#prediction-list").innerHTML = predictions.map((item) => `<div class="prediction-row"><span class="prediction-name" title="${html(item.name)}">${html(item.name)}</span><span class="prediction-track"><span class="prediction-bar" style="display:block;width:${(item.probability * 100).toFixed(2)}%"></span></span><span class="prediction-score">${(item.probability * 100).toFixed(1)}%</span></div>`).join("");
}

function togglePlayback() {
  clearInterval(state.timer);
  if (!$("#auto-play").checked) return;
  let index = Math.max(0, state.data?.layers.findIndex((layer) => layer.name === state.data.selected.path) || 0);
  state.timer = setInterval(() => {
    const layers = state.data?.layers || [];
    if (!layers.length || state.busy) return;
    index = (index + 1) % layers.length;
    explain(layers[index].name);
  }, 1800);
}

loadBootstrap().catch((error) => { toast(error.message); $("#flow-summary").textContent = "Could not initialize the local explorer."; });
