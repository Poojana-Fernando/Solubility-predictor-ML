const examples = JSON.parse(document.getElementById("examples-data").textContent);
const chartData = JSON.parse(document.getElementById("chart-data").textContent);

const form = document.getElementById("predict-form");
const error = document.getElementById("form-error");
const emptyState = document.getElementById("empty-state");
const resultBody = document.getElementById("result-body");
const reference = document.getElementById("reference");
const chips = document.getElementById("example-chips");
const submitButton = form.querySelector(".primary");

let activeExample = null;
let activeModel = "linear";

function tone(label) {
  if (label === "Very soluble" || label === "Soluble") return "good";
  if (label === "Slightly soluble") return "mid";
  return "poor";
}

function markerPosition(logS) {
  const min = -12;
  const max = 2;
  const clamped = Math.min(max, Math.max(min, logS));
  return ((clamped - min) / (max - min)) * 100;
}

function clearReference() {
  activeExample = null;
  chips.querySelectorAll(".chip").forEach((chip) => chip.classList.remove("active"));
  reference.hidden = true;
}

examples.forEach((example, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip";
  button.textContent = example.label;
  button.addEventListener("click", () => {
    activeExample = example;
    chips.querySelectorAll(".chip").forEach((chip) => chip.classList.remove("active"));
    button.classList.add("active");
    for (const [name, value] of Object.entries(example.values)) {
      form.elements[name].value = value;
    }
    error.hidden = true;
  });
  chips.appendChild(button);
  if (index === 1) button.click();
});

form.querySelectorAll("input").forEach((input) => {
  input.addEventListener("input", clearReference);
});

function showPrediction(data) {
  emptyState.hidden = true;
  resultBody.hidden = false;

  for (const key of ["linear", "forest"]) {
    document.getElementById(`${key}-value`).textContent = data[key].logS.toFixed(3);
    const badge = document.getElementById(`${key}-label`);
    badge.textContent = data[key].label;
    badge.className = `badge ${tone(data[key].label)}`;
    document.getElementById(`marker-${key}`).style.left = `${markerPosition(data[key].logS)}%`;
  }

  document.getElementById("result-note").textContent = data.note;
  if (activeExample) {
    reference.hidden = false;
    reference.textContent = `${activeExample.label} molecule in the dataset was measured at ${activeExample.logS.toFixed(2)} logS.`;
  } else {
    reference.hidden = true;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.hidden = true;
  submitButton.disabled = true;
  const originalLabel = submitButton.textContent;
  submitButton.textContent = "Predicting…";

  const payload = {};
  for (const name of ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]) {
    payload[name] = form.elements[name].value;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");
    showPrediction(data);
  } catch (err) {
    error.hidden = false;
    error.textContent = err.message;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = originalLabel;
  }
});

function drawChart(model) {
  const canvas = document.getElementById("chart");
  const points = chartData[model];
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = rect.width;
  const height = rect.height;
  const pad = { left: 58, right: 18, top: 16, bottom: 54 };
  const values = points.flat();
  const min = Math.min(...values) - 0.4;
  const max = Math.max(...values) + 0.4;
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xOf = (value) => pad.left + ((value - min) / (max - min)) * plotW;
  const yOf = (value) => pad.top + (1 - (value - min) / (max - min)) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 1;
  ctx.font = "12px Switzer, SF Pro Display, -apple-system, sans-serif";
  ctx.fillStyle = "#5e584f";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const value = min + ((max - min) * i) / 4;
    const y = yOf(value);
    const x = xOf(value);
    ctx.strokeStyle = "#efe8dc";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(0), pad.left - 8, y + 4);
    ctx.textAlign = "center";
    ctx.fillText(value.toFixed(0), x, pad.top + plotH + 18);
    ctx.textAlign = "right";
  }

  ctx.strokeStyle = "#e2d9cb";
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.fillText("Experimental logS", pad.left + plotW / 2, height - 8);
  ctx.save();
  ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Predicted logS", 0, 0);
  ctx.restore();

  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = "#b7aa98";
  ctx.beginPath();
  ctx.moveTo(xOf(min), yOf(min));
  ctx.lineTo(xOf(max), yOf(max));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = model === "forest" ? "rgba(61, 104, 68, 0.55)" : "rgba(14, 107, 98, 0.55)";
  for (const [actual, predicted] of points) {
    ctx.beginPath();
    ctx.arc(xOf(actual), yOf(predicted), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

document.querySelectorAll(".toggle button").forEach((button) => {
  button.addEventListener("click", () => {
    activeModel = button.dataset.model;
    document.querySelectorAll(".toggle button").forEach((item) => {
      const selected = item === button;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-pressed", selected ? "true" : "false");
    });
    drawChart(activeModel);
  });
});

window.addEventListener("resize", () => drawChart(activeModel));
drawChart(activeModel);
form.requestSubmit();
