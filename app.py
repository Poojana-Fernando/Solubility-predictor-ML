"""Local UI for the solubility models in my_first_projectml.py.

Train both models the same way as the notebook, then open a small page
where you can type molecular descriptors and compare the predictions.

    .venv\\Scripts\\python app.py

Then open http://127.0.0.1:5050
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/"
    "delaney_solubility_with_descriptors.csv"
)
DATA_PATH = Path(__file__).with_name("data") / "delaney_solubility_with_descriptors.csv"
FEATURES = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
LIMITS = {
    "MolLogP": (-12, 12),
    "MolWt": (10, 2000),
    "NumRotatableBonds": (0, 40),
    "AromaticProportion": (0, 1),
}

app = Flask(__name__)


def load_frame():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    frame = pd.read_csv(DATA_PATH)
    missing = [name for name in FEATURES + ["logS"] if name not in frame.columns]
    if missing:
        raise RuntimeError(f"Dataset is missing columns: {', '.join(missing)}")
    return frame


def solubility_label(log_s):
    if log_s > 0:
        return "Very soluble"
    if log_s > -2:
        return "Soluble"
    if log_s > -4:
        return "Slightly soluble"
    return "Poorly soluble"


def metric_row(name, y_train, pred_train, y_test, pred_test):
    return {
        "method": name,
        "train_mse": f"{mean_squared_error(y_train, pred_train):.3f}",
        "train_r2": f"{r2_score(y_train, pred_train):.3f}",
        "test_mse": f"{mean_squared_error(y_test, pred_test):.3f}",
        "test_r2": f"{r2_score(y_test, pred_test):.3f}",
        "test_r2_value": float(r2_score(y_test, pred_test)),
    }


def scatter_points(y_true, y_pred):
    return [
        [round(float(actual), 3), round(float(predicted), 3)]
        for actual, predicted in zip(y_true, y_pred)
    ]


def example_from(label, row):
    return {
        "label": label,
        "logS": round(float(row["logS"]), 2),
        "values": {name: round(float(row[name]), 4) for name in FEATURES},
    }


def pick_examples(frame):
    ranked = frame.sort_values("logS")
    median_gap = (frame["logS"] - frame["logS"].median()).abs()
    typical = frame.loc[median_gap.idxmin()]
    return [
        example_from("Least soluble", ranked.iloc[0]),
        example_from("Typical", typical),
        example_from("Most soluble", ranked.iloc[-1]),
    ]


def train_models():
    frame = load_frame()
    x = frame[FEATURES]
    y = frame["logS"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=100
    )

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    forest = RandomForestRegressor(max_depth=2, random_state=100)
    forest.fit(x_train, y_train)

    linear_row = metric_row(
        "Linear regression",
        y_train,
        linear.predict(x_train),
        y_test,
        linear.predict(x_test),
    )
    forest_row = metric_row(
        "Random forest",
        y_train,
        forest.predict(x_train),
        y_test,
        forest.predict(x_test),
    )
    better = (
        linear_row["method"]
        if linear_row["test_r2_value"] >= forest_row["test_r2_value"]
        else forest_row["method"]
    )

    ranges = {}
    for name in FEATURES:
        ranges[name] = {
            "min": round(float(frame[name].min()), 3),
            "max": round(float(frame[name].max()), 3),
        }

    chart = {
        "linear": scatter_points(y_test, linear.predict(x_test)),
        "forest": scatter_points(y_test, forest.predict(x_test)),
    }
    summary = {
        "rows": int(len(frame)),
        "train": int(len(x_train)),
        "test": int(len(x_test)),
        "better": better,
        "ranges": ranges,
        "examples": pick_examples(frame),
        "metrics": [linear_row, forest_row],
    }
    return {"linear": linear, "forest": forest}, summary, chart


MODELS, SUMMARY, CHART = train_models()


def read_features(payload):
    values = []
    for name in FEATURES:
        raw = payload.get(name, "")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} needs to be a number.")
        if not np.isfinite(number):
            raise ValueError(f"{name} needs to be a finite number.")
        low, high = LIMITS[name]
        if number < low or number > high:
            raise ValueError(f"{name} should be between {low} and {high}.")
        values.append(number)
    return pd.DataFrame([values], columns=FEATURES)


@app.route("/")
def index():
    return render_template("index.html", summary=SUMMARY, chart=CHART)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        features = read_features(payload)
    except ValueError as error:
        return jsonify(error=str(error)), 400

    linear_log_s = float(MODELS["linear"].predict(features)[0])
    forest_log_s = float(MODELS["forest"].predict(features)[0])
    gap = abs(linear_log_s - forest_log_s)
    note = (
        "The two models are close on this molecule."
        if gap < 0.5
        else "The two models disagree by more than 0.5 log units. Treat this as a rough estimate."
    )
    return jsonify(
        linear={"logS": round(linear_log_s, 3), "label": solubility_label(linear_log_s)},
        forest={"logS": round(forest_log_s, 3), "label": solubility_label(forest_log_s)},
        note=note,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
