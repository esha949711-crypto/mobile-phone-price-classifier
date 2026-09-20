from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
app = Flask(__name__)

model = joblib.load(MODEL_DIR / "best_model.joblib")
features = joblib.load(MODEL_DIR / "feature_order.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
labels = {int(k): v for k, v in metadata["labels"].items()}

FIELD_INFO = {
    "battery_power": ("Battery power (mAh)", 500, 2000, 1000),
    "blue": ("Bluetooth", 0, 1, 1),
    "clock_speed": ("Clock speed (GHz)", 0.1, 3.0, 1.5),
    "dual_sim": ("Dual SIM", 0, 1, 1),
    "fc": ("Front camera (MP)", 0, 20, 5),
    "four_g": ("4G", 0, 1, 1),
    "int_memory": ("Internal memory (GB)", 2, 64, 32),
    "m_dep": ("Mobile depth (cm)", 0.1, 1.0, 0.5),
    "mobile_wt": ("Mobile weight (g)", 80, 250, 150),
    "n_cores": ("Processor cores", 1, 8, 4),
    "pc": ("Primary camera (MP)", 0, 21, 12),
    "px_height": ("Pixel resolution height", 0, 2000, 800),
    "px_width": ("Pixel resolution width", 500, 2000, 1200),
    "ram": ("RAM (MB)", 256, 4000, 2000),
    "sc_h": ("Screen height (cm)", 5, 20, 12),
    "sc_w": ("Screen width (cm)", 0, 20, 7),
    "talk_time": ("Talk time (hours)", 2, 20, 12),
    "three_g": ("3G", 0, 1, 1),
    "touch_screen": ("Touch screen", 0, 1, 1),
    "wifi": ("Wi-Fi", 0, 1, 1),
}


def parse_payload(payload):
    values = {}
    for feature in features:
        if feature not in payload or str(payload[feature]).strip() == "":
            raise ValueError(f"Missing value for {FIELD_INFO[feature][0]}")
        try:
            value = float(payload[feature])
        except (TypeError, ValueError):
            raise ValueError(f"{FIELD_INFO[feature][0]} must be numeric")
        low, high = FIELD_INFO[feature][1], FIELD_INFO[feature][2]
        if value < low or value > high:
            raise ValueError(f"{FIELD_INFO[feature][0]} must be between {low} and {high}")
        values[feature] = value
    return pd.DataFrame([values], columns=features)


@app.get("/")
def home():
    return render_template(
        "index.html", features=features, field_info=FIELD_INFO,
        best_model=metadata["best_model"], results=metadata["results"],
        labels=labels,
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        row = parse_payload(payload)
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        probability_map = {
            labels[int(i)]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(probabilities))
        }
        return jsonify({
            "success": True,
            "prediction": prediction,
            "price_range": labels[prediction],
            "confidence": round(float(max(probabilities)) * 100, 2),
            "probabilities": probability_map,
            "model": metadata["best_model"],
        })
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": metadata["best_model"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
