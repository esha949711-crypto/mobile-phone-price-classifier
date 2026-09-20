"""Predict a mobile phone price range from a JSON file or command-line JSON string.

Examples:
    python predict.py
    python predict.py --json '{"battery_power":1500, ...}'
    python predict.py --file sample_phone.json
"""
from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
model = joblib.load(MODEL_DIR / "best_model.joblib")
features = joblib.load(MODEL_DIR / "feature_order.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
labels = {int(k): v for k, v in metadata["labels"].items()}

DEFAULT_PHONE = {
    "battery_power": 1500, "blue": 1, "clock_speed": 2.0, "dual_sim": 1,
    "fc": 8, "four_g": 1, "int_memory": 32, "m_dep": 0.5,
    "mobile_wt": 150, "n_cores": 4, "pc": 12, "px_height": 900,
    "px_width": 1400, "ram": 2000, "sc_h": 12, "sc_w": 7,
    "talk_time": 12, "three_g": 1, "touch_screen": 1, "wifi": 1,
}


def predict(phone):
    missing = [feature for feature in features if feature not in phone]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    row = pd.DataFrame([{feature: float(phone[feature]) for feature in features}], columns=features)
    class_id = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return {
        "price_range": labels[class_id],
        "class_id": class_id,
        "confidence_percent": round(float(max(probabilities)) * 100, 2),
        "probabilities_percent": {
            labels[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(probabilities))
        },
        "model": metadata["best_model"],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict mobile phone price range")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--json", help="Phone specifications as a JSON string")
    source.add_argument("--file", type=Path, help="Path to a JSON file containing phone specifications")
    args = parser.parse_args()

    if args.file:
        phone = json.loads(args.file.read_text())
    elif args.json:
        phone = json.loads(args.json)
    else:
        phone = DEFAULT_PHONE
        print("Using the built-in sample phone. Pass --json or --file for custom input.\n")

    print(json.dumps(predict(phone), indent=2))


if __name__ == "__main__":
    main()
