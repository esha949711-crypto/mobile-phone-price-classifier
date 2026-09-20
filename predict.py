from pathlib import Path

import joblib
import pandas as pd


# Project folders
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


# Saved model aur feature order load karna
model = joblib.load(MODEL_DIR / "best_model.joblib")
features = joblib.load(MODEL_DIR / "feature_order.joblib")


# Price labels
labels = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}


def get_number(message, default):
    """
    User se number input lena.
    Agar user sirf Enter press kare to default value use hogi.
    """
    while True:
        value = input(f"{message} [{default}]: ")

        if value.strip() == "":
            return default

        try:
            return float(value)
        except ValueError:
            print("Please sirf numeric value enter karein.")


def main():
    print("\n===== MOBILE PHONE PRICE PREDICTION =====")
    print("Har field ki value enter karein.")
    print("Agar default value use karni ho to sirf Enter press karein.\n")

    phone = {}

    phone["battery_power"] = get_number(
        "Battery power in mAh", 1500
    )

    phone["blue"] = get_number(
        "Bluetooth hai? 0 = No, 1 = Yes", 1
    )

    phone["clock_speed"] = get_number(
        "Clock speed in GHz", 2.0
    )

    phone["dual_sim"] = get_number(
        "Dual SIM hai? 0 = No, 1 = Yes", 1
    )

    phone["fc"] = get_number(
        "Front camera in MP", 8
    )

    phone["four_g"] = get_number(
        "4G hai? 0 = No, 1 = Yes", 1
    )

    phone["int_memory"] = get_number(
        "Internal memory in GB", 32
    )

    phone["m_dep"] = get_number(
        "Mobile depth in cm", 0.5
    )

    phone["mobile_wt"] = get_number(
        "Mobile weight in grams", 150
    )

    phone["n_cores"] = get_number(
        "Number of processor cores", 4
    )

    phone["pc"] = get_number(
        "Primary camera in MP", 12
    )

    phone["px_height"] = get_number(
        "Pixel resolution height", 900
    )

    phone["px_width"] = get_number(
        "Pixel resolution width", 1400
    )

    phone["ram"] = get_number(
        "RAM in MB", 2000
    )

    phone["sc_h"] = get_number(
        "Screen height in cm", 12
    )

    phone["sc_w"] = get_number(
        "Screen width in cm", 7
    )

    phone["talk_time"] = get_number(
        "Talk time in hours", 12
    )

    phone["three_g"] = get_number(
        "3G hai? 0 = No, 1 = Yes", 1
    )

    phone["touch_screen"] = get_number(
        "Touch screen hai? 0 = No, 1 = Yes", 1
    )

    phone["wifi"] = get_number(
        "Wi-Fi hai? 0 = No, 1 = Yes", 1
    )

    # Model ke required order mein DataFrame banana
    row = pd.DataFrame(
        [[phone[feature] for feature in features]],
        columns=features
    )

    # Price category predict karna
    class_id = int(model.predict(row)[0])
    predicted_category = labels[class_id]

    # Probabilities calculate karna
    probabilities = model.predict_proba(row)[0]
    confidence = max(probabilities) * 100

    print("\n========================================")
    print("          PREDICTION RESULT")
    print("========================================")
    print(f"Exact Category: {predicted_category}")
    print(f"Class ID: {class_id}")
    print(f"Confidence: {confidence:.2f}%")

    # Sirf broad result: Low ya High
    if class_id in [0, 1]:
        print("Simple Result: LOW PRICE GROUP")
    else:
        print("Simple Result: HIGH PRICE GROUP")

    print("\nCategory Probabilities:")
    for index, probability in enumerate(probabilities):
        print(
            f"{labels[index]}: {probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
