from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "mobile_price_train.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]
TARGET = "price_range"
LABELS = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost",
}


def main():
    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "SVM": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", random_state=42),
            cv=3,
        ),
    }

    results = []
    reports = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append({"model": name, "accuracy": round(float(accuracy), 4)})
        reports[name] = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=list(LABELS.values()), yticklabels=list(LABELS.values()))
        plt.title(f"{name} — Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close()

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_order.joblib")

    metadata = {
        "best_model": best_name,
        "dataset_rows": int(len(df)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "labels": {str(k): v for k, v in LABELS.items()},
        "results": results,
        "classification_reports": reports,
        "test_size": 0.20,
        "random_state": 42,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(results_df.to_string(index=False))
    print(f"\nSaved best model: {best_name}")


if __name__ == "__main__":
    main()
