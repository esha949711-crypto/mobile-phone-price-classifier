# Mobile Phone Price Range Classification

A complete machine-learning classification project that predicts a mobile phone's expected **price category** from its technical specifications. It trains and compares **Logistic Regression, Decision Tree, Random Forest, KNN, and SVM**, then serves the best-performing model through a Flask web app.

## Dataset

The project uses the real **Mobile Price Classification** dataset (2,000 records, 20 input features, and the `price_range` target) originally published through Kaggle and mirrored in the public repository listed below:

- Dataset source: [Mobile Price Classification repository](https://github.com/arpita-maji/Mobile-Price-Classification)
- Target labels: `0 = Low Cost`, `1 = Medium Cost`, `2 = High Cost`, `3 = Very High Cost`

## Project structure

```text
mobile-phone-price-classifier/
├── app.py                         # Flask prediction web app
├── predict.py                     # Standalone command-line prediction script
├── train_model.py                 # Train, compare, evaluate, and save models
├── requirements.txt               # Python dependencies
├── data/mobile_price_train.csv    # Training dataset
├── models/                        # Generated model artifacts
├── outputs/                       # Metrics and confusion matrices
├── templates/index.html           # Web form and prediction result UI
└── static/style.css               # Responsive styling
```

## Run locally

```bash
cd mobile-phone-price-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` in a browser. The form includes a pre-filled sample, so clicking **Classify price range** immediately demonstrates a prediction. The API is also available at `POST /predict` and the health check is at `GET /health`.

For a terminal prediction using the built-in sample:

```bash
python predict.py
```

For custom data, pass a JSON file containing all 20 feature names or use `--json` with the same payload format as the API:

```bash
python predict.py --file sample_phone.json
python predict.py --json '{"battery_power":1500,"blue":1,"clock_speed":2.0,"dual_sim":1,"fc":8,"four_g":1,"int_memory":32,"m_dep":0.5,"mobile_wt":150,"n_cores":4,"pc":12,"px_height":900,"px_width":1400,"ram":2000,"sc_h":12,"sc_w":7,"talk_time":12,"three_g":1,"touch_screen":1,"wifi":1}'
```

## Training and evaluation

`train_model.py` performs a stratified 80/20 train-test split with `random_state=42`. Scaling is applied inside pipelines for Logistic Regression, KNN, and SVM; tree-based models are trained without scaling. The script saves:

- `models/best_model.joblib`: the highest-accuracy fitted classifier.
- `models/feature_order.joblib`: the exact input order used by the model.
- `models/metadata.json`: model scores, labels, and classification reports.
- `outputs/model_comparison.csv`: accuracy ranking for all five algorithms.
- `outputs/confusion_*.png`: confusion matrix for each algorithm.

The selected best model is determined from the held-out test accuracy, not hard-coded. Re-run `python train_model.py` whenever the dataset or modeling choices change.

## API example

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "battery_power": 1500, "blue": 1, "clock_speed": 2.0, "dual_sim": 1,
    "fc": 8, "four_g": 1, "int_memory": 32, "m_dep": 0.5,
    "mobile_wt": 150, "n_cores": 4, "pc": 12, "px_height": 900,
    "px_width": 1400, "ram": 2000, "sc_h": 12, "sc_w": 7,
    "talk_time": 12, "three_g": 1, "touch_screen": 1, "wifi": 1
  }'
```

The response contains the predicted category, confidence, class probabilities, and the model name used for the prediction.

## Notes

This is a **price-range classifier**, not an exact-price estimator. The categories reflect the labels in the dataset and should not be interpreted as current market prices. For production use, retrain on current regional sales data and add model monitoring, authentication, and input logging controls.
