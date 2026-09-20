# Mobile Phone Price Range Classifier

A machine learning project that predicts a mobile phone's **price range** based on its technical specifications such as RAM, battery power, internal memory, camera, screen resolution, and connectivity.

## 🎯 Objective

To classify mobile phones into four price categories:

* Low Cost
* Medium Cost
* High Cost
* Very High Cost

## 🤖 Algorithms Used

The project compares five classification algorithms:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* K-Nearest Neighbors (KNN)

## 📊 Results

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression | **96.50%** |
| SVM                 |     89.50% |
| Random Forest       |     88.00% |
| Decision Tree       |     83.00% |
| KNN                 |     51.75% |

**Best Model:** Logistic Regression — **96.50% accuracy**

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Flask
* Joblib
* HTML/CSS

## 📁 Project Features

* Data preprocessing
* Multiple ML model training
* Model comparison
* Accuracy evaluation
* Confusion matrices
* Best-model selection
* Flask web application
* Command-line prediction

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 📌 Note

This project predicts a **price category**, not an exact mobile-phone market price.
