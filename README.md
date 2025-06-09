
# 🏙️ Dubai Property Price Prediction & Valuation Tool

This project predicts property prices in Dubai using machine learning and classifies listings as **underpriced**, **fairly priced**, or **overpriced**. It also includes a web interface for real-time valuation and prediction.

---

## 🔍 Overview

- Predicts prices using Random Forest Regressor
- Classifies listings based on price reasonability
- Provides validation messages using domain heuristics
- Interactive web app built with Flask
- Model trained on anonymized Dubai real estate data

---

## 🧠 Features

- **Data Preprocessing**: Handles missing values, high-cardinality features
- **Feature Engineering**: Adds price per sqft, total rooms, etc.
- **Model Evaluation**: R², MAE for performance insight
- **Price Valuation**: Determines if a listing is over/under/fairly priced
- **Validation Rules**: Warns about improbable values
- **Web Interface**: Predict and evaluate listings with a simple UI or JSON API

---

## 📁 Project Structure

```

Assignment-2-main/
├── api/
│   └── app.py                # Flask app
├── utils/
│   └── valuation.py          # Pricing logic
├── src/
│   └── predict.py            # Model loading and prediction
├── models/
│   └── price\_predictor.pkl   # Trained model
├── templates/
│   └── index.html            # Web UI
├── data/
│   └── ...                   # Raw dataset
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup & Run

### 🔧 Prerequisites
- Python 3.8+
- pip

### 🔌 Installation

```bash
git clone https://github.com/yourusername/dubai-property-price-predictor.git
cd dubai-property-price-predictor
pip install -r requirements.txt
````

### ▶️ Run Locally

```bash
python -m api.app
```

> ⚠️ **IMPORTANT:** You must run this command **from inside the `Assignment-2-main` folder**. Otherwise, Python will not find the required modules and may raise an `ImportError`.

Access at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠 Model Details

* **Algorithm**: Random Forest Regressor
* **Metrics**: R² Score, MAE
* **Valuation Heuristics**:

  * Underpriced: `actual < predicted - 10%`
  * Fairly Priced: `±10%`
  * Overpriced: `actual > predicted + 10%`

---

## 📌 Limitations & Extensions

### Known Limitations:

* Assumes static market
* Heuristic rules are fixed
* Model interpretability is limited

### Future Work:

* SHAP value explanations
* External data (interest rates, location scores)
* Cloud deployment & monitoring
* Docker integration

---

## 📄 License

This project is under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* Dubai Real Estate Dataset (anonymized)
* scikit-learn, Flask, pandas, gunicorn


