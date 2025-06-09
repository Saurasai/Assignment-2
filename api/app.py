import os
import requests

MODEL_URL = "https://www.dropbox.com/scl/fi/xr2z9v6e7ru31fs4u5m2v/price_predictor.pkl?rlkey=0r4df2i2iprbuseg22bkloliu&st=4r38e071&dl=1"
MODEL_PATH = "models/price_predictor.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Dropbox...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        r = requests.get(MODEL_URL)
        r.raise_for_status()  # optional, raise error if download fails
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")

download_model()





from flask import Flask, request, jsonify, render_template
from utils.valuation import classify_price, validate_price
from src.predict import predict_price
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    df = pd.read_csv('data/data_science_challenge_data.csv')  # Replace with your actual path
    neighbourhoods = sorted(df['neighbourhood'].dropna().unique())
    buildings = sorted(df['building'].dropna().unique())
    return render_template('index.html', neighbourhoods=neighbourhoods, buildings=buildings)

@app.route('/predict-ui', methods=['POST'])
def predict():
    # Accept both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    features = {
        'neighbourhood': data.get('neighbourhood'),
        'building': data.get('building'),
        'size': float(data.get('size')),
        'bedrooms': float(data.get('bedrooms')),
        'bathrooms': float(data.get('bathrooms')),
    }
    actual_price = float(data.get('listing_price', 0))

    predicted_price = predict_price(features)
    valuation = classify_price(predicted_price, actual_price)
    validation_msg = validate_price(features, predicted_price)

    df = pd.read_csv('data/data_science_challenge_data.csv')
    neighbourhoods = sorted(df['neighbourhood'].dropna().unique())
    buildings = sorted(df['building'].dropna().unique(), key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))


    return render_template(
        'index.html',
        predicted_price=predicted_price,
        valuation=valuation,
        validation=validation_msg,
        neighbourhoods=neighbourhoods,
        buildings=buildings
    )

if __name__ == '__main__':
    app.run(debug=True)
