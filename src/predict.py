import joblib
import pandas as pd

def predict_price(features_dict, model_path='models/price_predictor.pkl'):
    # Load the model and preprocessor together
    saved = joblib.load(model_path)
    model = saved['model']
    preprocessor = saved['preprocessor']
    input_df = pd.DataFrame([features_dict], columns=['neighbourhood', 'building', 'size', 'bedrooms', 'bathrooms'])
    X = preprocessor.transform(input_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(float)
    predicted_price = model.predict(X)[0]
    return predicted_price