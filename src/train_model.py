from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd
from preprocess import preprocess_data

def train_and_save_model(data_path, model_path='models/price_predictor.pkl'):
    # Load dataset
    df = pd.read_csv(data_path)

    # Preprocess
    X, y, preprocessor = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("R² Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    # Save model and preprocessor
    joblib.dump({'model': model, 'preprocessor': preprocessor}, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model('data/data_science_challenge_data.csv')
 
import pandas as pd
df = pd.read_csv('data/data_science_challenge_data.csv')
print(df.columns)


print(df.head())
print(df.dtypes)
