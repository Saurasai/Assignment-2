from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()

    # ====== 1. Basic cleaning ======
    df['bathrooms'] = df['bathrooms'].fillna(0).astype(int)
    df['bedrooms'] = df['bedrooms'].fillna(0).astype(int)
    df['size'] = df['size'].fillna(df['size'].median())

    # Drop or ignore size_bin (if created by binning size)
    if 'size_bin' in df.columns:
        df.drop(columns=['size_bin'], inplace=True)

    # ====== 2. Feature Engineering ======
    df['price_per_sqft'] = df['price'] / df['size']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # ====== 3. Simplify high-cardinality 'building' feature ======
    top_buildings = df['building'].value_counts().nlargest(20).index
    df['building_slim'] = df['building'].apply(lambda x: x if x in top_buildings else 'Other')

    # Optional: simplify neighbourhoods (if very many)
    top_neighbourhoods = df['neighbourhood'].value_counts().nlargest(30).index
    df['neighbourhood_slim'] = df['neighbourhood'].apply(lambda x: x if x in top_neighbourhoods else 'Other')

    # ====== 4. Define target ======
    y = df['price']
    
    # ====== 5. Select features ======
    features = [
        'size', 'bedrooms', 'bathrooms', 'price_per_sqft', 'total_rooms',
        'building_slim', 'neighbourhood_slim'
    ]
    X = df[features]

    # ====== 6. Column Transformer ======
    categorical_features = ['building_slim', 'neighbourhood_slim']
    numerical_features = ['size', 'bedrooms', 'bathrooms', 'price_per_sqft', 'total_rooms']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

    # ====== 7. Transform features ======
    X_transformed = preprocessor.fit_transform(X)

    return X_transformed, y, preprocessor
