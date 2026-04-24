import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import joblib
import os

def load_data(filepath='Delivery_Logistics.csv'):
    if not os.path.exists(filepath):
        print(f"Dataset not found at {filepath}.")
        return None
    return pd.read_csv(filepath)

def build_pipeline():
    num_features = ['distance_km', 'package_weight_kg']
    cat_features = ['delivery_partner', 'package_type', 'vehicle_type', 'delivery_mode', 'region', 'weather_condition']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, max_depth=5, n_estimators=100)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline

def train_model():
    print("Loading actual data...")
    delivery_dataframe = load_data()
    if delivery_dataframe is None: return

    delivery_dataframe['delayed_label'] = delivery_dataframe['delayed'].apply(lambda x: 1 if str(x).lower().strip() == 'yes' else 0)

    features_dataframe = delivery_dataframe[['delivery_partner', 'package_type', 'vehicle_type', 'delivery_mode', 'region', 'weather_condition', 'distance_km', 'package_weight_kg']]
    target_variable = delivery_dataframe['delayed_label']

    X_train, X_test, y_train, y_test = train_test_split(features_dataframe, target_variable, test_size=0.2, random_state=42)

    print(f"Training XGBoost Model on {len(X_train)} samples...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/delay_predictor_v2.joblib'
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
