import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_delay_prediction_model(data_path: str, model_output_path: str):
    delivery_data = pd.read_csv(data_path)
    
    target_labels = delivery_data['delayed'].apply(lambda status: 1 if str(status).lower().strip() == 'yes' else 0)
    features_df = delivery_data[['delivery_partner', 'package_type', 'vehicle_type', 
                                 'delivery_mode', 'region', 'weather_condition', 
                                 'distance_km', 'package_weight_kg']]

    features_train, features_test, labels_train, labels_test = train_test_split(
        features_df, target_labels, test_size=0.2, random_state=42
    )

    numeric_columns = ['distance_km', 'package_weight_kg']
    categorical_columns = ['delivery_partner', 'package_type', 'vehicle_type', 
                           'delivery_mode', 'region', 'weather_condition']

    column_preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numeric_columns),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )

    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1, 
        objective='binary:logistic',
        use_label_encoder=False, 
        eval_metric='logloss'
    )

    prediction_pipeline = Pipeline(steps=[
        ('preprocessor', column_preprocessor),
        ('classifier', xgb_classifier)
    ])

    prediction_pipeline.fit(features_train, labels_train)

    predicted_labels = prediction_pipeline.predict(features_test)
    predicted_probabilities = prediction_pipeline.predict_proba(features_test)[:, 1]

    print(classification_report(labels_test, predicted_labels, zero_division=0))
    print(f"ROC-AUC Score: {roc_auc_score(labels_test, predicted_probabilities):.4f}")

    conf_matrix = confusion_matrix(labels_test, predicted_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Validation Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(prediction_pipeline, model_output_path)

if __name__ == "__main__":
    dataset_file = '../Delivery_Logistics.csv'
    if not os.path.exists(dataset_file):
        dataset_file = '../../Delivery_Logistics.csv'
        
    train_delay_prediction_model(dataset_file, '../models/delay_predictor_v2.joblib')

