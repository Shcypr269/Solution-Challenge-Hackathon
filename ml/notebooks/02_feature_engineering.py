import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def engineer_features(data_path: str, output_model_path: str):
    delivery_data = pd.read_csv(data_path)
    delivery_data['delayed_label'] = delivery_data['delayed'].apply(lambda status: 1 if str(status).lower().strip() == 'yes' else 0)

    numeric_columns = ['distance_km', 'package_weight_kg']
    categorical_columns = ['delivery_partner', 'package_type', 'vehicle_type', 
                           'delivery_mode', 'region', 'weather_condition']

    features = delivery_data[numeric_columns + categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numeric_columns),
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ]
    )

    preprocessor.fit(features)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(preprocessor, output_model_path)
    print(f"Feature preprocessor pipeline saved to {output_model_path}")

if __name__ == "__main__":
    dataset_file = '../Delivery_Logistics.csv'
    if not os.path.exists(dataset_file):
        dataset_file = '../../Delivery_Logistics.csv'
        
    engineer_features(dataset_file, '../models/preprocessor.joblib')
