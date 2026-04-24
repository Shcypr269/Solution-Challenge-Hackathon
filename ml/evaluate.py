import joblib
import pandas as pd
import argparse
import os

def evaluate_model(model_path):

    model_pipeline = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    test_instance = pd.DataFrame({
        'delivery_partner': ['delhivery'],
        'package_type': ['electronics'],
        'vehicle_type': ['bike'],
        'delivery_mode': ['same day'],
        'region': ['west'],
        'weather_condition': ['stormy'],
        'distance_km': [350.5],
        'package_weight_kg': [15.2]
    })

    print("\nTest:")
    print("Input features:")
    print(test_instance.to_dict(orient='records')[0])

    pred = model_pipeline.predict(test_instance)
    prob = model_pipeline.predict_proba(test_instance)

    print(f"\nPrediction: {'Delayed' if pred[0] == 1 else 'On Time'}")
    print(f"Confidence (Delayed): {prob[0][1]:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/delay_predictor_v2.joblib', help='Path to model file')
    args = parser.parse_args()

    evaluate_model(args.model)
