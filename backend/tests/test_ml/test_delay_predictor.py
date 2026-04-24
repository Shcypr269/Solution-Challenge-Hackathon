"""Tests for the ML pipeline: model loading, feature engineering, prediction."""
import pandas as pd
import joblib
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_model_file_exists():
    """Model artifact should exist after training."""
    assert os.path.exists('models/delay_predictor_v2.joblib'), \
        "Model file not found. Run: python ml/train.py"

def test_model_loads():
    """Model pipeline should load without errors."""
    pipeline = joblib.load('models/delay_predictor_v2.joblib')
    assert pipeline is not None
    assert hasattr(pipeline, 'predict')
    assert hasattr(pipeline, 'predict_proba')

def test_model_prediction_shape():
    """Prediction should return correct shape."""
    pipeline = joblib.load('models/delay_predictor_v2.joblib')
    test_input = pd.DataFrame({
        'delivery_partner': ['delhivery'],
        'package_type': ['electronics'],
        'vehicle_type': ['bike'],
        'delivery_mode': ['same day'],
        'region': ['west'],
        'weather_condition': ['stormy'],
        'distance_km': [350.5],
        'package_weight_kg': [15.2]
    })
    pred = pipeline.predict(test_input)
    prob = pipeline.predict_proba(test_input)
    
    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0][1] <= 1.0

def test_model_handles_unknown_categories():
    """Pipeline should handle unseen categories gracefully."""
    pipeline = joblib.load('models/delay_predictor_v2.joblib')
    test_input = pd.DataFrame({
        'delivery_partner': ['unknown_company'],
        'package_type': ['mystery_box'],
        'vehicle_type': ['helicopter'],
        'delivery_mode': ['teleportation'],
        'region': ['mars'],
        'weather_condition': ['tornado'],
        'distance_km': [999.9],
        'package_weight_kg': [0.1]
    })
    pred = pipeline.predict(test_input)
    assert len(pred) == 1  # Should not crash

def test_high_risk_prediction():
    """Long distance + stormy weather should predict delay."""
    pipeline = joblib.load('models/delay_predictor_v2.joblib')
    test_input = pd.DataFrame({
        'delivery_partner': ['delhivery'],
        'package_type': ['electronics'],
        'vehicle_type': ['bike'],
        'delivery_mode': ['same day'],
        'region': ['west'],
        'weather_condition': ['stormy'],
        'distance_km': [500.0],
        'package_weight_kg': [50.0]
    })
    prob = pipeline.predict_proba(test_input)
    # High risk scenario should have elevated delay probability
    assert prob[0][1] > 0.5, f"Expected high delay probability, got {prob[0][1]:.2%}"

if __name__ == '__main__':
    test_model_file_exists()
    test_model_loads()
    test_model_prediction_shape()
    test_model_handles_unknown_categories()
    test_high_risk_prediction()
    print("All ML tests passed!")
