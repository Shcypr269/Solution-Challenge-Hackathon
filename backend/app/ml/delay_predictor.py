import os
from typing import Dict, Any, Tuple
import joblib

class DelayPredictor:
    def __init__(self, model_path: str = None):
        self.model = None
        self._try_load(model_path)
    
    def _try_load(self, model_path: str = None):
        """Try multiple paths to find the model artifact."""
        candidates = [
            model_path,
            "models/delay_predictor_v2.joblib",
            "ml/models/delay_predictor_v2.joblib",
            "../models/delay_predictor_v2.joblib",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "delay_predictor_v2.joblib"),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                self.model = joblib.load(path)
                print(f"Model loaded from {path}")
                return
        print("Warning: No model found, using fallback heuristic predictions.")
            
    def predict_delay_risk(self, features: "pd.DataFrame") -> Tuple[bool, float]:
        if self.model:
            prob = self.model.predict_proba(features)[0][1]
            return prob > 0.5, prob
    
        # Fallback heuristic if model missing
        default_prob = 0.85 if features["distance_km"].iloc[0] > 200 else 0.1
        return default_prob > 0.5, default_prob
