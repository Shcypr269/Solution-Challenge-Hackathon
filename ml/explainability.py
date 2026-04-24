import numpy as np
import pandas as pd
import joblib
import os
import json

class DelayExplainer:

    def __init__(self, model_path: str = "models/delay_predictor_v2.joblib"):
        self.pipeline = None

        candidates = [model_path, "../models/delay_predictor_v2.joblib"]
        for p in candidates:
            if os.path.exists(p):
                self.pipeline = joblib.load(p)
                break

    def _get_model(self):
        for name in ['classifier', 'model', 'regressor']:
            if name in self.pipeline.named_steps:
                return self.pipeline.named_steps[name]
        return self.pipeline.steps[-1][1]

    def _get_feature_names(self):
        preprocessor = self.pipeline.named_steps['preprocessor']
        num_cols = list(preprocessor.transformers_[0][2])
        cat_transformer = preprocessor.transformers_[1][1]
        cat_cols_original = list(preprocessor.transformers_[1][2])
        cat_cols_encoded = list(cat_transformer.get_feature_names_out(cat_cols_original))
        return num_cols, cat_cols_original, cat_cols_encoded

    def explain(self, features: dict) -> dict:

        if not self.pipeline:
            return {"error": "Model not loaded"}

        dataframe = pd.DataFrame([features])

        prob = float(self.pipeline.predict_proba(dataframe)[0][1])
        pred = "Delayed" if prob > 0.5 else "On Time"
        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

        model = self._get_model()
        importances = model.feature_importances_
        num_cols, cat_cols_original, cat_cols_encoded = self._get_feature_names()
        all_names = num_cols + cat_cols_encoded

        preprocessor = self.pipeline.named_steps['preprocessor']
        X_trans = preprocessor.transform(dataframe)

        factors = []

        for i, col in enumerate(num_cols):
            imp = float(importances[i])
            val = features.get(col, None)

            if val is not None and imp > 0.01:
                factors.append({
                    "feature": col,
                    "value": val,
                    "importance": round(imp * 100, 1),
                    "impact": f"+{imp*100:.1f}%" if prob > 0.5 else f"-{imp*100:.1f}%",
                    "direction": "increases_risk" if prob > 0.5 else "decreases_risk"
                })

        offset = len(num_cols)
        cat_importance_map = {}
        for j, enc_name in enumerate(cat_cols_encoded):
            idx = offset + j
            if idx < len(importances) and X_trans[0, idx] == 1.0:
                for orig_col in cat_cols_original:
                    if enc_name.startswith(orig_col + "_"):
                        cat_val = enc_name.replace(orig_col + "_", "")
                        if orig_col not in cat_importance_map or importances[idx] > cat_importance_map[orig_col][1]:
                            cat_importance_map[orig_col] = (cat_val, float(importances[idx]))

        for col, (val, imp) in cat_importance_map.items():
            if imp > 0.01:
                factors.append({
                    "feature": col,
                    "value": val,
                    "importance": round(imp * 100, 1),
                    "impact": f"+{imp*100:.1f}%" if prob > 0.5 else f"-{imp*100:.1f}%",
                    "direction": "increases_risk" if prob > 0.5 else "decreases_risk"
                })

        factors.sort(key=lambda x: x["importance"], reverse=True)
        top_factors = factors[:5]

        drivers = [f"{f['feature']}={f['value']} ({f['impact']})" for f in top_factors[:3]]
        explanation = (
            f"This shipment has a {prob:.0%} delay risk ({risk}). "
            f"Key drivers: {', '.join(drivers)}."
            if drivers else
            f"This shipment has a {prob:.0%} delay risk ({risk})."
        )

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "risk_level": risk,
            "top_factors": top_factors,
            "explanation": explanation,
        }

    def get_global_importance(self) -> list:

        if not self.pipeline:
            return []

        model = self._get_model()
        num_cols, _, cat_cols_encoded = self._get_feature_names()
        all_names = num_cols + cat_cols_encoded
        importances = model.feature_importances_

        ranked = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": sample_size, "importance": round(float(v)*100, 2)} for sample_size, v in ranked[:15]]

class ETAExplainer:

    def __init__(self, model_path: str = "models/eta_predictor_v2.joblib"):
        self.pipeline = None
        candidates = [model_path, "../models/eta_predictor_v2.joblib", "models/eta_predictor_v1.joblib"]
        for p in candidates:
            if os.path.exists(p):
                self.pipeline = joblib.load(p)
                break

    def explain(self, features: dict) -> dict:
        if not self.pipeline:
            return {"error": "Model not loaded"}

        dataframe = pd.DataFrame([features])
        eta = float(self.pipeline.predict(dataframe)[0])

        model = self.pipeline.steps[-1][1]
        importances = model.feature_importances_

        preprocessor = self.pipeline.named_steps['preprocessor']
        num_cols = list(preprocessor.transformers_[0][2])
        cat_transformer = preprocessor.transformers_[1][1]
        cat_cols = list(preprocessor.transformers_[1][2])
        cat_encoded = list(cat_transformer.get_feature_names_out(cat_cols))
        all_names = num_cols + cat_encoded

        factors = []
        for i, name in enumerate(all_names):
            imp = float(importances[i]) if i < len(importances) else 0
            if imp > 0.02:
                readable = name
                for c in cat_cols:
                    if name.startswith(c + "_"):
                        readable = c
                        break
                factors.append({
                    "feature": readable,
                    "importance": round(imp * 100, 1),
                    "impact_mins": round(imp * eta, 1),
                })

        factors.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "predicted_eta_mins": round(eta, 1),
            "top_factors": factors[:5],
            "explanation": f"Estimated delivery: {eta:.0f} min. "
                          f"Biggest factor: {factors[0]['feature']} ({factors[0]['importance']:.0f}% importance)."
                          if factors else f"Estimated delivery: {eta:.0f} min."
        }

if __name__ == "__main__":
    print("  Explainability Demo")

    delay_exp = DelayExplainer()

    high_risk = {
        "delivery_partner": "delhivery",
        "package_type": "electronics",
        "vehicle_type": "bike",
        "delivery_mode": "same day",
        "region": "west",
        "weather_condition": "stormy",
        "distance_km": 350.5,
        "package_weight_kg": 15.2,
    }

    print("\sample_size--- HIGH RISK Shipment ---")
    r1 = delay_exp.explain(high_risk)
    print(f"  {r1['explanation']}")
    print("  Factors:")
    for f in r1.get("top_factors", []):
        print(f"    {f['feature']:25s} = {str(f['value']):15s} importance: {f['importance']}%")

    low_risk = {
        "delivery_partner": "dhl",
        "package_type": "cosmetics",
        "vehicle_type": "ev van",
        "delivery_mode": "two day",
        "region": "south",
        "weather_condition": "clear",
        "distance_km": 25.0,
        "package_weight_kg": 2.0,
    }

    print("\sample_size--- LOW RISK Shipment ---")
    r2 = delay_exp.explain(low_risk)
    print(f"  {r2['explanation']}")

    print("\sample_size--- Global Feature Importance ---")
    for f in delay_exp.get_global_importance()[:10]:
        bar = '#' * int(f['importance'])
        print(f"  {f['feature']:30s} {f['importance']:>5.1f}% {bar}")
