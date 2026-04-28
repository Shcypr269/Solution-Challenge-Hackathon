import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import joblib
import os
import json


@dataclass
class AnomalyResult:
    shipment_id: str
    is_anomaly: bool
    anomaly_score: float           
    risk_level: str                # LOW, MEDIUM, HIGH, CRITICAL
    anomaly_reasons: List[str]     
    feature_scores: Dict[str, float]  # Per-feature z-scores


class SupplyChainAnomalyDetector:
    NUMERIC_FEATURES = ['distance_km', 'package_weight_kg']
    EXPECTED_RANGES = {
        'distance_km': (1, 2000),
        'package_weight_kg': (0.1, 500),
    }

    HIGH_RISK_CORRIDORS = {
        'north': {'base_risk': 0.15, 'monsoon_multiplier': 2.5},
        'south': {'base_risk': 0.08, 'monsoon_multiplier': 1.5},
        'east': {'base_risk': 0.20, 'monsoon_multiplier': 3.0},
        'west': {'base_risk': 0.18, 'monsoon_multiplier': 2.8},
        'central': {'base_risk': 0.12, 'monsoon_multiplier': 2.0},
    }
    
    WEATHER_RISK_SCORES = {
        'clear': 0.0,
        'cold': 0.1,
        'rainy': 0.4,
        'foggy': 0.5,
        'stormy': 0.9,
    }
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._feature_means = {}
        self._feature_stds = {}
    
    def fit(self, data: pd.DataFrame) -> 'SupplyChainAnomalyDetector':
       
        numeric_data = data[self.NUMERIC_FEATURES].copy()
        numeric_data = numeric_data.dropna()
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(numeric_data)
        
        for col in self.NUMERIC_FEATURES:
            self._feature_means[col] = float(numeric_data[col].mean())
            self._feature_stds[col] = float(numeric_data[col].std())
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
        )
        self.isolation_forest.fit(X_scaled)
        self._is_fitted = True
        
        return self
    
    def fit_on_csv(self, csv_path: str = 'Delivery_Logistics.csv') -> 'SupplyChainAnomalyDetector':
        
        candidates = [csv_path, f'../{csv_path}', f'data/raw/{csv_path}']
        for p in candidates:
            if os.path.exists(p):
                delivery_dataframe = pd.read_csv(p)
                return self.fit(delivery_dataframe)
        
        # Fallbac
        print("WARNING: No CSV found, fitting on synthetic baseline")
        synthetic = pd.DataFrame({
            'distance_km': np.random.lognormal(4.5, 1.0, 1000),
            'package_weight_kg': np.random.lognormal(2.0, 0.8, 1000),
        })
        return self.fit(synthetic)
    
    def detect(self, shipment: Dict[str, Any]) -> AnomalyResult:
        if not self._is_fitted:
            self.fit_on_csv()
        
        shipment_id = shipment.get('shipment_id', 'unknown')
        reasons = []
        feature_scores = {}
        
        numeric_df = pd.DataFrame([{
            'distance_km': shipment.get('distance_km', 100),
            'package_weight_kg': shipment.get('package_weight_kg', 5),
        }])
        
        X_scaled = self.scaler.transform(numeric_df)
        if_score = float(self.isolation_forest.score_samples(X_scaled)[0])
        if_prediction = int(self.isolation_forest.predict(X_scaled)[0])
        
        # Z-Scoree
        z_scores = {}
        for col in self.NUMERIC_FEATURES:
            val = shipment.get(col, 0)
            mean = self._feature_means.get(col, 100)
            std = self._feature_stds.get(col, 50)
            if std > 0:
                z = abs((val - mean) / std)
            else:
                z = 0
            z_scores[col] = round(z, 2)
            feature_scores[col] = round(z, 2)
            
            if z > 3:
                reasons.append(f"{col}={val} is extremely unusual (>3 std from mean)")
            elif z > 2:
                reasons.append(f"{col}={val} deviates significantly (>2 std from mean)")
        
        # Range 
        for col, (lo, hi) in self.EXPECTED_RANGES.items():
            val = shipment.get(col, 0)
            if val < lo:
                reasons.append(f"{col}={val} is below minimum expected ({lo})")
                feature_scores[f'{col}_range'] = -1.0
            elif val > hi:
                reasons.append(f"{col}={val} exceeds maximum expected ({hi})")
                feature_scores[f'{col}_range'] = 1.0
        
        # Weather + Region Risk
        weather = shipment.get('weather_condition', 'clear').lower()
        region = shipment.get('region', 'central').lower()
        
        weather_risk = self.WEATHER_RISK_SCORES.get(weather, 0.2)
        region_info = self.HIGH_RISK_CORRIDORS.get(region, {'base_risk': 0.1, 'monsoon_multiplier': 1.5})
        
        contextual_risk = region_info['base_risk']
        if weather in ('rainy', 'stormy'):
            contextual_risk *= region_info['monsoon_multiplier']
        
        feature_scores['weather_risk'] = round(weather_risk, 2)
        feature_scores['region_risk'] = round(contextual_risk, 2)
        
        if weather_risk > 0.7:
            reasons.append(f"Severe weather condition: {weather}")
        if contextual_risk > 0.3:
            reasons.append(f"High-risk corridor: {region} region during adverse weather")
        
        # Combine scores
        if_anomaly_prob = max(0, min(1, 0.5 - if_score))
        
        # Z-score component (average z-score, normalized)
        avg_z = np.mean(list(z_scores.values())) if z_scores else 0
        z_anomaly_prob = min(1.0, avg_z / 4.0)  # 4σ = 100% anomaly
        
        # Combined anomaly score (weighted ensemble)
        combined_score = (
            0.40 * if_anomaly_prob +
            0.30 * z_anomaly_prob +
            0.15 * weather_risk +
            0.15 * min(1.0, contextual_risk)
        )
        
        # Risk level
        if combined_score > 0.7:
            risk_level = 'CRITICAL'
        elif combined_score > 0.5:
            risk_level = 'HIGH'
        elif combined_score > 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        is_anomaly = combined_score > 0.4 or if_prediction == -1
        
        if not reasons:
            reasons.append("All metrics within normal ranges")
        
        return AnomalyResult(
            shipment_id=shipment_id,
            is_anomaly=is_anomaly,
            anomaly_score=round(combined_score, 4),
            risk_level=risk_level,
            anomaly_reasons=reasons,
            feature_scores=feature_scores,
        )
    
    def detect_batch(self, shipments: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = [self.detect(s) for s in shipments]
        
        anomaly_count = sum(1 for r in results if r.is_anomaly)
        critical_count = sum(1 for r in results if r.risk_level == 'CRITICAL')
        high_count = sum(1 for r in results if r.risk_level == 'HIGH')
        
        return {
            'summary': {
                'total_shipments': len(results),
                'anomalies_detected': anomaly_count,
                'anomaly_rate': round(anomaly_count / max(len(results), 1), 3),
                'critical_alerts': critical_count,
                'high_alerts': high_count,
                'avg_anomaly_score': round(np.mean([r.anomaly_score for r in results]), 4),
            },
            'results': [
                {
                    'shipment_id': r.shipment_id,
                    'is_anomaly': r.is_anomaly,
                    'anomaly_score': r.anomaly_score,
                    'risk_level': r.risk_level,
                    'reasons': r.anomaly_reasons,
                    'feature_scores': r.feature_scores,
                }
                for r in results
            ],
            'risk_distribution': {
                'LOW': sum(1 for r in results if r.risk_level == 'LOW'),
                'MEDIUM': sum(1 for r in results if r.risk_level == 'MEDIUM'),
                'HIGH': sum(1 for r in results if r.risk_level == 'HIGH'),
                'CRITICAL': sum(1 for r in results if r.risk_level == 'CRITICAL'),
            }
        }
    
    def save(self, path: str = 'models/anomaly_detector.joblib'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'feature_means': self._feature_means,
            'feature_stds': self._feature_stds,
            'contamination': self.contamination,
        }, path)
        print(f"Anomaly detector saved to {path}")
    
    def load(self, path: str = 'models/anomaly_detector.joblib') -> 'SupplyChainAnomalyDetector':
        candidates = [path, f'../{path}']
        for p in candidates:
            if os.path.exists(p):
                data = joblib.load(p)
                self.isolation_forest = data['isolation_forest']
                self.scaler = data['scaler']
                self._feature_means = data['feature_means']
                self._feature_stds = data['feature_stds']
                self.contamination = data['contamination']
                self._is_fitted = True
                return self
        raise FileNotFoundError(f"No model found at {path}")


def generate_test_fleet(n: int = 25) -> List[Dict]:
    np.random.seed(42)
    partners = ["delhivery", "shadowfax", "xpressbees", "dhl", "bluedart"]
    packages = ["electronics", "groceries", "automobile parts", "cosmetics", "medicines"]
    vehicles = ["truck", "ev van", "bike", "three wheeler"]
    modes = ["same day", "express", "standard", "two day"]
    regions = ["north", "south", "east", "west", "central"]
    weathers = ["clear", "rainy", "cold", "foggy", "stormy"]
    
    fleet = []
    for i in range(n):
        if i < int(n * 0.8):
            dist = np.random.uniform(20, 600)
            weight = np.random.uniform(1, 50)
            weather = np.random.choice(["clear", "clear", "rainy", "cold", "foggy"])
        else:
            # Inject anomalies
            anomaly_type = np.random.choice(['extreme_distance', 'extreme_weight', 'bad_weather_combo'])
            if anomaly_type == 'extreme_distance':
                dist = np.random.uniform(1500, 3000)
                weight = np.random.uniform(1, 50)
                weather = np.random.choice(weathers)
            elif anomaly_type == 'extreme_weight':
                dist = np.random.uniform(20, 600)
                weight = np.random.uniform(200, 600)
                weather = np.random.choice(weathers)
            else:
                dist = np.random.uniform(100, 800)
                weight = np.random.uniform(5, 30)
                weather = "stormy"
        
        fleet.append({
            'shipment_id': f'SHP-{i+1:03d}',
            'delivery_partner': np.random.choice(partners),
            'package_type': np.random.choice(packages),
            'vehicle_type': np.random.choice(vehicles),
            'delivery_mode': np.random.choice(modes),
            'region': np.random.choice(regions),
            'weather_condition': weather,
            'distance_km': round(dist, 1),
            'package_weight_kg': round(weight, 1),
        })
    
    return fleet

if __name__ == '__main__':
    print("  Supply Chain Anomaly Detection System")
    
    detector = SupplyChainAnomalyDetector(contamination=0.15)
    detector.fit_on_csv()
    
    fleet = generate_test_fleet(25)
    
    batch_result = detector.detect_batch(fleet)

    summary = batch_result['summary']
    print(f"\nAnalysis Summary:")
    print(f"   Total shipments:     {summary['total_shipments']}")
    print(f"   Anomalies detected:  {summary['anomalies_detected']}")
    print(f"   Anomaly rate:        {summary['anomaly_rate']:.1%}")
    print(f"   Critical alerts:     {summary['critical_alerts']}")
    print(f"   High alerts:         {summary['high_alerts']}")
    
    print(f"\n Distribution:")
    for level, count in batch_result['risk_distribution'].items():
        bar = '#' * count
        print(f"   {level:10s} {count:>3d} {bar}")
    
    print(f"\n Anomalous Shipments:")
    for r in batch_result['results']:
        if r['is_anomaly']:
            print(f"\n   {r['shipment_id']} -- Score: {r['anomaly_score']:.2f} [{r['risk_level']}]")
            for reason in r['reasons']:
                print(f"     ! {reason}")
    
    # Save model
    detector.save()
