import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from math import pi
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def convert_mercator_to_wgs84(x_coordinates: pd.Series, y_coordinates: pd.Series):
    longitude = (x_coordinates / 20037508.34) * 180.0
    latitude_radians = (y_coordinates / 20037508.34) * 180.0
    latitude = 180.0 / pi * (2 * np.arctan(np.exp(latitude_radians * pi / 180.0)) - pi / 2)
    return longitude, latitude

def calculate_haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    print("Computing haversine distances...")
    delivery_dataframe['delivery_distance_km'] = haversine_vec(
        delivery_dataframe['receipt_wgs_lng'], delivery_dataframe['receipt_wgs_lat'],
        delivery_dataframe['poi_wgs_lng'], delivery_dataframe['poi_wgs_lat']
    )

    delivery_dataframe = delivery_dataframe[(delivery_dataframe['delivery_distance_km'] > 0.01) & (delivery_dataframe['delivery_distance_km'] < 50)]
    print(f"Valid distances (0.01-50 km): {len(delivery_dataframe):,}")

    delivery_dataframe['log_distance'] = np.log1p(delivery_dataframe['delivery_distance_km'])

    delivery_dataframe['accept_hour'] = delivery_dataframe['receipt_time'].dt.hour
    delivery_dataframe['is_rush_hour'] = delivery_dataframe['accept_hour'].apply(lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 19) else 0)

    def time_period(h):
        if 6 <= h < 12: return 'morning'
        elif 12 <= h < 17: return 'afternoon'
        elif 17 <= h < 21: return 'evening'
        else: return 'night'
    delivery_dataframe['time_period'] = delivery_dataframe['accept_hour'].apply(time_period)

    delivery_dataframe['day_of_week'] = delivery_dataframe['ds'] % 7
    delivery_dataframe['is_weekend'] = (delivery_dataframe['day_of_week'] >= 5).astype(int)

    delivery_dataframe['city'] = delivery_dataframe['from_city_name'].fillna('unknown')

    delivery_dataframe['aoi_type'] = delivery_dataframe['typecode'].fillna('unknown').astype(str)
    aoi_counts = delivery_dataframe['aoi_type'].value_counts()
    rare_aois = aoi_counts[aoi_counts < 1000].index
    delivery_dataframe.loc[delivery_dataframe['aoi_type'].isin(rare_aois), 'aoi_type'] = 'other'

    courier_daily = delivery_dataframe.groupby(['delivery_user_id', 'ds']).size().reset_index(name='courier_daily_packages')
    delivery_dataframe = delivery_dataframe.merge(courier_daily, on=['delivery_user_id', 'ds'], how='left')

    courier_speed = delivery_dataframe.groupby('delivery_user_id').apply(
        lambda g: g['delivery_distance_km'].sum() / g['delivery_duration_mins'].sum()
        if g['delivery_duration_mins'].sum() > 0 else 0.02
    ).reset_index(name='courier_avg_speed')
    delivery_dataframe = delivery_dataframe.merge(courier_speed, on='delivery_user_id', how='left')
    delivery_dataframe['courier_avg_speed'] = delivery_dataframe['courier_avg_speed'].clip(0.001, 1.0)

    delivery_dataframe['distance_x_rush'] = delivery_dataframe['delivery_distance_km'] * delivery_dataframe['is_rush_hour']

    print(f"\nFinal dataset: {len(delivery_dataframe):,} rows")
    print(f"Cities: {len(delivery_dataframe['city'].unique())} unique")
    print(f"AOI types: {len(delivery_dataframe['aoi_type'].unique())} unique")
    print(f"Duration: mean={delivery_dataframe['delivery_duration_mins'].mean():.1f}, median={delivery_dataframe['delivery_duration_mins'].median():.1f} min")
    print(f"Distance: mean={delivery_dataframe['delivery_distance_km'].mean():.2f}, median={delivery_dataframe['delivery_distance_km'].median():.2f} km")

    return delivery_dataframe

def train_eta_model(csv_path: str = "data/raw/lade/delivery_five_cities.csv"):
    delivery_dataframe = load_and_engineer_features(csv_path)

    num_features = ['delivery_distance_km', 'log_distance', 'accept_hour', 'day_of_week',
                    'is_rush_hour', 'is_weekend', 'courier_daily_packages',
                    'courier_avg_speed', 'distance_x_rush']
    cat_features = ['city', 'time_period', 'aoi_type']
    target = 'delivery_duration_mins'

    all_features = num_features + cat_features
    features_dataframe = delivery_dataframe[all_features]
    target_variable = delivery_dataframe[target]

    X_train, X_test, y_train, y_test = train_test_split(features_dataframe, target_variable, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        ))
    ])

    print("Training XGBoost ETA model (v2)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    acc_30 = np.mean(np.abs(y_test - y_pred) < 30)
    acc_15 = np.mean(np.abs(y_test - y_pred) < 15)

    print(f"\n{'='*55}")
    print(f"  ETA Model v2 Results ({len(delivery_dataframe):,} real deliveries)")
    print(f"{'='*55}")
    print(f"  MAE:    {mae:.2f} minutes")
    print(f"  RMSE:   {rmse:.2f} minutes")
    print(f"  R2:     {r2:.4f}")
    print(f"  ACC@15: {acc_15:.2%}")
    print(f"  ACC@30: {acc_30:.2%}")
    print(f"{'='*55}")

    model = pipeline.named_steps['model']
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = list(ohe.get_feature_names_out(cat_features))
    all_names = num_features + cat_names

    importances = model.feature_importances_
    top_feats = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 feature importances:")
    for name, imp in top_feats:
        safe_name = name.encode('ascii', 'replace').decode('ascii')
        bar = '#' * int(imp * 100)
        print(f"  {safe_name:30s} {imp:.4f} {bar}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/eta_predictor_v2.joblib')
    print(f"\nModel saved to models/eta_predictor_v2.joblib")

    return pipeline

class ETAPredictor:
    def __init__(self, model_path='models/eta_predictor_v2.joblib'):
        self.model = None
        candidates = [model_path, 'models/eta_predictor_v1.joblib']
        for p in candidates:
            if os.path.exists(p):
                self.model = joblib.load(p)
                break

    def predict_eta(self, distance_km: float, hour: int, city: str = 'unknown',
                    day_of_week: int = 3, courier_daily_packages: int = 30,
                    courier_avg_speed: float = 0.05, aoi_type: str = 'other') -> float:
        is_rush = 1 if (8 <= hour <= 10) or (17 <= hour <= 19) else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        if 6 <= hour < 12: tp = 'morning'
        elif 12 <= hour < 17: tp = 'afternoon'
        elif 17 <= hour < 21: tp = 'evening'
        else: tp = 'night'

        if self.model:
            delivery_dataframe = pd.DataFrame([{
                'delivery_distance_km': distance_km,
                'log_distance': np.log1p(distance_km),
                'accept_hour': hour,
                'day_of_week': day_of_week,
                'is_rush_hour': is_rush,
                'is_weekend': is_weekend,
                'courier_daily_packages': courier_daily_packages,
                'courier_avg_speed': courier_avg_speed,
                'distance_x_rush': distance_km * is_rush,
                'city': city,
                'time_period': tp,
                'aoi_type': aoi_type,
            }])
            return float(self.model.predict(delivery_dataframe)[0])
        return distance_km * 4.5

if __name__ == '__main__':
    train_eta_model()

    print("\n--- Sample ETA Predictions ---")
    predictor = ETAPredictor()
    test_cases = [
        {"desc": "Short, rush hour, busy courier",   "dist": 1.0, "hour": 9,  "dow": 1, "pkgs": 50, "speed": 0.03},
        {"desc": "Medium, evening, light load",       "dist": 3.0, "hour": 18, "dow": 4, "pkgs": 15, "speed": 0.06},
        {"desc": "Long, midday, weekend",             "dist": 8.0, "hour": 14, "dow": 6, "pkgs": 20, "speed": 0.04},
        {"desc": "Very short, morning, fast courier",  "dist": 0.3, "hour": 10, "dow": 2, "pkgs": 35, "speed": 0.08},
        {"desc": "Medium, night, slow courier",        "dist": 5.0, "hour": 22, "dow": 0, "pkgs": 10, "speed": 0.02},
    ]
    for tc in test_cases:
        eta = predictor.predict_eta(tc["dist"], tc["hour"], 'unknown', tc["dow"], tc["pkgs"], tc["speed"])
        print(f"  {tc['desc']:40s} -> ETA: {eta:.1f} mins")
