import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, classification_report
import joblib
import os

def train_dataco_model(csv_path: str = 'data/processed/dataco_processed.csv'):
    print("Loading DataCo processed data...")
    delivery_dataframe = pd.read_csv(csv_path)
    print(f"Rows: {len(delivery_dataframe):,}, Columns: {len(delivery_dataframe.columns)}")

    delivery_dataframe = delivery_dataframe[delivery_dataframe['Delivery Status'] != 'Shipping canceled'].copy()
    print(f"After removing cancellations: {len(delivery_dataframe):,}")

    num_features = [
        'Days for shipment (scheduled)',
        'Order Item Quantity',
        'Sales',
        'Order Profit Per Order',
        'Benefit per order',
        'Latitude',
        'Longitude',
    ]

    cat_features = [
        'Shipping Mode',
        'Order Region',
        'Category Name',
        'Customer Segment',
    ]

    target = 'Late_delivery_risk'

    delivery_dataframe = delivery_dataframe.dropna(subset=num_features + cat_features + [target])

    features_dataframe = delivery_dataframe[num_features + cat_features]
    target_variable = delivery_dataframe[target]

    print(f"\nTarget distribution:")
    print(f"  Late: {target_variable.sum():,} ({target_variable.mean():.1%})")
    print(f"  On-time: {(~target_variable.astype(bool)).sum():,} ({1-target_variable.mean():.1%})")

    X_train, X_test, y_train, y_test = train_test_split(features_dataframe, target_variable, test_size=0.2, random_state=42, stratify=target_variable)
    print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
        ))
    ])

    print("Training XGBoost on DataCo (180K rows)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    print(f"\n{'='*55}")
    print(f"  DataCo Model Results ({len(delivery_dataframe):,} shipments)")
    print(f"{'='*55}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"{'='*55}")

    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    model = pipeline.named_steps['classifier']
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = list(ohe.get_feature_names_out(cat_features))
    all_names = num_features + cat_names

    importances = model.feature_importances_
    top_feats = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)[:15]
    print("\nTop 15 feature importances:")
    for name, imp in top_feats:
        bar = '#' * int(imp * 100)
        print(f"  {name:40s} {imp:.4f} {bar}")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/dataco_delay_predictor.joblib'
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    return pipeline

if __name__ == '__main__':
    train_dataco_model()
