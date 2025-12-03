# src/models/train_models.py

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib


def load_dataset():
    # Go from this file: src/models/train_models.py -> project root -> data/synthetic_files.csv
    data_path = Path(__file__).resolve().parents[2] / "data" / "synthetic_files.csv"
    df = pd.read_csv(data_path, parse_dates=["submission_date"])
    return df


def build_feature_sets(df: pd.DataFrame):
    # Columns we won't use as direct features
    drop_cols = [
        "file_id",
        "submission_date",
        "processing_time_hours",  # target for regression
        "processing_time_days",
        "delayed",               # target for classification
        "delay_ratio"
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y_reg = df["processing_time_hours"].copy()
    y_cls = df["delayed"].astype(int).copy()

    # Identify categorical vs numeric
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y_reg, y_cls, categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )
    return preprocessor


def train_regression_model(preprocessor, X_train, y_train, X_val, y_val):
    # LightGBM regressor pipeline
    reg_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", reg_model),
        ]
    )

    reg_pipeline.fit(X_train, y_train)
    preds = reg_pipeline.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))

    r2 = r2_score(y_val, preds)

    print("\n=== Regression model (processing_time_hours) ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    return reg_pipeline


def train_classification_model(preprocessor, X_train, y_train, X_val, y_val):
    cls_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    cls_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", cls_model),
        ]
    )

    cls_pipeline.fit(X_train, y_train)
    preds = cls_pipeline.predict(X_val)

    acc = accuracy_score(y_val, preds)
    print("\n=== Classification model (delayed) ===")
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_val, preds, target_names=["On time", "Delayed"]))

    return cls_pipeline


def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} rows")

    X, y_reg, y_cls, categorical_cols, numeric_cols = build_feature_sets(df)

    print("\nFeature columns:")
    print("Categorical:", categorical_cols)
    print("Numeric    :", numeric_cols)

    X_train, X_val, y_reg_train, y_reg_val = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # For classification, use the same split to keep it simple
    _, _, y_cls_train, y_cls_val = train_test_split(
        X, y_cls, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    # Train regression model
    reg_pipeline = train_regression_model(
        preprocessor, X_train, y_reg_train, X_val, y_reg_val
    )

    # Train classification model
    cls_pipeline = train_classification_model(
        preprocessor, X_train, y_cls_train, X_val, y_cls_val
    )

    # Save models
    models_dir = Path(__file__).resolve().parent
    reg_path = models_dir / "processing_time_model.pkl"
    cls_path = models_dir / "delay_risk_model.pkl"

    joblib.dump(reg_pipeline, reg_path)
    joblib.dump(cls_pipeline, cls_path)

    print(f"\nSaved regression model to: {reg_path}")
    print(f"Saved classification model to: {cls_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
