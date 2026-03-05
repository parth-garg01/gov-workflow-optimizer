# src/models/train_models.py
#
# Fixes applied vs. original:
#   1. Separate build_preprocessor() calls for regression and classification
#      pipelines — no shared, mutated ColumnTransformer state.
#   2. stratify=y_cls in the classification train/test split.
#   3. Permutation feature importance computed on the validation split and
#      saved as feature_importance_reg.csv / feature_importance_cls.csv.
#   4. LightGBM verbose=-1 to suppress noisy training output.
#   5. Model artifacts now saved as dicts with metadata
#      (trained_at, n_train_rows, feature_cols) so the dashboard can load
#      feature_cols directly from the pkl instead of re-reading the CSV.
#   6. feature_cols exported via FEATURE_COLS constant so it is available
#      when this module is imported.

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are targets or IDs — excluded from the feature set.
FEATURE_DROP_COLS: list[str] = [
    "file_id",
    "submission_date",
    "processing_time_hours",  # regression target
    "processing_time_days",
    "delayed",                # classification target
    "delay_ratio",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load synthetic_files.csv relative to this file's location."""
    data_path = Path(__file__).resolve().parents[2] / "data" / "synthetic_files.csv"
    df = pd.read_csv(data_path, parse_dates=["submission_date"])
    print(f"Loaded {len(df):,} rows from {data_path}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_sets(df: pd.DataFrame):
    """
    Returns
    -------
    X               : feature DataFrame
    y_reg           : Series — processing_time_hours
    y_cls           : Series (int) — delayed (0/1)
    categorical_cols: list of categorical column names
    numeric_cols    : list of numeric column names
    feature_cols    : ordered list of all feature column names
    """
    feature_cols = [c for c in df.columns if c not in FEATURE_DROP_COLS]

    X     = df[feature_cols].copy()
    y_reg = df["processing_time_hours"].copy()
    y_cls = df["delayed"].astype(int).copy()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols     = X.select_dtypes(exclude=["object"]).columns.tolist()

    return X, y_reg, y_cls, categorical_cols, numeric_cols, feature_cols


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------

def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    """
    Build a *fresh* ColumnTransformer each time so that regression and
    classification pipelines do not share mutable preprocessor state.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_regression_model(
    categorical_cols: list[str],
    numeric_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """Train LightGBM regression pipeline (fresh preprocessor)."""
    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
            ("model", lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbose=-1,          # suppress per-iteration output
            )),
        ]
    )

    reg_pipeline.fit(X_train, y_train)
    preds = reg_pipeline.predict(X_val)

    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    r2   = float(r2_score(y_val, preds))

    print("\n=== Regression model (processing_time_hours) ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    return reg_pipeline


def train_classification_model(
    categorical_cols: list[str],
    numeric_cols: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """Train RandomForest classification pipeline (fresh preprocessor)."""
    cls_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,          # was None → caused 53 MB pkl; cap keeps it reasonable
                min_samples_leaf=10,   # further regularisation
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )),
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


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def compute_and_save_importance(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    out_path: Path,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance on the *validation* set and save to CSV.

    Operates on the full pipeline (original feature names), so column names
    in the CSV correspond to the original input columns — not OHE-exploded names.
    """
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    fi_df = pd.DataFrame({
        "feature":          feature_cols,
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    fi_df.to_csv(out_path, index=False)
    print(f"Feature importance saved to: {out_path}")
    return fi_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_dataset()

    X, y_reg, y_cls, categorical_cols, numeric_cols, feature_cols = build_feature_sets(df)

    print("\nFeature columns:")
    print("  Categorical:", categorical_cols)
    print("  Numeric    :", numeric_cols)

    # --- Regression split (no stratification needed for continuous target) ---
    X_train, X_val, y_reg_train, y_reg_val = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # --- Classification split:  same indices but stratified on y_cls ---
    _, _, y_cls_train, y_cls_val = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    # --- Train models (each gets its OWN preprocessor instance) ---
    reg_pipeline = train_regression_model(
        categorical_cols, numeric_cols,
        X_train, y_reg_train, X_val, y_reg_val,
    )

    cls_pipeline = train_classification_model(
        categorical_cols, numeric_cols,
        X_train, y_cls_train, X_val, y_cls_val,
    )

    # --- Compute permutation importance on validation set ---
    models_dir = Path(__file__).resolve().parent
    print("\nComputing permutation importance (regression)…")
    compute_and_save_importance(
        reg_pipeline, X_val, y_reg_val, feature_cols,
        out_path=models_dir / "feature_importance_reg.csv",
    )

    print("Computing permutation importance (classification)…")
    compute_and_save_importance(
        cls_pipeline, X_val, y_cls_val, feature_cols,
        out_path=models_dir / "feature_importance_cls.csv",
    )

    # --- Save models as dicts with metadata so dashboard is self-contained ---
    trained_at   = datetime.now().isoformat(timespec="seconds")
    n_train_rows = len(X_train)

    reg_artifact = {
        "model":        reg_pipeline,
        "feature_cols": feature_cols,
        "trained_at":   trained_at,
        "n_train_rows": n_train_rows,
    }
    cls_artifact = {
        "model":        cls_pipeline,
        "feature_cols": feature_cols,
        "trained_at":   trained_at,
        "n_train_rows": n_train_rows,
    }

    reg_path = models_dir / "processing_time_model.pkl"
    cls_path = models_dir / "delay_risk_model.pkl"

    joblib.dump(reg_artifact, reg_path)
    joblib.dump(cls_artifact, cls_path)

    print(f"\nSaved regression model  → {reg_path}")
    print(f"Saved classification model → {cls_path}")
    print(f"\nTraining complete.  trained_at={trained_at}")


if __name__ == "__main__":
    main()
