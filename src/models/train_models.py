# src/models/train_models.py
# Overhauled training pipeline for the enhanced 50k-row government_files dataset.
#
# Key changes vs. original:
#   1. Loads government_files.csv (new realistic dataset, 22 features)
#   2. LightGBMClassifier replaces RandomForest for delay-risk prediction
#   3. XGBoost stacked with LightGBM for regression (averaged ensemble)
#   4. StratifiedKFold cross-validation (5-fold) for both models
#   5. ROC-AUC, F1, precision/recall added to classifier evaluation
#   6. Boolean features (online_submission etc.) cast to int before encoding
#   7. Model artifacts include cv_scores, roc_auc, r2 for dashboard display
#   8. feature_importance_reg.csv / feature_importance_cls.csv updated for new cols

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DROP_COLS: list[str] = [
    "file_id",
    "submission_date",
    "processing_time_hours",
    "processing_time_days",
    "delayed",
    "delay_ratio",
]

# Columns that are boolean and need int conversion before OneHotEncoder
BOOL_COLS: list[str] = [
    "online_submission",
    "incomplete_docs",
    "resubmission",
    "escalated",
]

DATA_FILE = "government_files.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run `python src/data_gen.py` to generate it first."
        )
    df = pd.read_csv(data_path, parse_dates=["submission_date"])
    print(f"Loaded {len(df):,} rows from {data_path.name}")
    print(f"Delayed rate: {df['delayed'].mean()*100:.1f}%")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_sets(df: pd.DataFrame):
    """
    Returns X, y_reg, y_cls, categorical_cols, numeric_cols, feature_cols.
    Boolean columns are cast to int so they're treated as numeric features.
    """
    feature_cols = [c for c in df.columns if c not in FEATURE_DROP_COLS]

    X = df[feature_cols].copy()
    # Cast bool flags to int (0/1) — pipelines handle them as numeric
    for col in BOOL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    y_reg = df["processing_time_hours"].copy()
    y_cls = df["delayed"].astype(int).copy()

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols     = X.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"\nFeatures: {len(feature_cols)}  "
          f"(cat={len(categorical_cols)}, num={len(numeric_cols)})")
    print("  Categorical:", categorical_cols)
    print("  Numeric    :", numeric_cols)

    return X, y_reg, y_cls, categorical_cols, numeric_cols, feature_cols


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------

def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_regression_model(
    categorical_cols, numeric_cols,
    X_train, y_train, X_val, y_val,
) -> Pipeline:
    """LightGBM + XGBoost averaged ensemble for processing-time regression."""

    # --- LightGBM pipeline ---
    lgbm_pipe = Pipeline([
        ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
        ("model", lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=8,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )),
    ])

    # --- XGBoost pipeline ---
    xgb_pipe = Pipeline([
        ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
        ("model", xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            verbosity=0,
        )),
    ])

    print("\nTraining LightGBM regressor…")
    lgbm_pipe.fit(X_train, y_train)
    print("Training XGBoost regressor…")
    xgb_pipe.fit(X_train, y_train)

    lgbm_preds = lgbm_pipe.predict(X_val)
    xgb_preds  = xgb_pipe.predict(X_val)
    avg_preds  = (lgbm_preds + xgb_preds) / 2.0

    rmse_lgbm = float(np.sqrt(mean_squared_error(y_val, lgbm_preds)))
    rmse_xgb  = float(np.sqrt(mean_squared_error(y_val, xgb_preds)))
    rmse_ens  = float(np.sqrt(mean_squared_error(y_val, avg_preds)))
    r2_ens    = float(r2_score(y_val, avg_preds))

    print(f"\n=== Regression (processing_time_hours) ===")
    print(f"RMSE  LightGBM : {rmse_lgbm:.2f}")
    print(f"RMSE  XGBoost  : {rmse_xgb:.2f}")
    print(f"RMSE  Ensemble : {rmse_ens:.2f}")
    print(f"R²    Ensemble : {r2_ens:.4f}")

    return lgbm_pipe, xgb_pipe, {"rmse": rmse_ens, "r2": r2_ens}


def train_classification_model(
    categorical_cols, numeric_cols,
    X_train, y_train, X_val, y_val,
) -> Pipeline:
    """LightGBM classifier for delay-risk prediction with CV evaluation."""

    cls_pipe = Pipeline([
        ("preprocessor", build_preprocessor(categorical_cols, numeric_cols)),
        ("model", lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=8,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )),
    ])

    print("\nTraining LightGBM classifier…")
    cls_pipe.fit(X_train, y_train)

    preds      = cls_pipe.predict(X_val)
    proba      = cls_pipe.predict_proba(X_val)[:, 1]
    acc        = accuracy_score(y_val, preds)
    roc_auc    = roc_auc_score(y_val, proba)
    f1         = f1_score(y_val, preds)

    print(f"\n=== Classification (delayed) ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, preds, target_names=["On time", "Delayed"]))

    return cls_pipe, {"accuracy": acc, "roc_auc": roc_auc, "f1": f1}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(pipeline, X, y, cv_type="regression") -> dict:
    """5-fold stratified (cls) or regular (reg) cross-validation."""
    print(f"\nRunning 5-fold CV ({cv_type})…")

    if cv_type == "classification":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_roc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        scores_f1  = cross_val_score(pipeline, X, y, cv=cv, scoring="f1",      n_jobs=-1)
        print(f"  ROC-AUC: {scores_roc.mean():.4f} ± {scores_roc.std():.4f}")
        print(f"  F1     : {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
        return {
            "roc_auc_mean": float(scores_roc.mean()),
            "roc_auc_std":  float(scores_roc.std()),
            "f1_mean":      float(scores_f1.mean()),
            "f1_std":       float(scores_f1.std()),
        }
    else:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2", n_jobs=-1)
        print(f"  R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return {
            "r2_mean": float(cv_scores.mean()),
            "r2_std":  float(cv_scores.std()),
        }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_and_save_importance(
    pipeline, X_val, y_val, feature_cols, out_path, n_repeats=10,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X_val, y_val,
        n_repeats=n_repeats, random_state=42, n_jobs=-1,
    )
    fi_df = pd.DataFrame({
        "feature":         feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    fi_df.to_csv(out_path, index=False)
    print(f"Feature importance saved: {out_path.name}")
    return fi_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_dataset()
    X, y_reg, y_cls, cat_cols, num_cols, feature_cols = build_feature_sets(df)

    # Single stratified split — same row indices for both targets
    X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val = train_test_split(
        X, y_reg, y_cls, test_size=0.20, random_state=42, stratify=y_cls,
    )

    # --- Train ---
    lgbm_reg, xgb_reg, reg_metrics = train_regression_model(
        cat_cols, num_cols, X_train, y_reg_train, X_val, y_reg_val,
    )
    cls_pipe, cls_metrics = train_classification_model(
        cat_cols, num_cols, X_train, y_cls_train, X_val, y_cls_val,
    )

    # --- Cross-validation (full dataset, robust estimate) ---
    reg_cv = run_cross_validation(lgbm_reg, X, y_reg, cv_type="regression")
    cls_cv = run_cross_validation(cls_pipe, X, y_cls, cv_type="classification")

    # --- Permutation importance (validation set only) ---
    models_dir = Path(__file__).resolve().parent
    print("\nComputing permutation importance (regression)…")
    compute_and_save_importance(
        lgbm_reg, X_val, y_reg_val, feature_cols,
        out_path=models_dir / "feature_importance_reg.csv",
    )
    print("Computing permutation importance (classification)…")
    compute_and_save_importance(
        cls_pipe, X_val, y_cls_val, feature_cols,
        out_path=models_dir / "feature_importance_cls.csv",
    )

    # --- Persist model artifacts ---
    trained_at   = datetime.now().isoformat(timespec="seconds")
    n_train_rows = len(X_train)

    reg_artifact = {
        "model":           lgbm_reg,
        "xgb_model":       xgb_reg,
        "feature_cols":    feature_cols,
        "trained_at":      trained_at,
        "n_train_rows":    n_train_rows,
        "metrics":         {**reg_metrics, **reg_cv},
        "dataset":         DATA_FILE,
    }
    cls_artifact = {
        "model":           cls_pipe,
        "feature_cols":    feature_cols,
        "trained_at":      trained_at,
        "n_train_rows":    n_train_rows,
        "metrics":         {**cls_metrics, **cls_cv},
        "dataset":         DATA_FILE,
    }

    reg_path = models_dir / "processing_time_model.pkl"
    cls_path = models_dir / "delay_risk_model.pkl"
    joblib.dump(reg_artifact, reg_path)
    joblib.dump(cls_artifact, cls_path)

    print(f"\nSaved regression model  : {reg_path}")
    print(f"Saved classification model : {cls_path}")
    print(f"\nTraining complete.  trained_at={trained_at}")
    print(f"\n--- Final metrics ---")
    print(f"Regression  R²: {reg_metrics['r2']:.4f}  RMSE: {reg_metrics['rmse']:.2f}")
    print(f"CV R²: {reg_cv['r2_mean']:.4f} ± {reg_cv['r2_std']:.4f}")
    print(f"Classification  ROC-AUC: {cls_metrics['roc_auc']:.4f}  F1: {cls_metrics['f1']:.4f}")
    print(f"CV ROC-AUC: {cls_cv['roc_auc_mean']:.4f} ± {cls_cv['roc_auc_std']:.4f}")


if __name__ == "__main__":
    main()
