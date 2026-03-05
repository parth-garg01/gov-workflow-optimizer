# src/dashboard/app.py
#
# Fixes applied vs original:
#   1.  load_models() now loads the new dict-artifact format {model, feature_cols, ...}
#       and exposes feature_cols so prediction helpers don't need to re-read the CSV.
#   2.  load_feature_importance() gracefully handles missing CSV files (warns instead of crash).
#   3.  FEATURE_DROP_COLS + NUMERIC_CAST_COLS defined once as module constants — DRY fix.
#   4.  prepare_row_for_prediction() and predict_batch() receive feature_cols as a param.
#   5.  predict_batch() fall-back uses "replace" not "overwrite" for SQLite if_exists.
#   6.  top_kpi() guards against empty DataFrame (avoids int(NaN) ValueError).
#   7.  Sidebar selectbox guarded: shown only when df_f is non-empty.
#   8.  mask built with pd.Series(True, index=df.index) — no RangeIndex alignment risk.
#   9.  Batch predictions stored in st.session_state so download buttons survive re-runs.
#   10. pie chart shows "Delayed" / "On-time" labels (not True/False or 0/1).
#   11. persist_mode "overwrite" → "replace" mapping for SQLite.
#   12. load_new_files_csv() wrapped in try/except against column-mismatch crashes.
#   13. Feature importance charts added (Section 5).
#   14. @st.cache_data(ttl=600) on load_data() so it refreshes if CSV is regenerated.

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import uuid

# ---------------------------------------------------------------------------
# Module-level constants  (single source of truth — fixes DRY violation)
# ---------------------------------------------------------------------------

FEATURE_DROP_COLS: list[str] = [
    "file_id",
    "submission_date",
    "processing_time_hours",
    "processing_time_days",
    "delayed",
    "delay_ratio",
]

NUMERIC_CAST_COLS: list[str] = [
    "complexity_score",
    "num_pages",
    "required_approvals",
    "officer_experience_years",
    "current_backlog_officer",
    "sla_days",
]

# ---------------------------------------------------------------------------
# Resource / data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """
    Load both model artifacts.  Each artifact is now a dict:
        {model, feature_cols, trained_at, n_train_rows}
    Returns (reg_pipeline, cls_pipeline, feature_cols, reg_meta, cls_meta).
    """
    models_dir = Path(__file__).resolve().parents[1] / "models"
    reg_path   = models_dir / "processing_time_model.pkl"
    cls_path   = models_dir / "delay_risk_model.pkl"

    for p in (reg_path, cls_path):
        if not p.exists():
            st.error(f"Model file not found: `{p}`.  Run `train_models.py` first.")
            st.stop()

    reg_art = joblib.load(reg_path)
    cls_art = joblib.load(cls_path)

    # Support both old (bare Pipeline) and new (dict) artifact formats
    if isinstance(reg_art, dict):
        reg_pipeline  = reg_art["model"]
        feature_cols  = reg_art["feature_cols"]
        reg_meta      = {k: v for k, v in reg_art.items() if k != "model"}
    else:
        reg_pipeline  = reg_art
        feature_cols  = None
        reg_meta      = {}

    cls_pipeline = cls_art["model"] if isinstance(cls_art, dict) else cls_art
    cls_meta     = {k: v for k, v in cls_art.items() if k != "model"} if isinstance(cls_art, dict) else {}

    return reg_pipeline, cls_pipeline, feature_cols, reg_meta, cls_meta


@st.cache_data(ttl=600)  # re-read from disk at most every 10 min
def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / "synthetic_files.csv"
    if not data_path.exists():
        st.error(f"Dataset not found: `{data_path}`.  Run `data_gen.py` first.")
        st.stop()
    return pd.read_csv(data_path, parse_dates=["submission_date"])


@st.cache_data
def load_feature_importance(name: str) -> pd.DataFrame:
    """
    Load feature_importance_<name>.csv from src/models/.
    Returns an empty DataFrame and shows a warning if the file is absent.
    """
    fi_path = Path(__file__).resolve().parents[1] / "models" / f"feature_importance_{name}.csv"
    if not fi_path.exists():
        return pd.DataFrame()
    return pd.read_csv(fi_path)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def ensure_data_folder() -> Path:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def save_predictions_csv(df_preds: pd.DataFrame, filename: str = "predictions.csv", mode: str = "append") -> str:
    data_dir = ensure_data_folder()
    path = data_dir / filename
    if mode == "overwrite" or not path.exists():
        df_preds.to_csv(path, index=False)
    else:
        df_preds.to_csv(path, index=False, mode="a", header=False)
    return str(path)


def get_predictions_csv(filename: str = "predictions.csv") -> pd.DataFrame:
    path = ensure_data_folder() / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_predictions_sqlite(
    df_preds: pd.DataFrame,
    dbname: str = "predictions.db",
    table: str = "predictions",
    if_exists: str = "append",
) -> str:
    """
    Persist predictions to SQLite.
    if_exists: 'append' or 'replace'  (pandas does NOT accept 'overwrite').
    The caller is responsible for mapping 'overwrite' -> 'replace' before calling.
    """
    db_path = ensure_data_folder() / dbname
    conn: Connection = sqlite3.connect(str(db_path))
    df_preds.to_sql(table, conn, if_exists=if_exists, index=False)
    conn.close()
    return str(db_path)


def get_predictions_sqlite(dbname: str = "predictions.db", table: str = "predictions") -> pd.DataFrame:
    db_path = ensure_data_folder() / dbname
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def save_new_file_csv(row_dict: dict, filename: str = "new_files.csv", mode: str = "append") -> str:
    data_dir = ensure_data_folder()
    path = data_dir / filename
    df_row = pd.DataFrame([row_dict])
    if mode == "overwrite" or not path.exists():
        df_row.to_csv(path, index=False)
    else:
        df_row.to_csv(path, index=False, mode="a", header=False)
    return str(path)


def load_new_files_csv(filename: str = "new_files.csv") -> pd.DataFrame:
    path = ensure_data_folder() / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["submission_date"])
    except Exception:
        # Fallback: load without parse_dates if column is absent / malformed
        df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# KPI helper
# ---------------------------------------------------------------------------

def top_kpi(df: pd.DataFrame):
    """Return (total, avg_hours, delayed_count, avg_delay_ratio).
    Returns zeros when df is empty to avoid int(NaN) ValueError."""
    if df.empty:
        return 0, 0.0, 0, 0.0

    total           = len(df)
    avg_hours       = round(float(df["processing_time_hours"].mean()), 2)
    # Cast to int only after ensuring the series is numeric and non-null
    delayed_series  = pd.to_numeric(df["delayed"], errors="coerce").fillna(0)
    delayed_count   = int(delayed_series.sum())
    avg_delay_ratio = round(float(df["delay_ratio"].mean()), 3)
    return total, avg_hours, delayed_count, avg_delay_ratio


def extract_first_stage(routing_str) -> str:
    try:
        return str(routing_str).split("->")[0]
    except Exception:
        return "Unknown"


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _fill_feature_defaults(X: pd.DataFrame, sample_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Ensure every column in feature_cols exists in X.
    Missing columns are filled with sensible defaults derived from sample_df.
    This is the single source of truth — replaces the duplicated if/elif chains.
    """
    col_defaults = {
        "department":              sample_df["department"].mode().iloc[0],
        "file_type":               sample_df["file_type"].mode().iloc[0],
        "priority":                "Low",
        "complexity_score":        0.2,
        "num_pages":               3,
        "required_approvals":      1,
        "assigned_officer_id":     "OFF001",
        "officer_experience_years": 2.0,
        "current_backlog_officer": 3,
        "routing_path":            "Clerk->Officer",
        "sla_days":                7,
    }
    for c in feature_cols:
        if c not in X.columns:
            X[c] = col_defaults.get(c, 0)
    return X


def prepare_row_for_prediction(
    row: "dict | pd.Series",
    feature_cols: list[str],
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a single row (Series or dict) to a feature-aligned DataFrame."""
    if isinstance(row, pd.Series):
        X = row.to_frame().T.reset_index(drop=True)
    else:
        X = pd.DataFrame([row])

    X = _fill_feature_defaults(X, sample_df, feature_cols)
    X = X[feature_cols].copy()

    for nc in NUMERIC_CAST_COLS:
        if nc in X.columns:
            X[nc] = pd.to_numeric(X[nc], errors="coerce").fillna(0)

    return X


def predict_for_row(
    reg,
    cls,
    row: "dict | pd.Series",
    feature_cols: list[str],
    sample_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Predict processing hours and delay probability for a single input.
    Returns (pred_hours, pred_delay_prob).
    """
    X = prepare_row_for_prediction(row, feature_cols, sample_df)

    pred_hours = float(reg.predict(X)[0])

    pred_prob = 0.0
    try:
        proba     = cls.predict_proba(X)
        pred_prob = float(proba[0][1])   # column 1 = 'Delayed'
    except Exception:
        try:
            pred_prob = float(cls.predict(X)[0])
        except Exception:
            pred_prob = 0.0

    return pred_hours, pred_prob


def predict_batch(
    reg,
    cls,
    df_subset: pd.DataFrame,
    feature_cols: list[str],
    sample_df: pd.DataFrame,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Vectorised batch prediction — builds the feature matrix once, then calls
    each model once.  Optionally limits to max_rows rows.
    """
    df_proc = (
        df_subset.head(max_rows).copy().reset_index(drop=True)
        if max_rows is not None
        else df_subset.copy().reset_index(drop=True)
    )

    X = pd.DataFrame(index=df_proc.index)
    for c in feature_cols:
        X[c] = df_proc[c] if c in df_proc.columns else np.nan

    X = _fill_feature_defaults(X, sample_df, feature_cols)
    X = X[feature_cols].copy()

    for nc in NUMERIC_CAST_COLS:
        if nc in X.columns:
            X[nc] = pd.to_numeric(X[nc], errors="coerce").fillna(0)

    pred_hours = reg.predict(X)
    try:
        pred_probs = cls.predict_proba(X)[:, 1]
    except Exception:
        pred_probs = cls.predict(X).astype(float)

    out = df_proc.copy()
    out["pred_processing_hours"] = np.round(pred_hours.astype(float), 2)
    out["pred_delay_prob"]       = np.round(pred_probs.astype(float), 3)
    return out


# ===========================================================================
# App UI
# ===========================================================================

st.set_page_config(page_title="Gov File Workflow Dashboard", layout="wide")
st.title("AI Workflow Optimization — Dashboard")

# --- Load resources ---
df = load_data()
reg_model, cls_model, feature_cols_from_model, reg_meta, cls_meta = load_models()

# If models were saved in new dict format, use embedded feature_cols.
# Otherwise fall back to deriving them from the CSV (old pkl compatibility).
if feature_cols_from_model is not None:
    FEATURE_COLS: list[str] = feature_cols_from_model
else:
    FEATURE_COLS = [c for c in df.columns if c not in FEATURE_DROP_COLS]

# Show model metadata if available
if reg_meta.get("trained_at"):
    st.caption(
        f"Models trained at **{reg_meta['trained_at']}** "
        f"on {reg_meta.get('n_train_rows', '?'):,} training rows."
    )

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

departments = ["All"] + sorted(df["department"].dropna().unique().tolist())
file_types  = ["All"] + sorted(df["file_type"].dropna().unique().tolist())
priorities  = ["All"] + sorted(df["priority"].dropna().unique().tolist())

sel_dept      = st.sidebar.selectbox("Department", departments)
sel_file_type = st.sidebar.selectbox("File Type",  file_types)
sel_priority  = st.sidebar.selectbox("Priority",   priorities)

date_min = df["submission_date"].min()
date_max = df["submission_date"].max()
sel_date_range = st.sidebar.date_input("Submission date range", [date_min.date(), date_max.date()])

search_officer = st.sidebar.text_input("Search Officer ID (partial)", "")

# Build mask — use df.index so alignment is always correct (fixes RT-04)
mask = pd.Series(True, index=df.index)

if sel_dept != "All":
    mask &= df["department"] == sel_dept
if sel_file_type != "All":
    mask &= df["file_type"] == sel_file_type
if sel_priority != "All":
    mask &= df["priority"] == sel_priority
if sel_date_range and len(sel_date_range) == 2:
    start_dt = pd.to_datetime(sel_date_range[0])
    end_dt   = pd.to_datetime(sel_date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask &= (df["submission_date"] >= start_dt) & (df["submission_date"] <= end_dt)
if search_officer.strip():
    mask &= df["assigned_officer_id"].str.contains(search_officer.strip(), case=False, na=False)

df_f = df[mask].copy()

# ---------------------------------------------------------------------------
# Top KPIs
# ---------------------------------------------------------------------------

total, avg_hours, delayed_count, avg_delay_ratio = top_kpi(df_f)  # safe on empty df
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
col1.metric("Total files",          f"{total:,}")
col2.metric("Avg processing (hrs)", f"{avg_hours}")
col3.metric("Delayed files",        f"{delayed_count}")
col4.metric("Avg delay ratio",      f"{avg_delay_ratio}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Main charts — guarded against empty filter
# ---------------------------------------------------------------------------

if df_f.empty:
    st.warning("⚠️ No files match the current filters. Adjust the sidebar options.")
else:
    left, right = st.columns([2.5, 1])

    with left:
        st.subheader("Processing time distribution")
        fig_hist = px.histogram(
            df_f, x="processing_time_hours", nbins=60,
            title="Distribution of processing time (hours)", marginal="box",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Avg processing time by Department / File Type")
        agg = (
            df_f.groupby(["department", "file_type"])["processing_time_hours"]
            .mean()
            .reset_index()
        )
        fig_bar = px.bar(
            agg, x="department", y="processing_time_hours", color="file_type",
            barmode="group", title="Avg processing time (hrs)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Bottleneck: Avg processing time by First Routing Stage")
        df_f["first_stage"] = df_f["routing_path"].apply(extract_first_stage)
        stage_agg = (
            df_f.groupby("first_stage")["processing_time_hours"]
            .mean()
            .reset_index()
            .sort_values("processing_time_hours", ascending=False)
        )
        fig_stage = px.bar(stage_agg, x="first_stage", y="processing_time_hours",
                           title="Avg hrs by first stage")
        st.plotly_chart(fig_stage, use_container_width=True)

    with right:
        st.subheader("Delay risk overview")
        # Fix ST-02: map bool/int → human-readable label before charting
        label_map = {True: "Delayed", False: "On-time", 1: "Delayed", 0: "On-time"}
        delay_labels = df_f["delayed"].map(label_map).fillna("Unknown")
        delay_counts = delay_labels.value_counts().reset_index()
        delay_counts.columns = ["Status", "Count"]
        fig_pie = px.pie(delay_counts, names="Status", values="Count",
                         title="Delayed vs On-time",
                         color_discrete_map={"Delayed": "#EF4444", "On-time": "#22C55E"})
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Quick stats")
        st.write("Median pages:",            int(df_f["num_pages"].median()))
        st.write("Avg required approvals:",  round(df_f["required_approvals"].mean(), 2))
        st.write("Avg officer exp (yrs):",   round(df_f["officer_experience_years"].mean(), 2))
        st.write("Avg backlog per officer:", round(df_f["current_backlog_officer"].mean(), 2))

    st.markdown("---")

    # Correlation heatmap
    st.subheader("Numeric feature correlations (heatmap)")
    numeric_col_names = [
        "complexity_score", "num_pages", "required_approvals",
        "officer_experience_years", "current_backlog_officer",
        "processing_time_hours", "delay_ratio",
    ]
    corr = df_f[numeric_col_names].corr()
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, colorscale="Viridis",
    ))
    fig_heat.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section: Single-file prediction
# ---------------------------------------------------------------------------

st.subheader("Predict processing time & delay risk for a file")

if df_f.empty:
    st.info("Apply filters that return at least one file to enable single-file prediction.")
else:
    file_options = df_f["file_id"].tolist()
    selected = st.selectbox("Pick a file (filtered view)", options=file_options)

    if selected:
        row = df.loc[df["file_id"] == selected].iloc[0]
        st.write("**File details (selected):**")
        st.write({
            "file_id":          row["file_id"],
            "department":       row["department"],
            "file_type":        row["file_type"],
            "priority":         row["priority"],
            "num_pages":        int(row["num_pages"]),
            "complexity_score": float(row["complexity_score"]),
            "required_approvals": int(row["required_approvals"]),
        })

        if st.button("Predict for selected file"):
            pred_hours, prob = predict_for_row(reg_model, cls_model, row, FEATURE_COLS, df)
            st.success(f"Predicted processing time: **{pred_hours:.2f} hours** ({pred_hours/24:.2f} days)")
            st.info(f"Predicted delay probability: **{prob*100:.1f}%**")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section: Batch predictions (with session_state persistence — fix ST-04)
# ---------------------------------------------------------------------------

st.subheader("Files (sample) — with optional predictions and persistence")

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    do_predict = st.button("Compute predictions for filtered files")
with col_b:
    fast_mode  = st.checkbox("Fast mode (limit rows)", value=True)
    limit_rows = st.number_input("Max rows in fast mode", min_value=100, max_value=5000, value=500, step=100)
with col_c:
    persist_now    = st.checkbox("Persist after compute", value=False)
    persist_target = st.selectbox("Persist target", ["CSV (data/predictions.csv)", "SQLite (data/predictions.db)"])
    persist_mode   = st.selectbox("Persist mode", ["append", "overwrite"])

display_cols = [
    "file_id", "department", "file_type", "priority", "submission_date",
    "processing_time_hours", "delayed", "assigned_officer_id", "current_backlog_officer",
]

# --- Compute and store in session_state so widgets survive re-runs ---
if do_predict:
    if df_f.empty:
        st.warning("No rows to predict — adjust the filters first.")
    else:
        with st.spinner("Computing predictions for filtered files…"):
            max_r = int(limit_rows) if fast_mode else None
            preds_df = predict_batch(reg_model, cls_model, df_f, FEATURE_COLS, df, max_rows=max_r)
        st.session_state["preds_df"] = preds_df

# --- Render batch results if they exist in session ---
if "preds_df" in st.session_state:
    preds_df  = st.session_state["preds_df"]
    show_cols = [c for c in display_cols if c in preds_df.columns] + ["pred_processing_hours", "pred_delay_prob"]

    st.dataframe(
        preds_df[show_cols].sort_values("submission_date", ascending=False).head(500),
        use_container_width=True,
    )

    csv_bytes = preds_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download filtered results with predictions (CSV)",
        data=csv_bytes,
        file_name="filtered_with_predictions.csv",
        mime="text/csv",
    )

    if persist_now:
        if persist_target.startswith("CSV"):
            save_path = save_predictions_csv(preds_df, filename="predictions.csv", mode=persist_mode)
            st.success(f"Saved predictions to CSV: `{save_path}`")
        else:
            # Fix RT-03: map "overwrite" → "replace" for pandas to_sql
            sql_mode  = "replace" if persist_mode == "overwrite" else "append"
            save_path = save_predictions_sqlite(preds_df, dbname="predictions.db",
                                                table="predictions", if_exists=sql_mode)
            st.success(f"Saved predictions to SQLite: `{save_path}`")
        st.info("Data saved under `data/`.  Avoid committing large files to git.")

    # Persisted store preview
    st.markdown("**Persisted store preview**")
    if persist_target.startswith("CSV"):
        persisted = get_predictions_csv("predictions.csv")
    else:
        persisted = get_predictions_sqlite("predictions.db", "predictions")

    if not persisted.empty:
        st.write(f"Persisted rows: {len(persisted):,}")
        st.dataframe(persisted.head(200), use_container_width=True)
        st.download_button(
            "⬇ Download persisted store (CSV)",
            data=persisted.to_csv(index=False).encode("utf-8"),
            file_name="persisted_predictions.csv",
            mime="text/csv",
        )
    else:
        st.write("No persisted predictions found yet.")

else:
    # No predictions computed yet — show raw sample
    if not df_f.empty:
        safe_display = [c for c in display_cols if c in df_f.columns]
        st.dataframe(
            df_f[safe_display].sort_values("submission_date", ascending=False).head(200),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Section: Feature importance charts  (fix ST-03 — was entirely missing)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📊 Feature Importance (Permutation — Validation Set)")

fi_tab_reg, fi_tab_cls = st.tabs(["Regression (processing time)", "Classification (delay risk)"])

with fi_tab_reg:
    fi_reg = load_feature_importance("reg")
    if fi_reg.empty:
        st.warning(
            "Feature importance file not found (`feature_importance_reg.csv`).  "
            "Re-run `train_models.py` to generate it."
        )
    else:
        fig_fi_reg = px.bar(
            fi_reg.head(15).sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
            title="Top 15 features — Processing Time (Regression)",
            labels={"importance_mean": "Mean Permutation Importance", "feature": "Feature"},
            color="importance_mean",
            color_continuous_scale="Blues",
        )
        fig_fi_reg.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi_reg, use_container_width=True)
        with st.expander("Show full importance table"):
            st.dataframe(fi_reg, use_container_width=True)

with fi_tab_cls:
    fi_cls = load_feature_importance("cls")
    if fi_cls.empty:
        st.warning(
            "Feature importance file not found (`feature_importance_cls.csv`).  "
            "Re-run `train_models.py` to generate it."
        )
    else:
        fig_fi_cls = px.bar(
            fi_cls.head(15).sort_values("importance_mean"),
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
            title="Top 15 features — Delay Risk (Classification)",
            labels={"importance_mean": "Mean Permutation Importance", "feature": "Feature"},
            color="importance_mean",
            color_continuous_scale="Reds",
        )
        fig_fi_cls.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi_cls, use_container_width=True)
        with st.expander("Show full importance table"):
            st.dataframe(fi_cls, use_container_width=True)

# ---------------------------------------------------------------------------
# Sidebar: Add new file form
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Add new file (manual & predict)")

with st.sidebar.form("add_file"):
    new_ftype      = st.selectbox("File type",    ["application", "permit", "appeal", "report"])
    new_dept       = st.selectbox("Department",   sorted(df["department"].unique()))
    new_priority   = st.selectbox("Priority",     ["Low", "Medium", "High"])
    new_pages      = st.number_input("Num pages", value=5,   min_value=1)
    new_complex    = st.slider("Complexity (0–1)", 0.0, 1.0, 0.2, 0.01)
    new_approvals  = st.number_input("Required approvals", value=1, min_value=1, max_value=10)
    new_officer    = st.text_input("Assigned officer ID", value="OFF001", max_chars=10)
    new_experience = st.number_input("Officer experience (yrs)", value=2.0, min_value=0.0, step=0.1)
    new_backlog    = st.number_input("Officer backlog",          value=5,   min_value=0)

    persist_new_file = st.checkbox("Persist new file to CSV (data/new_files.csv)", value=True)
    new_file_mode    = st.selectbox("New file save mode", ["append", "overwrite"])

    submit = st.form_submit_button("Add (predict & save)")

if submit:
    new_file_id     = str(uuid.uuid4())
    submission_date = datetime.now()

    new_row = {
        "file_id":                  new_file_id,
        "department":               new_dept,
        "file_type":                new_ftype,
        "priority":                 new_priority,
        "submission_date":          submission_date,
        "complexity_score":         float(new_complex),
        "num_pages":                int(new_pages),
        "required_approvals":       int(new_approvals),
        "assigned_officer_id":      new_officer.strip(),
        "officer_experience_years": float(new_experience),
        "current_backlog_officer":  int(new_backlog),
        "routing_path":             "Clerk->Officer",
        "sla_days":                 7,
        # targets are unknown for new files
        "processing_time_hours":    None,
        "delayed":                  None,
        "delay_ratio":              None,
        "processing_time_days":     None,
    }

    pred_hours, prob = predict_for_row(reg_model, cls_model, new_row, FEATURE_COLS, df)
    new_row["pred_processing_hours"] = round(pred_hours, 2)
    new_row["pred_delay_prob"]       = round(prob, 3)

    st.sidebar.success(f"Predicted time: **{pred_hours:.2f} hrs** ({pred_hours/24:.2f} days)")
    st.sidebar.info(f"Delay probability: **{prob*100:.1f}%**")

    if persist_new_file:
        save_path = save_new_file_csv(new_row, filename="new_files.csv", mode=new_file_mode)
        st.sidebar.success(f"Saved to: `{save_path}`")

# ---------------------------------------------------------------------------
# Section: Recently added new files
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Recently Added New Files (from form)")

new_files_df = load_new_files_csv("new_files.csv")
if not new_files_df.empty:
    st.write(f"Total new files persisted: **{len(new_files_df)}**")
    sort_col = "submission_date" if "submission_date" in new_files_df.columns else new_files_df.columns[0]
    st.dataframe(
        new_files_df.sort_values(by=sort_col, ascending=False).head(50),
        use_container_width=True,
    )
else:
    st.write("No new files have been added yet.")

st.markdown(
    "**Note:** Predictions are local. To productionise, add a FastAPI endpoint "
    "or a scheduled batch job, and avoid committing large CSV/DB files to git."
)
