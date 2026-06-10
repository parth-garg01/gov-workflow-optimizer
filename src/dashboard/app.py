# src/dashboard/app.py
# Rebuilt for the 50k-row government_files.csv dataset with 22 features.
# UI overhauled: tabbed navigation, colour-coded KPI cards, trend charts,
# regional analysis, officer heatmap, and a redesigned prediction panel.

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
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FlowGov AI — Government Workflow Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* KPI card styling */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 3px solid #3b82f6;
    padding-left: 0.6rem;
    margin-bottom: 0.8rem;
    margin-top: 0.5rem;
}

/* Risk badge */
.risk-high   { color: #ef4444; font-weight: 700; }
.risk-medium { color: #f59e0b; font-weight: 700; }
.risk-low    { color: #22c55e; font-weight: 700; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
}

/* Tab styling */
[data-testid="stTabs"] button {
    font-weight: 600;
}

/* Divider */
hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_DROP_COLS: list[str] = [
    "file_id", "submission_date",
    "processing_time_hours", "processing_time_days",
    "delayed", "delay_ratio",
]

BOOL_COLS: list[str] = [
    "online_submission", "incomplete_docs", "resubmission", "escalated",
]

NUMERIC_CAST_COLS: list[str] = [
    "complexity_score", "num_pages", "required_approvals",
    "officer_experience_years", "current_backlog_officer",
    "sla_days", "submission_month", "submission_weekday",
    "online_submission", "incomplete_docs", "resubmission", "escalated",
]

DATA_FILE = "government_files.csv"

CHART_COLORS = px.colors.qualitative.Set2
DELAY_COLOR  = {"Delayed": "#ef4444", "On-time": "#22c55e"}

DARK_LAYOUT = dict(
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font_color="#94a3b8",
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1e293b", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", zeroline=False),
)


def dark_layout(height=320, **extra) -> dict:
    layout = dict(DARK_LAYOUT)
    layout["height"] = height
    layout.update(extra)
    return layout

# ---------------------------------------------------------------------------
# Resource / data loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    models_dir = Path(__file__).resolve().parents[1] / "models"
    reg_path   = models_dir / "processing_time_model.pkl"
    cls_path   = models_dir / "delay_risk_model.pkl"

    for p in (reg_path, cls_path):
        if not p.exists():
            st.error(f"Model file not found: `{p}`. Run `train_models.py` first.")
            st.stop()

    reg_art = joblib.load(reg_path)
    cls_art = joblib.load(cls_path)

    reg_pipeline = reg_art["model"]        if isinstance(reg_art, dict) else reg_art
    xgb_model    = reg_art.get("xgb_model") if isinstance(reg_art, dict) else None
    feature_cols = reg_art.get("feature_cols") if isinstance(reg_art, dict) else None
    reg_meta     = {k: v for k, v in reg_art.items() if k not in ("model","xgb_model")} \
                   if isinstance(reg_art, dict) else {}

    cls_pipeline = cls_art["model"]        if isinstance(cls_art, dict) else cls_art
    cls_meta     = {k: v for k, v in cls_art.items() if k != "model"} \
                   if isinstance(cls_art, dict) else {}

    return reg_pipeline, xgb_model, cls_pipeline, feature_cols, reg_meta, cls_meta


@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / DATA_FILE
    if not data_path.exists():
        st.error(f"Dataset not found: `{data_path}`. Run `data_gen.py` first.")
        st.stop()
    df = pd.read_csv(data_path, parse_dates=["submission_date"])
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


@st.cache_data
def load_feature_importance(name: str) -> pd.DataFrame:
    fi_path = Path(__file__).resolve().parents[1] / "models" / f"feature_importance_{name}.csv"
    return pd.read_csv(fi_path) if fi_path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def ensure_data_folder() -> Path:
    d = Path(__file__).resolve().parents[2] / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_predictions_csv(df_preds: pd.DataFrame, mode: str = "append") -> str:
    path = ensure_data_folder() / "predictions.csv"
    if mode == "overwrite" or not path.exists():
        df_preds.to_csv(path, index=False)
    else:
        df_preds.to_csv(path, index=False, mode="a", header=False)
    return str(path)


def save_predictions_sqlite(df_preds: pd.DataFrame, if_exists: str = "append") -> str:
    db_path = ensure_data_folder() / "predictions.db"
    conn: Connection = sqlite3.connect(str(db_path))
    df_preds.to_sql("predictions", conn, if_exists=if_exists, index=False)
    conn.close()
    return str(db_path)


def save_new_file_csv(row_dict: dict, mode: str = "append") -> str:
    path = ensure_data_folder() / "new_files.csv"
    df_row = pd.DataFrame([row_dict])
    if mode == "overwrite" or not path.exists():
        df_row.to_csv(path, index=False)
    else:
        df_row.to_csv(path, index=False, mode="a", header=False)
    return str(path)


def load_new_files_csv() -> pd.DataFrame:
    path = ensure_data_folder() / "new_files.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["submission_date"])
    except Exception:
        return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

_COL_DEFAULTS = {
    "department": "Revenue", "file_type": "application", "priority": "Low",
    "region": "Central", "complexity_score": 0.2, "num_pages": 5,
    "required_approvals": 1, "assigned_officer_id": "OFF001",
    "officer_experience_years": 2.0, "current_backlog_officer": 5,
    "online_submission": 1, "incomplete_docs": 0,
    "resubmission": 0, "escalated": 0,
    "routing_path": "Clerk->Officer", "sla_days": 7,
    "submission_month": 6, "submission_weekday": 1,
}


def _prepare_X(row: "dict | pd.Series", feature_cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame([row if isinstance(row, dict) else row.to_dict()])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = _COL_DEFAULTS.get(c, 0)
    X = X[feature_cols].copy()
    for nc in NUMERIC_CAST_COLS:
        if nc in X.columns:
            X[nc] = pd.to_numeric(X[nc], errors="coerce").fillna(0)
    return X


def predict_single(
    reg, xgb_model, cls,
    row: "dict | pd.Series",
    feature_cols: list[str],
) -> Tuple[float, float]:
    X = _prepare_X(row, feature_cols)
    lgbm_pred = float(reg.predict(X)[0])
    if xgb_model is not None:
        xgb_pred = float(xgb_model.predict(X)[0])
        pred_hours = (lgbm_pred + xgb_pred) / 2.0
    else:
        pred_hours = lgbm_pred
    try:
        pred_prob = float(cls.predict_proba(X)[0][1])
    except Exception:
        pred_prob = float(cls.predict(X)[0])
    return pred_hours, pred_prob


def predict_batch(
    reg, xgb_model, cls,
    df_subset: pd.DataFrame,
    feature_cols: list[str],
    max_rows: int | None = None,
) -> pd.DataFrame:
    df_proc = (df_subset.head(max_rows) if max_rows else df_subset).copy().reset_index(drop=True)
    X = pd.DataFrame(index=df_proc.index)
    for c in feature_cols:
        X[c] = df_proc[c] if c in df_proc.columns else np.nan
    for c in feature_cols:
        if c not in X.columns:
            X[c] = _COL_DEFAULTS.get(c, 0)
    X = X[feature_cols].copy()
    for nc in NUMERIC_CAST_COLS:
        if nc in X.columns:
            X[nc] = pd.to_numeric(X[nc], errors="coerce").fillna(0)

    lgbm_preds = reg.predict(X)
    if xgb_model is not None:
        xgb_preds = xgb_model.predict(X)
        pred_hours = (lgbm_preds + xgb_preds) / 2.0
    else:
        pred_hours = lgbm_preds

    try:
        pred_probs = cls.predict_proba(X)[:, 1]
    except Exception:
        pred_probs = cls.predict(X).astype(float)

    out = df_proc.copy()
    out["pred_processing_hours"] = np.round(pred_hours.astype(float), 2)
    out["pred_delay_prob"]       = np.round(pred_probs.astype(float), 3)
    return out


# ---------------------------------------------------------------------------
# KPI helper
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame):
    if df.empty:
        return 0, 0.0, 0, 0.0, 0.0
    total        = len(df)
    avg_hours    = round(float(df["processing_time_hours"].mean()), 1)
    delayed      = int(pd.to_numeric(df["delayed"], errors="coerce").fillna(0).sum())
    delay_pct    = round(delayed / total * 100, 1)
    avg_ratio    = round(float(df["delay_ratio"].mean()), 3)
    return total, avg_hours, delayed, delay_pct, avg_ratio


def extract_first_stage(routing_str) -> str:
    try:
        return str(routing_str).split("->")[0]
    except Exception:
        return "Unknown"


# ===========================================================================
# LOAD RESOURCES
# ===========================================================================

df = load_data()
reg_model, xgb_model, cls_model, feature_cols_from_model, reg_meta, cls_meta = load_models()
FEATURE_COLS: list[str] = feature_cols_from_model or [
    c for c in df.columns if c not in FEATURE_DROP_COLS
]

# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("## 🏛️ FlowGov AI")
    st.markdown("*Government Workflow Intelligence*")
    st.divider()

    st.markdown("### Filters")
    departments = ["All"] + sorted(df["department"].dropna().unique().tolist())
    regions     = ["All"] + sorted(df["region"].dropna().unique().tolist()) \
                  if "region" in df.columns else ["All"]
    file_types  = ["All"] + sorted(df["file_type"].dropna().unique().tolist())
    priorities  = ["All"] + sorted(df["priority"].dropna().unique().tolist())

    sel_dept      = st.selectbox("Department",  departments)
    sel_region    = st.selectbox("Region",       regions) if "region" in df.columns else "All"
    sel_file_type = st.selectbox("File Type",    file_types)
    sel_priority  = st.selectbox("Priority",     priorities)

    date_min = df["submission_date"].min().date()
    date_max = df["submission_date"].max().date()
    sel_dates = st.date_input("Date range", [date_min, date_max])
    search_officer = st.text_input("Search Officer ID", "")

    st.divider()

    # Model metadata
    if reg_meta.get("trained_at"):
        st.markdown("### Model Info")
        st.caption(f"Trained: `{reg_meta['trained_at']}`")
        st.caption(f"Training rows: `{reg_meta.get('n_train_rows', '?'):,}`")
        m = reg_meta.get("metrics", {})
        if m:
            st.caption(f"Reg R²: `{m.get('r2', '?'):.3f}`")
        m2 = cls_meta.get("metrics", {})
        if m2:
            st.caption(f"Cls AUC: `{m2.get('roc_auc', '?'):.3f}`")


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

mask = pd.Series(True, index=df.index)
if sel_dept != "All":
    mask &= df["department"] == sel_dept
if "region" in df.columns and sel_region != "All":
    mask &= df["region"] == sel_region
if sel_file_type != "All":
    mask &= df["file_type"] == sel_file_type
if sel_priority != "All":
    mask &= df["priority"] == sel_priority
if isinstance(sel_dates, (list, tuple)) and len(sel_dates) == 2:
    s = pd.to_datetime(sel_dates[0])
    e = pd.to_datetime(sel_dates[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask &= (df["submission_date"] >= s) & (df["submission_date"] <= e)
if search_officer.strip():
    mask &= df["assigned_officer_id"].str.contains(search_officer.strip(), case=False, na=False)

df_f = df[mask].copy()

# ===========================================================================
# HEADER & KPIs
# ===========================================================================

st.markdown('<h1 style="font-size:1.8rem;font-weight:800;color:#f1f5f9;">🏛️ FlowGov AI — Workflow Dashboard</h1>', unsafe_allow_html=True)

total, avg_hours, delayed, delay_pct, avg_ratio = compute_kpis(df_f)

c1, c2, c3, c4, c5 = st.columns(5)
overall_delay_pct = round(df["delayed"].astype(int).mean() * 100, 1)
delta_delay = round(delay_pct - overall_delay_pct, 1)
c1.metric("Total Files",    f"{total:,}",            f"{total/len(df)*100:.0f}% of dataset")
c2.metric("Avg Processing", f"{avg_hours:.0f} hrs",  f"{avg_hours/24:.1f} days")
c3.metric("Delayed Files",  f"{delayed:,}",          f"{delay_pct:.1f}% (overall {overall_delay_pct}%)")
c4.metric("Avg SLA Usage",  f"{avg_ratio*100:.1f}%", delta=f"{(avg_ratio-0.68)*100:+.1f}% vs baseline")
c5.metric("On-Time Rate",   f"{100-delay_pct:.1f}%", delta=f"{-delta_delay:+.1f}% vs overall")

# ===========================================================================
# MAIN TABS
# ===========================================================================

tab_overview, tab_analytics, tab_predict, tab_features, tab_newfile = st.tabs([
    "📊 Overview",
    "🔍 Analytics",
    "🤖 Predictions",
    "📈 Feature Insights",
    "➕ Add New File",
])

# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================

with tab_overview:
    if df_f.empty:
        st.warning("No files match the current filters.")
    else:
        col_left, col_right = st.columns([3, 1])

        with col_left:
            # --- Processing time distribution ---
            st.markdown('<p class="section-header">Processing Time Distribution</p>', unsafe_allow_html=True)
            fig_hist = px.histogram(
                df_f, x="processing_time_hours", nbins=50,
                marginal="box",
                color_discrete_sequence=["#3b82f6"],
                labels={"processing_time_hours": "Processing Hours"},
            )
            fig_hist.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#94a3b8", showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10), height=320,
            )
            fig_hist.update_xaxes(gridcolor="#1e293b")
            fig_hist.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- Avg processing by dept & file type ---
            st.markdown('<p class="section-header">Avg Processing Time by Department & File Type</p>', unsafe_allow_html=True)
            agg = (
                df_f.groupby(["department", "file_type"])["processing_time_hours"]
                .mean().reset_index()
            )
            fig_bar = px.bar(
                agg, x="department", y="processing_time_hours",
                color="file_type", barmode="group",
                color_discrete_sequence=CHART_COLORS,
                labels={"processing_time_hours": "Avg Hours", "department": "Department"},
            )
            fig_bar.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#94a3b8", legend_title_text="File Type",
                margin=dict(l=10, r=10, t=20, b=10), height=320,
            )
            fig_bar.update_xaxes(gridcolor="#1e293b")
            fig_bar.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            # --- Delay pie ---
            st.markdown('<p class="section-header">Delay Status</p>', unsafe_allow_html=True)
            label_map = {True: "Delayed", False: "On-time", 1: "Delayed", 0: "On-time"}
            dl = df_f["delayed"].map(label_map).fillna("Unknown").value_counts().reset_index()
            dl.columns = ["Status", "Count"]
            fig_pie = px.pie(
                dl, names="Status", values="Count",
                color="Status", color_discrete_map=DELAY_COLOR,
                hole=0.5,
            )
            fig_pie.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#94a3b8", showlegend=True,
                margin=dict(l=10, r=10, t=10, b=10), height=260,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # --- Quick stats ---
            st.markdown('<p class="section-header">Quick Stats</p>', unsafe_allow_html=True)
            stats = {
                "Median pages": f"{int(df_f['num_pages'].median())}",
                "Avg approvals": f"{df_f['required_approvals'].mean():.1f}",
                "Avg experience": f"{df_f['officer_experience_years'].mean():.1f} yrs",
                "Avg backlog": f"{df_f['current_backlog_officer'].mean():.0f} files",
            }
            if "online_submission" in df_f.columns:
                stats["Online submissions"] = f"{df_f['online_submission'].mean()*100:.0f}%"
            if "incomplete_docs" in df_f.columns:
                stats["Incomplete docs"] = f"{df_f['incomplete_docs'].mean()*100:.0f}%"
            for k, v in stats.items():
                st.markdown(f"**{k}:** {v}")

        # --- Submission trend over time ---
        st.markdown('<p class="section-header">Monthly Filing Volume & Delay Rate</p>', unsafe_allow_html=True)
        trend = (
            df_f.assign(month=df_f["submission_date"].dt.to_period("M").astype(str))
            .groupby("month")
            .agg(
                total=("file_id", "count"),
                delayed_count=("delayed", "sum"),
            )
            .reset_index()
        )
        trend["delay_rate"] = trend["delayed_count"] / trend["total"] * 100

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=trend["month"], y=trend["total"],
            name="Total Files", marker_color="#3b82f6", opacity=0.7,
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend["month"], y=trend["delay_rate"],
            name="Delay Rate %", yaxis="y2",
            line=dict(color="#ef4444", width=2),
            mode="lines+markers",
        ))
        fig_trend.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Delay Rate %", range=[0, 80]),
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#94a3b8", height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_trend.update_xaxes(gridcolor="#1e293b")
        fig_trend.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_trend, use_container_width=True)


# ===========================================================================
# TAB 2: ANALYTICS
# ===========================================================================

with tab_analytics:
    if df_f.empty:
        st.warning("No files match the current filters.")
    else:
        col_a, col_b = st.columns(2)

        # --- Regional analysis ---
        if "region" in df_f.columns:
            with col_a:
                st.markdown('<p class="section-header">Delay Rate by Region</p>', unsafe_allow_html=True)
                reg_agg = (
                    df_f.groupby("region")
                    .agg(total=("file_id","count"), delayed=("delayed","sum"))
                    .reset_index()
                )
                reg_agg["delay_rate"] = reg_agg["delayed"] / reg_agg["total"] * 100
                fig_reg = px.bar(
                    reg_agg.sort_values("delay_rate", ascending=True),
                    x="delay_rate", y="region", orientation="h",
                    color="delay_rate",
                    color_continuous_scale="RdYlGn_r",
                    labels={"delay_rate": "Delay Rate %", "region": "Region"},
                )
                fig_reg.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font_color="#94a3b8", coloraxis_showscale=False,
                    height=280, margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_reg, use_container_width=True)

        # --- Priority breakdown ---
        with col_b:
            st.markdown('<p class="section-header">Processing Time by Priority</p>', unsafe_allow_html=True)
            fig_box = px.box(
                df_f, x="priority", y="processing_time_hours",
                color="priority",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
                category_orders={"priority": ["High", "Medium", "Low"]},
                labels={"processing_time_hours": "Hours", "priority": "Priority"},
            )
            fig_box.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#94a3b8", showlegend=False,
                height=280, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # --- Officer performance heatmap ---
        st.markdown('<p class="section-header">Top 20 Officers — Delay Rate Heatmap</p>', unsafe_allow_html=True)
        officer_agg = (
            df_f.groupby("assigned_officer_id")
            .agg(
                total=("file_id","count"),
                delayed_count=("delayed","sum"),
                avg_hours=("processing_time_hours","mean"),
                avg_exp=("officer_experience_years","mean"),
            )
            .reset_index()
        )
        officer_agg["delay_rate"] = officer_agg["delayed_count"] / officer_agg["total"] * 100
        top_officers = officer_agg.nlargest(20, "total")

        fig_off = px.scatter(
            top_officers,
            x="avg_exp", y="delay_rate",
            size="total", color="avg_hours",
            hover_name="assigned_officer_id",
            hover_data={"total": True, "delayed_count": True},
            color_continuous_scale="RdYlGn_r",
            labels={
                "avg_exp": "Avg Experience (yrs)",
                "delay_rate": "Delay Rate %",
                "avg_hours": "Avg Processing Hrs",
                "total": "Files Handled",
            },
        )
        fig_off.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#94a3b8", height=350,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_off, use_container_width=True)

        # --- Boolean flag impact ---
        bool_flags = [c for c in ["online_submission","incomplete_docs","resubmission","escalated"]
                      if c in df_f.columns]
        if bool_flags:
            st.markdown('<p class="section-header">Process Flag Impact on Delay Rate</p>', unsafe_allow_html=True)
            flag_rows = []
            for flag in bool_flags:
                for val in [0, 1]:
                    sub = df_f[df_f[flag] == val]
                    if len(sub) > 0:
                        flag_rows.append({
                            "Flag":       flag.replace("_", " ").title(),
                            "Value":      "Yes" if val == 1 else "No",
                            "Delay Rate": sub["delayed"].astype(int).mean() * 100,
                            "Count":      len(sub),
                        })
            flag_df = pd.DataFrame(flag_rows)
            fig_flags = px.bar(
                flag_df, x="Flag", y="Delay Rate", color="Value",
                barmode="group",
                color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
                text=flag_df["Delay Rate"].round(1),
                labels={"Delay Rate": "Delay Rate %"},
            )
            fig_flags.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_flags.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#94a3b8", height=300,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_flags, use_container_width=True)

        # --- Top officers by delay rate (min 20 files) ---
        st.markdown('<p class="section-header">Officers with Highest Delay Rate (min 20 files)</p>', unsafe_allow_html=True)
        officer_risk = officer_agg[officer_agg["total"] >= 20].nlargest(10, "delay_rate")[
            ["assigned_officer_id","total","delayed_count","delay_rate","avg_hours","avg_exp"]
        ].rename(columns={
            "assigned_officer_id": "Officer",
            "total": "Files",
            "delayed_count": "Delayed",
            "delay_rate": "Delay %",
            "avg_hours": "Avg Hrs",
            "avg_exp": "Avg Exp Yrs",
        })
        officer_risk["Delay %"]   = officer_risk["Delay %"].round(1)
        officer_risk["Avg Hrs"]   = officer_risk["Avg Hrs"].round(1)
        officer_risk["Avg Exp Yrs"] = officer_risk["Avg Exp Yrs"].round(1)
        st.dataframe(
            officer_risk.style.background_gradient(subset=["Delay %"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

        # --- Correlation heatmap ---
        st.markdown('<p class="section-header">Feature Correlation Matrix</p>', unsafe_allow_html=True)
        num_cols_heatmap = [
            c for c in [
                "complexity_score", "num_pages", "required_approvals",
                "officer_experience_years", "current_backlog_officer",
                "online_submission", "incomplete_docs", "resubmission",
                "escalated", "sla_days", "processing_time_hours", "delay_ratio",
            ] if c in df_f.columns
        ]
        corr = df_f[num_cols_heatmap].corr()
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 9},
        ))
        fig_heat.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#94a3b8", height=460,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # --- Seasonality charts ---
        seas_col1, seas_col2 = st.columns(2)
        if "submission_month" in df_f.columns:
            with seas_col1:
                st.markdown('<p class="section-header">Monthly Delay Rate (Seasonality)</p>', unsafe_allow_html=True)
                month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
                mo_agg = (
                    df_f.groupby("submission_month")
                    .agg(total=("file_id","count"), delayed=("delayed","sum"))
                    .reset_index()
                )
                mo_agg["delay_rate"] = mo_agg["delayed"] / mo_agg["total"] * 100
                mo_agg["month_name"] = mo_agg["submission_month"].map(month_names)
                fig_mo = px.bar(
                    mo_agg, x="month_name", y="delay_rate",
                    color="delay_rate", color_continuous_scale="RdYlGn_r",
                    labels={"delay_rate":"Delay Rate %","month_name":"Month"},
                    text=mo_agg["delay_rate"].round(1),
                )
                fig_mo.update_traces(texttemplate="%{text}%", textposition="outside")
                fig_mo.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font_color="#94a3b8", coloraxis_showscale=False,
                    height=280, margin=dict(l=10,r=10,t=30,b=10),
                    xaxis=dict(categoryorder="array",
                               categoryarray=list(month_names.values())),
                )
                st.plotly_chart(fig_mo, use_container_width=True)

        if "submission_weekday" in df_f.columns:
            with seas_col2:
                st.markdown('<p class="section-header">Weekday Delay Rate</p>', unsafe_allow_html=True)
                day_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
                wd_agg = (
                    df_f.groupby("submission_weekday")
                    .agg(total=("file_id","count"), delayed=("delayed","sum"))
                    .reset_index()
                )
                wd_agg["delay_rate"] = wd_agg["delayed"] / wd_agg["total"] * 100
                wd_agg["day_name"]   = wd_agg["submission_weekday"].map(day_names)
                fig_wd = px.bar(
                    wd_agg, x="day_name", y="delay_rate",
                    color="delay_rate", color_continuous_scale="RdYlGn_r",
                    labels={"delay_rate":"Delay Rate %","day_name":"Weekday"},
                    text=wd_agg["delay_rate"].round(1),
                )
                fig_wd.update_traces(texttemplate="%{text}%", textposition="outside")
                fig_wd.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font_color="#94a3b8", coloraxis_showscale=False,
                    height=280, margin=dict(l=10,r=10,t=30,b=10),
                    xaxis=dict(categoryorder="array",
                               categoryarray=list(day_names.values())),
                )
                st.plotly_chart(fig_wd, use_container_width=True)

        # --- Bottleneck: first routing stage ---
        st.markdown('<p class="section-header">Bottleneck: Avg Processing by First Routing Stage</p>', unsafe_allow_html=True)
        df_f_copy = df_f.copy()
        df_f_copy["first_stage"] = df_f_copy["routing_path"].apply(extract_first_stage)
        stage_agg = (
            df_f_copy.groupby("first_stage")["processing_time_hours"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_hours", "count": "total_files"})
            .sort_values("avg_hours", ascending=False)
        )
        fig_stage = px.bar(
            stage_agg, x="first_stage", y="avg_hours",
            color="avg_hours", color_continuous_scale="Reds",
            text="total_files",
            labels={"first_stage": "First Stage", "avg_hours": "Avg Hours"},
        )
        fig_stage.update_traces(texttemplate="%{text} files", textposition="outside")
        fig_stage.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#94a3b8", coloraxis_showscale=False,
            height=320, margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_stage, use_container_width=True)


# ===========================================================================
# TAB 3: PREDICTIONS
# ===========================================================================

with tab_predict:
    if df_f.empty:
        st.info("Adjust filters to load files for prediction.")
    else:
        pred_col1, pred_col2 = st.columns([1, 1])

        # --- Single-file prediction ---
        with pred_col1:
            st.markdown('<p class="section-header">Single-File Prediction</p>', unsafe_allow_html=True)
            file_options = df_f["file_id"].tolist()
            selected = st.selectbox("Select a file", options=file_options[:500])

            if selected:
                row = df.loc[df["file_id"] == selected].iloc[0]

                with st.expander("File details", expanded=True):
                    detail_cols = [
                        "department", "file_type", "priority", "complexity_score",
                        "num_pages", "required_approvals", "sla_days",
                    ]
                    detail_cols += [c for c in ["region", "online_submission", "incomplete_docs", "resubmission", "escalated"] if c in row.index]
                    st.json({k: (bool(v) if k in BOOL_COLS else v)
                             for k, v in row[detail_cols].to_dict().items()})

                if st.button("Run Prediction", type="primary"):
                    ph, prob = predict_single(reg_model, xgb_model, cls_model, row, FEATURE_COLS)
                    sla_h = float(row.get("sla_days", 7)) * 24
                    ratio = ph / sla_h

                    # Risk level
                    risk_label = "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.35 else "LOW")
                    risk_class = "risk-high" if prob > 0.6 else ("risk-medium" if prob > 0.35 else "risk-low")

                    st.success(f"Predicted processing time: **{ph:.1f} hrs** ({ph/24:.1f} days)")

                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        title={"text": "Delay Risk %", "font": {"color": "#f1f5f9"}},
                        delta={"reference": 34, "valueformat": ".1f"},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                            "bar": {"color": "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.35 else "#22c55e"},
                            "bgcolor": "#1e293b",
                            "steps": [
                                {"range": [0, 35],  "color": "#052e16"},
                                {"range": [35, 60], "color": "#451a03"},
                                {"range": [60, 100],"color": "#450a0a"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.85,
                                "value": prob * 100,
                            },
                        },
                        number={"suffix": "%", "font": {"color": "#f1f5f9"}},
                    ))
                    gauge.update_layout(
                        paper_bgcolor="#0f172a", font_color="#94a3b8",
                        height=280, margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(gauge, use_container_width=True)

                    st.markdown(f"""
                    | Metric | Value |
                    |--------|-------|
                    | SLA hours | {sla_h:.0f} h |
                    | Predicted hours | {ph:.1f} h |
                    | SLA utilisation | {ratio*100:.1f}% |
                    | Risk level | <span class="{risk_class}">{risk_label}</span> |
                    """, unsafe_allow_html=True)

                    if prob > 0.6:
                        st.warning("**Recommendation:** Escalate to senior officer or reassign to reduce backlog impact.")
                    elif prob > 0.35:
                        st.info("**Recommendation:** Monitor closely — check documentation completeness.")

        # --- Batch predictions ---
        with pred_col2:
            st.markdown('<p class="section-header">Batch Predictions</p>', unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            with b1:
                fast_mode  = st.checkbox("Fast mode (limit rows)", value=True)
                limit_rows = st.number_input("Max rows", 100, 5000, 500, 100) if fast_mode else None
            with b2:
                persist_now    = st.checkbox("Persist results", value=False)
                persist_target = st.selectbox("Target", ["CSV", "SQLite"])
                persist_mode   = st.selectbox("Mode",   ["append", "overwrite"])

            if st.button("Compute Batch Predictions", type="primary"):
                with st.spinner("Running predictions..."):
                    preds_df = predict_batch(
                        reg_model, xgb_model, cls_model, df_f,
                        FEATURE_COLS, max_rows=int(limit_rows) if fast_mode else None,
                    )
                st.session_state["preds_df"] = preds_df

            if "preds_df" in st.session_state:
                preds_df = st.session_state["preds_df"]
                show_cols = [c for c in [
                    "file_id","department","file_type","priority","submission_date",
                    "processing_time_hours","delayed","assigned_officer_id",
                ] if c in preds_df.columns] + ["pred_processing_hours","pred_delay_prob"]

                st.dataframe(
                    preds_df[show_cols].sort_values("submission_date", ascending=False).head(300),
                    use_container_width=True, height=320,
                )

                # Download
                st.download_button(
                    "Download predictions (CSV)",
                    data=preds_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions_export.csv",
                    mime="text/csv",
                )

                if persist_now:
                    sql_mode = "replace" if persist_mode == "overwrite" else "append"
                    if persist_target == "CSV":
                        path = save_predictions_csv(preds_df, mode=persist_mode)
                    else:
                        path = save_predictions_sqlite(preds_df, if_exists=sql_mode)
                    st.success(f"Saved to: `{path}`")

                # Prediction summary chart
                fig_pred_hist = px.histogram(
                    preds_df, x="pred_delay_prob", nbins=30,
                    color_discrete_sequence=["#ef4444"],
                    labels={"pred_delay_prob": "Predicted Delay Probability"},
                )
                fig_pred_hist.update_layout(
                    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                    font_color="#94a3b8", height=220,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_pred_hist, use_container_width=True)


# ===========================================================================
# TAB 4: FEATURE INSIGHTS
# ===========================================================================

with tab_features:
    fi_col1, fi_col2 = st.columns(2)

    with fi_col1:
        st.markdown('<p class="section-header">Processing Time — Feature Importance</p>', unsafe_allow_html=True)
        fi_reg = load_feature_importance("reg")
        if fi_reg.empty:
            st.warning("Run `train_models.py` to generate feature importance files.")
        else:
            fig_fi_reg = px.bar(
                fi_reg.head(15).sort_values("importance_mean"),
                x="importance_mean", y="feature",
                error_x="importance_std",
                orientation="h",
                color="importance_mean",
                color_continuous_scale="Blues",
                labels={"importance_mean": "Permutation Importance", "feature": "Feature"},
            )
            fig_fi_reg.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#94a3b8", coloraxis_showscale=False,
                height=420, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_fi_reg, use_container_width=True)
            with st.expander("Full importance table"):
                st.dataframe(fi_reg, use_container_width=True)

    with fi_col2:
        st.markdown('<p class="section-header">Delay Risk — Feature Importance</p>', unsafe_allow_html=True)
        fi_cls = load_feature_importance("cls")
        if fi_cls.empty:
            st.warning("Run `train_models.py` to generate feature importance files.")
        else:
            fig_fi_cls = px.bar(
                fi_cls.head(15).sort_values("importance_mean"),
                x="importance_mean", y="feature",
                error_x="importance_std",
                orientation="h",
                color="importance_mean",
                color_continuous_scale="Reds",
                labels={"importance_mean": "Permutation Importance", "feature": "Feature"},
            )
            fig_fi_cls.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#94a3b8", coloraxis_showscale=False,
                height=420, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_fi_cls, use_container_width=True)
            with st.expander("Full importance table"):
                st.dataframe(fi_cls, use_container_width=True)

    # --- Model performance summary ---
    st.markdown('<p class="section-header">Model Performance Summary</p>', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        rm = reg_meta.get("metrics", {})
        if rm:
            st.markdown("**Regression (Processing Time)**")
            st.metric("R² Score",  f"{rm.get('r2',0):.4f}")
            st.metric("RMSE",      f"{rm.get('rmse',0):.1f} hrs")
            st.metric("CV R² (5-fold)", f"{rm.get('r2_mean',0):.4f} ± {rm.get('r2_std',0):.4f}")
    with mc2:
        cm = cls_meta.get("metrics", {})
        if cm:
            st.markdown("**Classification (Delay Risk)**")
            st.metric("ROC-AUC",   f"{cm.get('roc_auc',0):.4f}")
            st.metric("F1-Score",  f"{cm.get('f1',0):.4f}")
            st.metric("CV AUC (5-fold)", f"{cm.get('roc_auc_mean',0):.4f} ± {cm.get('roc_auc_std',0):.4f}")


# ===========================================================================
# TAB 5: ADD NEW FILE
# ===========================================================================

with tab_newfile:
    nf_col1, nf_col2 = st.columns([1, 1])

    with nf_col1:
        st.markdown('<p class="section-header">Submit New File for Prediction</p>', unsafe_allow_html=True)

        with st.form("add_file_form"):
            f1, f2 = st.columns(2)
            with f1:
                new_dept     = st.selectbox("Department",   sorted(df["department"].unique()))
                new_ftype    = st.selectbox("File Type",    ["application","permit","appeal","report"])
                new_priority = st.selectbox("Priority",     ["Low","Medium","High"])
                new_region   = st.selectbox("Region",       ["North","South","East","West","Central"]) \
                               if "region" in df.columns else "Central"
            with f2:
                new_pages     = st.number_input("Num pages",        value=5,   min_value=1, max_value=300)
                new_complex   = st.slider("Complexity (0-1)",        0.0, 1.0, 0.2, 0.01)
                new_approvals = st.number_input("Required approvals", value=2, min_value=1, max_value=6)
                new_officer   = st.text_input("Officer ID",          "OFF001")

            f3, f4 = st.columns(2)
            with f3:
                new_exp      = st.number_input("Officer experience (yrs)", value=2.0, step=0.5)
                new_backlog  = st.number_input("Officer backlog",          value=5,   min_value=0)
            with f4:
                new_online   = st.checkbox("Online submission",   value=True)
                new_incdocs  = st.checkbox("Incomplete docs",     value=False)
                new_resub    = st.checkbox("Resubmission",        value=False)
                new_escalate = st.checkbox("Escalated",           value=False)

            persist_new = st.checkbox("Save to new_files.csv", value=True)
            submitted   = st.form_submit_button("Predict & Submit", type="primary")

    with nf_col2:
        if submitted:
            month   = datetime.now().month
            weekday = datetime.now().weekday()
            sla_map = {
                "application": {"Low": 7,  "Medium": 5,  "High": 3},
                "permit":      {"Low": 14, "Medium": 10, "High": 5},
                "appeal":      {"Low": 21, "Medium": 14, "High": 7},
                "report":      {"Low": 10, "Medium": 7,  "High": 4},
            }
            sla_d = sla_map[new_ftype][new_priority]
            stages = ["Clerk","Officer","SectionHead","Finance","Director"]
            routing = "->".join(stages[: min(len(stages), 1 + new_approvals)])

            new_row = {
                "file_id":                  str(uuid.uuid4()),
                "department":               new_dept,
                "file_type":                new_ftype,
                "priority":                 new_priority,
                "region":                   new_region,
                "submission_date":          datetime.now(),
                "submission_month":         month,
                "submission_weekday":       weekday,
                "complexity_score":         float(new_complex),
                "num_pages":                int(new_pages),
                "required_approvals":       int(new_approvals),
                "assigned_officer_id":      new_officer.strip(),
                "officer_experience_years": float(new_exp),
                "current_backlog_officer":  int(new_backlog),
                "online_submission":        int(new_online),
                "incomplete_docs":          int(new_incdocs),
                "resubmission":             int(new_resub),
                "escalated":                int(new_escalate),
                "routing_path":             routing,
                "sla_days":                 sla_d,
                "processing_time_hours":    None,
                "delayed":                  None,
                "delay_ratio":              None,
                "processing_time_days":     None,
            }

            ph, prob = predict_single(reg_model, xgb_model, cls_model, new_row, FEATURE_COLS)
            new_row["pred_processing_hours"] = round(ph, 2)
            new_row["pred_delay_prob"]       = round(prob, 3)

            risk = "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.35 else "LOW")
            risk_color = "#ef4444" if prob > 0.6 else "#f59e0b" if prob > 0.35 else "#22c55e"

            st.markdown(f"""
            ### Prediction Result

            | | |
            |---|---|
            | **Predicted Processing Time** | {ph:.1f} hrs ({ph/24:.1f} days) |
            | **Delay Probability** | {prob*100:.1f}% |
            | **Risk Level** | <span style="color:{risk_color};font-weight:bold">{risk}</span> |
            | **SLA** | {sla_d} days ({sla_d*24} hrs) |
            | **SLA Usage** | {ph/(sla_d*24)*100:.1f}% |
            """, unsafe_allow_html=True)

            gauge2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Delay Risk", "font": {"color": "#f1f5f9"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": risk_color},
                    "bgcolor": "#1e293b",
                    "steps": [
                        {"range": [0, 35],  "color": "#052e16"},
                        {"range": [35, 60], "color": "#451a03"},
                        {"range": [60, 100],"color": "#450a0a"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "#f1f5f9"}},
            ))
            gauge2.update_layout(
                paper_bgcolor="#0f172a", font_color="#94a3b8",
                height=250, margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(gauge2, use_container_width=True)

            if persist_new:
                path = save_new_file_csv(new_row)
                st.success(f"Saved: `{path}`")

        # --- Recently added files ---
        st.markdown('<p class="section-header">Recently Added Files</p>', unsafe_allow_html=True)
        new_files_df = load_new_files_csv()
        if not new_files_df.empty:
            st.write(f"Total: **{len(new_files_df)}** files")
            sort_col = "submission_date" if "submission_date" in new_files_df.columns else new_files_df.columns[0]
            st.dataframe(
                new_files_df.sort_values(sort_col, ascending=False).head(20),
                use_container_width=True, height=280,
            )
        else:
            st.info("No new files submitted yet.")
