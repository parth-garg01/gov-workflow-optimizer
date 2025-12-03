# src/dashboard/app.py
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- Helpers ----------
@st.cache_data
def load_data():
    # robust path: go up from this file to project root and find data/synthetic_files.csv
    data_path = Path(__file__).resolve().parents[2] / "data" / "synthetic_files.csv"
    df = pd.read_csv(data_path, parse_dates=["submission_date"])
    return df

def top_kpi(df):
    total = len(df)
    avg_hours = df["processing_time_hours"].mean().round(2)
    delayed_count = int(df["delayed"].sum())
    avg_delay_ratio = df["delay_ratio"].mean().round(3)
    return total, avg_hours, delayed_count, avg_delay_ratio

def extract_first_stage(routing_str):
    # routing_path like "Clerk->Officer->SectionHead"
    try:
        return str(routing_str).split("->")[0]
    except:
        return "Unknown"

# ---------- App UI ----------
st.set_page_config(page_title="Gov File Workflow Dashboard", layout="wide")
st.title("AI Workflow Optimization — Dashboard (Mockup)")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
departments = ["All"] + sorted(df["department"].dropna().unique().tolist())
file_types = ["All"] + sorted(df["file_type"].dropna().unique().tolist())
priorities = ["All"] + sorted(df["priority"].dropna().unique().tolist())

sel_dept = st.sidebar.selectbox("Department", departments, index=0)
sel_file_type = st.sidebar.selectbox("File Type", file_types, index=0)
sel_priority = st.sidebar.selectbox("Priority", priorities, index=0)
date_min = df["submission_date"].min()
date_max = df["submission_date"].max()
sel_date_range = st.sidebar.date_input("Submission date range", [date_min.date(), date_max.date()])

search_officer = st.sidebar.text_input("Search Officer ID (partial)", "")

# Apply filters
mask = pd.Series([True] * len(df))
if sel_dept != "All":
    mask &= df["department"] == sel_dept
if sel_file_type != "All":
    mask &= df["file_type"] == sel_file_type
if sel_priority != "All":
    mask &= df["priority"] == sel_priority
if sel_date_range and len(sel_date_range) == 2:
    start_dt = pd.to_datetime(sel_date_range[0])
    end_dt = pd.to_datetime(sel_date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    mask &= (df["submission_date"] >= start_dt) & (df["submission_date"] <= end_dt)
if search_officer.strip():
    mask &= df["assigned_officer_id"].str.contains(search_officer.strip(), case=False, na=False)

df_f = df[mask].copy()

# Top KPIs
total, avg_hours, delayed_count, avg_delay_ratio = top_kpi(df_f)
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
col1.metric("Total files", f"{total:,}")
col2.metric("Avg processing (hrs)", f"{avg_hours}")
col3.metric("Delayed files", f"{delayed_count}")
col4.metric("Avg delay ratio", f"{avg_delay_ratio}")

st.markdown("---")

# Main charts area
left, right = st.columns([2.5, 1])

with left:
    st.subheader("Processing time distribution")
    fig_hist = px.histogram(df_f, x="processing_time_hours", nbins=60,
                            title="Distribution of processing time (hours)", marginal="box")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Avg processing time by Department / File Type")
    agg = df_f.groupby(["department", "file_type"])["processing_time_hours"].mean().reset_index()
    fig_bar = px.bar(agg, x="department", y="processing_time_hours", color="file_type",
                     barmode="group", title="Avg processing time (hrs)")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Bottleneck: Avg processing time by First Routing Stage")
    df_f["first_stage"] = df_f["routing_path"].apply(extract_first_stage)
    stage_agg = df_f.groupby("first_stage")["processing_time_hours"].mean().reset_index().sort_values(by="processing_time_hours", ascending=False)
    fig_stage = px.bar(stage_agg, x="first_stage", y="processing_time_hours", title="Avg hrs by first stage")
    st.plotly_chart(fig_stage, use_container_width=True)

with right:
    st.subheader("Delay risk overview")
    delay_counts = df_f["delayed"].value_counts().rename_axis("delayed").reset_index(name="count")
    fig_pie = px.pie(delay_counts, names="delayed", values="count", title="Delayed vs On-time")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Quick stats")
    st.write("Median pages:", int(df_f["num_pages"].median()))
    st.write("Avg required approvals:", round(df_f["required_approvals"].mean(), 2))
    st.write("Avg officer experience (yrs):", round(df_f["officer_experience_years"].mean(), 2))
    st.write("Avg backlog per officer:", round(df_f["current_backlog_officer"].mean(), 2))

st.markdown("---")

# Correlation heatmap (numeric)
st.subheader("Numeric feature correlations (heatmap)")
numeric_cols = ["complexity_score", "num_pages", "required_approvals", "officer_experience_years",
                "current_backlog_officer", "processing_time_hours", "delay_ratio"]
corr = df_f[numeric_cols].corr()
fig_heat = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.index,
    colorscale="Viridis"
))
fig_heat.update_layout(height=400, margin=dict(l=40, r=40, t=40, b=40))
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# File list / table with simple actions
st.subheader("Files (sample)")
display_cols = ["file_id", "department", "file_type", "priority", "submission_date",
                "processing_time_hours", "delayed", "assigned_officer_id", "current_backlog_officer"]
st.dataframe(df_f[display_cols].sort_values(by="submission_date", ascending=False).head(200), use_container_width=True)

# Upload new file (basic) - placeholder for API integration later
st.sidebar.markdown("---")
st.sidebar.subheader("Add new file (manual)")
with st.sidebar.form("add_file"):
    new_ftype = st.selectbox("File type", ["application", "permit", "appeal", "report"])
    new_dept = st.selectbox("Department", sorted(df["department"].unique()))
    new_priority = st.selectbox("Priority", ["Low", "Medium", "High"])
    new_pages = st.number_input("Num pages", value=5, min_value=1)
    new_complex = st.slider("Complexity (0-1)", 0.0, 1.0, 0.2, 0.01)
    submit = st.form_submit_button("Add (mock)")
if submit:
    st.sidebar.success("File added (mock). In next step we'll wire this to FastAPI + model prediction.")

st.markdown("**Note:** This is a prototype mockup for visualization and manual testing. Next steps: wire to FastAPI endpoints, add live predictions, and include stage-level timing traces for Gantt/Sankey visualizations.")
