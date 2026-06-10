# FlowGov AI — Government File Processing Intelligence

A machine-learning system for predicting and optimising government file processing workflows.  
Built with a realistic 50,000-row dataset, LightGBM + XGBoost ensemble models, and an interactive Streamlit dashboard.

## Overview

| Capability | Description |
|------------|-------------|
| **Processing Time Regression** | Predict how many hours a file will take (LightGBM + XGBoost ensemble, R²=0.69) |
| **Delay Risk Classification** | Predict whether a file will breach SLA (LightGBM, ROC-AUC=0.797, F1=0.64) |
| **Interactive Dashboard** | 5-tab dark-theme UI: Overview, Analytics, Predictions, Feature Insights, New File |
| **Regional Analysis** | North/South/East/West/Central performance breakdown |
| **Officer Performance** | Experience vs delay-rate scatter, workload heatmap |

## Project Structure

```
.
├── data/                           # gitignored — regenerate locally
│   ├── government_files.csv        # 50,000-row realistic dataset (run data_gen.py)
│   ├── predictions.csv             # batch prediction outputs
│   ├── predictions.db              # SQLite store (optional)
│   └── new_files.csv               # form submissions
├── src/
│   ├── data_gen.py                 # Realistic dataset generator (50k rows, 22 features)
│   ├── validate_data.py            # Dataset statistics & sanity checker
│   ├── models/
│   │   ├── train_models.py         # Training pipeline (LGBM + XGBoost + 5-fold CV)
│   │   ├── feature_importance_reg.csv   # Permutation importance (regression)
│   │   └── feature_importance_cls.csv   # Permutation importance (classification)
│   └── dashboard/
│       └── app.py                  # Streamlit dashboard (5 tabs, dark theme)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the 50k-row realistic dataset
python src/data_gen.py

# 3. (Optional) Validate dataset statistics
python src/validate_data.py

# 4. Train models (LightGBM + XGBoost, ~5 min on CPU)
python src/models/train_models.py

# 5. Launch dashboard
streamlit run src/dashboard/app.py
```

## Dataset Schema (22 features + 2 targets)

| Column | Type | Description |
|--------|------|-------------|
| `file_id` | UUID | Unique file identifier |
| `department` | String | Revenue / Transport / Health / Education / Welfare / UrbanDev |
| `file_type` | String | application / permit / appeal / report |
| `priority` | String | Low / Medium / High |
| `region` | String | North / South / East / West / Central |
| `submission_date` | DateTime | When the file was submitted |
| `submission_month` | Int | Month (1–12), captures filing peaks |
| `submission_weekday` | Int | Day of week (0=Mon), captures workload patterns |
| `complexity_score` | Float | 0–1 score (Beta-distributed) |
| `num_pages` | Int | Pages in document |
| `required_approvals` | Int | Number of approval stages (1–6) |
| `assigned_officer_id` | String | Processing officer identifier |
| `officer_experience_years` | Float | Years on the job |
| `current_backlog_officer` | Int | Officer's current queue depth |
| `online_submission` | Bool | Submitted online (62% = faster) |
| `incomplete_docs` | Bool | Missing documentation (22% = +50% time) |
| `resubmission` | Bool | Previously rejected (14% = +30% time) |
| `escalated` | Bool | Escalated to senior officer (10% = +40% time) |
| `routing_path` | String | Processing stages (Clerk->Officer->...) |
| `sla_days` | Int | SLA deadline in days |
| **`processing_time_hours`** | Float | **Target: Regression** |
| **`delayed`** | Bool | **Target: Classification** |

### Realistic Dataset Properties
- **50,000 rows** spanning 2 years (Jan 2023 – Dec 2024)
- **34.3% delayed** — well-balanced for classification
- **Seasonal patterns**: Jan/Apr/Oct/Dec filing peaks (1.15–1.22× multiplier)
- **Regional variation**: West (+12% slower) vs Central (–5% faster)
- **Heteroscedastic noise**: std = 18% of predicted time + 3h floor
- **1.5% extreme outliers** (lost files, bureaucratic bottlenecks)

## Model Performance

| Model | Algorithm | Val Metric | CV (5-fold) |
|-------|-----------|------------|-------------|
| Processing Time | LGBM + XGBoost ensemble | R²=0.688, RMSE=71h | R²=0.691 ± 0.005 |
| Delay Risk | LightGBM classifier | ROC-AUC=0.797, F1=0.643 | AUC=0.793 ± 0.005 |

Both CV and holdout metrics agree — no data leakage.

## Dashboard Tabs

| Tab | Contents |
|-----|---------|
| **Overview** | KPI cards, processing-time histogram, dept/file-type bar chart, monthly trend |
| **Analytics** | Regional delay rates, priority box-plots, officer performance scatter, correlation heatmap |
| **Predictions** | Single-file risk gauge + SLA utilisation, batch prediction with download |
| **Feature Insights** | Permutation importance charts (regression + classification), model metrics |
| **Add New File** | 22-field form, instant risk gauge prediction, submission history |

## License

MIT License
