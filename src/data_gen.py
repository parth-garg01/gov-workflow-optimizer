# src/data_gen.py
# Overhauled: realistic government-file dataset with 50 000 rows, 22 features,
# seasonal patterns, regional variation, and proper statistical properties.

from pathlib import Path
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Config tables
# ---------------------------------------------------------------------------

DEPARTMENTS = ["Revenue", "Transport", "Health", "Education", "Welfare", "UrbanDev"]
DEPT_WEIGHTS = [0.20, 0.15, 0.15, 0.20, 0.20, 0.10]

FILE_TYPES = ["application", "permit", "appeal", "report"]
FILE_TYPE_WEIGHTS = [0.45, 0.20, 0.15, 0.20]

PRIORITIES = ["Low", "Medium", "High"]
PRIORITY_WEIGHTS = [0.60, 0.30, 0.10]

REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHTS = [0.22, 0.18, 0.20, 0.15, 0.25]

STAGES = ["Clerk", "Officer", "SectionHead", "Finance", "Director"]

# SLA (days) by file_type × priority
SLA_MAP = {
    "application": {"Low": 7,  "Medium": 5,  "High": 3},
    "permit":      {"Low": 14, "Medium": 10, "High": 5},
    "appeal":      {"Low": 21, "Medium": 14, "High": 7},
    "report":      {"Low": 10, "Medium": 7,  "High": 4},
}

# Base processing multipliers per department
DEPT_FACTOR = {
    "Revenue": 1.05, "Transport": 1.00, "Health": 1.10,
    "Education": 0.95, "Welfare": 1.08, "UrbanDev": 1.12,
}

# Regional efficiency multipliers (< 1 = faster, > 1 = slower)
REGION_FACTOR = {
    "North": 1.08, "South": 0.97, "East": 1.02,
    "West": 1.12, "Central": 0.95,
}

# Month seasonality multipliers (government filing peaks: Jan, Apr, Oct, Dec)
MONTH_FACTOR = {
    1: 1.18, 2: 0.95, 3: 0.92, 4: 1.15, 5: 0.90, 6: 0.88,
    7: 0.85, 8: 0.87, 9: 1.05, 10: 1.20, 11: 1.08, 12: 1.22,
}

# Weekday factor: Mon=0 heavy, Fri=4 light
WEEKDAY_FACTOR = {0: 1.12, 1: 1.00, 2: 0.98, 3: 0.97, 4: 0.93, 5: 0.80, 6: 0.78}


def _weighted_choice(rng: np.random.Generator, choices, weights):
    weights = np.array(weights, dtype=float)
    return rng.choice(choices, p=weights / weights.sum())


def generate_dataset(
    n_rows: int = 50_000,
    seed: int = 42,
    save_path: Path = None,
) -> pd.DataFrame:
    """
    Generate a realistic government file workflow dataset and save to disk.

    New features vs. original synthetic generator:
      - region (5 values with regional efficiency multipliers)
      - online_submission (bool, 62% online = faster)
      - incomplete_docs (bool, 22% = significant delay penalty)
      - resubmission (bool, 14% = already rejected once)
      - escalated (bool, 9% = escalated to senior officer)
      - submission_month / submission_weekday derived from submission_date
      - month & weekday seasonality factored into processing time
      - heteroscedastic noise (variance proportional to complexity)
      - realistic outliers (1% extreme delay cases)
      - ~35% delayed rate (calibrated for balanced classification)
    """
    if save_path is None:
        save_path = Path(__file__).resolve().parent.parent / "data" / "government_files.csv"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    base_date = datetime(2023, 1, 1)

    rows = []
    for _ in range(n_rows):
        # ---- Categorical draws ----
        dept     = _weighted_choice(rng, DEPARTMENTS,  DEPT_WEIGHTS)
        ftype    = _weighted_choice(rng, FILE_TYPES,   FILE_TYPE_WEIGHTS)
        priority = _weighted_choice(rng, PRIORITIES,   PRIORITY_WEIGHTS)
        region   = _weighted_choice(rng, REGIONS,      REGION_WEIGHTS)

        # ---- Submission date (2 years, realistic exponential gap) ----
        day_offset = int(rng.exponential(scale=180) % 730)
        submission_date = base_date + timedelta(days=day_offset)
        month   = submission_date.month
        weekday = submission_date.weekday()

        # ---- File attributes ----
        complexity_score   = float(np.clip(rng.beta(2, 5), 0.01, 0.99))
        num_pages          = int(np.clip(rng.normal(5 + complexity_score * 12, 4), 1, 300))
        required_approvals = int(np.clip(rng.poisson(1.2 + complexity_score * 2.5), 1, 6))

        # ---- Officer attributes ----
        n_officers = 200
        officer_id = f"OFF{rng.integers(1, n_officers + 1):03d}"
        exp_base   = 3.0 - 0.6 * (priority == "High")
        officer_exp = float(np.clip(rng.normal(exp_base, 1.8), 0.5, 12.0))

        # Backlog: dept and region influence workload
        backlog_lambda = 6 + complexity_score * 10 + (1.5 if region == "West" else 0)
        current_backlog = int(np.clip(rng.poisson(backlog_lambda), 0, 200))

        # ---- Binary feature flags ----
        online_submission = bool(rng.random() < 0.62)
        incomplete_docs   = bool(rng.random() < 0.22)
        resubmission      = bool(rng.random() < 0.14)
        escalated         = bool(rng.random() < (0.09 + 0.10 * (priority == "High")))

        # ---- SLA and routing ----
        sla_days     = SLA_MAP[ftype][priority]
        routing_path = "->".join(STAGES[: min(len(STAGES), 1 + required_approvals)])

        # ---- Processing time formula ----
        base_hours = (
            2.0
            + 0.6 * num_pages
            + 12.0 * complexity_score
            + 9.0 * (required_approvals - 1)
        )

        # Modifiers
        backlog_penalty      = 0.25 * current_backlog
        exp_reduction        = max(0.0, (officer_exp - 1.5) * -1.8)
        priority_adj         = {"Low": 1.0, "Medium": 0.88, "High": 0.72}[priority]
        dept_mult            = DEPT_FACTOR[dept]
        region_mult          = REGION_FACTOR[region]
        month_mult           = MONTH_FACTOR[month]
        weekday_mult         = WEEKDAY_FACTOR[weekday]
        online_factor        = 0.88 if online_submission else 1.0
        incomplete_factor    = 1.45 if incomplete_docs   else 1.0
        resubmission_factor  = 1.25 if resubmission      else 1.0
        escalation_factor    = 1.35 if escalated         else 1.0

        combined = (
            (base_hours + backlog_penalty + exp_reduction)
            * priority_adj
            * dept_mult
            * region_mult
            * month_mult
            * weekday_mult
            * online_factor
            * incomplete_factor
            * resubmission_factor
            * escalation_factor
        )

        # Heteroscedastic noise: variance proportional to combined (realistic)
        noise_std = 0.15 * combined + 5.0
        noise     = float(rng.normal(0, noise_std))

        # 1% chance of extreme outlier (bureaucratic bottleneck, lost file, etc.)
        if rng.random() < 0.01:
            noise += float(rng.uniform(50, 200))

        processing_time_hours = float(max(1.0, combined + noise))
        sla_hours             = sla_days * 24
        delay_ratio           = processing_time_hours / max(1, sla_hours)

        # Delay: based on delay_ratio but with additional independent factors
        # This ensures the classifier has to learn a non-trivial decision boundary
        base_delay_prob = 1.0 / (1.0 + np.exp(-5.0 * (delay_ratio - 0.75)))
        adj_delay_prob  = float(np.clip(
            base_delay_prob
            + 0.12 * incomplete_docs
            + 0.08 * resubmission
            + 0.06 * escalated
            - 0.05 * online_submission
            + 0.001 * current_backlog,
            0.01, 0.99,
        ))
        delayed = bool(rng.random() < adj_delay_prob)

        rows.append({
            "file_id":                  str(uuid.uuid4()),
            "department":               dept,
            "file_type":                ftype,
            "priority":                 priority,
            "region":                   region,
            "submission_date":          submission_date,
            "submission_month":         month,
            "submission_weekday":       weekday,
            "complexity_score":         round(complexity_score, 3),
            "num_pages":                num_pages,
            "required_approvals":       required_approvals,
            "assigned_officer_id":      officer_id,
            "officer_experience_years": round(officer_exp, 1),
            "current_backlog_officer":  current_backlog,
            "online_submission":        online_submission,
            "incomplete_docs":          incomplete_docs,
            "resubmission":             resubmission,
            "escalated":                escalated,
            "routing_path":             routing_path,
            "sla_days":                 sla_days,
            "processing_time_hours":    round(processing_time_hours, 2),
            "delayed":                  delayed,
            "delay_ratio":              round(float(delay_ratio), 4),
        })

    df = pd.DataFrame(rows)
    df["processing_time_days"] = (df["processing_time_hours"] / 24).round(3)

    df.to_csv(save_path, index=False)
    delayed_pct = df["delayed"].mean() * 100
    print(f"Dataset saved to: {save_path}  ({len(df):,} rows, {delayed_pct:.1f}% delayed)")
    return df


if __name__ == "__main__":
    generate_dataset()
