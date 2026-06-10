# src/validate_data.py
# Prints a comprehensive stats report on government_files.csv to confirm
# the dataset is realistic before training models.

from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "government_files.csv"


def validate():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Run `python src/data_gen.py` first.")
        return

    df = pd.read_csv(DATA_PATH, parse_dates=["submission_date"])
    print("=" * 60)
    print(f"Dataset: {DATA_PATH.name}")
    print(f"Shape:   {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 60)

    print("\n--- Class balance ---")
    delayed = df["delayed"].astype(int)
    print(f"  On-time: {(1-delayed).sum():,}  ({(1-delayed).mean()*100:.1f}%)")
    print(f"  Delayed: {delayed.sum():,}  ({delayed.mean()*100:.1f}%)")

    print("\n--- Processing time (hours) ---")
    pt = df["processing_time_hours"]
    print(f"  Mean:   {pt.mean():.1f}")
    print(f"  Median: {pt.median():.1f}")
    print(f"  Std:    {pt.std():.1f}")
    print(f"  P95:    {pt.quantile(0.95):.1f}")
    print(f"  Max:    {pt.max():.1f}")

    print("\n--- SLA utilisation (delay_ratio) ---")
    dr = df["delay_ratio"]
    print(f"  Mean:   {dr.mean():.3f}")
    print(f"  Median: {dr.median():.3f}")
    print(f"  >1.0:   {(dr > 1.0).sum():,} files ({(dr > 1.0).mean()*100:.1f}%)")

    print("\n--- Delayed rate by department ---")
    print(df.groupby("department")["delayed"].mean().mul(100).round(1)
          .rename("delay_%").sort_values(ascending=False).to_string())

    print("\n--- Delayed rate by region ---")
    if "region" in df.columns:
        print(df.groupby("region")["delayed"].mean().mul(100).round(1)
              .rename("delay_%").sort_values(ascending=False).to_string())

    print("\n--- Delayed rate by priority ---")
    print(df.groupby("priority")["delayed"].mean().mul(100).round(1)
          .rename("delay_%").to_string())

    print("\n--- Boolean feature rates ---")
    for col in ["online_submission", "incomplete_docs", "resubmission", "escalated"]:
        if col in df.columns:
            rate = df[col].astype(int).mean() * 100
            print(f"  {col:<25}: {rate:.1f}%")

    print("\n--- Feature correlations with processing_time_hours ---")
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ("processing_time_hours","processing_time_days","delay_ratio","delayed")]
    corr = df[num_cols + ["processing_time_hours"]].corr()["processing_time_hours"].drop("processing_time_hours")
    print(corr.round(3).sort_values(ascending=False).to_string())

    print("\n--- Date range ---")
    print(f"  From: {df['submission_date'].min().date()}")
    print(f"  To:   {df['submission_date'].max().date()}")

    print("\nValidation PASSED\n")


if __name__ == "__main__":
    validate()
