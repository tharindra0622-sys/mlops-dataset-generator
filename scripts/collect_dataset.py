"""
Collect all individual JSON run logs into one master CSV dataset
Run this after downloading all artifacts from GitHub Actions
"""

import json
import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

INPUT_DIR  = "data/all_logs"
OUTPUT_CSV = "data/mlops_dataset.csv"
SUMMARY_FILE = "data/dataset_summary.txt"

# All expected feature columns in order
COLUMNS = [
    # Identity
    "pipeline_id", "pipeline_name", "trigger_event",
    "runner_os", "runner_arch", "python_version",
    # Timing
    "created_at", "started_at", "completed_at",
    "duration_sec", "queue_wait_sec",
    # Target
    "conclusion",
    # Failure info
    "failure_scenario", "failure_reason",
    # Data features
    "dataset_size", "dataset_row_count", "dataset_feature_count",
    "dataset_size_mb", "data_validation_passed",
    "missing_value_pct", "data_drift_injected", "schema_mismatch",
    "data_load_duration_sec", "data_split_ratio",
    # Model features
    "model_framework", "model_type", "algorithm_name",
    "training_duration_sec", "hyperparameter_count",
    "learning_rate", "n_estimators", "max_depth",
    "batch_size", "epochs_configured",
    "gpu_required", "gpu_available",
    # Eval features
    "eval_metric_name", "eval_metric_value", "eval_threshold",
    "eval_passed", "train_metric_value", "val_metric_value",
    "metric_gap", "overfitting_detected",
    # Log/error features
    "steps_with_errors", "has_oom_error", "has_import_error",
    "has_data_error", "has_schema_error", "has_convergence_warning",
    "has_threshold_fail", "has_gpu_error", "has_dependency_error",
    "log_total_lines", "max_step_duration_sec",
    # Engineered
    "random_seed",
]

def load_all_logs(input_dir):
    records = []
    json_files = glob.glob(f"{input_dir}/**/*.json", recursive=True)
    print(f"Found {len(json_files)} JSON log files")

    for fpath in json_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            print(f"  [WARN] Failed to load {fpath}: {e}")

    return records

def build_dataset(records):
    rows = []
    for rec in records:
        row = {col: rec.get(col, None) for col in COLUMNS}
        # Add binary label
        row["label"] = 1 if rec.get("conclusion") == "failure" else 0
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add engineered features
    df["duration_min"] = df["duration_sec"] / 60
    df["error_count"]  = (
        df["has_oom_error"].astype(int) +
        df["has_import_error"].astype(int) +
        df["has_data_error"].astype(int) +
        df["has_schema_error"].astype(int) +
        df["has_convergence_warning"].astype(int) +
        df["has_threshold_fail"].astype(int) +
        df["has_gpu_error"].astype(int) +
        df["has_dependency_error"].astype(int)
    )
    df["is_linux"]   = (df["runner_os"] == "linux").astype(int)
    df["is_windows"] = (df["runner_os"] == "windows").astype(int)
    df["is_macos"]   = (df["runner_os"] == "darwin").astype(int)
    df["is_rf"]      = (df["algorithm_name"] == "random_forest").astype(int)
    df["is_gb"]      = (df["algorithm_name"] == "gradient_boosting").astype(int)
    df["is_lr"]      = (df["algorithm_name"] == "logistic_regression").astype(int)

    # Historical fail rates per scenario
    scenario_fail_rate = df.groupby("failure_scenario")["label"].mean()
    df["scenario_hist_fail_rate"] = df["failure_scenario"].map(scenario_fail_rate)

    # Historical fail rates per model
    model_fail_rate = df.groupby("algorithm_name")["label"].mean()
    df["model_hist_fail_rate"] = df["algorithm_name"].map(model_fail_rate)

    return df

def print_summary(df):
    total    = len(df)
    failures = df["label"].sum()
    success  = total - failures

    summary = f"""
╔══════════════════════════════════════════════════════╗
║         MLOps Dataset Generation Summary             ║
╚══════════════════════════════════════════════════════╝

Generated    : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
Total runs   : {total:,}
Successes    : {success:,}  ({success/total*100:.1f}%)
Failures     : {failures:,}  ({failures/total*100:.1f}%)
Features     : {len(df.columns)}

── Failure breakdown by scenario ──
{df[df['label']==1]['failure_scenario'].value_counts().to_string()}

── Distribution by OS ──
{df['runner_os'].value_counts().to_string()}

── Distribution by model ──
{df['algorithm_name'].value_counts().to_string()}

── Conclusion counts ──
{df['conclusion'].value_counts().to_string()}
"""
    print(summary)
    return summary

def main():
    Path("data").mkdir(exist_ok=True)

    records = load_all_logs(INPUT_DIR)
    if not records:
        print(f"No logs found in {INPUT_DIR}")
        print("Make sure you've downloaded artifacts from GitHub Actions first")
        return

    df = build_dataset(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset saved to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")

    summary = print_summary(df)
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)

    # Also save a quick stats CSV
    stats = df.describe().round(4)
    stats.to_csv("data/dataset_stats.csv")
    print(f"Stats saved to data/dataset_stats.csv")

if __name__ == "__main__":
    main()
