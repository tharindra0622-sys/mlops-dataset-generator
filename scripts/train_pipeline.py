"""
MLOps Training Pipeline — with controlled failure injection
Logs all features needed for failure prediction research dataset
"""

import os
import sys
import json
import time
import random
import argparse
import traceback
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_failure_study")
FAILURE_SCENARIO    = os.getenv("FAILURE_SCENARIO", "none")
RANDOM_SEED         = int(os.getenv("RANDOM_SEED", str(random.randint(0, 9999))))
EVAL_THRESHOLD      = float(os.getenv("EVAL_THRESHOLD", "0.70"))
DATASET_SIZE        = int(os.getenv("DATASET_SIZE", "5000"))
MODEL_TYPE          = os.getenv("MODEL_TYPE", "random_forest")
TASK_TYPE           = os.getenv("TASK_TYPE", "classification")
LEARNING_RATE       = float(os.getenv("LEARNING_RATE", "0.1"))
N_ESTIMATORS        = int(os.getenv("N_ESTIMATORS", "100"))
MAX_DEPTH           = int(os.getenv("MAX_DEPTH", "5"))
TEST_SIZE           = float(os.getenv("TEST_SIZE", "0.2"))
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS              = int(os.getenv("EPOCHS", "10"))
GPU_REQUIRED        = os.getenv("GPU_REQUIRED", "false").lower() == "true"
INJECT_DRIFT        = os.getenv("INJECT_DRIFT", "false").lower() == "true"
INJECT_MISSING      = float(os.getenv("INJECT_MISSING", "0.0"))
WRONG_SCHEMA        = os.getenv("WRONG_SCHEMA", "false").lower() == "true"
FORCE_OOM           = os.getenv("FORCE_OOM", "false").lower() == "true"
MIN_ROWS_REQUIRED   = int(os.getenv("MIN_ROWS_REQUIRED", "100"))

# ── Feature log collector ─────────────────────────────────────────────
RUN_LOG = {
    "pipeline_id":              os.getenv("GITHUB_RUN_ID", f"local_{int(time.time())}"),
    "pipeline_name":            os.getenv("GITHUB_WORKFLOW", "mlops_pipeline"),
    "trigger_event":            os.getenv("GITHUB_EVENT_NAME", "manual"),
    "runner_os":                platform.system().lower(),
    "runner_arch":              platform.machine(),
    "python_version":           platform.python_version(),
    "created_at":               datetime.utcnow().isoformat(),
    "started_at":               None,
    "completed_at":             None,
    "duration_sec":             None,
    "queue_wait_sec":           float(os.getenv("QUEUE_WAIT_SEC", "0")),
    "conclusion":               None,
    "failure_scenario":         FAILURE_SCENARIO,
    "random_seed":              RANDOM_SEED,
    # Data features
    "dataset_size":             DATASET_SIZE,
    "dataset_row_count":        None,
    "dataset_feature_count":    None,
    "dataset_size_mb":          None,
    "data_validation_passed":   None,
    "missing_value_pct":        INJECT_MISSING,
    "data_drift_injected":      INJECT_DRIFT,
    "schema_mismatch":          WRONG_SCHEMA,
    "data_load_duration_sec":   None,
    "data_split_ratio":         1.0 - TEST_SIZE,
    # Model features
    "model_framework":          "sklearn",
    "model_type":               TASK_TYPE,
    "algorithm_name":           MODEL_TYPE,
    "training_duration_sec":    None,
    "hyperparameter_count":     3,
    "learning_rate":            LEARNING_RATE,
    "n_estimators":             N_ESTIMATORS,
    "max_depth":                MAX_DEPTH,
    "batch_size":               BATCH_SIZE,
    "epochs_configured":        EPOCHS,
    "gpu_required":             GPU_REQUIRED,
    "gpu_available":            False,
    # Eval features
    "eval_metric_name":         "accuracy" if TASK_TYPE == "classification" else "r2",
    "eval_metric_value":        None,
    "eval_threshold":           EVAL_THRESHOLD,
    "eval_passed":              None,
    "train_metric_value":       None,
    "val_metric_value":         None,
    "metric_gap":               None,
    "overfitting_detected":     None,
    # Log/error features
    "steps_with_errors":        0,
    "has_oom_error":            False,
    "has_import_error":         False,
    "has_data_error":           False,
    "has_schema_error":         False,
    "has_convergence_warning":  False,
    "has_threshold_fail":       False,
    "has_gpu_error":            False,
    "has_dependency_error":     False,
    "log_total_lines":          0,
    "max_step_duration_sec":    0,
}

def log(msg):
    print(msg, flush=True)
    RUN_LOG["log_total_lines"] += 1

def fail(reason, error_type):
    RUN_LOG["steps_with_errors"] += 1
    RUN_LOG[error_type] = True
    RUN_LOG["conclusion"] = "failure"
    RUN_LOG["failure_reason"] = reason
    log(f"[FAILURE] {reason}")
    return False

def step_time(start):
    d = time.time() - start
    RUN_LOG["max_step_duration_sec"] = max(RUN_LOG["max_step_duration_sec"], d)
    return d

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Environment validation
# ══════════════════════════════════════════════════════════════════════
def step_validate_environment():
    log("=" * 60)
    log("STEP 1: Validating environment")
    t = time.time()

    log(f"  Python     : {platform.python_version()}")
    log(f"  OS         : {platform.system()} {platform.release()}")
    log(f"  Arch       : {platform.machine()}")
    log(f"  Scenario   : {FAILURE_SCENARIO}")

    # Check GPU requirement
    if GPU_REQUIRED:
        try:
            import torch
            if not torch.cuda.is_available():
                RUN_LOG["has_gpu_error"] = True
                return fail("GPU required but not available on this runner", "has_gpu_error")
        except ImportError:
            RUN_LOG["has_gpu_error"] = True
            return fail("GPU required but torch not installed", "has_gpu_error")

    # Simulate dependency conflict
    if FAILURE_SCENARIO == "dependency_conflict":
        RUN_LOG["has_dependency_error"] = True
        return fail(
            "Dependency conflict: numpy==1.20 required but numpy==1.24 installed",
            "has_dependency_error"
        )

    step_time(t)
    log("  Environment OK")
    return True


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Data loading & validation
# ══════════════════════════════════════════════════════════════════════
def step_load_data():
    log("=" * 60)
    log("STEP 2: Loading and validating data")
    t = time.time()

    # Missing data file scenario
    if FAILURE_SCENARIO == "missing_data_file":
        RUN_LOG["has_data_error"] = True
        return fail(
            "FileNotFoundError: dataset.csv not found at data/dataset.csv",
            "has_data_error"
        )

    # Generate synthetic dataset
    np.random.seed(RANDOM_SEED)
    n_features = random.randint(8, 25)

    if TASK_TYPE == "classification":
        X, y = make_classification(
            n_samples=DATASET_SIZE,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=2,
            random_state=RANDOM_SEED
        )
    else:
        X, y = make_regression(
            n_samples=DATASET_SIZE,
            n_features=n_features,
            noise=0.1,
            random_state=RANDOM_SEED
        )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y

    # Inject data drift
    if INJECT_DRIFT:
        log("  [WARN] Injecting data drift — shifting feature distributions")
        df.iloc[:, :3] = df.iloc[:, :3] + np.random.normal(5, 2, (len(df), 3))
        RUN_LOG["has_convergence_warning"] = True

    # Inject missing values
    if INJECT_MISSING > 0:
        mask = np.random.rand(*df.shape) < INJECT_MISSING
        df[mask] = np.nan
        missing_pct = df.isnull().sum().sum() / df.size
        RUN_LOG["missing_value_pct"] = round(missing_pct, 4)
        log(f"  [WARN] Missing values injected: {missing_pct*100:.1f}%")

    # Schema mismatch scenario
    if WRONG_SCHEMA or FAILURE_SCENARIO == "schema_mismatch":
        df.columns = [f"wrong_col_{i}" for i in range(len(df.columns))]
        RUN_LOG["has_schema_error"] = True
        return fail(
            "SchemaError: Expected columns ['feature_0'...'feature_N'] "
            "but got ['wrong_col_0'...'wrong_col_N']",
            "has_schema_error"
        )

    # Data validation checks
    if len(df) < MIN_ROWS_REQUIRED:
        RUN_LOG["has_data_error"] = True
        return fail(
            f"DataValidationError: Dataset has {len(df)} rows, "
            f"minimum required is {MIN_ROWS_REQUIRED}",
            "has_data_error"
        )

    if df.isnull().sum().sum() / df.size > 0.5:
        RUN_LOG["has_data_error"] = True
        return fail(
            "DataValidationError: More than 50% missing values detected",
            "has_data_error"
        )

    # OOM simulation
    if FORCE_OOM or FAILURE_SCENARIO == "memory_overflow":
        RUN_LOG["has_oom_error"] = True
        return fail(
            "MemoryError: Unable to allocate 128 GiB for dataset array",
            "has_oom_error"
        )

    # Log data stats
    d = step_time(t)
    RUN_LOG["dataset_row_count"]     = len(df)
    RUN_LOG["dataset_feature_count"] = n_features
    RUN_LOG["dataset_size_mb"]       = round(df.memory_usage(deep=True).sum() / 1e6, 3)
    RUN_LOG["data_validation_passed"] = True
    RUN_LOG["data_load_duration_sec"] = round(d, 3)

    log(f"  Rows       : {len(df):,}")
    log(f"  Features   : {n_features}")
    log(f"  Size       : {RUN_LOG['dataset_size_mb']} MB")
    log(f"  Duration   : {d:.2f}s")
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Feature engineering & splitting
# ══════════════════════════════════════════════════════════════════════
def step_feature_engineering(df):
    log("=" * 60)
    log("STEP 3: Feature engineering & splitting")
    t = time.time()

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    step_time(t)
    log(f"  Train size : {len(X_train):,}")
    log(f"  Test size  : {len(X_test):,}")
    return X_train_sc, X_test_sc, y_train, y_test


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Model training
# ══════════════════════════════════════════════════════════════════════
def step_train_model(X_train, y_train):
    log("=" * 60)
    log(f"STEP 4: Training model — {MODEL_TYPE}")
    t = time.time()

    # Wrong hyperparameter scenario
    if FAILURE_SCENARIO == "wrong_hyperparameters":
        n_est = -1   # invalid
        lr    = 999  # absurd
        log(f"  [WARN] Using invalid hyperparameters: n_estimators={n_est}, lr={lr}")

    if TASK_TYPE == "classification":
        models = {
            "random_forest":      RandomForestClassifier(
                                    n_estimators=N_ESTIMATORS,
                                    max_depth=MAX_DEPTH,
                                    random_state=RANDOM_SEED
                                  ),
            "gradient_boosting":  GradientBoostingClassifier(
                                    n_estimators=N_ESTIMATORS,
                                    learning_rate=LEARNING_RATE,
                                    max_depth=MAX_DEPTH,
                                    random_state=RANDOM_SEED
                                  ),
            "logistic_regression": LogisticRegression(
                                    C=1.0/LEARNING_RATE if LEARNING_RATE > 0 else 1.0,
                                    max_iter=1000,
                                    random_state=RANDOM_SEED
                                   ),
            "decision_tree":      DecisionTreeClassifier(
                                    max_depth=MAX_DEPTH,
                                    random_state=RANDOM_SEED
                                  ),
        }
    else:
        models = {
            "ridge":              Ridge(alpha=1.0/LEARNING_RATE if LEARNING_RATE > 0 else 1.0),
            "random_forest":      RandomForestClassifier(
                                    n_estimators=N_ESTIMATORS,
                                    random_state=RANDOM_SEED
                                  ),
        }

    model = models.get(MODEL_TYPE, models[list(models.keys())[0]])

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        RUN_LOG["has_convergence_warning"] = True
        return fail(f"TrainingError: {str(e)}", "has_convergence_warning")

    d = step_time(t)
    RUN_LOG["training_duration_sec"] = round(d, 3)

    # Simulate divergence for wrong_hyperparameters
    if FAILURE_SCENARIO == "wrong_hyperparameters":
        RUN_LOG["has_convergence_warning"] = True
        log("  [WARN] Model diverged — loss = NaN after epoch 1")

    log(f"  Training complete in {d:.2f}s")
    return model


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Evaluation
# ══════════════════════════════════════════════════════════════════════
def step_evaluate(model, X_train, X_test, y_train, y_test):
    log("=" * 60)
    log("STEP 5: Evaluating model")
    t = time.time()

    if TASK_TYPE == "classification":
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)
        train_score = accuracy_score(y_train, train_pred)
        test_score  = accuracy_score(y_test,  test_pred)
        metric_name = "accuracy"
    else:
        train_pred  = model.predict(X_train)
        test_pred   = model.predict(X_test)
        train_score = r2_score(y_train, train_pred)
        test_score  = r2_score(y_test,  test_pred)
        metric_name = "r2"

    # Inject divergence scenario
    if FAILURE_SCENARIO == "wrong_hyperparameters":
        test_score  = random.uniform(0.3, 0.45)
        train_score = random.uniform(0.3, 0.45)

    gap = round(abs(train_score - test_score), 4)
    overfitting = gap > 0.15

    RUN_LOG["train_metric_value"]    = round(train_score, 4)
    RUN_LOG["val_metric_value"]      = round(test_score,  4)
    RUN_LOG["eval_metric_value"]     = round(test_score,  4)
    RUN_LOG["eval_metric_name"]      = metric_name
    RUN_LOG["metric_gap"]            = gap
    RUN_LOG["overfitting_detected"]  = overfitting

    if overfitting:
        log(f"  [WARN] Overfitting detected — gap={gap:.4f}")
        RUN_LOG["has_convergence_warning"] = True

    passed = test_score >= EVAL_THRESHOLD

    # Threshold fail scenario
    if FAILURE_SCENARIO == "eval_threshold_fail":
        passed = False
        RUN_LOG["eval_metric_value"] = round(random.uniform(0.30, 0.55), 4)
        log(f"  [FAIL] Metric {RUN_LOG['eval_metric_value']:.4f} < threshold {EVAL_THRESHOLD}")

    RUN_LOG["eval_passed"] = passed

    step_time(t)
    log(f"  Train {metric_name}: {train_score:.4f}")
    log(f"  Test  {metric_name}: {test_score:.4f}")
    log(f"  Threshold       : {EVAL_THRESHOLD}")
    log(f"  Passed          : {passed}")

    if not passed:
        RUN_LOG["has_threshold_fail"] = True
        return fail(
            f"EvaluationError: {metric_name}={test_score:.4f} "
            f"below threshold {EVAL_THRESHOLD}",
            "has_threshold_fail"
        )
    return True


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — MLflow logging
# ══════════════════════════════════════════════════════════════════════
def step_log_mlflow(model):
    log("=" * 60)
    log("STEP 6: Logging to MLflow")
    t = time.time()

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{MODEL_TYPE}_{FAILURE_SCENARIO}_{RANDOM_SEED}"):
            # Log params
            mlflow.log_params({
                "model_type":       MODEL_TYPE,
                "task_type":        TASK_TYPE,
                "learning_rate":    LEARNING_RATE,
                "n_estimators":     N_ESTIMATORS,
                "max_depth":        MAX_DEPTH,
                "dataset_size":     DATASET_SIZE,
                "random_seed":      RANDOM_SEED,
                "failure_scenario": FAILURE_SCENARIO,
                "eval_threshold":   EVAL_THRESHOLD,
            })

            # Log metrics
            if RUN_LOG.get("eval_metric_value") is not None:
                mlflow.log_metric("test_metric",  RUN_LOG["eval_metric_value"])
                mlflow.log_metric("train_metric", RUN_LOG.get("train_metric_value", 0))
                mlflow.log_metric("metric_gap",   RUN_LOG.get("metric_gap", 0))

            # Log model if eval passed
            if RUN_LOG.get("eval_passed"):
                mlflow.sklearn.log_model(model, "model")

        log("  MLflow logging complete")
    except Exception as e:
        log(f"  [WARN] MLflow logging failed: {e}")

    step_time(t)
    return True


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    RUN_LOG["started_at"] = datetime.utcnow().isoformat()
    pipeline_start = time.time()
    log(f"MLOps Pipeline started at {RUN_LOG['started_at']}")
    log(f"Scenario: {FAILURE_SCENARIO} | Model: {MODEL_TYPE} | Seed: {RANDOM_SEED}")

    success = True
    model   = None

    # Run pipeline steps
    if not step_validate_environment():
        success = False

    if success:
        result = step_load_data()
        if result is False:
            success = False
        else:
            df = result

    if success:
        X_train, X_test, y_train, y_test = step_feature_engineering(df)

    if success:
        model = step_train_model(X_train, y_train)
        if model is False:
            success = False

    if success:
        if not step_evaluate(model, X_train, X_test, y_train, y_test):
            success = False

    if success and model is not None:
        step_log_mlflow(model)

    # Finalise run log
    total_duration = time.time() - pipeline_start
    RUN_LOG["completed_at"]  = datetime.utcnow().isoformat()
    RUN_LOG["duration_sec"]  = round(total_duration, 3)

    if success and RUN_LOG["conclusion"] is None:
        RUN_LOG["conclusion"] = "success"

    log("=" * 60)
    log(f"Pipeline finished: {RUN_LOG['conclusion'].upper()} in {total_duration:.2f}s")

    # Save run log to JSON
    out_dir = Path("data/run_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"run_{RUN_LOG['pipeline_id']}_{RANDOM_SEED}.json"
    with open(out_file, "w") as f:
        json.dump(RUN_LOG, f, indent=2)
    log(f"Run log saved to {out_file}")

    # Exit with correct code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
