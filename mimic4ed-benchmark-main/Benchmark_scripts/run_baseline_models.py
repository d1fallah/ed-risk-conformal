"""
Baseline Model Evaluation — MIMIC-IV-ED Benchmark
==================================================
Runs Logistic Regression, Random Forest, and XGBoost on the three
canonical prediction tasks from the benchmark paper:

  Task 1 — Hospitalization (ED triage features)
  Task 2 — Critical outcome  (ED triage + early vitals features)
  Task 3 — 72-h ED revisit   (ED triage features)

Outputs AUROC and AUPRC for each model × task to stdout and
saves results to ../data/baseline_results.csv.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

# ── Feature sets (mirroring the three task notebooks) ─────────────────────────
TRIAGE_FEATURES = [
    "age", "gender",
    "triage_temperature", "triage_heartrate", "triage_resprate",
    "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
    "n_ed_30d", "n_ed_90d", "n_ed_365d",
    "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
    "n_icu_30d", "n_icu_90d", "n_icu_365d",
    "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
    "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
    "chiefcom_nausea_vomiting", "chiefcom_fever_chills",
    "chiefcom_syncope", "chiefcom_dizziness",
]

ED_VITALS_FEATURES = [
    "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last",
    "ed_o2sat_last", "ed_sbp_last", "ed_dbp_last", "ed_pain_last",
    "n_med", "n_medrecon",
]

# Task definitions: (name, target_column, feature_set)
TASKS = [
    ("Task 1 — Hospitalization",  "outcome_hospitalization", TRIAGE_FEATURES),
    ("Task 2 — Critical Outcome", "outcome_critical",        TRIAGE_FEATURES + ED_VITALS_FEATURES),
    ("Task 3 — 72h ED Revisit",   "outcome_ed_revisit_3d",   TRIAGE_FEATURES),
]

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0, solver="lbfgs"),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    "XGBoost":             xgb.XGBClassifier(
                               n_estimators=100, random_state=0,
                               eval_metric="logloss", verbosity=0,
                               use_label_encoder=False),
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)
print(f"  Train: {df_train.shape}  |  Test: {df_test.shape}\n")

# Encode gender (M/F -> 0/1)
gender_map = {"M": 1, "F": 0, "m": 1, "f": 0}
for df in (df_train, df_test):
    df["gender"] = df["gender"].map(gender_map).fillna(0).astype(int)

# ── Helpers ───────────────────────────────────────────────────────────────────
def prepare_xy(df, features, target):
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"    [WARN] Missing features (will be skipped): {missing}")
    X = df[available].copy()
    # Median-impute remaining NaNs
    X = X.fillna(X.median(numeric_only=True))
    y = df[target].astype(int)
    return X, y

def metrics_str(y_true, y_prob, label):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    return auroc, auprc

# ── Run ───────────────────────────────────────────────────────────────────────
rows = []
sep  = "=" * 72

for task_name, target_col, features in TASKS:
    print(sep)
    print(f"  {task_name}")
    print(f"  Target: {target_col}")

    if target_col not in df_train.columns:
        print(f"  [SKIP] Target column not found in data.\n")
        continue

    X_train, y_train = prepare_xy(df_train, features, target_col)
    X_test,  y_test  = prepare_xy(df_test,  features, target_col)

    pos_rate = y_test.mean()
    print(f"  Train positive rate: {y_train.mean():.1%}  |  "
          f"Test positive rate: {pos_rate:.1%}")
    print(f"  {'Model':<25} {'AUROC':>8}  {'AUPRC':>8}")
    print(f"  {'-'*25} {'-'*8}  {'-'*8}")

    for model_name, model in MODELS.items():
        try:
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            auroc, auprc = metrics_str(y_test, y_prob, model_name)
            print(f"  {model_name:<25} {auroc:>8.4f}  {auprc:>8.4f}")
            rows.append({
                "task":        task_name,
                "target":      target_col,
                "model":       model_name,
                "auroc":       round(auroc, 4),
                "auprc":       round(auprc, 4),
                "n_train":     len(X_train),
                "n_test":      len(X_test),
                "pos_rate_test": round(pos_rate, 4),
            })
        except Exception as e:
            print(f"  {model_name:<25} ERROR: {e}")

    print()

print(sep)

# ── Save results ──────────────────────────────────────────────────────────────
df_results = pd.DataFrame(rows)
out_path   = os.path.join(DATA_DIR, "baseline_results.csv")
df_results.to_csv(out_path, index=False)
print(f"\nResults saved -> {out_path}")
print("\nFull results table:")
print(df_results.to_string(index=False))
