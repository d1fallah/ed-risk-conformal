"""
M2 Classical Baseline 5-Seed Sweep (Task 2 — Critical Outcome only)
======================================================================
GRU-D is reported as a 5-seed mean +/- std, but the classical baselines in
run_baseline_models.py train once at a fixed random_state=0 -- comparing a
5-seed distribution against a single point (via "gap in GRU-D's own std"
framing) is not a fair two-sample comparison: we don't know whether the
classical models' own seed-to-seed variance is smaller, comparable to, or
larger than GRU-D's.

This script re-runs Logistic Regression, Random Forest, and XGBoost across
the SAME 5 seeds used for GRU-D (42, 123, 7, 456, 789), varying only each
model's own random_state -- the Gate-mandated train/test cohort stays fixed
throughout, identical to every other multiseed script in this pipeline. Only
Task 2 (Critical Outcome) is re-run, since that's the only task compared
against GRU-D in the report.

Output: data/m2_baseline_multiseed.csv (per-seed rows) and printed mean+/-std
plus a Welch's two-sample t-test (GRU-D vs. each baseline) using GRU-D's real
per-seed AUROC values from data/m2_seed_results/seed_*.json.
"""
import os, json, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ttest_ind
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SEED_DIR   = os.path.join(DATA_DIR, "m2_seed_results")
OUT_PATH   = os.path.join(DATA_DIR, "m2_baseline_multiseed.csv")
XGB_TUNED_PATH = os.path.join(DATA_DIR, "m2_xgboost_tuned_config.json")

SEEDS = [42, 123, 7, 456, 789]
TARGET = "outcome_critical"

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
FEATURES = TRIAGE_FEATURES + ED_VITALS_FEATURES

print("Loading data (same Gate-mandated patient-level split as GRU-D)...")
df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

gender_map = {"M": 1, "F": 0, "m": 1, "f": 0}
for df in (df_train, df_test):
    df["gender"] = df["gender"].map(gender_map).fillna(0).astype(int)

def prepare_xy(df, features, target):
    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df[target].astype(int)
    return X, y

X_train, y_train = prepare_xy(df_train, FEATURES, TARGET)
X_test,  y_test  = prepare_xy(df_test,  FEATURES, TARGET)

rows = []
for seed in SEEDS:
    print(f"\nSeed {seed}:")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "XGBoost":             xgb.XGBClassifier(
                                    n_estimators=100, random_state=seed,
                                    eval_metric="logloss", verbosity=0),
        # Tuned XGBoost, added so Sec 3.2's comparison is methodologically
        # symmetrical: GRU-D receives a staged validation-only search, so the
        # strongest baseline must too. Config comes from
        # m2_xgboost_tuning.py (same protocol as m2_grud_tuning.py: same
        # patient-level carve, same 3 tune seeds, same noise guard, same
        # calibration guard, never opens test.csv). Loaded from the saved
        # JSON rather than retyped, so the numbers here cannot drift from the
        # search that produced them.
        "XGBoost (tuned)":     xgb.XGBClassifier(
                                    random_state=seed,
                                    **json.load(open(XGB_TUNED_PATH))["effective"]),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        print(f"  {name:<22} AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        rows.append({"seed": seed, "model": name, "auroc": round(auroc, 4), "auprc": round(auprc, 4)})

df_results = pd.DataFrame(rows)
df_results.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")

# ── Aggregate + Welch's t-test vs. GRU-D ─────────────────────────────────────
grud_files = sorted(glob.glob(os.path.join(SEED_DIR, "seed_*.json")))
grud_aurocs = [json.load(open(f))["overall"]["auroc"] for f in grud_files]
print(f"\nGRU-D (tuned config) per-seed AUROC: {grud_aurocs}")
print(f"GRU-D mean +/- std: {np.mean(grud_aurocs):.4f} +/- {np.std(grud_aurocs, ddof=1):.4f}")

print("\n" + "="*78)
print(f"{'Model':<22} {'AUROC mean':>12} {'std':>8} {'vs GRU-D: t-stat':>18} {'p-value':>10} {'Sig. (a=0.05)':>15}")
print("="*78)
summary_rows = []
for name in ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (tuned)"]:
    vals = df_results[df_results["model"] == name]["auroc"].values
    mean, std = vals.mean(), vals.std(ddof=1)
    t_stat, p_val = ttest_ind(grud_aurocs, vals, equal_var=False)
    sig = "Yes" if p_val < 0.05 else "No"
    print(f"{name:<22} {mean:>12.4f} {std:>8.4f} {t_stat:>18.3f} {p_val:>10.4f} {sig:>15}")
    summary_rows.append({
        "model": name, "auroc_mean": round(mean, 4), "auroc_std": round(std, 4),
        "grud_vs_model_t_stat": round(t_stat, 3), "p_value": round(p_val, 4),
        "significant_at_0.05": sig,
    })

summary_path = os.path.join(DATA_DIR, "m2_baseline_multiseed_summary.csv")
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")
