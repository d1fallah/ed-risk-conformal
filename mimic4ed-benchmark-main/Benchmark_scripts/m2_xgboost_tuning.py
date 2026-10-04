"""
M2 XGBoost Tuning Search (validation-only -- never reads data/test.csv)
=======================================================================
Why this script exists
----------------------
`M2_Findings_Dossier.md` Sec 3.2 compares tuned GRU-D against Logistic
Regression, Random Forest and XGBoost. Until now GRU-D received a five-stage
hyperparameter search while all three baselines ran at library defaults.
That asymmetry originally ran AGAINST GRU-D's headline (XGBoost won anyway)
and was disclosed as a caveat. After Deviation 7's repair the comparison
reversed, so the same asymmetry now runs IN GRU-D's favour -- which makes it
a live threat to the finding rather than a conservative footnote. A tuned
neural model beating untuned tree baselines is weak evidence for anything.

This script removes that asymmetry for XGBoost, the strongest and only
previously-significant baseline.

Protocol symmetry with m2_grud_tuning.py (deliberate, point by point)
---------------------------------------------------------------------
  * Reads ONLY data/train.csv; asserts it never opens data/test.csv.
  * Same patient-level carve: X_fit (80%) / X_cal (20%), then X_fit_train
    (85%) / X_fit_val (15%) within the fit pool, re-drawn per seed.
  * Same 3 tuning seeds (42, 123, 7), distinct from the 5-seed confirmatory
    protocol.
  * Same staged/sequential structure with an explicit UNTUNED_BASE merged
    into every candidate, so no stage's control arm can silently inherit a
    previous winner (the failure recorded as Deviation 8).
  * Same noise guard: a candidate replaces the stage baseline only if it
    beats the baseline mean by more than one baseline validation SD.
  * Comparable budget: GRU-D's search evaluated 14 candidates x 3 seeds.

Calibration guard, applied symmetrically
----------------------------------------
Dossier Sec 3.1 rejected `pos_weight` for GRU-D because it bought AUROC while
destroying calibration, and M2's contribution is the conformal interval.
`scale_pos_weight` is XGBoost's exact analogue and carries the same risk, so
this search reports Brier/ECE/mean-predicted-probability for every candidate
and applies the SAME rejection rule. Selecting XGBoost on AUROC alone while
holding GRU-D to a calibration bar would simply invert the unfairness this
script exists to remove.

Output: data/m2_xgboost_tuning_trace.csv (full effective config per row)
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TRACE_CSV = os.path.join(DATA_DIR, "m2_xgboost_tuning_trace.csv")

assert not any("test.csv" in str(a) for a in sys.argv), "This script must never touch test.csv."
print(f"Integrity check: TEST_CSV path is defined ({TEST_PATH}) but will NOT be opened by this script.")

TUNE_SEEDS = [42, 123, 7]
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

# The untuned reference: XGBoost exactly as m2_baseline_multiseed.py runs it.
# Stated explicitly so no stage's baseline can drift (Deviation 8's lesson).
UNTUNED_BASE = {
    "n_estimators": 100, "eval_metric": "logloss", "verbosity": 0,
    "max_depth": 6, "learning_rate": 0.3, "subsample": 1.0,
    "colsample_bytree": 1.0, "min_child_weight": 1,
    "reg_lambda": 1.0, "scale_pos_weight": 1.0,
}

print("Loading data/train.csv only...", flush=True)
df_train = pd.read_csv(TRAIN_PATH)
df_train["gender"] = df_train["gender"].map({"M": 1, "F": 0, "m": 1, "f": 0}).fillna(0).astype(int)
available = [f for f in FEATURES if f in df_train.columns]
X_all = df_train[available].copy()
X_all = X_all.fillna(X_all.median(numeric_only=True))
y_all = df_train[TARGET].astype(int).values
subjects_all = df_train["subject_id"].values

pos_rate = y_all.mean()
SCALE_POS_WEIGHT = round((1 - pos_rate) / pos_rate, 2)
print(f"Train positive rate: {pos_rate:.4f} -> scale_pos_weight = {SCALE_POS_WEIGHT}")

trace_rows = []


def patient_level_split(X, y, subjects, test_size, seed):
    """Identical convention to m2_grud_tuning.py: split by unique subject_id,
    never by row, since 34% of patients have more than one ED visit."""
    unique_subj = np.unique(subjects)
    subj_a, subj_b = train_test_split(unique_subj, test_size=test_size, random_state=seed)
    mask_a = np.isin(subjects, subj_a)
    mask_b = np.isin(subjects, subj_b)
    return (X[mask_a], X[mask_b], y[mask_a], y[mask_b],
            subjects[mask_a], subjects[mask_b])


def ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    out = 0.0
    for b in range(n_bins):
        msk = idx == b
        if msk.sum() == 0:
            continue
        out += (msk.sum() / len(y_true)) * abs(y_true[msk].mean() - y_prob[msk].mean())
    return float(out)


def evaluate_candidate(stage, name, overrides, seeds=TUNE_SEEDS):
    aurocs, auprcs, briers, eces, meanps = [], [], [], [], []
    cfg = {**UNTUNED_BASE, **overrides}
    for seed in seeds:
        X_fit, X_cal, y_fit, y_cal, subj_fit, subj_cal = patient_level_split(
            X_all, y_all, subjects_all, test_size=0.2, seed=seed)
        X_ft, X_fv, y_ft, y_fv, subj_ft, subj_fv = patient_level_split(
            X_fit, y_fit, subj_fit, test_size=0.15, seed=seed)
        assert not (set(subj_fv.tolist()) & set(subj_cal.tolist())), \
            f"X_fit_val overlaps X_cal for seed={seed}"

        model = xgb.XGBClassifier(random_state=seed, **cfg)
        model.fit(X_ft, y_ft)
        p = model.predict_proba(X_fv)[:, 1]

        aurocs.append(roc_auc_score(y_fv, p))
        auprcs.append(average_precision_score(y_fv, p))
        briers.append(brier_score_loss(y_fv, p))
        eces.append(ece(y_fv, p))
        meanps.append(float(p.mean()))
        trace_rows.append({
            "stage": stage, "candidate": name, "seed": seed,
            "val_auroc": round(aurocs[-1], 4), "val_auprc": round(auprcs[-1], 4),
            "val_brier": round(briers[-1], 4), "val_ece": round(eces[-1], 4),
            "val_mean_pred": round(meanps[-1], 4),
            **{f"cfg_{k}": v for k, v in cfg.items()},
        })
    am, asd = float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))
    print(f"  [{stage}] {name:<34} val AUROC={am:.4f}+/-{asd:.4f}  "
          f"AUPRC={np.mean(auprcs):.4f}  Brier={np.mean(briers):.4f}  "
          f"ECE={np.mean(eces):.4f}  mean_p={np.mean(meanps):.4f}", flush=True)
    return am, asd, float(np.mean(eces)), float(np.mean(meanps))


def pick_winner(base, candidates):
    """Same rule as m2_grud_tuning.py: beat the baseline mean by more than one
    baseline SD. Additionally rejects any candidate whose ECE exceeds 5x the
    baseline's -- the calibration guard applied to GRU-D in dossier Sec 3.1,
    applied here for symmetry."""
    base_mean, base_sd, base_ece = base
    best = ("baseline", base_mean, base_sd)
    for name, mean, sd, cand_ece in candidates:
        if cand_ece > 5 * max(base_ece, 1e-6):
            print(f"      -> {name} REJECTED by calibration guard "
                  f"(ECE {cand_ece:.4f} vs baseline {base_ece:.4f})")
            continue
        if mean - base_mean > base_sd:
            if best[0] == "baseline" or mean > best[1]:
                best = (name, mean, sd)
    return best


best_cfg = {}

print("\n" + "=" * 78 + "\nSTAGE 1: Ensemble size and learning rate\n" + "=" * 78)
b = evaluate_candidate("1_capacity", "defaults (n=100, lr=0.3)", {**best_cfg})
c1 = evaluate_candidate("1_capacity", "n=300, lr=0.1", {**best_cfg, "n_estimators": 300, "learning_rate": 0.1})
c2 = evaluate_candidate("1_capacity", "n=600, lr=0.05", {**best_cfg, "n_estimators": 600, "learning_rate": 0.05})
c3 = evaluate_candidate("1_capacity", "n=1000, lr=0.03", {**best_cfg, "n_estimators": 1000, "learning_rate": 0.03})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("n=300, lr=0.1", c1[0], c1[1], c1[2]),
    ("n=600, lr=0.05", c2[0], c2[1], c2[2]),
    ("n=1000, lr=0.03", c3[0], c3[1], c3[2])])
print(f"  -> Stage 1 winner: {w} (val AUROC {wm:.4f})")
for nm, nn, ll in [("n=300, lr=0.1", 300, 0.1), ("n=600, lr=0.05", 600, 0.05), ("n=1000, lr=0.03", 1000, 0.03)]:
    if w == nm:
        best_cfg["n_estimators"], best_cfg["learning_rate"] = nn, ll

print("\n" + "=" * 78 + "\nSTAGE 2: Tree depth\n" + "=" * 78)
b = evaluate_candidate("2_depth", "depth=6 (baseline)", {**best_cfg})
c1 = evaluate_candidate("2_depth", "depth=4", {**best_cfg, "max_depth": 4})
c2 = evaluate_candidate("2_depth", "depth=8", {**best_cfg, "max_depth": 8})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("depth=4", c1[0], c1[1], c1[2]), ("depth=8", c2[0], c2[1], c2[2])])
print(f"  -> Stage 2 winner: {w} (val AUROC {wm:.4f})")
if w == "depth=4":
    best_cfg["max_depth"] = 4
elif w == "depth=8":
    best_cfg["max_depth"] = 8

print("\n" + "=" * 78 + "\nSTAGE 3: Stochastic regularisation (subsampling)\n" + "=" * 78)
b = evaluate_candidate("3_subsample", "no subsampling (baseline)", {**best_cfg})
c1 = evaluate_candidate("3_subsample", "subsample=0.8, colsample=0.8",
                        {**best_cfg, "subsample": 0.8, "colsample_bytree": 0.8})
c2 = evaluate_candidate("3_subsample", "subsample=0.6, colsample=0.6",
                        {**best_cfg, "subsample": 0.6, "colsample_bytree": 0.6})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("subsample=0.8, colsample=0.8", c1[0], c1[1], c1[2]),
    ("subsample=0.6, colsample=0.6", c2[0], c2[1], c2[2])])
print(f"  -> Stage 3 winner: {w} (val AUROC {wm:.4f})")
if w.startswith("subsample=0.8"):
    best_cfg.update({"subsample": 0.8, "colsample_bytree": 0.8})
elif w.startswith("subsample=0.6"):
    best_cfg.update({"subsample": 0.6, "colsample_bytree": 0.6})

print("\n" + "=" * 78 + "\nSTAGE 4: Complexity penalties\n" + "=" * 78)
b = evaluate_candidate("4_penalty", "min_child_weight=1, lambda=1 (baseline)", {**best_cfg})
c1 = evaluate_candidate("4_penalty", "min_child_weight=10", {**best_cfg, "min_child_weight": 10})
c2 = evaluate_candidate("4_penalty", "reg_lambda=10", {**best_cfg, "reg_lambda": 10.0})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("min_child_weight=10", c1[0], c1[1], c1[2]), ("reg_lambda=10", c2[0], c2[1], c2[2])])
print(f"  -> Stage 4 winner: {w} (val AUROC {wm:.4f})")
if w == "min_child_weight=10":
    best_cfg["min_child_weight"] = 10
elif w == "reg_lambda=10":
    best_cfg["reg_lambda"] = 10.0

print("\n" + "=" * 78 + "\nSTAGE 5: Class imbalance (the GRU-D `pos_weight` analogue)\n" + "=" * 78)
b = evaluate_candidate("5_imbalance", "scale_pos_weight=1 (baseline)", {**best_cfg})
c1 = evaluate_candidate("5_imbalance", f"scale_pos_weight={SCALE_POS_WEIGHT}",
                        {**best_cfg, "scale_pos_weight": SCALE_POS_WEIGHT})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    (f"scale_pos_weight={SCALE_POS_WEIGHT}", c1[0], c1[1], c1[2])])
print(f"  -> Stage 5 winner: {w} (val AUROC {wm:.4f})")
if w != "baseline":
    best_cfg["scale_pos_weight"] = SCALE_POS_WEIGHT

trace_df = pd.DataFrame(trace_rows)
trace_df.to_csv(TRACE_CSV, index=False)
print(f"\nFull search trace ({len(trace_df)} rows) saved to: {TRACE_CSV}")

print("\n" + "=" * 78)
print("FINAL RECOMMENDED XGBOOST CONFIG (validation-only; test.csv never read)")
print("=" * 78)
print(json.dumps(best_cfg, indent=2) if best_cfg else "{}  (library defaults win every stage)")
print(f"\nFull effective config: {json.dumps({**UNTUNED_BASE, **best_cfg})}")
with open(os.path.join(DATA_DIR, "m2_xgboost_tuned_config.json"), "w") as f:
    json.dump({"overrides": best_cfg, "effective": {**UNTUNED_BASE, **best_cfg}}, f, indent=2)
print("\nRolling this out to the 5-seed test comparison is a separate step "
      "(m2_baseline_multiseed.py).")
