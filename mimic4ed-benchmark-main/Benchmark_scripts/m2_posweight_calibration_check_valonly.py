"""
M2 Track A -- validation-only replication of the pos_weight/hid attribution study
==================================================================================
Why this script exists
-----------------------
An external statistical review of this module (2026-09-01) found that
`m2_posweight_calibration_check.py` -- the study `M2_Findings_Dossier.md` Sec
3.1 cites as the evidence for adopting `hid=64, pos_weight=None` -- computes
every metric (AUROC, ECE, Brier, conformal width) directly on `data/test.csv`.
That is a config-selection decision informed by the held-out test set, which
is exactly the leak `m2_grud_tuning.py` is architected to prevent everywhere
else in this module (it asserts at startup that it never opens `test.csv`).

This script re-runs the identical 4-arm comparison using ONLY a validation
fold carved from the training pool -- the same `X_fit_train`/`X_fit_val`
construction `m2_grud_tuning.py` uses (patient-level, disjoint from the MAPIE
calibration fold, asserted every seed) -- so the adoption decision has a
validation-data provenance. `test.csv` is never opened.

This does not replace `m2_posweight_calibration_check.py` or its saved CSV;
that file's test-set numbers remain a valid CONFIRMATORY measurement of the
decision's effect on the actual held-out set, reported as such, not as the
basis for the decision. This script establishes whether the decision itself
would have been the same one made on data the model never touched.

Output: data/m2_posweight_calibration_check_valonly.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

assert not any("test.csv" in str(a) for a in sys.argv), "This script must never touch test.csv."
print(f"Integrity check: TEST_CSV path is defined ({TEST_PATH}) but will NOT be opened by this script.")

sys.path.insert(0, SCRIPT_DIR)
from m2_ablations import TARGET, prepare_data, SKLearnGRUD

SEEDS = [42, 123, 7]  # the same 3 tuning seeds m2_grud_tuning.py uses, not the 5-seed confirmatory set
OUT_CSV = os.path.join(DATA_DIR, "m2_posweight_calibration_check_valonly.csv")

ARMS = [
    ("A: hid=16, no pos_weight", {"hid": 16, "pos_weight": None}),
    ("B: hid=64, no pos_weight", {"hid": 64, "pos_weight": None}),
    ("C: hid=16, pos_weight",    {"hid": 16, "pos_weight": 15.75}),
    ("D: hid=64, pos_weight",    {"hid": 64, "pos_weight": 15.75}),
]
COMMON = {"epochs": 150, "lr": 0.01, "clf_hidden": 32, "dropout": 0.0,
          "weight_decay": 0.0, "batch_size": None, "early_stopping_patience": None}


def patient_level_split(X, y, subjects, test_size, seed):
    unique_subj = np.unique(subjects)
    subj_a, subj_b = train_test_split(unique_subj, test_size=test_size, random_state=seed)
    mask_a = np.isin(subjects, subj_a)
    mask_b = np.isin(subjects, subj_b)
    return X[mask_a], X[mask_b], y[mask_a], y[mask_b], subjects[mask_a], subjects[mask_b]


def ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out += (m.sum() / len(y_true)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(out)


def main():
    print("Loading data/train.csv only...", flush=True)
    df_train = pd.read_csv(TRAIN_PATH)
    X_train, y_train = prepare_data(df_train)
    train_subjects = df_train["subject_id"].values

    rows = []
    for arm_name, arm_cfg in ARMS:
        print(f"\n{'='*66}\n{arm_name}\n{'='*66}", flush=True)
        for seed in SEEDS:
            # Identical carve to m2_grud_tuning.py: X_fit (80%) / X_cal (20%),
            # then X_fit_train (85%) / X_fit_val (15%) within X_fit, re-drawn
            # per seed. X_fit_val never overlaps X_cal -- asserted directly.
            X_fit, X_cal, y_fit, y_cal, subj_fit, subj_cal = patient_level_split(
                X_train, y_train, train_subjects, test_size=0.2, seed=seed)
            X_ft, X_fv, y_ft, y_fv, subj_ft, subj_fv = patient_level_split(
                X_fit, y_fit, subj_fit, test_size=0.15, seed=seed)
            assert not (set(subj_fv.tolist()) & set(subj_cal.tolist())), \
                f"X_fit_val overlaps X_cal for seed={seed}"

            model = SKLearnGRUD(seed=seed, **COMMON, **arm_cfg)
            model.fit(X_ft, y_ft)
            p = model.predict_proba(X_fv)[:, 1]

            auroc = float(roc_auc_score(y_fv, p))
            auprc = float(average_precision_score(y_fv, p))
            brier = float(brier_score_loss(y_fv, p))
            e = ece(y_fv, p)
            rows.append({"arm": arm_name, "seed": seed, "auroc": auroc, "auprc": auprc,
                         "brier": brier, "ece": e, "mean_pred": float(p.mean())})
            print(f"  seed={seed}: val AUROC={auroc:.4f} AUPRC={auprc:.4f} Brier={brier:.4f} "
                  f"ECE={e:.4f} mean_p={p.mean():.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    print(f"\n{'='*90}\nSummary (mean +/- SD across {len(SEEDS)} seeds, VALIDATION DATA ONLY)\n{'='*90}")
    summary = df.groupby("arm").agg(
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"), brier=("brier", "mean"), ece=("ece", "mean"),
        mean_pred=("mean_pred", "mean"),
    ).round(4)
    print(summary.to_string())
    print("\ntest.csv was never opened by this script.")


if __name__ == "__main__":
    main()
