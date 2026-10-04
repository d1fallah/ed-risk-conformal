"""
M2 Track A -- does pos_weight buy AUROC at the cost of the conformal intervals?
==============================================================================
Why this script exists
----------------------
`m2_grud_tuning.py`'s 2026-09-01 re-search (under Deviation 7's corrected
architecture) selected `pos_weight=15.75, hid=64`, and the 5-seed
confirmatory run at that config reported a large discrimination gain
(AUROC 0.9009 -> 0.9261, AUPRC 0.4506 -> 0.5640) together with a 2.7x
widening of the conformal intervals (mean width 0.2824 -> 0.7692 on a
[0,1] scale), at unchanged nominal coverage.

That pairing is exactly what a class-weighted loss predicts, and it matters
more here than it would elsewhere:

  * The tuning search selects on validation AUROC only. AUROC is invariant
    to any monotone rescaling of the scores, so it is structurally BLIND to
    the decalibration a `pos_weight` term introduces. A config can therefore
    win the search while degrading the very quantity M2's Track A primary
    contribution is about.
  * Both conformal procedures here score on the residual |y - p|.
    Inflating p toward the positive class inflates those residuals for the
    ~94% negative majority, which widens the intervals. Coverage does NOT
    detect this -- split-conformal and ECI both hold marginal coverage
    essentially by construction, and they hold it by paying in width.

So "coverage is still 0.90" is not evidence the intervals are healthy. This
script separates the two factors that changed at once (hid=16->64 and
pos_weight off->on) so the width blow-up can be attributed to one of them
rather than assumed.

Design
------
Four arms, 5 seeds each, on the Gate-mandated fixed patient-level split --
identical data, identical conformal code, identical everything except the
two factors under test:

    A. hid=16, pos_weight=None   (the untuned reference)
    B. hid=64, pos_weight=None   (capacity only)
    C. hid=16, pos_weight=15.75  (weighting only)
    D. hid=64, pos_weight=15.75  (the config the search selected)

Every arm passes its configuration EXPLICITLY rather than relying on
SKLearnGRUD's constructor defaults. That is deliberate: letting unspecified
kwargs fall through to mutable module defaults is precisely the bug that
invalidated the first re-run of the tuning search (see Deviation 8), and an
attribution study is the last place that ambiguity belongs.

Output: data/m2_posweight_calibration_check.csv
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
sys.path.insert(0, SCRIPT_DIR)

from m2_ablations import (
    TARGET, prepare_data, patient_level_split, SKLearnGRUD,
    run_mapie_bounds, run_eci,
)
from m2_hybrid_common import sort_chronologically

SEEDS = [42, 123, 7, 456, 789]
OUT_CSV = os.path.join(DATA_DIR, "m2_posweight_calibration_check.csv")

ARMS = [
    ("A: hid=16, no pos_weight", {"hid": 16, "pos_weight": None}),
    ("B: hid=64, no pos_weight", {"hid": 64, "pos_weight": None}),
    ("C: hid=16, pos_weight",    {"hid": 16, "pos_weight": 15.75}),
    ("D: hid=64, pos_weight",    {"hid": 64, "pos_weight": 15.75}),
]

# Held fixed across every arm, stated here rather than inherited.
COMMON = {"epochs": 150, "lr": 0.01, "clf_hidden": 32, "dropout": 0.0,
          "weight_decay": 0.0, "batch_size": None,
          "early_stopping_patience": None}


def ece(y_true, y_prob, n_bins=10):
    """Expected calibration error, equal-width bins -- same definition the
    rest of the M2 pipeline uses."""
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
    print("Loading data (fixed Gate-mandated patient-level split)...", flush=True)
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_test = sort_chronologically(df_test)

    X_train, y_train = prepare_data(df_train)
    X_test, y_test = prepare_data(df_test)
    train_subjects = df_train["subject_id"].values

    rows = []
    for arm_name, arm_cfg in ARMS:
        print(f"\n{'='*66}\n{arm_name}\n{'='*66}", flush=True)
        for seed in SEEDS:
            X_fit, X_cal, y_fit, y_cal, _, _ = patient_level_split(
                X_train, y_train, train_subjects, test_size=0.2, seed=42)

            model = SKLearnGRUD(seed=seed, **COMMON, **arm_cfg)
            model.fit(X_fit, y_fit)
            p = model.predict_proba(X_test)[:, 1]

            auroc = float(roc_auc_score(y_test, p))
            auprc = float(average_precision_score(y_test, p))
            brier = float(brier_score_loss(y_test, p))
            e = ece(y_test, p)

            _, cov_m, wid_m, _, _, _ = run_mapie_bounds(model, X_cal, y_cal, X_test, y_test)
            _, cov_e, wid_e, _, _, _ = run_eci(model, X_test, y_test)

            rows.append({"arm": arm_name, "seed": seed, "auroc": auroc, "auprc": auprc,
                         "brier": brier, "ece": e, "mean_pred": float(p.mean()),
                         "coverage_mapie": cov_m, "width_mapie": wid_m,
                         "coverage_eci": cov_e, "width_eci": wid_e})
            print(f"  seed={seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f} Brier={brier:.4f} "
                  f"ECE={e:.4f} mean_p={p.mean():.4f} | MAPIE cov={cov_m:.4f} wid={wid_m:.4f} "
                  f"| ECI cov={cov_e:.4f} wid={wid_e:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    print(f"\n{'='*98}\nSummary (mean +/- SD across {len(SEEDS)} seeds)\n{'='*98}")
    summary = df.groupby("arm").agg(
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"), brier=("brier", "mean"), ece=("ece", "mean"),
        mean_pred=("mean_pred", "mean"),
        cov_mapie=("coverage_mapie", "mean"), wid_mapie=("width_mapie", "mean"),
        cov_eci=("coverage_eci", "mean"), wid_eci=("width_eci", "mean"),
    ).round(4)
    print(summary.to_string())
    print(f"\nTest-set positive rate (for reference): {y_test.mean():.4f}")


if __name__ == "__main__":
    main()
