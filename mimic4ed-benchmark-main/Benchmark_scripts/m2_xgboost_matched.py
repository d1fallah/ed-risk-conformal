"""
M2 Hybrid Guide v3.8 -- XGBoost-Matched (Sec 4.3/4.5, "Reuse Criteria")
========================================================================
"XGBoost-Matched receives matched underlying clinical variables as Hybrid
GRU-D, but represented through a different predefined structure." Features:
vitals aggregated as mean/min/max/last/count per vital, + the 4 Guide static
features (age, sex, triage acuity, Elixhauser comorbidity count). No lab
values, medication history, admission diagnoses beyond the 4 statics, or any
engineered feature not also available to Hybrid GRU-D (Sec 4.3's explicit
exclusion list). "Last observation" tie-breaking (Sec 4.3) is moot given
this pipeline's shape -- there is exactly one named triage value and one
named "latest ED" value per vital, never multiple same-timestamp raw
measurements to break ties among.

RETRAINED under v3.8 (was previously fit on the FULL 80% dev pool, all
seeds identical since XGBoost's own hyperparameters make random_state a
no-op -- see M2_Findings_Dossier.md Sec 13 for that history). Sec 4.5's
reuse criteria require "same patient-level train/validation/test
partitions" as the new fixed 4-way split; the prior run didn't carve out
any validation/calibration subset at all, so it's retrained here fitting
on exactly the fixed `training` subject subset (Sec 5.1) -- for strict
patient-level parity with the temporal models, whose gradient-fitting
likewise only ever touches `training`, never `validation`/calibration
subjects. This matters here specifically because "matched underlying
clinical variables" is this whole experiment's central framing; letting
XGBoost-Matched see more patients during fitting than the temporal models
do would be a different, if subtler, form of the same mismatch the 4
static features were chosen to avoid.

Aggregation caveat (documented, not a shortfall): the underlying M2 pipeline
only ever captures 2 observed timesteps per vital (triage, latest ED) -- see
M2_Outcome_Definition_Gate.md Sec 4 and m2_ablations.py's prepare_data(). So
"mean/min/max/last/count" here aggregates over exactly those same 2 points
Hybrid GRU-D also sees -- an accurate reflection of shared data reality,
not a richer trajectory this model is failing to exploit.

Per Guide Sec 8/10, XGBoost-Matched gets discrimination (AUROC/AUPRC) and
probabilistic calibration (Brier/ECE/slope/intercept) only -- no MAPIE/ECI
coverage or set-size (scoped to "Hybrid GRU-D, Temporal-Only GRU-D, and
Frozen-Temporal Head Control").

Hyperparameters match the existing XGBoost-Rich definition exactly
(run_baseline_models.py: n_estimators=100, eval_metric="logloss",
verbosity=0) -- only random_state varies per seed and only the FEATURE SET
+ training-subject subset differ from XGBoost-Rich, so any AUROC/AUPRC
difference reflects information content, not a differently-tuned model.
XGBoost-Rich itself is reused as-is per the Guide's explicit "do not
retrain it" -- this script never touches it.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR = os.path.join(DATA_DIR, "m2_hybrid_seed_results", "xgboost_matched")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    TARGET, HYBRID_STATIC_COLS, add_hybrid_static_features, build_fixed_dev_split,
    expected_calibration_error, calibration_slope_intercept,
    save_predictions, save_seed_metrics, save_manifest,
)
import m2_hybrid_common as mhc

SEEDS = [42, 123, 7, 456, 789]
VITALS_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]


def build_matched_features(df):
    """7 vitals x {mean,min,max,last,count} over the 2 observed timesteps
    (triage, latest ED) + the 4 Guide static features = 39 columns. NaN is
    left as-is for XGBoost's native missing-value handling (no imputation),
    matching the existing pipeline's own convention of not hand-imputing
    vitals before a tree model sees them.

    FIXED (M2_Hybrid_Deviation_Log.md Deviation 4): vitals are now sourced
    from master_dataset.csv's genuine pre-imputation values (via
    m2_hybrid_common's cached lookup, joined on stay_id) instead of from
    df's already-imputed train.csv/test.csv columns. Confirmed the bug
    directly before fixing it: every `_count` column was exactly 2 for
    every row (zero information) under the old, already-imputed source --
    the whole point of a completeness-count feature was silently defeated.
    The 4 static features are untouched (never part of the vitals-
    imputation step).
    """
    lookup = mhc._load_master_vitals_lookup()
    real_vitals = lookup.loc[df["stay_id"].values]
    df = add_hybrid_static_features(df)
    feats = {}
    for v in VITALS_COLS:
        t = real_vitals[f"triage_{v}"].values
        e = real_vitals[f"ed_{v}_last"].values
        both = pd.DataFrame({"t": t, "e": e}, index=df.index)
        feats[f"{v}_mean"] = both.mean(axis=1)
        feats[f"{v}_min"] = both.min(axis=1)
        feats[f"{v}_max"] = both.max(axis=1)
        feats[f"{v}_last"] = both["e"]  # "last" = the later (ED) observation
        feats[f"{v}_count"] = both.notna().sum(axis=1)
    feat_df = pd.DataFrame(feats, index=df.index)
    for c in HYBRID_STATIC_COLS:
        feat_df[c] = df[c].values
    return feat_df


def main():
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Fixed 4-way dev-pool split (Sec 5.1/Sec 7): fit on the `training`
    # subject subset only, not the full 80% dev pool -- see module
    # docstring for why this parity choice matters here.
    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    df_train_fit = df_train[df_train["subject_id"].isin(train_ids)]

    X_train = build_matched_features(df_train_fit)
    X_test = build_matched_features(df_test)
    y_train = df_train_fit[TARGET].astype(int).values
    y_test = df_test[TARGET].astype(int).values
    subj_test = df_test["subject_id"].values
    stay_test = df_test["stay_id"].values

    print(f"Feature matrix: {X_train.shape[1]} columns "
          f"({len(VITALS_COLS)} vitals x 5 aggregations + {len(HYBRID_STATIC_COLS)} statics)")
    print(f"Train (fixed `training` subset only): {len(X_train)} rows / {len(train_ids)} patients, "
          f"pos rate {y_train.mean():.4f}  Test: {len(X_test)} rows, pos rate {y_test.mean():.4f}")

    for seed in SEEDS:
        clf = xgb.XGBClassifier(
            n_estimators=100, random_state=seed, eval_metric="logloss", verbosity=0,
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test, y_prob)
        slope, intercept = calibration_slope_intercept(y_test, y_prob)

        print(f"  seed={seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f} Brier={brier:.4f} "
              f"ECE={ece:.4f} slope={slope:.3f} intercept={intercept:.3f}")

        result = {
            "seed": seed, "model": "XGBoost-Matched",
            "config": {"n_estimators": 100, "random_state": seed},
            "n_features": int(X_train.shape[1]),
            "metrics": {
                "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
                "ece": float(ece), "calibration_slope": slope, "calibration_intercept": intercept,
            },
        }
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)

        save_predictions(subj_test, "XGBoost-Matched", seed, y_test, y_prob, stay_id=stay_test)
        save_seed_metrics("XGBoost-Matched", seed, result["metrics"])
        save_manifest("XGBoost-Matched", seed, result["config"])

    df_res = pd.read_csv(os.path.join(DATA_DIR, "results_hybrid.csv"))
    df_res = df_res[df_res["model"] == "XGBoost-Matched"]
    print(f"\n5-seed summary (XGBoost-Matched): "
          f"AUROC {df_res['auroc'].mean():.4f} +/- {df_res['auroc'].std():.4f}  "
          f"AUPRC {df_res['auprc'].mean():.4f} +/- {df_res['auprc'].std():.4f}  "
          f"Brier {df_res['brier'].mean():.4f} +/- {df_res['brier'].std():.4f}  "
          f"ECE {df_res['ece'].mean():.4f} +/- {df_res['ece'].std():.4f}")


if __name__ == "__main__":
    main()
