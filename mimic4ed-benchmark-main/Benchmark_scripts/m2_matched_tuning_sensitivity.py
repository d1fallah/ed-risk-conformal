"""
M2 -- Does the primary comparison survive tuning symmetry?
==========================================================
Why this script exists
----------------------
`m2_xgboost_tuning.py` established that the Sec 3.2 baseline comparison was
decided by a tuning asymmetry rather than by representation quality: with a
staged search of its own, XGBoost goes from 0.9184 to 0.9239 and reverses
the finding, significantly beating tuned GRU-D.

The SAME asymmetry sits inside the module's primary comparison. Hybrid GRU-D
received a two-stage search (Guide Sec 6.2, Stage 2A/2B) while
XGBoost-Matched runs at the fixed XGBoost-Rich hyperparameters the Guide
locks for it (Sec 4.3: "Hyperparameters match the existing XGBoost-Rich
definition exactly"). `Delta_representation` -- on which the whole Scenario
B classification rests -- is therefore a tuned model against an untuned one.

Having just measured that this asymmetry is worth ~+0.0055 AUROC to XGBoost
on the Sec 3.2 feature set, leaving it unexamined in the headline comparison
would be indefensible.

What this is and is not
-----------------------
This is a SENSITIVITY ANALYSIS, not a replacement for the primary result.
The Guide locks XGBoost-Matched's hyperparameters, and this script does not
modify `m2_xgboost_matched.py` or overwrite any primary artefact. It fits a
second, tuned XGBoost-Matched on the identical fixed `training` subject
subset and the identical matched feature set, then recomputes
`Delta_representation` against the already-saved Hybrid GRU-D predictions
using the same patient-cluster bootstrap. Both versions are reported.

The tuned config is read from `data/m2_xgboost_tuned_config.json` -- the
result of a search that never opened test.csv. Note it is transferred from
the Sec 3.2 feature set to the matched feature set rather than re-searched;
that is stated as a limitation of this check rather than hidden, and it is
conservative in the sense that a config searched directly on the matched
features could only do better.

Output: data/m2_matched_tuning_sensitivity.json
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
sys.path.insert(0, SCRIPT_DIR)

from m2_hybrid_common import (
    TARGET, add_hybrid_static_features, build_fixed_dev_split,
)
from m2_xgboost_matched import build_matched_features
from m2_hybrid_analyze import patient_cluster_bootstrap, RNG_SEED

SEEDS = [42, 123, 7, 456, 789]
OUT = os.path.join(DATA_DIR, "m2_matched_tuning_sensitivity.json")


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
    tuned = json.load(open(os.path.join(DATA_DIR, "m2_xgboost_tuned_config.json")))["effective"]
    print(f"Tuned XGBoost config (from m2_xgboost_tuning.py): {tuned}\n", flush=True)

    print("Loading data...", flush=True)
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_train = add_hybrid_static_features(df_train)
    df_test = add_hybrid_static_features(df_test)

    # Identical fit population to the primary XGBoost-Matched run.
    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    df_fit = df_train[df_train["subject_id"].isin(train_ids)]

    X_fit = build_matched_features(df_fit)
    y_fit = df_fit[TARGET].astype(int).values
    X_test = build_matched_features(df_test)
    y_test = df_test[TARGET].astype(int).values
    print(f"Fit: {len(X_fit)} rows / {len(train_ids)} patients | Test: {len(X_test)} rows",
          flush=True)

    preds = {}
    rows = []
    for seed in SEEDS:
        clf = xgb.XGBClassifier(random_state=seed, **tuned)
        clf.fit(X_fit, y_fit)
        p = clf.predict_proba(X_test)[:, 1]
        preds[seed] = p
        rows.append({
            "seed": seed,
            "auroc": float(roc_auc_score(y_test, p)),
            "auprc": float(average_precision_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "ece": ece(y_test, p),
        })
        print(f"  seed={seed}: AUROC={rows[-1]['auroc']:.4f} AUPRC={rows[-1]['auprc']:.4f} "
              f"Brier={rows[-1]['brier']:.4f} ECE={rows[-1]['ece']:.4f}", flush=True)

    auroc_mean = float(np.mean([r["auroc"] for r in rows]))
    auroc_sd = float(np.std([r["auroc"] for r in rows], ddof=1))
    print(f"\nXGBoost-Matched (tuned): AUROC {auroc_mean:.4f} +/- {auroc_sd:.4f}", flush=True)

    # Recompute Delta_representation against the SAVED Hybrid predictions, so
    # the Hybrid side is byte-identical to the primary analysis.
    # predictions_all_models.csv is WIDE (one column per model), not long.
    allp = pd.read_csv(os.path.join(DATA_DIR, "predictions_all_models.csv"))

    print(f"\n{'='*74}\nDelta_representation = Hybrid GRU-D - XGBoost-Matched\n{'='*74}")
    out_deltas = {"default": [], "tuned": []}
    for seed in SEEDS:
        h = allp[allp["seed"] == seed][
            ["subject_id", "stay_id", "y_true",
             "prob_hybrid_gru_d", "prob_xgboost_matched"]].rename(
            columns={"prob_hybrid_gru_d": "p_hybrid",
                     "prob_xgboost_matched": "p_xgb_default"})
        # Align the freshly-computed tuned predictions onto the same rows.
        tuned_df = pd.DataFrame({"stay_id": df_test["stay_id"].values, "p_xgb_tuned": preds[seed]})
        n_before = len(h)
        d = h.merge(tuned_df, on="stay_id", how="inner")
        assert len(d) == n_before, f"row alignment lost for seed {seed}: {len(d)} vs {n_before}"

        for label, col in (("default", "p_xgb_default"), ("tuned", "p_xgb_tuned")):
            # seed=RNG_SEED+seed matches m2_hybrid_analyze.py's own convention
            # exactly, so the "default" arm here uses the SAME bootstrap
            # resamples as Sec 18's own Delta_representation, not merely the
            # same data -- an external statistical review found the previous
            # unseeded call reproduced Sec 18's point estimate but not its CI
            # endpoints (differing by ~0.0002-0.0003), which was itself a
            # demonstration of Monte Carlo error at N_BOOTSTRAP=1000 rather
            # than a bug in either bootstrap run.
            # n_boot raised from the shared default of 1000: an external
            # statistical review measured the Monte Carlo SD of a similarly-
            # sized bootstrap's CI endpoints and found it comparable to the
            # effect size for a subgroup-scale delta; 10000 replicates cuts
            # that SD by roughly root(10).
            point, lo, hi, _ = patient_cluster_bootstrap(
                d, "subject_id", "p_hybrid", col, "y_true", n_boot=10000, seed=RNG_SEED + seed)
            out_deltas[label].append({"seed": seed, "point_estimate": point,
                                      "ci_lo": lo, "ci_hi": hi})
            excl = not (lo <= 0 <= hi)
            print(f"  seed={seed} vs {label:<7}: {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {'EXCLUDES zero' if excl else 'includes zero'}", flush=True)

    result = {
        "tuned_config": tuned,
        "config_source": "m2_xgboost_tuning.py, transferred from the Sec 3.2 feature set",
        "xgboost_matched_tuned": {"per_seed": rows, "auroc_mean": auroc_mean,
                                  "auroc_sd": auroc_sd},
        "delta_representation": out_deltas,
    }
    for label in ("default", "tuned"):
        pts = [d["point_estimate"] for d in out_deltas[label]]
        n_excl = sum(1 for d in out_deltas[label] if not (d["ci_lo"] <= 0 <= d["ci_hi"]))
        result[f"summary_{label}"] = {
            "mean": float(np.mean(pts)), "sd": float(np.std(pts, ddof=1)),
            "n_seeds_ci_excludes_zero": n_excl,
        }
        print(f"\nDelta_representation vs {label}: mean {np.mean(pts):+.4f} "
              f"+/- {np.std(pts, ddof=1):.4f}, CI excludes zero at {n_excl}/5 seeds")

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
