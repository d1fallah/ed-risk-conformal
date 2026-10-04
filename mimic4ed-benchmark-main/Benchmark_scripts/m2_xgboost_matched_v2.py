"""
M2 -- XGBoost-Matched, irregular-series (v2) cohort, TUNED CONFIRMATORY RUN
==============================================================================
Same feature-parity contract as the T=2 pipeline's XGBoost-Matched (7
vitals x {mean,min,max,last,count} + 4 Guide static features = 39
columns, NaN left native for XGBoost, no lab values/meds/diagnoses beyond
the 4 statics) -- extended here to aggregate over the FULL irregular
sequence (up to 11 timesteps: triage + up to 10 ED observations) instead
of just the fixed 2 (triage, latest-ED). "last" is now the row's actual
most-recent real observation of that vital, not a fixed second column.

Loads the tuned config selected by m2_xgboost_tuning_v2.py (Phase 2a of
the session's "Clearing the Runway to the Manuscript" plan,
data/m2_v2_xgboost_tuned_config.json), confirms it exists (mirroring
m2_mtan_final_v2.py's own discipline -- this script refuses to silently
fall back to the untuned pilot config), and runs the full 5-seed
confirmatory protocol. This REPLACES the untuned pilot run that
M2_Findings_Dossier.md Sec 24.2 originally reported (n_estimators=100,
lr=0.3 defaults) -- same MODEL_NAME "XGBoost-Matched v2", so
save_predictions_v2/save_seed_metrics_v2's own (model, seed) dedup
overwrites the old untuned rows in place rather than duplicating them.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CONFIG_JSON = os.path.join(DATA_DIR, "m2_v2_xgboost_tuned_config.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import HYBRID_STATIC_COLS, add_hybrid_static_features, expected_calibration_error, calibration_slope_intercept
from m2_v2_common import (load_split_frame, load_dev_pool_split_v2, VITALS_COLS, save_predictions_v2, save_seed_metrics_v2,
                          HORIZON, TARGET, SUFFIX, apply_horizon_filter)

SEED_DIR = os.path.join(DATA_DIR, f"m2_v2_seed_results{SUFFIX}", "xgboost_matched")
os.makedirs(SEED_DIR, exist_ok=True)

FINAL_SEEDS = [42, 123, 7, 456, 789]
MODEL_NAME = "XGBoost-Matched v2"


def build_matched_features_v2(df, split_name):
    full_df = load_split_frame(split_name)
    with np.load(os.path.join(DATA_DIR, f"m2_irregular_sequences_{split_name}.npz")) as z:
        values, obs_mask = z["values"], z["obs_mask"]
    assert len(full_df) == len(values)
    pos = df.index.values
    v = values[pos]      # (n, 11, 7), NaN where not observed
    m = obs_mask[pos].astype(bool)

    feats = {}
    for i, vit in enumerate(VITALS_COLS):
        col = v[:, :, i]        # (n, 11)
        col_m = m[:, :, i]      # (n, 11)
        with np.errstate(all="ignore"):
            feats[f"{vit}_mean"] = np.nanmean(col, axis=1)
            feats[f"{vit}_min"] = np.nanmin(np.where(col_m, col, np.nan), axis=1)
            feats[f"{vit}_max"] = np.nanmax(np.where(col_m, col, np.nan), axis=1)
        feats[f"{vit}_count"] = col_m.sum(axis=1)
        # "last" = the value at the highest timestep index where this
        # variable was actually observed (chronologically most recent).
        last_val = np.full(len(df), np.nan)
        any_obs = col_m.any(axis=1)
        if any_obs.any():
            last_idx = np.where(col_m, np.arange(col_m.shape[1]), -1).max(axis=1)
            rows = np.where(any_obs)[0]
            last_val[rows] = col[rows, last_idx[rows]]
        feats[f"{vit}_last"] = last_val

    feat_df = pd.DataFrame(feats, index=df.index)
    df_static = add_hybrid_static_features(df)
    for c in HYBRID_STATIC_COLS:
        feat_df[c] = df_static[c].values
    return feat_df


def main():
    assert os.path.exists(CONFIG_JSON), (
        f"{CONFIG_JSON} not found -- run m2_xgboost_tuning_v2.py first. "
        f"Refusing to run the confirmatory grid on the untuned pilot config."
    )
    with open(CONFIG_JSON) as f:
        tuned_cfg = json.load(f)["effective"]
    print(f"Using tuned config: {tuned_cfg}")

    df_train = load_split_frame("train_v2")
    df_test = load_split_frame("test_v2")
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())

    df_fit = apply_horizon_filter(df_train[df_train["subject_id"].isin(train_ids)])
    df_test = apply_horizon_filter(df_test)
    print(f"target={TARGET} (horizon={HORIZON or 'composite'})  fit={len(df_fit)}, test={len(df_test)}")
    X_fit = build_matched_features_v2(df_fit, "train_v2")
    X_test = build_matched_features_v2(df_test, "test_v2")
    y_fit = df_fit[TARGET].astype(int).values
    y_test = df_test[TARGET].astype(int).values

    print(f"X_fit: {X_fit.shape}, X_test: {X_test.shape}, "
          f"mean non-null count col: {X_fit[[c for c in X_fit.columns if c.endswith('_count')]].mean().mean():.2f}")

    results = []
    for seed in FINAL_SEEDS:
        model = xgb.XGBClassifier(random_state=seed, **tuned_cfg)
        model.fit(X_fit, y_fit)
        prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, prob)
        auprc = average_precision_score(y_test, prob)
        brier = brier_score_loss(y_test, prob)
        ece = expected_calibration_error(y_test, prob)
        slope, intercept = calibration_slope_intercept(y_test, prob)
        m = {"seed": seed, "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
             "ece": float(ece), "cal_slope": float(slope), "cal_intercept": float(intercept)}
        results.append(m)
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(m, f, indent=2)
        save_predictions_v2(df_test["subject_id"].values, MODEL_NAME, seed, y_test, prob,
                             stay_id=df_test["stay_id"].values)
        save_seed_metrics_v2(MODEL_NAME, seed, {
            "auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece,
            "cal_slope": slope, "cal_intercept": intercept,
        })
        print(f"seed {seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f}", flush=True)

    aurocs = [r["auroc"] for r in results]
    print(f"\nXGBoost-Matched v2 tuned confirmatory ({len(results)} seeds): "
          f"AUROC mean={np.mean(aurocs):.4f} std={np.std(aurocs):.4f}")


if __name__ == "__main__":
    main()
