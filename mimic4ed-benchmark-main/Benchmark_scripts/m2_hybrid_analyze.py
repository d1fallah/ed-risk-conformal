"""
M2 Hybrid Guide v3.8 -- Patient-Cluster Bootstrap for the 5 Primary Deltas
========================================================================
Guide's "Statistical Comparison -- Patient-Cluster Bootstrap (Primary)":
because repeat ED encounters can belong to the same patient, all bootstrap
inference is performed at the patient-cluster level -- sample PATIENTS
(subject_id) with replacement from the test set, include ALL encounters
for each sampled patient with multiplicity (never deduplicate), compute
each model's AUROC within that resampled set, and take the difference.

DeLong is omitted per the guide's own explicitly sanctioned fallback -- no
validated patient-cluster-adjusted implementation exists (see
M2_Hybrid_Deviation_Log.md Deviation 2). This bootstrap is the sole
inferential procedure.

The 5 deltas (Sec 9):
  Delta_representation      = Hybrid - XGBoost-Matched          (PRIMARY)
  Delta_static_contribution = Hybrid - Frozen-Temporal Head Control  (cleanest ablation)
  Delta_temporal_alone      = Temporal-Only GRU-D - XGBoost-Matched
  Delta_static_only         = Static-Only MLP - Temporal-Only GRU-D
  Delta_rich                = XGBoost-Rich - Hybrid

XGBoost-Rich has no saved per-patient predictions anywhere in the existing
M2 pipeline (only aggregate metrics were ever persisted) and the Guide
says "do not retrain it." Its exact original fit (run_baseline_models.py:
TRIAGE_FEATURES+ED_VITALS_FEATURES, full 80% dev pool, n_estimators=100,
random_state=0) is fully deterministic (already confirmed for the Matched
variant), so reproducing it recovers the identical historical result and
merely extracts the per-patient predictions that were never saved --
verified below by checking the reproduced AUROC matches the recorded
0.9184 exactly before using it for Delta_rich.

Reports, per Guide Sec 8/10: point estimate (mean across seeds), SD across
seeds (training-seed variability), and 95% CI from the patient-cluster
bootstrap (population uncertainty) -- kept as two explicitly distinct
quantities, never conflated.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
N_BOOTSTRAP = 1000
RNG_SEED = 42

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import TARGET

MODEL_COL = {
    "Temporal-Only GRU-D": "prob_temporal_only_gru_d",
    "Frozen-Temporal Head Control": "prob_frozen_temporal_head_control",
    "Static-Only MLP": "prob_static_only_mlp",
    "Hybrid GRU-D": "prob_hybrid_gru_d",
    "XGBoost-Matched": "prob_xgboost_matched",
}
FINAL_SEEDS = [42, 123, 7, 456, 789]

DELTAS = [
    ("Delta_representation", "Hybrid GRU-D", "XGBoost-Matched",
     "PRIMARY: does temporal representation add value over matched tabular aggregation?"),
    ("Delta_static_contribution", "Hybrid GRU-D", "Frozen-Temporal Head Control",
     "Cleanest ablation: incremental contribution of the static MLP branch alone."),
    ("Delta_temporal_alone", "Temporal-Only GRU-D", "XGBoost-Matched",
     "How strong is temporal representation alone vs. matched tabular aggregation?"),
    ("Delta_static_only", "Static-Only MLP", "Temporal-Only GRU-D",
     "Relative contribution of static characteristics vs. temporal representation."),
]


def reproduce_xgboost_rich():
    """Reproduces the historical XGBoost-Rich fit exactly (never retrained
    -- this recovers per-patient predictions that were simply never saved,
    from an already-deterministic, already-frozen result)."""
    TRIAGE_FEATURES = [
        "age", "gender", "triage_temperature", "triage_heartrate", "triage_resprate",
        "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
        "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
        "n_icu_30d", "n_icu_90d", "n_icu_365d", "chiefcom_chest_pain", "chiefcom_abdominal_pain",
        "chiefcom_headache", "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
        "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", "chiefcom_dizziness",
    ]
    ED_VITALS_FEATURES = [
        "ed_temperature_last", "ed_heartrate_last", "ed_resprate_last", "ed_o2sat_last",
        "ed_sbp_last", "ed_dbp_last", "ed_pain_last", "n_med", "n_medrecon",
    ]
    features = TRIAGE_FEATURES + ED_VITALS_FEATURES
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    gender_map = {"M": 1, "F": 0, "m": 1, "f": 0}
    for df in (df_train, df_test):
        df["gender"] = df["gender"].map(gender_map).fillna(0).astype(int)

    def prepare_xy(df):
        X = df[features].copy().fillna(df[features].median(numeric_only=True))
        y = df[TARGET].astype(int)
        return X, y

    X_train, y_train = prepare_xy(df_train)
    X_test, y_test = prepare_xy(df_test)
    clf = xgb.XGBClassifier(n_estimators=100, random_state=0, eval_metric="logloss",
                             verbosity=0, use_label_encoder=False)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    print(f"Reproduced XGBoost-Rich: AUROC={auroc:.4f} (recorded historical value: 0.9184)")
    assert abs(auroc - 0.9184) < 0.0001, "Reproduction does not match the recorded historical AUROC -- STOP."
    return pd.DataFrame({"stay_id": df_test["stay_id"].values, "y_true": y_test.values, "prob_xgboost_rich": y_prob})


def patient_cluster_bootstrap(df, subj_col, model_a_col, model_b_col, y_col, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Guide's patient-cluster bootstrap: resample PATIENTS with
    replacement, include all of a sampled patient's encounters with
    multiplicity (no dedup), compute AUROC_a - AUROC_b within each
    resample. Returns (point_estimate, ci_lo, ci_hi, bootstrap_deltas)."""
    df = df.reset_index(drop=True)  # groupby(...).groups below must return
    # positions matching the 0-based numpy arrays (y, pa, pb), not df's
    # original (possibly non-contiguous, since df is often a filtered
    # slice of a larger frame) pandas index.
    rng = np.random.RandomState(seed)
    patients = df[subj_col].unique()
    point_a = roc_auc_score(df[y_col], df[model_a_col])
    point_b = roc_auc_score(df[y_col], df[model_b_col])
    point_estimate = point_a - point_b

    by_patient = {p: idx.values for p, idx in df.groupby(subj_col).groups.items()}
    y = df[y_col].values
    pa = df[model_a_col].values
    pb = df[model_b_col].values

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([by_patient[p] for p in sampled_patients])
        auroc_a = roc_auc_score(y[idx], pa[idx])
        auroc_b = roc_auc_score(y[idx], pb[idx])
        deltas[b] = auroc_a - auroc_b
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return point_estimate, ci_lo, ci_hi, deltas


def main():
    preds = pd.read_csv(os.path.join(DATA_DIR, "predictions_all_models.csv"))
    print(f"Loaded predictions_all_models.csv: {len(preds)} rows, {preds['seed'].nunique()} seeds")

    rich = reproduce_xgboost_rich()

    results = {}
    for delta_name, model_a, model_b, question in DELTAS:
        print(f"\n{'='*60}\n{delta_name}: {model_a} - {model_b}\n{question}\n{'='*60}")
        col_a, col_b = MODEL_COL[model_a], MODEL_COL[model_b]
        per_seed = []
        for seed in FINAL_SEEDS:
            df_seed = preds[preds["seed"] == seed]
            point, ci_lo, ci_hi, _ = patient_cluster_bootstrap(
                df_seed, "subject_id", col_a, col_b, "y_true", seed=RNG_SEED + seed,
            )
            per_seed.append({"seed": seed, "point_estimate": point, "ci_lo": ci_lo, "ci_hi": ci_hi})
            print(f"  seed={seed}: {delta_name} = {point:+.4f}  bootstrap 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        points = np.array([r["point_estimate"] for r in per_seed])
        print(f"  Cross-seed: mean={points.mean():+.4f} +/- SD={points.std(ddof=1):.4f} "
              f"(training-seed variability, distinct from the bootstrap CI's population uncertainty)")
        results[delta_name] = {"model_a": model_a, "model_b": model_b, "question": question,
                                "per_seed": per_seed, "mean": float(points.mean()), "sd": float(points.std(ddof=1))}

    # Delta_rich: XGBoost-Rich (single, no seed variation) vs Hybrid (5 seeds)
    print(f"\n{'='*60}\nDelta_rich: XGBoost-Rich - Hybrid GRU-D\n"
          f"How much performance is attributable to richer features vs. representation choice?\n{'='*60}")
    per_seed = []
    for seed in FINAL_SEEDS:
        df_seed = preds[preds["seed"] == seed][["subject_id", "stay_id", "y_true", MODEL_COL["Hybrid GRU-D"]]]
        df_seed = df_seed.merge(rich[["stay_id", "prob_xgboost_rich"]], on="stay_id", how="inner")
        assert len(df_seed) == len(preds[preds["seed"] == seed]), "XGBoost-Rich merge lost rows -- STOP."
        point, ci_lo, ci_hi, _ = patient_cluster_bootstrap(
            df_seed, "subject_id", "prob_xgboost_rich", MODEL_COL["Hybrid GRU-D"], "y_true", seed=RNG_SEED + seed,
        )
        per_seed.append({"seed": seed, "point_estimate": point, "ci_lo": ci_lo, "ci_hi": ci_hi})
        print(f"  seed={seed}: Delta_rich = {point:+.4f}  bootstrap 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    points = np.array([r["point_estimate"] for r in per_seed])
    print(f"  Cross-seed: mean={points.mean():+.4f} +/- SD={points.std(ddof=1):.4f}")
    results["Delta_rich"] = {"model_a": "XGBoost-Rich", "model_b": "Hybrid GRU-D",
                              "question": "How much performance is attributable to richer features vs. representation choice?",
                              "per_seed": per_seed, "mean": float(points.mean()), "sd": float(points.std(ddof=1))}

    with open(os.path.join(DATA_DIR, "m2_hybrid_deltas.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved data/m2_hybrid_deltas.json")


if __name__ == "__main__":
    main()
