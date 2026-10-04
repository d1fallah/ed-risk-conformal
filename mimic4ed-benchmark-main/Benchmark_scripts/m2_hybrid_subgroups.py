"""
M2 Hybrid Guide v3.8 -- Predefined Subgroup Analysis (Sec 11)
========================================================================
Primary, formally-tested hypothesis: "GRU-D's temporal representation
provides greater incremental value for patients with sparse/missing
vital-sign observations compared to those with dense observations."
Tested via a difference-in-differences (Sec 11's "Formal interaction
test"): DiD = (AUROC_Hybrid - AUROC_XGBoost-Matched)_sparse -
(AUROC_Hybrid - AUROC_XGBoost-Matched)_dense, using the SAME
patient-cluster bootstrap as m2_hybrid_analyze.py (resample patients with
replacement, keep all encounters, recompute DiD per replicate). A 95% CI
excluding zero is evidence of a heterogeneous effect; subgroup
point-estimate differences alone are NOT interpreted as such (Sec 11's
explicit rule).

Observation-density thresholds are estimated from the DEVELOPMENT
population only (train.csv, via pre-imputation counts recovered from
master_dataset.csv -- see observation_count()'s docstring for why this
recovery is necessary) and frozen before being applied to the test set --
Sec 11's explicit requirement, not adjusted after seeing results. The
literal 25th/75th-percentile scheme degenerates against this dataset's
real, heavily-skewed distribution; see observation_density_thresholds()
for the natural-break resolution used instead (Deviation 5, full record in
M2_Hybrid_Deviation_Log.md).

All other subgroups (age, triage acuity, NEWS2 risk group) are reported
as descriptive/exploratory only, per Sec 11's rule against uncorrected
significance testing across many subgroups.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
N_BOOTSTRAP = 1000
RNG_SEED = 42

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import VITALS_COLS, _load_master_vitals_lookup

HYBRID_COL, XGB_COL = "prob_hybrid_gru_d", "prob_xgboost_matched"
FINAL_SEEDS = [42, 123, 7, 456, 789]


def observation_count(df):
    """Count of non-null pre-cutoff vital measurements per encounter,
    across both timesteps (triage + latest ED) x 7 vitals -- nominally
    0 to 14.

    IMPORTANT FINDING: train.csv/test.csv's vitals columns have been
    median-imputed by data_general_processing.py (lines 71-76,
    SimpleImputer(strategy='median'), fit on train/applied to test) BEFORE
    these files are ever written -- confirmed directly (0 nulls across all
    14 vitals columns, for all 334,693 training rows). This means every
    GRU-D model in this entire M2 module (this Hybrid experiment and the
    original Track A pipeline alike) has always received fully-observed
    vitals: the mask M is always all-ones and the decay-based imputation
    branch (xc = m*x + (1-m)*(gamma*x)) is never actually invoked, since
    m=1 everywhere. This is a genuine, pipeline-wide limitation on GRU-D's
    stated missingness-handling rationale, not specific to this analysis
    -- flagged prominently in M2_Findings_Dossier.md.

    For the observation-density subgroup split specifically, this
    function instead recovers the TRUE pre-imputation missingness from
    data/master_dataset.csv (joined on stay_id, verified as a unique key
    present for every train/test row) -- real missingness there ranges
    ~4-12% per vital, confirmed directly -- so the intended sparse-vs-dense
    analysis can still be computed from the genuine underlying data-
    generating process, rather than declared a dead end when a clean,
    non-arbitrary fix exists.
    """
    triage_cols = [f"triage_{c}" for c in VITALS_COLS]
    ed_cols = [f"ed_{c}_last" for c in VITALS_COLS]
    # Counted from the SAME cleaned lookup the models' observation mask M is
    # built from (m2_hybrid_common._load_master_vitals_lookup), not from a
    # separate raw read of master_dataset.csv. Deviation 7's cleaning turns
    # physiologically impossible values into NaN, so a raw read would define
    # the density subgroups from a slightly different missingness pattern
    # than the one the models actually see -- and this subgroup split exists
    # precisely to test how the models behave as a function of that pattern.
    master = _load_master_vitals_lookup()
    raw_counts = master[triage_cols].notna().sum(axis=1) + master[ed_cols].notna().sum(axis=1)
    counts = df["stay_id"].map(raw_counts)
    assert counts.notna().all(), "Some stay_ids not found in master_dataset.csv -- STOP."
    return counts


def observation_density_thresholds():
    """Estimated from the development population (train.csv, via the
    recovered pre-imputation counts) only -- frozen before test
    evaluation, per Sec 11.

    DEVIATION from Sec 11's literal 25th/75th-percentile scheme (full
    record: M2_Hybrid_Deviation_Log.md Deviation 5). The real
    distribution is heavily right-skewed, not smoothly spread: ~80% of
    the development population has all 14 vitals fully observed
    (count==14), so BOTH the 25th and 75th percentiles land exactly on
    14 -- the standard tertile split degenerates (a percentile-based
    Sparse/Moderate/Dense would not actually separate anything). Resolved
    with natural breaks matching the data's own visible structure (see
    bucket_density()): Dense = fully observed (count==14, the majority);
    Moderate = 8-13 observed (some missingness, still mostly complete);
    Sparse = <=7 observed (half or more of the 14 measurements missing).
    The formal interaction test compares the two POLES (Sparse vs. Dense),
    excluding Moderate -- directly analogous to how this same guide's own
    "Outcome risk group" subgroup (Sec 11's table: NEWS2<3 vs. NEWS2>=7)
    also compares extremes with an excluded middle band (NEWS2 3-6), not
    a novel pattern introduced here.
    """
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    counts = observation_count(df_train)
    frac_dense = float((counts == 14).mean())
    print(f"Observation-density (development population, n={len(df_train)}): "
          f"{frac_dense:.1%} fully observed (count==14); {1-frac_dense:.1%} have any missingness")
    return frac_dense


def bucket_density(counts):
    """Dense = fully observed; Sparse = any missingness; Moderate (8-13
    observed) reported only in the exploratory breakdown, not the formal
    interaction test -- see observation_density_thresholds()'s docstring."""
    return pd.cut(counts, bins=[-np.inf, 7, 13, np.inf], labels=["Sparse", "Moderate", "Dense"])


def did_bootstrap(df, subj_col, group_col, y_col, model_a_col, model_b_col,
                   group_a="Sparse", group_b="Dense", n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Difference-in-differences via patient-cluster bootstrap: for each
    replicate, resample patients with replacement (all encounters, no
    dedup) from the FULL set (preserving natural subgroup proportions),
    then compute DiD = (AUROC_a-AUROC_b)_group_a - (AUROC_a-AUROC_b)_group_b
    within that resample."""
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    patients = df[subj_col].unique()
    by_patient = {p: idx.values for p, idx in df.groupby(subj_col).groups.items()}
    y = df[y_col].values
    pa = df[model_a_col].values
    pb = df[model_b_col].values
    grp = df[group_col].values

    def point_did(idx):
        m_a = idx[grp[idx] == group_a]
        m_b = idx[grp[idx] == group_b]
        d_a = roc_auc_score(y[m_a], pa[m_a]) - roc_auc_score(y[m_a], pb[m_a])
        d_b = roc_auc_score(y[m_b], pa[m_b]) - roc_auc_score(y[m_b], pb[m_b])
        return d_a - d_b

    full_idx = np.arange(len(df))
    point = point_did(full_idx)

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([by_patient[p] for p in sampled_patients])
        deltas[b] = point_did(idx)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return point, ci_lo, ci_hi


def main():
    frac_dense = observation_density_thresholds()
    preds = pd.read_csv(os.path.join(DATA_DIR, "predictions_all_models.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_test["observation_count"] = observation_count(df_test)
    df_test["density_group"] = bucket_density(df_test["observation_count"])
    df_test["age_group"] = pd.cut(df_test["age"], bins=[-np.inf, 65, 80, np.inf], labels=["<65", "65-80", ">80"])
    df_test["acuity_group"] = pd.cut(df_test["triage_acuity"], bins=[0, 2.5, 3.5, 5], labels=["High(1-2)", "Intermediate(3)", "Low(4-5)"])
    if "score_NEWS2" in df_test.columns:
        df_test["news2_risk_group"] = pd.cut(df_test["score_NEWS2"], bins=[-np.inf, 3, 7, np.inf],
                                              labels=["Low(<3)", "Intermediate", "High(>=7)"])
    print(f"\nDensity group sizes (test set): {df_test['density_group'].value_counts().to_dict()}")

    merged = preds.merge(df_test[["stay_id", "density_group", "age_group", "acuity_group"]
                                  + (["news2_risk_group"] if "score_NEWS2" in df_test.columns else [])],
                          on="stay_id", how="left")
    assert merged["density_group"].notna().all(), "density_group merge produced nulls -- STOP."

    # ── Primary: formal interaction test (Sparse vs Dense), per seed ────────
    print(f"\n{'='*60}\nPRIMARY HYPOTHESIS: Sparse-vs-Dense interaction test\n"
          f"DiD = (Hybrid-XGBoost-Matched)_Sparse - (Hybrid-XGBoost-Matched)_Dense\n{'='*60}")
    did_results = []
    for seed in FINAL_SEEDS:
        df_seed = merged[merged["seed"] == seed]
        point, ci_lo, ci_hi = did_bootstrap(df_seed, "subject_id", "density_group", "y_true", HYBRID_COL, XGB_COL,
                                             seed=RNG_SEED + seed)
        evidence = "excludes zero (evidence of heterogeneous effect)" if (ci_lo > 0 or ci_hi < 0) else "includes zero (no formal evidence)"
        print(f"  seed={seed}: DiD={point:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  -- {evidence}")
        did_results.append({"seed": seed, "did": point, "ci_lo": ci_lo, "ci_hi": ci_hi})

    # ── Exploratory: per-subgroup AUROC by model (age/acuity/density/NEWS2) ─
    print(f"\n{'='*60}\nExploratory: per-subgroup AUROC (Hybrid GRU-D), seed 42 only, descriptive\n{'='*60}")
    exploratory = {}
    df_seed42 = merged[merged["seed"] == 42]
    for group_col in ["age_group", "acuity_group", "density_group"] + (["news2_risk_group"] if "news2_risk_group" in merged.columns else []):
        rows = []
        for g, sub in df_seed42.groupby(group_col, observed=True):
            if sub["y_true"].nunique() < 2:
                continue
            auroc_hybrid = roc_auc_score(sub["y_true"], sub[HYBRID_COL])
            auroc_xgb = roc_auc_score(sub["y_true"], sub[XGB_COL])
            rows.append({"group": str(g), "n": len(sub), "auroc_hybrid": auroc_hybrid, "auroc_xgboost_matched": auroc_xgb})
            print(f"  {group_col}={g}: n={len(sub)}  AUROC_Hybrid={auroc_hybrid:.4f}  AUROC_XGBoost-Matched={auroc_xgb:.4f}")
        exploratory[group_col] = rows

    output = {
        "density_bucketing": "natural breaks: Sparse<=7, Moderate=8-13, Dense=14 (see Deviation 5)",
        "frac_fully_observed_dev_population": frac_dense,
        "density_group_sizes": df_test["density_group"].value_counts().to_dict(),
        "primary_interaction_test": did_results,
        "exploratory_subgroups": exploratory,
    }
    with open(os.path.join(DATA_DIR, "m2_hybrid_subgroup_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved data/m2_hybrid_subgroup_results.json")


if __name__ == "__main__":
    main()
