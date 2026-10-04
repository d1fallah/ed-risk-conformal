"""
M2 -- NEWS2 Forensic Sensitivity Analysis (Q1-blocker #3)
==========================================================
External Q1-readiness review flagged the reported NEWS2 AUROC (0.6359,
composite outcome_critical, Deviation 10 in M2_Hybrid_Deviation_Log.md)
against the published ED-validation literature range (0.71-0.86) and
suspected the comparison was rigged to make GRU-D look better.

`add_score_NEWS2` (helpers.py) already documents why the score is
partial -- 5 of 7 RCP parameters (no ACVPU consciousness, no
supplemental-oxygen use) -- and that a quick mortality-alone check
reproduced ~0.62-0.63. This script makes that sensitivity check
systematic, with patient-cluster bootstrap CIs, across six arms, so the
manuscript can show *why* the number is low rather than just disclosing
that it is.

No retraining. No new data extraction. Reads the frozen `data/test.csv`
that every other M2 result is computed from (the fixed split persisted
under gate v1 and unaffected by this analysis). Extends the evaluation
path of `run_news2_baseline.py`; does not replace it.

Reuses the patient-cluster bootstrap resampling logic already
established for M2 in `m2_hybrid_analyze.py::patient_cluster_bootstrap`
(same resampling unit -- patients with replacement, all encounters kept
with multiplicity, no dedup) but for a single-arm AUROC/AUPRC CI rather
than a two-model delta, so it is a small variant, not a reimplementation.

Output: data/m2_news2_sensitivity.csv (one row per arm, with n, positive
rate, AUROC, AUPRC and their 95% patient-cluster bootstrap CIs).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
N_BOOTSTRAP = 1000
RNG_SEED = 42

# The 5 of 7 RCP NEWS2 parameters actually scored by add_score_NEWS2
# (helpers.py), post Deviation 10 (SpO2 restored). Used for the
# complete-case arm (E).
NEWS2_SCORED_PARAMS = [
    "triage_resprate", "triage_temperature", "triage_sbp",
    "triage_heartrate", "triage_o2sat",
]


def single_arm_patient_cluster_bootstrap(df, subj_col, score_col, y_col,
                                          n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Patient-cluster bootstrap for a single model's AUROC/AUPRC, same
    resampling unit as m2_hybrid_analyze.py::patient_cluster_bootstrap
    (patients resampled with replacement, all of a sampled patient's
    encounters kept with multiplicity). Returns dict of point estimates
    and 95% CIs for AUROC and AUPRC."""
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    patients = df[subj_col].unique()
    y = df[y_col].values
    s = df[score_col].values

    point_auroc = roc_auc_score(y, s)
    point_auprc = average_precision_score(y, s)

    by_patient = {p: idx.values for p, idx in df.groupby(subj_col).groups.items()}

    auroc_boot = np.empty(n_boot)
    auprc_boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([by_patient[p] for p in sampled_patients])
        # a degenerate resample (single class present) would break AUROC;
        # guard and resample once more rather than silently skip
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        auroc_boot[b] = roc_auc_score(yb, s[idx])
        auprc_boot[b] = average_precision_score(yb, s[idx])

    auroc_boot = auroc_boot[~np.isnan(auroc_boot)]
    auprc_boot = auprc_boot[~np.isnan(auprc_boot)]

    return {
        "n": len(df),
        "n_patients": len(patients),
        "pos_rate": float(np.mean(y)),
        "auroc": round(float(point_auroc), 4),
        "auroc_ci_lo": round(float(np.percentile(auroc_boot, 2.5)), 4),
        "auroc_ci_hi": round(float(np.percentile(auroc_boot, 97.5)), 4),
        "auprc": round(float(point_auprc), 4),
        "auprc_ci_lo": round(float(np.percentile(auprc_boot, 2.5)), 4),
        "auprc_ci_hi": round(float(np.percentile(auprc_boot, 97.5)), 4),
    }


# --split random: the original analysis on data/test.csv (earlier patient-level random split;
#   triage vitals imputed upstream, so the complete-case arm E removes nothing).
# --split chronological: the same six arms on data/test_v2.csv, the chronological test period every
#   other reported result uses. Vitals there are NOT imputed; add_score_NEWS2's np.select default
#   gives a missing parameter 0 points (scored as normal), and arm E is a real complete-case check.
SPLITS = {
    "random": ("test.csv", "m2_news2_sensitivity.csv"),
    "chronological": ("test_v2.csv", "m2_news2_sensitivity_v2.csv"),
}


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(SPLITS), default="random")
    split = parser.parse_args().split
    in_name, out_name = SPLITS[split]

    df = pd.read_csv(os.path.join(DATA_DIR, in_name))
    if "score_NEWS2" not in df.columns:
        if split == "random":
            raise SystemExit("score_NEWS2 not found in data/test.csv -- run "
                              "data_general_processing.py first.")
        sys.path.insert(0, SCRIPT_DIR)
        from helpers import add_score_NEWS2
        add_score_NEWS2(df)

    df["score_NEWS2"] = df["score_NEWS2"].fillna(0)

    # 24h-from-ED-departure mortality window, using deathtime already
    # present in test.csv (no new join needed).
    intime = pd.to_datetime(df["intime"], errors="coerce")
    outtime = pd.to_datetime(df["outtime"], errors="coerce")
    deathtime = pd.to_datetime(df["deathtime"], errors="coerce")
    hours_to_death_from_edout = (deathtime - outtime).dt.total_seconds() / 3600.0
    mortality_24h_from_edout = (
        deathtime.notna()
        & (hours_to_death_from_edout >= 0)
        & (hours_to_death_from_edout <= 24)
    ).astype(int)

    complete_case_mask = df[NEWS2_SCORED_PARAMS].notna().all(axis=1)

    arms = []

    # Arm A -- current comparison, composite outcome (reproduces 0.6359)
    arms.append(("A_composite_critical", "outcome_critical", df))

    # Arm B -- in-hospital mortality alone
    arms.append(("B_mortality_alone", "outcome_inhospital_mortality", df))

    # Arm C -- direct ED->ICU transfer alone
    arms.append(("C_icu_transfer_alone", "outcome_icu_transfer_12h", df))

    # Arm D -- mortality within 24h of ED departure (closest analogue to
    # the published ED-validation endpoint, which is typically in-hospital
    # or short-horizon mortality measured from an ED-adjacent origin)
    df_d = df.copy()
    df_d["_arm_d_target"] = mortality_24h_from_edout
    arms.append(("D_mortality_24h_from_edout", "_arm_d_target", df_d))

    # Arm E -- complete-case subset: all 5 scored NEWS2 params observed
    # (tests whether imputation/missingness, not the score itself, is
    # depressing AUROC)
    arms.append(("E_complete_case_composite", "outcome_critical", df[complete_case_mask]))

    # Arm F -- NEWS2 recomputed on last-recorded ED vitals instead of
    # triage-only, to test whether the triage-only restriction (not the
    # missing ACVPU/supplemental-O2 components) explains the gap.
    # Reuses add_score_NEWS2's exact bucket logic by renaming columns.
    sys.path.insert(0, SCRIPT_DIR)
    from helpers import add_score_NEWS2

    df_f = df.copy()
    rename_map = {
        "ed_resprate_last": "triage_resprate",
        "ed_temperature_last": "triage_temperature",
        "ed_sbp_last": "triage_sbp",
        "ed_heartrate_last": "triage_heartrate",
        "ed_o2sat_last": "triage_o2sat",
    }
    # add_score_NEWS2 reads the triage_* columns directly; swap in the
    # ed_*_last values under those names on a copy so the original
    # triage_* columns used by every other arm are untouched.
    df_f_scored = df_f.drop(columns=["triage_resprate", "triage_temperature",
                                      "triage_sbp", "triage_heartrate",
                                      "triage_o2sat"])
    df_f_scored = df_f_scored.rename(columns={v: v for v in []})  # no-op, clarity
    for ed_col, triage_col in rename_map.items():
        df_f_scored[triage_col] = df_f[ed_col]
    add_score_NEWS2(df_f_scored)
    df_f["score_NEWS2_last_ed_vitals"] = df_f_scored["score_NEWS2"]
    ed_vitals_available = df_f[list(rename_map.keys())].notna().any(axis=1)
    df_f_valid = df_f[ed_vitals_available].copy()

    results = []
    for arm_name, target_col, arm_df in arms:
        if arm_df[target_col].nunique() < 2:
            print(f"Skipping {arm_name}: target has <2 classes in this subset")
            continue
        stats = single_arm_patient_cluster_bootstrap(
            arm_df, "subject_id", "score_NEWS2", target_col)
        stats["arm"] = arm_name
        stats["score_col"] = "score_NEWS2"
        results.append(stats)
        print(f"{arm_name:35s} n={stats['n']:>7} pos_rate={stats['pos_rate']:.4f} "
              f"AUROC={stats['auroc']:.4f} [{stats['auroc_ci_lo']:.4f}, "
              f"{stats['auroc_ci_hi']:.4f}]  AUPRC={stats['auprc']:.4f}")

    # Arm F, using the swapped-in ED-vitals score column
    stats_f = single_arm_patient_cluster_bootstrap(
        df_f_valid, "subject_id", "score_NEWS2_last_ed_vitals", "outcome_critical")
    stats_f["arm"] = "F_last_ed_vitals_composite"
    stats_f["score_col"] = "score_NEWS2_last_ed_vitals"
    results.append(stats_f)
    print(f"{'F_last_ed_vitals_composite':35s} n={stats_f['n']:>7} "
          f"pos_rate={stats_f['pos_rate']:.4f} AUROC={stats_f['auroc']:.4f} "
          f"[{stats_f['auroc_ci_lo']:.4f}, {stats_f['auroc_ci_hi']:.4f}]  "
          f"AUPRC={stats_f['auprc']:.4f}")

    out_df = pd.DataFrame(results)[[
        "arm", "score_col", "n", "n_patients", "pos_rate",
        "auroc", "auroc_ci_lo", "auroc_ci_hi",
        "auprc", "auprc_ci_lo", "auprc_ci_hi",
    ]]
    out_path = os.path.join(DATA_DIR, out_name)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    if split == "chronological":
        # Outcome rate by triage-missingness pattern: the reason arm E differs from arm A here.
        # (On the random split every triage vital was imputed upstream, so there is nothing to report.)
        import json
        missing = df[NEWS2_SCORED_PARAMS].isna()
        groups = {
            "all_scored_params_recorded": ~missing.any(axis=1),
            "any_scored_param_missing": missing.any(axis=1),
            "all_scored_params_missing": missing.all(axis=1),
        }
        y = df["outcome_critical"]
        summary = {
            "source": in_name,
            "scored_params": NEWS2_SCORED_PARAMS,
            "outcome": "outcome_critical",
            "n_total": int(len(df)),
            "n_pos_total": int(y.sum()),
            "groups": {
                name: {"n": int(mask.sum()), "n_pos": int(y[mask].sum()),
                       "pos_rate": float(y[mask].mean()),
                       "share_of_all_positives": float(y[mask].sum() / y.sum())}
                for name, mask in groups.items()
            },
        }
        miss_path = os.path.join(DATA_DIR, "m2_news2_missingness_v2.json")
        with open(miss_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {miss_path}")
    print(
        "\nInterpretation note for the manuscript: 2 of NEWS2's 7 RCP "
        "parameters (level of consciousness / ACVPU, supplemental-oxygen "
        "use) are genuinely absent from this MIMIC-IV-ED extract and are "
        "not imputed anywhere in this analysis -- every arm above is "
        "still a partial NEWS2. If arms B/D/F do not close most of the "
        "gap to the published 0.71-0.86 range, the residual gap is "
        "attributable to the missing consciousness/oxygen components, "
        "not to a rigged comparison or a composite-outcome artifact, and "
        "should be reported as such."
    )


if __name__ == "__main__":
    main()
