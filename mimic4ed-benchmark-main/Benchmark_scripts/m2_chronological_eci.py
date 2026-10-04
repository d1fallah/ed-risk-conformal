"""
M2 — Auxiliary chronological evaluation for the ECI vs. MAPIE contribution
==========================================================================
Why this script exists
----------------------
M2's primary research contribution is replacing split-conformal calibration
(MAPIE) with ECI, on the grounds that ECI stays valid under *temporal
non-stationarity*. The frozen primary benchmark cannot test that claim: its
outer train/test split is a RANDOM patient-level split
(`data_general_processing.py`), so calibration and test are temporally
interleaved draws from one distribution. Under exchangeability an adaptive
online method has nothing to adapt to, and "ECI ≈ MAPIE" is a property of
the evaluation design rather than a finding about ECI.

Per the supervisor-directed split strategy, the frozen random split is
RETAINED unchanged as the primary benchmark (see
`M2_Outcome_Definition_Gate.md` §7's disclosure), and this script adds an
*auxiliary* past→future evaluation used ONLY for the ECI-vs-MAPIE
non-exchangeability analysis. No headline discrimination number is
reported from here.

Both arms are computed in this one script, by the same code, so that the
only difference between them is the split and the stream order:

  Arm R (random)         — the frozen patient-level split, as the primary
                           benchmark uses it.
  Arm C (chronological)  — train on the past, calibrate on the most recent
                           slice of the past, test on the future.

Recovering real time from MIMIC-IV
----------------------------------
MIMIC-IV shifts every patient's dates by a random per-patient offset
(`anchor_year` spans 2110–2208 in this cohort), so raw `intime` is not
comparable across patients and a "chronological" sort on it is arbitrary.
`m2_hybrid_common._load_real_time_lookup()` de-shifts via
`anchor_year_group`; see its docstring for the recipe and, importantly, its
±1-year resolution limit. That limit is why the cut below is a multi-year
boundary rather than a precise timestamp, and why no within-year ordering
is claimed.

Model
-----
`SKLearnGRUD` — the Track A architecture (7 vitals × 2 timesteps + 24
static features), because ECI-vs-MAPIE is a Track A deliverable and this
must be directly comparable to the primary Track A result. The conformal
code is the same `run_mapie_bounds`/`run_eci` used everywhere else in M2.

Outputs: `data/m2_chronological_eci.csv` (per-seed, per-arm, per-method)
and `data/m2_chronological_shift.json` (the drift evidence).
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
sys.path.insert(0, SCRIPT_DIR)

from m2_ablations import (
    TARGET, prepare_data, patient_level_split, SKLearnGRUD,
    run_mapie_bounds, run_eci,
)
from m2_hybrid_common import add_temporal_axis, sort_chronologically, conformal_coverage_report

SEEDS = [42, 123, 7, 456, 789]
CUT_YEAR = 2016          # train period = years <= CUT_YEAR; test period = > CUT_YEAR
CAL_FRACTION = 0.20      # most-recent share of the train period held out for calibration
OUT_CSV = os.path.join(DATA_DIR, "m2_chronological_eci.csv")
OUT_SHIFT = os.path.join(DATA_DIR, "m2_chronological_shift.json")


def load_cohort():
    """The full frozen cohort (train.csv + test.csv), with the de-shifted
    approximate real year attached. Reassembling both halves is safe and
    necessary here: Arm C needs to re-partition the SAME encounters along a
    different axis, and Arm R re-derives the original split from the same
    frame, so neither arm sees an encounter the frozen cohort excludes."""
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_train["_frozen_side"] = "train"
    df_test["_frozen_side"] = "test"
    cohort = pd.concat([df_train, df_test], ignore_index=True)
    return add_temporal_axis(cohort)


def chronological_split(cohort):
    """Past→future, patient-disjoint.

    A patient with encounters on both sides of the cut cannot be placed
    without breaking either the temporal ordering (their future visit would
    be in train) or patient disjointness (the model would have seen them).
    Those patients are assigned wholly to the TRAIN period and their
    post-cut encounters are dropped from the evaluation entirely — the
    conservative choice, since the alternative (letting a seen patient into
    the future test set) is exactly the leakage the patient-level split
    exists to prevent. The number dropped is reported, not hidden.
    """
    past_patients = set(cohort.loc[cohort["approx_real_year"] <= CUT_YEAR, "subject_id"])
    is_past = cohort["approx_real_year"] <= CUT_YEAR
    train = cohort[is_past]
    future = cohort[~is_past]
    test = future[~future["subject_id"].isin(past_patients)]
    dropped = future[future["subject_id"].isin(past_patients)]
    assert not set(train["subject_id"]) & set(test["subject_id"]), "patient overlap across the chronological cut"
    return train, test, dropped


def chronological_fit_cal(train):
    """Within the train period, hold out the MOST RECENT patients as the
    conformal calibration fold — matching how a deployed system would
    calibrate (on recent history) before predicting forward, rather than on
    a random slice of the past."""
    first_seen = train.groupby("subject_id")["approx_real_year"].min().sort_values()
    n_cal = int(round(len(first_seen) * CAL_FRACTION))
    cal_ids = set(first_seen.index[-n_cal:])
    df_cal = train[train["subject_id"].isin(cal_ids)]
    df_fit = train[~train["subject_id"].isin(cal_ids)]
    return df_fit, df_cal


def quantify_shift(train, test):
    """Is there any distribution shift to adapt to? Reported as evidence
    either way — a null here would mean the auxiliary arm cannot support
    the contribution any better than the random split does."""
    out = {
        "cut_year": CUT_YEAR,
        "train_period_rows": int(len(train)), "test_period_rows": int(len(test)),
        "train_patients": int(train["subject_id"].nunique()),
        "test_patients": int(test["subject_id"].nunique()),
        "train_positive_rate": float(train[TARGET].mean()),
        "test_positive_rate": float(test[TARGET].mean()),
        "positive_rate_by_year": {
            int(y): float(g[TARGET].mean())
            for y, g in pd.concat([train, test]).groupby("approx_real_year")},
        "n_by_year": {
            int(y): int(len(g))
            for y, g in pd.concat([train, test]).groupby("approx_real_year")},
    }
    for col in ("age", "triage_acuity", "triage_heartrate", "triage_sbp", "triage_resprate"):
        if col in train.columns:
            out[f"mean_{col}_train"] = float(train[col].mean())
            out[f"mean_{col}_test"] = float(test[col].mean())
    return out


def evaluate(arm, df_fit, df_cal, df_test, seed):
    """Fit GRU-D on df_fit, calibrate MAPIE on df_cal, and run both
    conformal procedures on df_test (already chronologically ordered)."""
    X_fit, y_fit = prepare_data(df_fit)
    X_cal, y_cal = prepare_data(df_cal)
    X_test, y_test = prepare_data(df_test)

    model = SKLearnGRUD(epochs=150, lr=0.01, seed=seed)
    model.fit(X_fit, y_fit)

    y_prob = model.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, y_prob))
    auprc = float(average_precision_score(y_test, y_prob))

    _, cov_m, wid_m, lat_m, lo_m, hi_m = run_mapie_bounds(model, X_cal, y_cal, X_test, y_test)
    mapie_cov = conformal_coverage_report(
        y_test, (y_test >= lo_m) & (y_test <= hi_m), prefix="")

    _, cov_e, wid_e, lat_e, lo_e, hi_e = run_eci(model, X_test, y_test)
    eci_cov = conformal_coverage_report(
        y_test, (y_test >= lo_e) & (y_test <= hi_e), prefix="")

    rows = [
        {"arm": arm, "seed": seed, "method": "MAPIE", "auroc": auroc, "auprc": auprc,
         "coverage": cov_m, "coverage_class0": mapie_cov["coverage_class0"],
         "coverage_class1": mapie_cov["coverage_class1"],
         "width": wid_m, "latency_ms": lat_m, "n_test": len(y_test)},
        {"arm": arm, "seed": seed, "method": "ECI", "auroc": auroc, "auprc": auprc,
         "coverage": cov_e, "coverage_class0": eci_cov["coverage_class0"],
         "coverage_class1": eci_cov["coverage_class1"],
         "width": wid_e, "latency_ms": lat_e, "n_test": len(y_test)},
    ]
    for r in rows:
        print(f"    {r['method']:<6} cov={r['coverage']:.4f} "
              f"(c0={r['coverage_class0']:.4f} c1={r['coverage_class1']:.4f}) "
              f"width={r['width']:.4f}", flush=True)
    return rows


def main():
    print("Loading cohort...", flush=True)
    cohort = load_cohort()
    print(f"Cohort: {len(cohort)} encounters, {cohort['subject_id'].nunique()} patients, "
          f"years {int(cohort['approx_real_year'].min())}-{int(cohort['approx_real_year'].max())}", flush=True)

    # ---- Arm C: chronological -------------------------------------------
    c_train, c_test, c_dropped = chronological_split(cohort)
    print(f"\nArm C (chronological, cut at {CUT_YEAR}):")
    print(f"  train period: {len(c_train)} rows / {c_train['subject_id'].nunique()} patients")
    print(f"  test  period: {len(c_test)} rows / {c_test['subject_id'].nunique()} patients")
    print(f"  dropped (post-cut encounters of patients already seen pre-cut): "
          f"{len(c_dropped)} rows / {c_dropped['subject_id'].nunique()} patients", flush=True)

    shift = quantify_shift(c_train, c_test)
    shift["dropped_rows"] = int(len(c_dropped))
    shift["dropped_patients"] = int(c_dropped["subject_id"].nunique())
    with open(OUT_SHIFT, "w") as f:
        json.dump(shift, f, indent=2)
    print(f"  positive rate: train {shift['train_positive_rate']:.4f} -> "
          f"test {shift['test_positive_rate']:.4f}")
    print(f"  saved shift evidence to {OUT_SHIFT}", flush=True)

    c_fit, c_cal = chronological_fit_cal(c_train)
    c_test_sorted = sort_chronologically(c_test)

    # ---- Arm R: the frozen random split ---------------------------------
    r_train = cohort[cohort["_frozen_side"] == "train"]
    r_test = cohort[cohort["_frozen_side"] == "test"]
    r_test_sorted = sort_chronologically(r_test)

    all_rows = []
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}", flush=True)

        # Arm R's fit/cal split is patient-level and random, matching the
        # primary pipeline exactly (m2_multiseed_run.py).
        idx = np.arange(len(r_train))
        _, _, _, _, fit_subj, cal_subj = patient_level_split(
            idx, idx, r_train["subject_id"].values, test_size=0.2, seed=seed)
        r_fit = r_train[r_train["subject_id"].isin(set(fit_subj))]
        r_cal = r_train[r_train["subject_id"].isin(set(cal_subj))]

        print("  Arm R (random split):", flush=True)
        all_rows += evaluate("random", r_fit, r_cal, r_test_sorted, seed)
        print("  Arm C (chronological split):", flush=True)
        all_rows += evaluate("chronological", c_fit, c_cal, c_test_sorted, seed)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    print(f"\n{'='*78}\nSummary (mean +/- SD across {len(SEEDS)} seeds)\n{'='*78}")
    summary = df.groupby(["arm", "method"]).agg(
        auroc=("auroc", "mean"),
        coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
        cov_class0=("coverage_class0", "mean"), cov_class1=("coverage_class1", "mean"),
        width=("width", "mean"), width_sd=("width", "std"),
    ).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
