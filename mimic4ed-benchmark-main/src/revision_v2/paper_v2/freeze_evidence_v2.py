"""
Copies the final result files of the revision into one flat folder (results/revision_v2/evidence/) and writes a SHA-256 manifest.
The provenance accessor (src/paper_data.py::Evidence) resolves artifacts only as flat files in one directory; paper_data_v2.py points it
here. Re-run after any result changes; the manifest records the hashes the manuscript assets were generated from.
"""
import hashlib
import json
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "results", "revision_v2")
EVD = os.path.join(RES, "evidence")
SOURCES = {
    "audit_inputs.json": "phase0/p04_truncation_timing.json",
    "audit_cohort.json": "phase0/p05_cohort_counts.json",
    "audit_elixhauser.json": "phase0/p02_elixhauser_audit.json",
    "audit_elixhauser_anomaly.json": "phase0/p02b_index_admission_anomaly.json",
    "audit_ed_admit_gap.json": "phase0/p06_ed_admit_gap.json",
    "audit_eci_streams.json": "phase0/p07_eci_stream_vs_lag.json",
    "ph1_shortcuts.json": "phase1/p1_2_shortcuts.json",
    "ph1_elixhauser.json": "phase1/p1_3_elixhauser_ablation.json",
    "ph1_duration.json": "phase1/p1_4_duration_strata.json",
    "ph1_scores.json": "phase1/p1_5_score_distributions.json",
    "ph1_sets_alpha.json": "phase1/p1_6_7_operating_and_alpha.json",
    "ph1_stream_order.json": "phase1/p1_8_stream_order.json",
    "ph2_cohorts.json": "phase2/p2_1_cohorts.json",
    "ph2_classical.json": "phase2/p2_2_classical.json",
    "ph2_contrasts.json": "phase2/p2_4_contrasts.json",
    "ph2_conformal.json": "phase2/p2_5_conformal.json",
    "ph2_news2.json": "phase2/p2_6_news2.json",
    "ph2_stream_order.json": "phase2/p2_8_stream_order.json",
    "ph2_sweep.json": "phase2/p2_10_landmark_sweep.json",
    "ph2_hist.json": "phase2/p2_11_score_histogram.json",
    "ph2_scores.json": "phase2/p2_9_score_distribution.json",
    "ph2_calibration.json": "phase2/p2_7_calibration_subgroups.json",
    "ph2_reliability.json": "phase2/p2_12_reliability.json",
    "ph2_years_decision.json": "phase2/p2_13_year_blocks_decision.json",
    "ph2_search.json": "phase2/p2_15_screening.json",
    "ph2_acuity_shortcut.json": "phase2/p2_16_shortcut_by_acuity.json",
    "ph2_fixed_cohort.json": "phase2/p2_17_fixed_cohort_landmarks.json",
    "ph2_characteristics.json": "phase2/p2_18_cohort_characteristics.json",
    "ph2_recording.json": "phase2/p2_19_recording_by_year.json",
    "ph2_arrival_increment.json": "phase2/p2_20_arrival_increment.json",
    "ph2_arrival_h2.json": "phase2/p2_21_arrival_on_h2_population.json",
    "ph2_static_cov_boot.json": "phase2/p2_22_static_coverage_bootstrap.json",
    "ph2_pr_points.json": "phase2/p2_23_pr_operating_points.json",
    "ph2_intercept_recal.json": "phase2/p2_24_intercept_recalibration.json",
    "ph2_overlap.json": "phase2/p2_25_outcome_overlap.json",
    "ph2_dedup.json": "phase2/p2_26_dedup_and_ablation_classical.json",
    "ph2_grud_dedup.json": "phase2/p2_27_grud_dedup_inference.json",
    "ph2_conformal_xgb.json": "phase2/p2_5_conformal_h2_xgboost_timing.json",
    "ph2_conformal_lr.json": "phase2/p2_5_conformal_h2_logreg.json",
}
EXTRA_DATA = {"orig_chronological_shift.json": "data/m2_chronological_shift.json", "ph2_sel_temporal.json": "data/revision_v2/runs/temporal_selected.json", "ph2_sel_e2e.json": "data/revision_v2/runs/e2e_selected.json",
              "ph2_sel_hybrid_frozen.json": "data/revision_v2/runs/hybrid_frozen_selected.json"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def derived():
    """Quantities that are sums or ratios of stored values, computed here from the frozen files (never typed)."""
    g = lambda n: json.load(open(os.path.join(RES, n)))  # noqa: E731
    an = g("phase0/p02b_index_admission_anomaly.json")
    co = g("phase2/p2_1_cohorts.json")
    h2 = co["landmarks"]["h2"]
    out = {"own_admission_cohort_stays": an["train_v2_stays_with_own_admission_accepted"] + an["test_v2_stays_with_own_admission_accepted"],
           "own_admission_cohort_changed": an["train_v2_stays_count_changes"] + an["test_v2_stays_count_changes"],
           "h2_zero_rows_share_train_period": co["h2_tensors"]["train_v2"]["stays_with_zero_ed_rows"] / co["h2_tensors"]["train_v2"]["rows"],
           "h2_zero_rows_share_test": co["h2_tensors"]["test_v2"]["stays_with_zero_ed_rows"] / co["h2_tensors"]["test_v2"]["rows"],
           "h2_train_period_eligible": sum(h2["splits"][r]["eligible_stays"] for r in ("fit", "val", "cs", "fc")),
           "h2_train_period_primary_positives": sum(h2["splits"][r]["primary_positives"] for r in ("fit", "val", "cs", "fc"))}
    k = g("phase2/p2_4_contrasts.json") if os.path.exists(os.path.join(RES, "phase2/p2_4_contrasts.json")) else None
    if k:
        c1 = k["landmarks"]["h2"]["contrasts"]["C1"]
        out["trees_minus_hybrid_delta"] = -c1["delta_seed_mean"]
        out["trees_minus_hybrid_lo"] = -c1["delta_bootstrap_ci95_of_seed_mean"][1]
        out["trees_minus_hybrid_hi"] = -c1["delta_bootstrap_ci95_of_seed_mean"][0]
        c4 = k["landmarks"]["h2"]["contrasts"]["C4"]
        out["frozen_minus_e2e_delta"] = -c4["delta_seed_mean"]
    import pandas as pd
    rec = pd.read_csv(os.path.join(ROOT, "data", "m2_v2_eci_vs_recalibration.csv"))
    mp = pd.read_csv(os.path.join(ROOT, "data", "m2_v2_conformal_summary.csv")).set_index(["model", "seed"])
    m1 = rec[rec["method"] == "M1"].set_index(["model", "seed"])["coverage"]
    out["static_vs_mapie_max_abs_coverage_diff"] = float((m1 - mp.loc[m1.index, "mapie_coverage"]).abs().max())
    out["h2_test_ineligible_share"] = 1.0 - h2["splits"]["test"]["eligible_stays"] / h2["splits"]["test"]["stays_in_split"]
    out["h2_train_period_primary_prevalence"] = out["h2_train_period_primary_positives"] / out["h2_train_period_eligible"]
    return out


def main():
    os.makedirs(EVD, exist_ok=True)
    man = {}
    missing = []
    for dst, src in list(SOURCES.items()):
        p = os.path.join(RES, src)
        if not os.path.exists(p):
            missing.append(src); continue
        shutil.copyfile(p, os.path.join(EVD, dst)); man[dst] = {"source": f"results/revision_v2/{src}", "sha256": sha(p)}
    for dst, src in EXTRA_DATA.items():
        p = os.path.join(ROOT, src)
        if not os.path.exists(p):
            missing.append(src); continue
        shutil.copyfile(p, os.path.join(EVD, dst)); man[dst] = {"source": src, "sha256": sha(p)}
    d = derived()
    json.dump(d, open(os.path.join(EVD, "derived.json"), "w"), indent=2)
    man["derived.json"] = {"source": "computed by freeze_evidence_v2.py from the frozen files", "sha256": sha(os.path.join(EVD, "derived.json"))}
    json.dump({"files": man, "missing_at_freeze": missing}, open(os.path.join(EVD, "MANIFEST.json"), "w"), indent=2)
    print(len(man), "files frozen; missing:", missing)


if __name__ == "__main__":
    main()
