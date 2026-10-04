"""
M2 -- Named macro registry for numbers quoted in manuscript PROSE (not inside a generated table,
which already carries its own numbers). See src/paper_data.py:emit_macro_file for the design
rationale (explicit, stable names -- never positional) and for the CSV-kind extension this module
needs, since most M2 result artifacts are long-format CSV rather than JSON.

Only headline/repeated quantities that a manuscript SENTENCE states directly are registered here --
a number that only ever appears inside a table cell is already traced by that table's own sidecar
(src/generate_latex_tables.py) and does not need a second, separate macro (the manuscript's own
style is to cite \\Cref{tab:...} for exact per-cell values and use a macro only for the number a
sentence actually asserts, per the plan's evidence-tracing-granularity note).

Cohort-flow counts (pre-filter/gate-eligible/horizon-eligible cohort sizes, the observation-window
percentile statistics) are sourced as ev.literal() with the reason citing
M2_Outcome_Definition_Gate.md's own "Measured by the actual implementation" sections (Sec.3/Sec.4) --
these ARE real, script-computed numbers (Benchmark_scripts/m2_build_irregular_cohort.py), just not
persisted to a small independently-replayable artifact the way M1's m1_cohort_flow.json was (no such
artifact exists in module2/mimic4ed-benchmark-main/data/ -- confirmed by search in this session; the
only place these counts live is the signed-off gate document's own prose). literal() with a stated
reason is the same treatment table_headline_discrimination() already gives the T=2 reference AUROCs
for an analogous reason (a real, computed number without its own small artifact to replay from).

Output: paper_assets/tables/m2_numbers.tex + paper_assets/provenance/m2_numbers.json
Usage: python generate_paper_macros.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_data import emit_macro_file  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "..", "paper_assets", "tables", "m2_numbers.tex")

RESULTS_CSV = "m2_v2_results.csv"
CONFORMAL_CSV = "m2_v2_conformal_summary.csv"
ER_CSV = "m2_v2_eci_vs_recalibration.csv"
LAG_CSV = "m2_v2_label_lag.csv"
HZ_CSV = "m2_v2_horizon_summary.csv"
BLOCKS_CSV = "m2_v2_eci_vs_recalibration_blocks.csv"
CMP_CSV = "m2_v2_mondrian_eci_comparators.csv"
E3_CSV = "m2_v2_rev_e3_adaptive_sets.csv"
E4_CSV = "m2_v2_rev_e4_drift_stress.csv"
S1_JSON = "m2_v2_rev_s1_width_bootstrap.json"
S2_JSON = "m2_v2_rev_s2_xgb_bootstrap.json"
FROZEN = "Frozen-Temporal Head Control v2"

HYBRID = "Hybrid GRU-D v2"
XGB = "XGBoost-Matched v2"
TEMPORAL = "Temporal-Only GRU-D v2"

# CLAUDE.md rule 5: every number derived from the MIMIC-IV-ED-derived cohort must be labeled Proxy
# wherever it appears -- checked mechanically by verify_paper_assets.py:check_qc_labels, which scans
# this asset's combined provenance notes for the literal substring "Proxy" the same way it scans a
# table's caption. Appended to every note below (matching generate_latex_tables.py's own PROXY use).
PROXY = "Proxy cohort (MIMIC-IV-ED-derived)."


def _n(text: str) -> str:
    return f"{text} {PROXY}" if text else PROXY


# (macro_suffix, kind, artifact, pointer, fmt, note)
# kind = "get"/"max"/"min"/"mean"/"len" (JSON, pointer: str) or
#        "get_csv"/"mean_csv"/"std_csv"/"max_csv"/"min_csv" (CSV, pointer: (filters_dict, column)).
MACROS: list[tuple[str, str, str, object, str | None, str]] = [
    # ---- Dataset / split (guide "cohort/task/outcome" backbone item 1) -------------------
    ("TrainSubjects", "len", "m2_v2_dev_pool_split.json", "/training/subject_ids/*", ",",
     _n("v2 irregular-series cohort, patient-level chronological-primary split.")),
    ("ValSubjects", "len", "m2_v2_dev_pool_split.json", "/validation/subject_ids/*", ",", _n("")),
    ("CalSelSubjects", "len", "m2_v2_dev_pool_split.json", "/calibration_selection/subject_ids/*", ",", _n("")),
    ("FinalCalSubjects", "len", "m2_v2_dev_pool_split.json", "/final_conformal_calibration/subject_ids/*", ",", _n("")),

    # ---- Backbone item 1: predictive performance under temporal shift --------------------
    ("HybridAurocMean", "mean_csv", RESULTS_CSV, ({"model": HYBRID}, "auroc"), ".4f",
     _n("Primary model (Hybrid GRU-D), composite critical-outcome AUROC, 5-seed mean.")),
    ("HybridAurocSD", "std_csv", RESULTS_CSV, ({"model": HYBRID}, "auroc"), ".4f", _n("")),
    ("XgbAurocMean", "mean_csv", RESULTS_CSV, ({"model": XGB}, "auroc"), ".4f",
     _n("Tuned XGBoost-Matched comparator, composite critical-outcome AUROC, 5-seed mean.")),
    ("DeltaReprMean", "get", "m2_v2_deltas.json", "/Delta_representation/mean", "+.4f",
     _n("PRIMARY delta: Hybrid GRU-D minus XGBoost-Matched (composite outcome).")),
    ("DeltaTemporalAloneMean", "get", "m2_v2_deltas.json", "/Delta_temporal_alone/mean", "+.4f",
     _n("The reversal finding: Temporal-Only minus XGBoost-Matched (composite outcome, negative).")),
    ("HzSixHReprDeltaMean", "get", "m2_v2_horizon_deltas.json", "/6h/deltas/D_repr/mean", "+.4f",
     _n("Time-anchored 6 h label: Hybrid GRU-D minus XGBoost-Matched.")),
    ("HzTwelveHReprDeltaMean", "get", "m2_v2_horizon_deltas.json", "/12h/deltas/D_repr/mean", "+.4f",
     _n("Time-anchored 12 h label: Hybrid GRU-D minus XGBoost-Matched.")),
    ("HzTwelveHTemporalDeltaMean", "get", "m2_v2_horizon_deltas.json", "/12h/deltas/D_temporal/mean", "+.4f",
     _n("Time-anchored 12 h label: Temporal-Only GRU-D minus XGBoost-Matched (not significant).")),
    ("HzSixHTemporalDeltaMean", "get", "m2_v2_horizon_deltas.json", "/6h/deltas/D_temporal/mean", "+.4f",
     _n("Horizon-dependent nuance: at 6h the temporal-only representation beats tuned XGBoost, "
        "reversing the composite-outcome result above.")),

    # ---- Backbone item 2+3: calibration under temporal shift; marginal vs. class-conditional --
    ("MapieMarginalCovMean", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "mapie_coverage"), ".4f",
     _n("MAPIE split-conformal marginal coverage, primary model.")),
    ("EciMarginalCovMean", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "eci_coverage"), ".4f",
     _n("ECI marginal coverage, primary model.")),
    ("MapieClassOneCovMean", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "mapie_coverage_class1"), ".4f",
     _n("MAPIE class-1 (deterioration) coverage, primary model -- the marginal/"
        "class-conditional gap this backbone item is about.")),
    ("EciClassOneCovMean", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "eci_coverage_class1"), ".4f", _n("")),
    ("MondrianClassOneCovMean", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "mondrian_lac_coverage_class1"), ".4f",
     _n("Mondrian-LAC restores class-1 coverage near target.")),
    # Static Mondrian-LAC's OWN class-0/marginal coverage, all 3 GRU-D-family models -- the
    # "restores class-1 but doesn't cleanly restore validity overall" nuance (Findings Dossier
    # Sec.24.3); not in T3 (which only carries Mondrian-LAC's class-1 column) or T5 (Hybrid only).
    ("MondrianClassZeroCovHybrid", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "mondrian_lac_coverage_class0"), ".4f",
     _n("Static Mondrian-LAC class-0 coverage, Hybrid GRU-D.")),
    ("MondrianClassZeroCovTemporal", "mean_csv", CONFORMAL_CSV, ({"model": TEMPORAL}, "mondrian_lac_coverage_class0"), ".4f",
     _n("Static Mondrian-LAC class-0 coverage, Temporal-Only GRU-D.")),
    ("MondrianClassZeroCovFrozen", "mean_csv", CONFORMAL_CSV, ({"model": "Frozen-Temporal Head Control v2"}, "mondrian_lac_coverage_class0"), ".4f",
     _n("Static Mondrian-LAC class-0 coverage, Frozen-Temporal Head Control.")),

    # ---- Backbone item 4: is the adaptive machinery needed vs. periodic recalibration ------
    ("EciWidthMean", "mean_csv", ER_CSV, ({"model": HYBRID, "method": "M2"}, "width"), ".3f",
     _n("ECI mean clipped interval width, primary model (headline negative-result comparison).")),
    ("RecalTenKWidthMean", "mean_csv", ER_CSV, ({"model": HYBRID, "method": "M3"}, "width"), ".3f",
     _n("Periodic recalibration (W=10k) mean interval width, primary model.")),
    ("RecalTenKCovMean", "mean_csv", ER_CSV, ({"model": HYBRID, "method": "M3"}, "coverage"), ".4f", _n("")),

    # ---- Conformal prose numbers added after the 2026-10-01 alignment-fix rerun -------------
    ("MapieMarginalCovTemporal", "mean_csv", CONFORMAL_CSV, ({"model": TEMPORAL}, "mapie_coverage"), ".4f", _n("")),
    ("EciMarginalCovTemporal", "mean_csv", CONFORMAL_CSV, ({"model": TEMPORAL}, "eci_coverage"), ".4f", _n("")),
    ("MapieClassOneCovTemporal", "mean_csv", CONFORMAL_CSV, ({"model": TEMPORAL}, "mapie_coverage_class1"), ".4f", _n("")),
    ("EciClassOneCovTemporal", "mean_csv", CONFORMAL_CSV, ({"model": TEMPORAL}, "eci_coverage_class1"), ".4f", _n("")),
    ("MondrianClassOneCovTemporal", "mean_csv", CONFORMAL_CSV, ({"model": TEMPORAL}, "mondrian_lac_coverage_class1"), ".4f", _n("")),
    ("MapieMarginalCovFrozen", "mean_csv", CONFORMAL_CSV, ({"model": FROZEN}, "mapie_coverage"), ".4f", _n("")),
    ("EciMarginalCovFrozen", "mean_csv", CONFORMAL_CSV, ({"model": FROZEN}, "eci_coverage"), ".4f", _n("")),
    ("MapieClassOneCovFrozen", "mean_csv", CONFORMAL_CSV, ({"model": FROZEN}, "mapie_coverage_class1"), ".4f", _n("")),
    ("EciClassOneCovFrozen", "mean_csv", CONFORMAL_CSV, ({"model": FROZEN}, "eci_coverage_class1"), ".4f", _n("")),
    ("MondrianClassOneCovFrozen", "mean_csv", CONFORMAL_CSV, ({"model": FROZEN}, "mondrian_lac_coverage_class1"), ".4f", _n("")),
    ("MondrianSetSizeTemporal", "mean_csv", CONFORMAL_CSV, ({"model": TEMPORAL}, "mondrian_mean_set_size"), ".3f", _n("")),
    ("MondrianSetSizeFrozen", "mean_csv", CONFORMAL_CSV, ({"model": FROZEN}, "mondrian_mean_set_size"), ".3f", _n("")),
    ("MondrianSetSizeHybrid", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "mondrian_mean_set_size"), ".3f", _n("")),
    ("MapieEmptyClassOneHybrid", "mean_csv", CONFORMAL_CSV, ({"model": HYBRID}, "mapie_empty_set_rate_class1"), ".1%", _n("Share of deteriorating patients (class 1) given the empty set by static split-conformal, primary model.")),
    ("StaticWidthHybrid", "mean_csv", ER_CSV, ({"model": HYBRID, "method": "M1"}, "width"), ".3f", _n("Static split-conformal mean clipped interval width, primary model.")),
    ("RecalFortyKWidthMean", "mean_csv", ER_CSV, ({"model": HYBRID, "method": "M4"}, "width"), ".3f", _n("")),
    ("RecalFortyKCovMean", "mean_csv", ER_CSV, ({"model": HYBRID, "method": "M4"}, "coverage"), ".4f", _n("")),
    ("StaticYearCovFirst", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M1", "scheme": "year", "block": "0"}, "coverage"), ".3f", _n("Static split-conformal coverage by approximate calendar year of the test period, primary model.")),
    ("EciYearCovFirst", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M2", "scheme": "year", "block": "0"}, "coverage"), ".3f", _n("ECI coverage by approximate calendar year of the test period, primary model.")),
    ("StaticYearCovSecond", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M1", "scheme": "year", "block": "1"}, "coverage"), ".3f", _n("Static split-conformal coverage by approximate calendar year of the test period, primary model.")),
    ("EciYearCovSecond", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M2", "scheme": "year", "block": "1"}, "coverage"), ".3f", _n("ECI coverage by approximate calendar year of the test period, primary model.")),
    ("StaticYearCovThird", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M1", "scheme": "year", "block": "2"}, "coverage"), ".3f", _n("Static split-conformal coverage by approximate calendar year of the test period, primary model.")),
    ("EciYearCovThird", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M2", "scheme": "year", "block": "2"}, "coverage"), ".3f", _n("ECI coverage by approximate calendar year of the test period, primary model.")),
    ("StaticYearCovFourth", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M1", "scheme": "year", "block": "3"}, "coverage"), ".3f", _n("Static split-conformal coverage by approximate calendar year of the test period, primary model.")),
    ("EciYearCovFourth", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M2", "scheme": "year", "block": "3"}, "coverage"), ".3f", _n("ECI coverage by approximate calendar year of the test period, primary model.")),
    ("YearBlockNFirst", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M1", "scheme": "year", "block": "0"}, "n"), ",.0f", _n("Stays in the first (largest) calendar-year block of the test period.")),
    ("YearBlockNFourth", "mean_csv", BLOCKS_CSV, ({"model": HYBRID, "method": "M1", "scheme": "year", "block": "3"}, "n"), ",.0f", _n("Stays in the fourth (smallest) calendar-year block of the test period.")),
    ("CmpStaticInformative", "mean_csv", CMP_CSV, ({"model": HYBRID, "method": "A"}, "informative_class1"), ".1%", _n("Share of deteriorating patients given the singleton set {1} by static Mondrian-LAC, primary model.")),
    ("CmpEciSize", "mean_csv", CMP_CSV, ({"model": HYBRID, "method": "B"}, "mean_set_size"), ".3f", _n("")),
    ("CmpEciInformative", "mean_csv", CMP_CSV, ({"model": HYBRID, "method": "B"}, "informative_class1"), ".1%", _n("")),
    ("CmpRecalSize", "mean_csv", CMP_CSV, ({"model": HYBRID, "method": "C1"}, "mean_set_size"), ".3f", _n("")),
    ("CmpRecalInformative", "mean_csv", CMP_CSV, ({"model": HYBRID, "method": "C1"}, "informative_class1"), ".1%", _n("")),
    ("CmpScoreBinStaticClassOne", "mean_csv", CMP_CSV, ({"model": HYBRID, "method": "D"}, "coverage_class1"), ".3f", _n("")),
    ("CmpScoreBinOnlineClassOne", "mean_csv", CMP_CSV, ({"model": HYBRID, "method": "E"}, "coverage_class1"), ".3f", _n("")),
    ("LagSixtyEciWidth", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "60", "method": "M2"}, "size"), ".3f", _n("")),
    ("LagThousandEciWidth", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "1000", "method": "M2"}, "size"), ".3f", _n("")),
    ("LagThousandEciCov", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "1000", "method": "M2"}, "coverage"), ".4f", _n("")),
    ("LagFourKEciCov", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "4000", "method": "M2"}, "coverage"), ".4f", _n("")),
    ("LagFourKEciWidth", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "4000", "method": "M2"}, "size"), ".3f", _n("")),
    ("LagSixteenKEciWidth", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "16000", "method": "M2"}, "size"), ".3f", _n("")),
    ("LagInfEciCov", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "inf", "method": "M2"}, "coverage"), ".4f", _n("")),
    ("LagInfEciWidth", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "inf", "method": "M2"}, "size"), ".3f", _n("")),
    ("LagFourKRecalCov", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "4000", "method": "M3"}, "coverage"), ".4f", _n("")),
    ("LagFourKRecalWidth", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "marginal", "lag": "4000", "method": "M3"}, "size"), ".3f", _n("")),
    ("LagThousandMondEciSize", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "class", "lag": "1000", "method": "B"}, "size"), ".3f", _n("")),
    ("LagFourKMondEciClassZero", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "class", "lag": "4000", "method": "B"}, "coverage_class0"), ".3f", _n("")),
    ("LagFourKMondEciInformative", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "class", "lag": "4000", "method": "B"}, "informative_class1"), ".1%", _n("")),
    ("LagSixteenKMondEciClassZero", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "class", "lag": "16000", "method": "B"}, "coverage_class0"), ".3f", _n("")),
    ("LagFourKMondRecalInformative", "mean_csv", LAG_CSV, ({"model": HYBRID, "family": "class", "lag": "4000", "method": "C1"}, "informative_class1"), ".1%", _n("")),
    ("LagRealisticLow", "literal", None, 60, None,
     _n("Pre-specified realistic label-lag band, low end (arrivals) -- M2_LabelLag_PreRegistration.md.")),
    ("LagRealisticHigh", "literal", None, 1000, None,
     _n("Pre-specified realistic label-lag band, high end (arrivals).")),


    # ---- Revision analyses (M2_Revision_PreSpecification.md; supervisor sign-off 2026-10-03) -----
    # E3: adaptive set-valued baselines on the marginal LAC score
    ("RevStaticSetSize", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "static_lac", "lag": "0"}, "mean_set_size"), ".3f", _n("Static LAC mean set size, primary model.")),
    ("RevAciSetSize", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "aci_lac", "lag": "0"}, "mean_set_size"), ".3f", _n("ACI on LAC, immediate labels.")),
    ("RevAciCov", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "aci_lac", "lag": "0"}, "coverage_marginal"), ".4f", _n("")),
    ("RevEciSetSize", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "eci_lac", "lag": "0"}, "mean_set_size"), ".3f", _n("ECI on LAC, immediate labels.")),
    ("RevEciSetCov", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "eci_lac", "lag": "0"}, "coverage_marginal"), ".4f", _n("")),
    ("RevAciLagSetSize", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "aci_lac", "lag": "240"}, "mean_set_size"), ".2f", _n("ACI on LAC, labels delayed 240 arrivals.")),
    ("RevAciLagCov", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "aci_lac", "lag": "240"}, "coverage_marginal"), ".4f", _n("")),
    ("RevAciLagEmpty", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "aci_lac", "lag": "240"}, "rate_empty"), ".1%", _n("")),
    ("RevAciLagInformative", "mean_csv", E3_CSV, ({"model": HYBRID, "method": "aci_lac", "lag": "240"}, "informative_class1"), ".1%", _n("")),
    # S1: width differences against static split-conformal (patient-cluster bootstrap, Holm over 9)
    ("RevWidthEciHybrid", "get", S1_JSON, f"/contrasts/{HYBRID} | ECI - static/mean", "+.4f", _n("")),
    ("RevWidthEciTemporal", "get", S1_JSON, f"/contrasts/{TEMPORAL} | ECI - static/mean", "+.4f", _n("")),
    ("RevWidthEciFrozen", "get", S1_JSON, f"/contrasts/{FROZEN} | ECI - static/mean", "+.4f", _n("")),
    ("RevWidthRecalTenHybrid", "get", S1_JSON, f"/contrasts/{HYBRID} | recal(10k) - static/mean", "+.4f", _n("")),
    ("RevWidthRecalFortyHybrid", "get", S1_JSON, f"/contrasts/{HYBRID} | recal(40k) - static/mean", "+.4f", _n("")),
    ("RevWidthRecalTenFrozen", "get", S1_JSON, f"/contrasts/{FROZEN} | recal(10k) - static/mean", "+.4f", _n("")),
    ("RevWidthRecalFortyTemporal", "get", S1_JSON, f"/contrasts/{TEMPORAL} | recal(40k) - static/mean", "+.4f", _n("")),
    ("RevWidthFamilySize", "get", S1_JSON, "/family_size", None, _n("")),
    # E4: controlled within-period drift (stress test, supplementary)
    ("RevDriftStaticPostHalf", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "shift+0.5", "method": "static"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftStaticPostOne", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "shift+1.0", "method": "static"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftStaticPostMix", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "casemix", "method": "static"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftStaticPreMix", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "casemix", "method": "static"}, "cov_pre"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftRecalPostMix", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "casemix", "method": "recal_10k"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftRecalPostOne", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "shift+1.0", "method": "recal_10k"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftAciPostMix", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "casemix", "method": "aci_lac"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftEciPostMix", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "casemix", "method": "eci_interval"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftEciPostOne", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "shift+1.0", "method": "eci_interval"}, "cov_post"), ".3f", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftBaseRatePreMix", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "casemix", "method": "static"}, "base_rate_pre"), ".1%", _n("E4 drift stress test (synthetic drift), primary model.")),
    ("RevDriftBaseRatePostMix", "mean_csv", E4_CSV, ({"model": HYBRID, "scenario": "casemix", "method": "static"}, "base_rate_post"), ".1%", _n("E4 drift stress test (synthetic drift), primary model.")),
    # S2: XGBoost-Matched bootstrap interval
    ("RevXgbAuroc", "get", S2_JSON, "/auroc/point", ".4f", _n("")),
    ("RevXgbAurocLo", "get", S2_JSON, "/auroc/ci_lo", ".4f", _n("")),
    ("RevXgbAurocHi", "get", S2_JSON, "/auroc/ci_hi", ".4f", _n("")),
    # ---- Streaming latency budget (CLAUDE.md M2 mandatory-core requirement: <2.0s/update) --
    ("LatencyBudgetMs", "get", "m2_v2_latency.json", "/methodology/budget_ms", ",.0f", _n("")),
    ("LatencyPNinetyNineMs", "get", "m2_v2_latency.json", "/paths/end_to_end_gpu_forward_plus_eci/p99_ms", ".2f",
     _n("End-to-end (GPU forward + ECI step + set), p99.")),
    ("LatencyCpuMedianMs", "get", "m2_v2_latency.json", "/paths/forward_cpu/median_ms", ".2f",
     _n("Forward pass alone on the CPU, batch size 1, median.")),
    ("LatencyGpuMedianMs", "get", "m2_v2_latency.json", "/paths/forward_gpu/median_ms", ".2f",
     _n("Forward pass alone on the GPU, batch size 1, median.")),

    # ---- Cohort/outcome/observation-window flow (Methods Sec: cohort construction) -----------
    # All literal() with the reason citing M2_Outcome_Definition_Gate.md's own "Measured by the
    # actual implementation" sections -- see this file's module docstring for why these are not
    # ev.get()/reduce_csv() (no small replayable artifact persists them independently).
    ("CohortPreFilter", "literal", None, 425088, None,
     _n("Full cohort before gate filtering (Outcome-Definition Gate v2 Sec.3).")),
    ("CohortGateEligible", "literal", None, 418025, None,
     _n("Gate-eligible cohort after inclusion/exclusion criteria (Gate v2 Sec.5-6).")),
    ("CohortHorizonEligible", "literal", None, 417964, None,
     _n("Horizon-labeled-analysis cohort: gate-eligible minus 54 unrecoverable-event_time minus "
        "7 event_time-before-intime encounters (Gate v2 Sec.3).")),
    ("NoEventTimeExclusions", "literal", None, 54, None,
     _n("Gate-eligible encounters with outcome_critical=1 but no recoverable event_time, excluded "
        "from horizon-labeled analyses only (Gate v2 Sec.3).")),
    ("EventBeforeIntimeExclusions", "literal", None, 7, None,
     _n("Encounters with event_time before intime (date-consistency artifact), excluded from "
        "horizon-labeled analyses (Gate v2 Sec.3).")),
    ("TMaxEd", "literal", None, 10, None,
     _n("T_max: measured p90 of ED-vitalsign observations/stay, right-truncated sequence length "
        "(Gate v2 Sec.4).")),
    ("ChronologicalCutYear", "literal", None, 2016, None,
     _n("Chronological-primary split cut year (M2_Findings_Dossier.md Sec.24.1).")),
    ("TrainPeriodRows", "literal", None, 288208, None,
     _n("Chronological train-period rows (Dossier Sec.24.1).")),
    ("TrainPeriodPatients", "literal", None, 139371, None,
     _n("Chronological train-period patients (Dossier Sec.24.1) -- the pool the 4-way "
        "training/validation/calibration-selection/final-conformal-calibration split partitions.")),
    ("TestPeriodRows", "literal", None, 84241, None,
     _n("Chronological test-period rows (Dossier Sec.24.1).")),
    ("TestPeriodPatients", "literal", None, 61843, None,
     _n("Chronological test-period patients (Dossier Sec.24.1).")),
    ("DroppedRows", "literal", None, 45576, None,
     _n("Rows straddling the chronological cut, dropped from evaluation (Dossier Sec.24.1).")),
    ("DroppedPatients", "literal", None, 18968, None,
     _n("Patients straddling the chronological cut (Dossier Sec.24.1).")),
    ("TrainAgeMedian", "get", "m2_cohort_demographics.json", "/train_period/age_median", ".0f",
     _n("Median age in years, train period (src/cohort_demographics.py).")),
    ("TrainAgeQOne", "get", "m2_cohort_demographics.json", "/train_period/age_q1", ".0f",
     _n("First quartile of age, train period.")),
    ("TrainAgeQThree", "get", "m2_cohort_demographics.json", "/train_period/age_q3", ".0f",
     _n("Third quartile of age, train period.")),
    ("TrainFemalePct", "get", "m2_cohort_demographics.json", "/train_period/female_pct", ".1f",
     _n("Share of female stays, train period.")),
    ("TrainOutcomePct", "get", "m2_cohort_demographics.json", "/train_period/outcome_pct", ".1f",
     _n("Composite critical-event prevalence, train period.")),
    ("TestAgeMedian", "get", "m2_cohort_demographics.json", "/test_period/age_median", ".0f",
     _n("Median age in years, test period (src/cohort_demographics.py).")),
    ("TestAgeQOne", "get", "m2_cohort_demographics.json", "/test_period/age_q1", ".0f",
     _n("First quartile of age, test period.")),
    ("TestAgeQThree", "get", "m2_cohort_demographics.json", "/test_period/age_q3", ".0f",
     _n("Third quartile of age, test period.")),
    ("TestFemalePct", "get", "m2_cohort_demographics.json", "/test_period/female_pct", ".1f",
     _n("Share of female stays, test period.")),
    ("TestOutcomePct", "get", "m2_cohort_demographics.json", "/test_period/outcome_pct", ".1f",
     _n("Composite critical-event prevalence, test period.")),
    ("TestOutcomeEvents", "get", "m2_cohort_demographics.json", "/test_period/outcome_events", ",d",
     _n("Composite critical events in the test period.")),
    ("ZeroEdObsTrainPct", "literal", None, "4.62", None,
     _n("Zero-ED-vitalsign-observation rate, train period (pre-2016) -- itself a temporal-"
        "non-stationarity finding (Dossier Sec.24.1).")),
    ("ZeroEdObsTestPct", "literal", None, "0.02", None,
     _n("Zero-ED-vitalsign-observation rate, test period (post-2016) (Dossier Sec.24.1).")),
]


def main() -> int:
    literals = [m for m in MACROS if m[1] == "literal"]
    sourced = [m for m in MACROS if m[1] != "literal"]

    out_path, ev = emit_macro_file(sourced, OUT_PATH)
    # literal() entries bypass emit_macro_file's artifact-based kinds (they are declared design
    # constants, not measurements) -- appended to the same file/provenance sidecar by hand, same
    # \MTwo<suffix> naming and isalpha() guard as emit_macro_file applies to the sourced ones.
    if literals:
        with open(out_path, "a", encoding="utf-8") as f:
            for suffix, _kind, _artifact, value, _fmt, note in literals:
                assert suffix.isalpha(), f"macro suffix {suffix!r} contains a non-letter"
                v = ev.literal(value, reason=note)
                f.write("\\newcommand{\\MTwo" + suffix + "}{" + str(v) + "}\n")

    sidecar_path = ev.write_provenance()
    print(f"wrote {out_path} ({len(MACROS)} macros)")
    print(f"  provenance -> {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
