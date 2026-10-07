"""
Generates the revision manuscript's number macros and generated tables from the frozen evidence (results/revision_v2/evidence/), through the
repository's provenance accessor. Nothing is typed: every value in a macro or a table cell is a recorded `Evidence.get` call, and each asset writes
a provenance sidecar (paper_assets_v2/provenance/<asset>.json) that verify_assets_v2.py replays against the frozen files.

Outputs: module2/paper-m2-rev/rv_numbers.tex (macros \\RV...), module2/paper-m2-rev/floats/tab_*.tex (generated tables).
Part B (results that depend on the Phase 2 neural runs) is skipped when its evidence file does not exist yet.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data_v2 as PD  # noqa: E402

PAPER = os.path.abspath(os.path.join(PD.ROOT, "..", "paper-m2-rev"))
FLOATS = os.path.join(PAPER, "floats")
os.makedirs(FLOATS, exist_ok=True)
EVD = PD.EVIDENCE_DIR


def have(name):
    return os.path.exists(os.path.join(EVD, name))


# ----------------------------------------------------------------------------------------------- macros
def macros_A():
    C, I, E = "audit_cohort.json", "audit_inputs.json", "ph2_cohorts.json"
    L = [("MasterStays", C, "/master_rows", ",d"), ("EligibleStays", C, "/after_age_ge18_and_acuity_notnull_stays", ",d"),
         ("TrainStays", C, "/train_period/stays", ",d"), ("TrainPatients", C, "/train_period/patients", ",d"),
         ("TestStays", C, "/test_period/stays", ",d"), ("TestPatients", C, "/test_period/patients", ",d"),
         ("ExcludedStays", C, "/excluded_returning_patient_stays/stays", ",d"), ("ExcludedPatients", C, "/excluded_returning_patient_stays/patients", ",d"),
         ("DevTrainPatients", C, "/dev_subsets/training/patients_in_split_file", ",d"), ("DevValPatients", C, "/dev_subsets/validation/patients_in_split_file", ",d"),
         ("DevCsPatients", C, "/dev_subsets/calibration_selection/patients_in_split_file", ",d"),
         ("DevFcPatients", C, "/dev_subsets/final_conformal_calibration/patients_in_split_file", ",d"),
         ("PrevFit", E, "/landmarks/h2/splits/fit/primary_prevalence", "pct"), ("PrevFc", E, "/landmarks/h2/splits/fc/primary_prevalence", "pct"),
         ("PrevTest", E, "/landmarks/h2/splits/test/primary_prevalence", "pct"),
         ("HtwoEligibleTotal", E, "/landmarks/h2/totals/eligible_stays", ",d"), ("HtwoPrimaryPositivesTotal", E, "/landmarks/h2/totals/primary_positives", ",d"),
         ("HtwoSecondaryPositivesTotal", E, "/landmarks/h2/totals/secondary_positives", ",d"),
         ("RetainedPrimary", E, "/landmarks/h2/totals/original_positives_retained_primary", ",d"),
         ("RetainedSecondary", E, "/landmarks/h2/totals/original_positives_retained_secondary", ",d"),
         ("OriginalPositivesTotal", E, "/landmarks/h2/totals/original_composite_positives_in_split", ",d"),
         ("TEdHtwo", E, "/h2_sequence_rows_within_2h/T_ED_chosen", "d"),
         ("OwnAdmissionStays", "audit_elixhauser_anomaly.json", "/stays", "d"), ("OwnAdmissionCohortChanged", "derived.json", "/own_admission_cohort_changed", "d"),
         ("ZeroRowsTrainPct", "derived.json", "/h2_zero_rows_share_train_period", "pct"), ("ZeroRowsTestPct", "derived.json", "/h2_zero_rows_share_test", "pct"),
         ("CutYear", "orig_chronological_shift.json", "/cut_year", "d"),
         ("RefNewsLo", "EXT", "0.71", "mhdyunin2025news2", "reported AUROC range across outcomes in one ED cohort"),
         ("RefNewsHi", "EXT", "0.86", "mhdyunin2025news2", "reported AUROC range across outcomes in one ED cohort"),
         ("RefBurninLo", "EXT", "100", "wu2025eci", "tests/configs/*.yaml of the authors' repository (T_burnin)"),
         ("RefBurninHi", "EXT", "1{,}000", "wu2025eci", "tests/configs/*.yaml of the authors' repository (T_burnin)"),
         ("RefEtaLo", "EXT", "0.05", "wu2025eci", "tests/configs/*.yaml of the authors' repository (ECI learning rates)"),
         ("RefEtaHi", "EXT", "1", "wu2025eci", "tests/configs/*.yaml of the authors' repository (ECI learning rates)"),
         ("BootMain", "ph2_contrasts.json", "/bootstrap_primary_resamples", ",d") if have("ph2_contrasts.json") else None,
         # input audit (end-of-stay inputs)
         ("TruncTrainPct", I, "/train_v2/share_stays_truncated", "pct"), ("TruncTestPct", I, "/test_v2/share_stays_truncated", "pct"),
         ("TruncTrainN", I, "/train_v2/stays_with_more_than_10_rows_truncated", ",d"), ("TruncTestN", I, "/test_v2/stays_with_more_than_10_rows_truncated", ",d"),
         ("AfterDepRowsTrain", I, "/train_v2/retained_rows_charted_after_departure", ",d"), ("AfterDepRowsTest", I, "/test_v2/retained_rows_charted_after_departure", ",d"),
         ("AfterDepStaysTrain", I, "/train_v2/stays_with_retained_row_after_departure", ",d"), ("AfterDepStaysTest", I, "/test_v2/stays_with_retained_row_after_departure", ",d"),
         ("AfterOneHTest", I, "/test_v2/share_stays_with_retained_measurement_after_1h", "pct"), ("AfterTwoHTest", I, "/test_v2/share_stays_with_retained_measurement_after_2h", "pct"),
         ("AfterSixHTest", I, "/test_v2/share_stays_with_retained_measurement_after_6h", "pct"), ("AfterTwelveHTest", I, "/test_v2/share_stays_with_retained_measurement_after_12h", "pct"),
         ("AfterTwoHTrain", I, "/train_v2/share_stays_with_retained_measurement_after_2h", "pct"),
         ("ZeroRowsEosTrain", I, "/train_v2/stays_zero_ed_rows", ",d"), ("ZeroRowsEosTest", I, "/test_v2/stays_zero_ed_rows", ",d"),
         # Elixhauser audit
         ("ElixAccepted", "audit_elixhauser.json", "/n_source_admission_links", ",d"), ("ElixWindowDays", "audit_elixhauser.json", "/timerange_days_in_code", ",d"),
         ("ElixFlagsIdentical", "audit_elixhauser.json", "/recompute_flags_n_stays_identical_to_stored", ",d"),
         ("ElixMostRecentWithinSevenDays", "audit_elixhauser.json", "/stays_whose_most_recent_source_ended_within_7d", ",d"),
         ("EdAdmitMedian", "audit_ed_admit_gap.json", "/all_linked_stays/edouttime_minus_admittime/quantiles_minutes/0.5", ".0f"),
         # shortcut diagnostic
         ("ShortAAuroc", "ph1_shortcuts.json", "/baselines/A_only/test_auroc", ".3f"), ("ShortALo", "ph1_shortcuts.json", "/baselines/A_only/test_auroc_ci95/0", ".3f"),
         ("ShortAHi", "ph1_shortcuts.json", "/baselines/A_only/test_auroc_ci95/1", ".3f"),
         ("ShortABAuroc", "ph1_shortcuts.json", "/baselines/A_plus_B/test_auroc", ".3f"), ("ShortABLo", "ph1_shortcuts.json", "/baselines/A_plus_B/test_auroc_ci95/0", ".3f"),
         ("ShortABHi", "ph1_shortcuts.json", "/baselines/A_plus_B/test_auroc_ci95/1", ".3f"),
         ("ShortBAuroc", "ph1_shortcuts.json", "/baselines/B_only/test_auroc", ".3f"), ("ShortDurAuroc", "ph1_shortcuts.json", "/baselines/duration_alone/test_auroc", ".3f"),
         ("EosHybridAuroc", "ph1_shortcuts.json", "/models/Hybrid GRU-D v2/auroc_seed_summary/mean", ".3f"),
         ("EosXgbAuroc", "ph1_shortcuts.json", "/models/XGBoost-Matched v2/auroc_seed_summary/mean", ".3f"),
         ("ElixDeltaHybrid", "ph1_elixhauser.json", "/hybrid_seed_summary/delta_auroc/mean", ".4f"), ("ElixDeltaXgb", "ph1_elixhauser.json", "/xgboost/delta_auroc", ".4f"),
         ]
    return [m for m in L if m is not None]


def write_macros(macros, out_path, asset_id):
    ev = PD.EV(asset_id)
    lines = ["% AUTO-GENERATED by src/revision_v2/paper_v2/gen_assets_v2.py; do not edit. Every value is an Evidence.get record.", ""]
    seen = set()
    for m in macros:
        suffix, art = m[0], m[1]
        assert suffix.isalpha() and suffix not in seen, suffix
        seen.add(suffix)
        if art == "EXT":
            _s, _a, value, cite, locus = m
            v = ev.external(value, citation=cite, locus=locus)
        else:
            if len(m) == 5:
                _s, art, ptr, fmt, how = m
                v = ev.reduce(how, art, ptr, fmt=fmt)
            else:
                _s, art, ptr, fmt = m
                v = ev.get(art, ptr, fmt=fmt)
        lines.append("\\newcommand{\\RV" + suffix + "}{" + str(v) + "}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    ev.write_provenance()
    return len(macros)


# ----------------------------------------------------------------------------------------------- tables
def table_cohort():
    ev = PD.EV("tab_cohort")
    E = "ph2_cohorts.json"
    names = [("fit", "Training"), ("val", "Validation"), ("cs", "Calibration selection"), ("fc", "Final calibration"), ("test", "Test period")]
    rows = []
    for k, label in names:
        b = f"/landmarks/h2/splits/{k}"
        rows.append(" & ".join([label, ev.get(E, b + "/stays_in_split", fmt=",d"), ev.get(E, b + "/eligible_stays", fmt=",d"),
                                ev.get(E, b + "/primary_positives", fmt=",d"), ev.get(E, b + "/primary_prevalence", fmt="pct"),
                                ev.get(E, b + "/secondary_positives", fmt=",d"), ev.get(E, b + "/secondary_prevalence", fmt="pct")]) + " \\\\")
    t = "/landmarks/h2/totals"
    rows.append("\\midrule")
    rows.append(" & ".join(["All", ev.get(E, t + "/stays_in_split", fmt=",d"), ev.get(E, t + "/eligible_stays", fmt=",d"), ev.get(E, t + "/primary_positives", fmt=",d"), "",
                            ev.get(E, t + "/secondary_positives", fmt=",d"), ""]) + " \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Cohort at the 2-hour landmark (MIMIC-IV-ED, retrospective single-center proxy). Eligible stays are in the ED at $L$ and event-free at $L$. "
           "Primary outcome: ICU admission or in-hospital death in $(L, L+24\\,\\mathrm{h}]$; secondary outcome: the benchmark composite restricted to events after $L$.}\n\\label{tab:cohort}\n"
           "\\begin{tabular}{lrrrrrr}\n\\toprule\nSubset & Stays & Eligible & Primary events & Rate & Secondary events & Rate \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(FLOATS, "tab_cohort.tex"), "w", encoding="utf-8").write(tex)
    ev.write_provenance()


def table_shortcuts():
    ev = PD.EV("tab_shortcuts")
    S = "ph1_shortcuts.json"
    rows = []
    for k, label in (("duration_alone", "ED duration alone"), ("B_only", "Triage-time missingness"), ("A_only", "Workflow quantities"),
                     ("A_plus_B", "Workflow quantities and triage missingness")):
        b = f"/baselines/{k}"
        rows.append(f"{label} & {ev.get(S, b + '/test_auroc', fmt='.3f')} & [{ev.get(S, b + '/test_auroc_ci95/0', fmt='.3f')}, {ev.get(S, b + '/test_auroc_ci95/1', fmt='.3f')}] & "
                    f"{ev.get(S, b + '/test_auprc', fmt='.3f')} \\\\")
    rows.append("\\midrule")
    for k, label in (("Static-Only MLP v2", "Static features, neural network"), ("XGBoost-Matched v2", "Trees, whole stay"),
                     ("Temporal-Only GRU-D v2", "GRU-D, whole stay"), ("Hybrid GRU-D v2", "Hybrid GRU-D, whole stay")):
        b = f"/models/{k}"
        rows.append(f"{label} & {ev.get(S, b + '/auroc_seed_summary/mean', fmt='.3f')} & seed SD {ev.get(S, b + '/auroc_seed_summary/sd', fmt='.3f')} & "
                    f"{ev.get(S, b + '/auprc_seed_summary/mean', fmt='.3f')} \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Shortcut diagnostic on the end-of-stay task of the original analysis (test period; outcome: death or ICU transfer within 12 h of ED departure). "
           "Upper block: logistic regression on quantities that describe how a stay was monitored (workflow quantities: ED duration, number of retained measurements, and "
           "per-vital measurement counts); interval: patient-cluster bootstrap 95\\%. Lower block: models that read the last ten vital-sign rows of the whole stay; "
           "interval: standard deviation over five seeds.}\n\\label{tab:shortcuts}\n\\begin{tabular}{lccc}\n\\toprule\nInputs & AUROC & Interval & AUPRC \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(FLOATS, "tab_shortcuts.tex"), "w", encoding="utf-8").write(tex)
    ev.write_provenance()


def main():
    macros = macros_A()
    partB = have("ph2_contrasts.json") and have("ph2_conformal.json") and have("ph2_news2.json")
    if partB:
        import gen_assets_v2_b as B
        import gen_assets_v2_c as Cc
        macros = macros + B.macros_B() + B.macros_C() + Cc.macros_D()
    n = write_macros(macros, os.path.join(PAPER, "rv_numbers.tex"), "rv_numbers")
    print("macros:", n)
    table_cohort()
    table_shortcuts()
    print("tables A written")
    if partB:
        B.table_discrimination(); B.table_contrasts(); B.table_conformal_main(); B.table_intervals(); B.table_alpha(); B.table_lag_drift(); B.table_tuning(); B.table_calibration(); B.table_elixhauser(); B.table_duration(); B.table_stream(); B.table_other_models(); B.table_sweep(); B.table_years(); B.table_search(); B.table_acuity_shortcut(); B.table_fixed_cohort(); B.table_characteristics(); Cc.table_recording(); Cc.table_arrival(); Cc.table_static_cov(); Cc.table_intercept(); Cc.table_dedup()
        print("tables B written")
    label_proxy()


PROXY = "MIMIC-IV-ED, retrospective single-center proxy"


def label_proxy():
    """Rule 5 of CLAUDE.md: every table built from MIMIC-IV-ED states that the data are a retrospective single-center proxy. Appends the label to each
    caption that lacks it. The literature-search table holds no patient data and is left unlabelled."""
    import glob
    for p in sorted(glob.glob(os.path.join(FLOATS, "tab_*.tex"))):
        if os.path.basename(p) == "tab_search.tex":
            continue
        s = open(p, encoding="utf-8").read()
        out, i = [], 0
        while True:
            j = s.find("\\caption{", i)
            if j < 0:
                out.append(s[i:]); break
            k, depth = j + len("\\caption{"), 1
            while depth:
                depth += {"{": 1, "}": -1}.get(s[k], 0); k += 1
            cap = s[j:k - 1]
            if "proxy" not in cap:
                cap = cap.rstrip() + ("" if cap.rstrip().endswith(".") else ".") + " " + PROXY + "."
            out.append(s[i:j] + cap + "}"); i = k
        open(p, "w", encoding="utf-8").write("".join(out))


if __name__ == "__main__":
    main()
