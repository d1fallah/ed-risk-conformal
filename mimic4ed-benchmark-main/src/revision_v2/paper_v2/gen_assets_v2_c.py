"""Part C of the asset generator: macros, tables and the figure for the analyses added after the external review (recording practice by year, arrival
information, bootstrap intervals of static coverage, precision-recall operating points, intercept recalibration)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data_v2 as PD  # noqa: E402
from gen_assets_v2_b import _wrap, write, NL, SD, FLOATS  # noqa: E402

RC_, AI, AH, SB, PR, IR = "ph2_recording.json", "ph2_arrival_increment.json", "ph2_arrival_h2.json", "ph2_static_cov_boot.json", "ph2_pr_points.json", "ph2_intercept_recal.json"
YEARS = [str(y) for y in range(2009, 2021)]
YL = dict(zip(YEARS, "ABCDEFGHIJKL"))


def macros_D():
    L = []
    for y in YEARS:
        b = f"/by_year/{y}"
        L += [(f"RecZero{YL[y]}", RC_, b + "/zero_rows_by_L_share", "pct"), (f"RecEqTri{YL[y]}", RC_, b + "/first_row_equals_triage_share_of_comparable", "pct"),
              (f"RecDelay{YL[y]}", RC_, b + "/first_delay_h_q/2", ".2f"), (f"RecStays{YL[y]}", RC_, b + "/stays", ",d")]
    for k, nm in (("train", "Train"), ("test", "Test")):
        b = f"/by_period/{k}"
        L += [(f"RecEqTri{nm}", RC_, b + "/first_row_equals_triage_share_of_comparable", "pct"), (f"RecDelay{nm}", RC_, b + "/first_delay_h_q/2", ".2f"),
              (f"RecZero{nm}", RC_, b + "/zero_rows_by_L_share", "pct")]
    # arrival information on the end-of-stay task
    for k, nm in (("Arr", "Arr"), ("W", "Work"), ("Arr+W", "ArrWork"), ("Arr+duration", "ArrDur")):
        b = f"/models/{k}"
        L += [(f"Inc{nm}", AI, b + "/auroc", ".3f"), (f"Inc{nm}Lo", AI, b + "/auroc_ci95/0", ".3f"), (f"Inc{nm}Hi", AI, b + "/auroc_ci95/1", ".3f")]
    for k, nm in (("Arr+W minus Arr", "WorkOverArr"), ("Arr+duration minus Arr", "DurOverArr")):
        b = f"/differences/{k}"
        L += [(f"Inc{nm}", AI, b + "/delta", "+.3f"), (f"Inc{nm}Lo", AI, b + "/ci95/0", "+.3f"), (f"Inc{nm}Hi", AI, b + "/ci95/1", "+.3f")]
    # arrival versus two hours on the same stays and outcome
    for k, nm in (("arrival_trees", "ArrTrees"), ("arrival_logreg", "ArrLr"), ("h2_trees", "TwoTrees"), ("h2_logreg", "TwoLr"), ("h2_hybrid", "TwoHybrid")):
        b = f"/models/{k}"
        L += [(f"Common{nm}", AH, b + "/auroc", ".3f"), (f"Common{nm}Lo", AH, b + "/auroc_ci95/0", ".3f"), (f"Common{nm}Hi", AH, b + "/auroc_ci95/1", ".3f")]
    for k, nm in (("h2_trees minus arrival_trees", "TreesGain"), ("h2_logreg minus arrival_logreg", "LrGain")):
        b = f"/differences/{k}"
        L += [(f"Common{nm}", AH, b + "/delta", "+.3f"), (f"Common{nm}Lo", AH, b + "/ci95/0", "+.3f"), (f"Common{nm}Hi", AH, b + "/ci95/1", "+.3f")]
    L += [("CommonN", AH, "/n_common", ",d")]
    # bootstrap intervals of the static coverage
    for k, nm in (("hybrid_e2e", "Hyb"), ("xgboost_timing", "Xgb"), ("logreg", "Lr")):
        b = f"/models/{k}"
        L += [(f"StaticBoot{nm}", SB, b + "/point/marginal", ".3f"), (f"StaticBoot{nm}Lo", SB, b + "/marginal_ci95/0", ".3f"), (f"StaticBoot{nm}Hi", SB, b + "/marginal_ci95/1", ".3f"),
              (f"StaticBoot{nm}Share", SB, b + "/marginal_share_resamples_at_or_above_threshold", "pct"),
              (f"StaticBootMond{nm}", SB, b + "/point/class1_static_mondrian", ".3f"), (f"StaticBootMond{nm}Lo", SB, b + "/class1_static_mondrian_ci95/0", ".3f"),
              (f"StaticBootMond{nm}Hi", SB, b + "/class1_static_mondrian_ci95/1", ".3f")]
    # precision-recall operating points
    for k, nm in (("static_lac", "Lac"), ("static_mondrian_lac", "Mond"), ("mondrian_eci", "Eci"), ("mondrian_recal_w10k", "Recal")):
        b = f"/operating_points/{k}"
        L += [(f"Pr{nm}Ppv", PR, b + "/ppv", ".3f"), (f"Pr{nm}Sens", PR, b + "/sensitivity", ".3f"), (f"Pr{nm}Same", PR, b + "/same_sensitivity_threshold_ppv", ".3f")]
    # intercept recalibration (mean and SD over five seeds)
    for nm, ptr, f in (("MeanPredOrig", "calibration_original/mean_pred", "pct"), ("MeanPredRecal", "calibration_recalibrated/mean_pred", "pct"),
                       ("BrierOrig", "calibration_original/brier", ".4f"), ("BrierRecal", "calibration_recalibrated/brier", ".4f"),
                       ("EceOrig", "calibration_original/ece", ".3f"), ("EceRecal", "calibration_recalibrated/ece", ".3f"),
                       ("CovOrig", "static_original/coverage", ".3f"), ("CovOne", "variant_III_marginal_coverage", ".3f"), ("CovTwo", "variant_II_marginal_coverage", ".3f"),
                       ("OneOrig", "static_original/class1", ".3f"), ("OneOne", "variant_III_class1_coverage", ".3f"), ("OneTwo", "variant_II_class1_coverage", ".3f"),
                       ("MondOne", "variant_III_mondrian_class1", ".3f")):
        L += [(f"Ir{nm}", IR, f"/runs/*/{ptr}", f, "mean")]
    L += [("IrObserved", IR, "/runs/0/calibration_original/observed", "pct")]
    L += [("OverlapPrimaryOnly", "ph2_overlap.json", "/primary_only", ",d"), ("OverlapOriginalOnly", "ph2_overlap.json", "/original_only", ",d"),
          ("OverlapBoth", "ph2_overlap.json", "/primary_and_original", ",d"), ("OverlapOriginalEligible", "ph2_overlap.json", "/original_positive_among_eligible", ",d")]
    # de-duplicated and ablated classical models
    DD = "ph2_dedup.json"
    for v, nm in (("original", "Orig"), ("dd", "Dd"), ("nt", "Nt"), ("vo", "Vo")):
        for m, mn in (("trees", "Trees"), ("logreg", "Lr")):
            b = f"/auroc/{v}_{m}"
            L += [(f"{nm}{mn}", DD, b + "/auroc", ".3f"), (f"{nm}{mn}Lo", DD, b + "/ci95/0", ".3f"), (f"{nm}{mn}Hi", DD, b + "/ci95/1", ".3f")]
        b = f"/differences/{v}_trees minus {v}_logreg"
        L += [(f"{nm}TreesOverLr", DD, b + "/delta", "+.4f"), (f"{nm}TreesOverLrLo", DD, b + "/ci95/0", "+.4f"), (f"{nm}TreesOverLrHi", DD, b + "/ci95/1", "+.4f")]
        if v != "original":
            for m, mn in (("trees", "Trees"), ("logreg", "Lr")):
                b = f"/differences/{v}_{m} minus original_{m}"
                L += [(f"{nm}{mn}Change", DD, b + "/delta", "+.5f"), (f"{nm}{mn}ChangeLo", DD, b + "/ci95/0", "+.5f"), (f"{nm}{mn}ChangeHi", DD, b + "/ci95/1", "+.5f")]
    GD = "ph2_grud_dedup.json"
    for kind, nm in (("e2e", "GrudHybrid"), ("temporal", "GrudTemp")):
        b = f"/models/{kind}"
        L += [(f"{nm}Stored", GD, b + "/auroc_stored_seed_mean", ".4f"), (f"{nm}Dd", GD, b + "/auroc_dedup_seed_mean", ".4f"),
              (f"{nm}Change", GD, b + "/change_seed_mean", "+.5f"), (f"{nm}ChangeLo", GD, b + "/change_seed_mean_ci95/0", "+.5f"), (f"{nm}ChangeHi", GD, b + "/change_seed_mean_ci95/1", "+.5f"),
              (f"{nm}ChangeMaxAbs", GD, b + "/max_abs_change_over_seeds", ".4f")]
    L += [("GrudHybridSeedLow", GD, "/models/e2e/per_seed/1/change", "+.4f"), ("GrudHybridSeedHigh", GD, "/models/e2e/per_seed/2/change", "+.4f"),
          ("GrudTempSeedLow", GD, "/models/temporal/per_seed/2/change", "+.4f"), ("GrudTempSeedHigh", GD, "/models/temporal/per_seed/0/change", "+.4f"),
          ("GrudDdFlaggedTest", GD, "/flagged_share_test", "pct"), ("GrudRule", GD, "/rule_threshold", ".3f")]
    L += [("DdFlaggedFit", DD, "/variants/dd/flagged_share_by_role/fit", "pct"), ("DdFlaggedTest", DD, "/variants/dd/flagged_share_by_role/test", "pct")]
    return L


def ci(ev, art, b, f=".3f"):
    return ev.get(art, b + "/0", fmt=f) + ", " + ev.get(art, b + "/1", fmt=f)


def table_recording():
    ev = PD.EV("tab_recording")
    rows = []
    for y in YEARS:
        b = f"/by_year/{y}"
        rows.append(" & ".join([y, ev.get(RC_, b + "/stays", fmt=",d"), ev.get(RC_, b + "/zero_rows_by_L_share", fmt="pct"),
                                ev.get(RC_, b + "/first_delay_h_q/2", fmt=".2f") + " (" + ev.get(RC_, b + "/first_delay_h_q/1", fmt=".2f") + "--" + ev.get(RC_, b + "/first_delay_h_q/3", fmt=".2f") + ")",
                                ev.get(RC_, b + "/first_row_equals_triage_share_of_comparable", fmt="pct")]) + NL)
    tex = _wrap("tab:recording", "Recording of ED vital signs by approximate calendar year (stays eligible at the two-hour landmark; the training period ends in 2016 and the test period begins in 2017; "
                + ev.get(RC_, "/by_year/2008/stays", fmt=",d") + " stay of 2008 omitted). Delay: hours from arrival to the first charted ED vital sign, median (interquartile range), among stays with one. "
                "Copy of triage: share of stays whose first ED vital-sign row has the same values as the triage measurement (stays with at least three comparable vital signs).",
                "lrrcc", "Year & Stays & No ED vital sign by $L$ & Delay to first ED row, h & Copy of triage", rows)
    write("tab_recording", tex, ev)


def table_arrival():
    ev = PD.EV("tab_arrival")
    rows = ["\\multicolumn{3}{l}{\\emph{End-of-stay task (logistic regression; test period)}}" + NL]
    for k, label in (("Arr", "Arrival information"), ("W", "Workflow quantities"), ("Arr+duration", "Arrival information and ED duration"), ("Arr+W", "Arrival information and workflow quantities")):
        rows.append(" & ".join([label, ev.get(AI, f"/models/{k}/auroc", fmt=".3f"), "[" + ci(ev, AI, f"/models/{k}/auroc_ci95") + "]"]) + NL)
    for k, label in (("Arr+duration minus Arr", "ED duration added to arrival information"), ("Arr+W minus Arr", "Workflow quantities added to arrival information")):
        rows.append(" & ".join([label + " (difference)", ev.get(AI, f"/differences/{k}/delta", fmt="+.3f"), "[" + ci(ev, AI, f"/differences/{k}/ci95", "+.3f") + "]"]) + NL)
    rows.append("\\midrule\n\\multicolumn{3}{l}{\\emph{Same " + ev.get(AH, "/n_common", fmt=",d") + " stays and the two-hour outcome}}" + NL)
    for k, label in (("arrival_logreg", "Arrival landmark, logistic regression"), ("arrival_trees", "Arrival landmark, trees"), ("h2_logreg", "Two-hour landmark, logistic regression"),
                     ("h2_trees", "Two-hour landmark, timing-aware trees"), ("h2_hybrid", "Two-hour landmark, Hybrid GRU-D")):
        rows.append(" & ".join([label, ev.get(AH, f"/models/{k}/auroc", fmt=".3f"), "[" + ci(ev, AH, f"/models/{k}/auroc_ci95") + "]"]) + NL)
    for k, label in (("h2_logreg minus arrival_logreg", "Two hours over arrival, logistic regression"), ("h2_trees minus arrival_trees", "Two hours over arrival, trees")):
        rows.append(" & ".join([label + " (difference)", ev.get(AH, f"/differences/{k}/delta", fmt="+.3f"), "[" + ci(ev, AH, f"/differences/{k}/ci95", "+.3f") + "]"]) + NL)
    tex = _wrap("tab:arrival", "Information at arrival and what is added later. Upper block: end-of-stay task of the original analysis; arrival information is the triage vital signs and their "
                "missingness, age, sex, triage acuity, and comorbidity count. Lower block: arrival-landmark models and two-hour models scored on the same test stays against the same outcome "
                "(events in the 24 hours after the two-hour landmark). AUROC with the patient-cluster bootstrap 95\\% interval; differences use paired resamples.",
                "lcc", "Inputs or model & AUROC & 95\\% CI", rows)
    write("tab_arrival", tex, ev)


def table_static_cov():
    ev = PD.EV("tab_static_cov")
    rows = []
    for k, label in (("hybrid_e2e", "Hybrid GRU-D, end to end"), ("xgboost_timing", "Timing-aware trees"), ("logreg", "Logistic regression")):
        b = f"/models/{k}"
        rows.append(" & ".join([label, ev.get(SB, b + "/point/marginal", fmt=".3f"), "[" + ci(ev, SB, b + "/marginal_ci95") + "]", ev.get(SB, b + "/marginal_share_resamples_at_or_above_threshold", fmt="pct"),
                                ev.get(SB, b + "/point/class1_static_mondrian", fmt=".3f"), "[" + ci(ev, SB, b + "/class1_static_mondrian_ci95") + "]"]) + NL)
    tex = _wrap("tab:static-cov", "Sampling uncertainty of the coverage of the static conformal sets on the test period ($\\alpha = 0.10$; seed-averaged coverage; "
                + ev.get(SB, "/bootstrap_resamples", fmt=",d") + " patient-cluster bootstrap resamples, 95\\% percentile interval). Share: resamples whose marginal coverage reaches the validity "
                "threshold of 0.89. The bootstrap is valid for static methods, which apply a fixed threshold, and is not used for the online methods.",
                "lccccc", "Model & Marginal coverage & 95\\% CI & Share $\\geq 0.89$ & Class-1 coverage, Mondrian & 95\\% CI", rows)
    write("tab_static_cov", tex, ev)


def table_intercept():
    ev = PD.EV("tab_intercept")

    def ms(ptr, f):
        return ev.reduce("mean", IR, f"/runs/*/{ptr}", fmt=f) + SD + ev.reduce("std", IR, f"/runs/*/{ptr}", fmt=f)
    rows = [
        " & ".join(["Mean predicted risk"] + [ms("calibration_original/mean_pred", "pct"), ms("calibration_recalibrated/mean_pred", "pct"), "as matched"]) + NL,
        " & ".join(["Brier score"] + [ms("calibration_original/brier", ".4f"), ms("calibration_recalibrated/brier", ".4f"), "as matched"]) + NL,
        " & ".join(["Expected calibration error"] + [ms("calibration_original/ece", ".3f"), ms("calibration_recalibrated/ece", ".3f"), "as matched"]) + NL,
        "\\midrule",
        " & ".join(["Marginal coverage, LAC sets"] + [ms("static_original/coverage", ".3f"), ms("variant_III_marginal_coverage", ".3f"), ms("variant_II_marginal_coverage", ".3f")]) + NL,
        " & ".join(["Class-1 coverage, LAC sets"] + [ms("static_original/class1", ".3f"), ms("variant_III_class1_coverage", ".3f"), ms("variant_II_class1_coverage", ".3f")]) + NL,
        " & ".join(["Mean set size, LAC sets"] + [ms("static_original/size", ".2f"), ms("variant_III_set_size", ".2f"), "--"]) + NL,
        " & ".join(["Class-1 coverage, static Mondrian"] + [ev.reduce("mean", "ph2_conformal.json", "/runs/*/sweep/0.1/static_mondrian_lac/metrics/coverage_class1", fmt=".3f") + SD + ev.reduce("std", "ph2_conformal.json", "/runs/*/sweep/0.1/static_mondrian_lac/metrics/coverage_class1", fmt=".3f"), ms("variant_III_mondrian_class1", ".3f"), "--"]) + NL,
    ]
    tex = _wrap("tab:intercept", "Periodic recalibration of the intercept of the primary model's score (test period, mean $\\pm$ SD over five seeds). The logit of the score is shifted by an intercept estimated on "
                "the most recent labelled arrivals and re-estimated every 2{,}000 arrivals. Original: static LAC sets on the original score. Matched: for each block of 2{,}000 arrivals the same intercept shift is applied to the final-calibration scores and to the block, "
                "and the static quantile is recomputed on the shifted calibration scores. Rolling quantile: the recalibrated score with a quantile recomputed every 2{,}000 arrivals from the most recent 10{,}000. Observed event rate "
                + ev.get(IR, "/runs/0/calibration_original/observed", fmt="pct") + ".",
                "lccc", "& Original & Intercept, matched & Intercept, rolling quantile", rows)
    write("tab_intercept", tex, ev)


def table_dedup():
    ev = PD.EV("tab_dedup")
    DD = "ph2_dedup.json"
    rows = []
    for v, label in (("original", "Primary analysis (all rows, timing features)"), ("dd", "Copied triage row removed"), ("nt", "No timing features"), ("vo", "Vital-sign values and static profile only")):
        cells = [label]
        for m in ("logreg", "trees"):
            b = f"/auroc/{v}_{m}"
            cells.append(ev.get(DD, b + "/auroc", fmt=".3f") + " [" + ci(ev, DD, b + "/ci95") + "]")
        b = f"/differences/{v}_trees minus {v}_logreg"
        cells.append(ev.get(DD, b + "/delta", fmt="+.4f") + " [" + ci(ev, DD, b + "/ci95", "+.4f") + "]")
        for m in ("logreg", "trees"):
            if v == "original":
                cells.append("--")
            else:
                b = f"/differences/{v}_{m} minus original_{m}"
                cells.append(ev.get(DD, b + "/delta", fmt="+.5f") + " [" + ci(ev, DD, b + "/ci95", "+.5f") + "]")
        rows.append(" & ".join(cells) + NL)
    tex = _wrap("tab:dedup", "Logistic regression and gradient-boosted trees at the two-hour landmark with the copy of the triage measurement removed and with features removed "
                "(test period; same stays, outcome, tuning budget and seeds as the primary analysis; AUROC with the patient-cluster bootstrap 95\% interval, paired for differences). "
                "Copied triage row: the first ED vital-sign row removed when it repeats the triage measurement, "
                + ev.get(DD, "/variants/dd/flagged_share_by_role/fit", fmt="pct") + " of stays of the fitting subset and " + ev.get(DD, "/variants/dd/flagged_share_by_role/test", fmt="pct")
                + " of the test period. The change columns are relative to the primary analysis.", "lcccll",
                "Variant & Logistic regression & Trees & Trees minus logistic regression & Change, logistic regression & Change, trees", rows)
    write("tab_dedup", tex, ev)
