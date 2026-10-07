"""Part B of the asset generator: tables and macros that depend on the Phase 2 model runs (contrasts, conformal suite, NEWS2)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data_v2 as PD  # noqa: E402

PAPER = os.path.abspath(os.path.join(PD.ROOT, "..", "paper-m2-rev"))
FLOATS = os.path.join(PAPER, "floats")
K, N, CF = "ph2_contrasts.json", "ph2_news2.json", "ph2_conformal.json"
H2 = [("logreg", "Logistic regression"), ("xgboost_timing", "Timing-aware trees"), ("temporal_grud", "Temporal-Only GRU-D"),
      ("hybrid_frozen", "Hybrid GRU-D, frozen encoder"), ("hybrid_e2e", "Hybrid GRU-D, end to end")]
TR = [("logreg", "Logistic regression"), ("xgboost", "Gradient-boosted trees")]
SD = "\\,$\\pm$\\,"


def write(name, tex, ev):
    open(os.path.join(FLOATS, name + ".tex"), "w", encoding="utf-8").write(tex)
    ev.write_provenance()


def table_discrimination():
    ev = PD.EV("tab_discrimination")
    rows = []

    def block(lm, models, news_key):
        for k, label in models:
            b = f"/landmarks/{lm}/models/{k}"
            sd = f"{SD}{ev.get(K, b + '/auroc_seed_sd', fmt='.4f')}" if k not in ("logreg",) else ""
            sec = f"{ev.get(K, f'/landmarks/{lm}/secondary_outcome/models/{k}/auroc_seed_mean', fmt='.3f')}"
            rows.append(f"{label} & {ev.get(K, b + '/auroc_seed_mean', fmt='.3f')}{sd} & [{ev.get(K, b + '/auroc_bootstrap_ci95_of_seed_mean/0', fmt='.3f')}, "
                        f"{ev.get(K, b + '/auroc_bootstrap_ci95_of_seed_mean/1', fmt='.3f')}] & {ev.get(K, b + '/auprc_seed_mean', fmt='.3f')} & {sec} \\\\")
        rows.append(f"NEWS2 (five parameters) & {ev.get(N, f'/landmarks/{news_key}/auroc', fmt='.3f')} & [{ev.get(N, f'/landmarks/{news_key}/auroc_ci95/0', fmt='.3f')}, "
                    f"{ev.get(N, f'/landmarks/{news_key}/auroc_ci95/1', fmt='.3f')}] & -- & -- \\\\")
    rows.append("\\multicolumn{5}{l}{\\emph{2-hour landmark}} \\\\")
    block("h2", H2, "h2")
    rows.append("\\midrule\n\\multicolumn{5}{l}{\\emph{Arrival (triage) landmark}} \\\\")
    block("triage", TR, "triage")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Discrimination on the test period for the primary outcome $Y_L$ (ICU admission or in-hospital death within 24\\,h of the landmark; "
           "MIMIC-IV-ED, retrospective single-center proxy). AUROC is the mean over five seeds with the standard deviation across seeds; the interval is the patient-cluster "
           "bootstrap 95\\% interval of the seed-averaged AUROC. Sec.: AUROC for the secondary outcome (benchmark composite after $L$). The primary model is Hybrid GRU-D trained end to end.}\n\\label{tab:discrimination}\n"
           "\\begin{tabular}{lcccc}\n\\toprule\nModel & AUROC & 95\\% CI & AUPRC & Sec. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_discrimination", tex, ev)


def table_contrasts():
    ev = PD.EV("tab_contrasts")
    spec = [("C1", "h2", "Hybrid GRU-D minus timing-aware trees"), ("C2", "h2", "Hybrid GRU-D minus logistic regression"),
            ("C3", "h2", "Hybrid GRU-D minus Temporal-Only GRU-D"), ("C4", "h2", "Hybrid GRU-D minus frozen-encoder variant"),
            ("C5", "triage", "Trees minus logistic regression (arrival landmark)")]
    rows = []
    for cid, lm, label in spec:
        b = f"/landmarks/{lm}/contrasts/{cid}"
        sig = ev.get(K, b + "/significant", note="not displayed: selects yes or no")
        rows.append(f"{cid} & {label} & {ev.get(K, b + '/delta_seed_mean', fmt='+.4f')}{SD}{ev.get(K, b + '/delta_seed_sd', fmt='.4f')} & "
                    f"[{ev.get(K, b + '/delta_bootstrap_ci95_of_seed_mean/0', fmt='+.4f')}, {ev.get(K, b + '/delta_bootstrap_ci95_of_seed_mean/1', fmt='+.4f')}] & "
                    f"{ev.get(K, b + '/holm_adjusted_p', fmt='.4f')} & {'yes' if sig else 'no'} \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Pre-specified AUROC contrasts on the test period. Difference: mean $\\pm$ SD over five seeds (training variability); interval: patient-cluster "
           "bootstrap 95\\% interval of the seed-averaged difference (sampling uncertainty). Holm correction over the five contrasts (the smallest attainable adjusted value with " + ev.get(K, "/bootstrap_primary_resamples", fmt=",d") + " resamples is 0.0005). A contrast is called significant only if it is "
           "Holm-significant and all five per-seed intervals exclude zero.}\n\\label{tab:contrasts}\n\\begin{tabular}{llccccc}\n\\toprule\n"
           "ID & Contrast & Difference & 95\\% CI & Holm $p$ & Significant \\\\\n\\midrule\n" + "\n".join(r.replace(" & ", " & ", 1) for r in rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    tex = tex.replace("\\begin{tabular}{llccccc}", "\\begin{tabular}{llcccc}")
    write("tab_contrasts", tex, ev)


def table_conformal_main():
    ev = PD.EV("tab_conformal")
    methods = [("static_lac", "Static LAC (marginal)"), ("eci_sets", "ECI sets (marginal)"), ("static_mondrian_lac", "Static Mondrian-LAC"),
               ("mondrian_eci", "Mondrian-ECI"), ("mondrian_recal_w10k", "Mondrian recalibration, $W = 10$k")]
    rows = []
    for k, label in methods:
        b = f"/runs/*/sweep/0.1/{k}"
        rows.append(" & ".join([label, f"{ev.reduce('mean', CF, b + '/metrics/coverage_marginal', fmt='.3f')}{SD}{ev.reduce('std', CF, b + '/metrics/coverage_marginal', fmt='.3f')}",
                                f"{ev.reduce('mean', CF, b + '/metrics/coverage_class1', fmt='.3f')}{SD}{ev.reduce('std', CF, b + '/metrics/coverage_class1', fmt='.3f')}",
                                ev.reduce("mean", CF, b + "/metrics/mean_set_size", fmt=".2f"), ev.reduce("mean", CF, b + "/metrics/rate_singleton_1", fmt="pct"),
                                ev.reduce("mean", CF, b + "/operating/ppv_only1", fmt=".2f"), ev.reduce("mean", CF, b + "/operating/sensitivity_only1", fmt=".2f"),
                                ev.reduce("mean", CF, b + "/operating/false_alarms_per_100_patients", fmt=".1f"),
                                ev.reduce("mean", CF, b + "/operating/abstain_any_class1", fmt="pct")]) + " \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Conformal prediction sets for the primary model at $\\alpha = 0.10$ (target coverage 0.90), test period, mean over five seeds "
           "($\\pm$ SD across seeds). $\\{1\\}$ denotes the singleton set containing the deterioration label; PPV and sensitivity refer to that set; false alarms are $\\{1\\}$ sets "
           "for stays without the outcome, per 100 patients; abstention is a set with both labels or none, among stays with the outcome.}\n\\label{tab:conformal}\n"
           "\\begin{tabular}{lcccccccc}\n\\toprule\nMethod & Coverage & Class-1 coverage & Set size & $\\{1\\}$ rate & PPV & Sensitivity & False alarms / 100 & Abstention, class 1 \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write("tab_conformal", tex, ev)


def table_intervals():
    ev = PD.EV("tab_intervals")
    rows = []
    for k, label in (("static", "Static split-conformal"), ("eci", "ECI"), ("recal_w10k", "Recalibration, $W = 10$k"), ("recal_w40k", "Recalibration, $W = 40$k")):
        b = f"/runs/*/intervals/{k}"
        rows.append(" & ".join([label, f"{ev.reduce('mean', CF, b + '/coverage', fmt='.3f')}{SD}{ev.reduce('std', CF, b + '/coverage', fmt='.3f')}",
                                ev.reduce("mean", CF, b + "/coverage_class1", fmt=".3f"), f"{ev.reduce('mean', CF, b + '/width', fmt='.3f')}{SD}{ev.reduce('std', CF, b + '/width', fmt='.3f')}"]) + " \\\\")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Score intervals for the primary model at $\\alpha = 0.10$ on the test period: marginal and class-1 coverage and mean width "
           "(mean over five seeds, $\\pm$ SD across seeds). Validity threshold 0.89.}\n\\label{tab:intervals}\n\\begin{tabular}{lccc}\n\\toprule\nMethod & Coverage & Class-1 coverage & Width \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_intervals", tex, ev)


def table_alpha():
    ev = PD.EV("tab_alpha")
    rows = []
    for a in ("0.2", "0.1", "0.05", "0.01"):
        for k, label in (("static_lac", "Static LAC"), ("static_mondrian_lac", "Static Mondrian-LAC"), ("mondrian_eci", "Mondrian-ECI")):
            b = f"/runs/*/sweep/{a}/{k}"
            rows.append(" & ".join([f"{1 - float(a):.2f}", label, ev.reduce("mean", CF, b + "/metrics/coverage_marginal", fmt=".3f"), ev.reduce("mean", CF, b + "/metrics/coverage_class1", fmt=".3f"),
                                    ev.reduce("mean", CF, b + "/metrics/mean_set_size", fmt=".2f"), ev.reduce("mean", CF, b + "/operating/ppv_only1", fmt=".2f"),
                                    ev.reduce("mean", CF, b + "/operating/false_alarms_per_100_patients", fmt=".1f")]) + " \\\\")
        rows.append("\\addlinespace")
    npos = ev.get(CF, "/runs/0/pos_fc", fmt=",d")
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Behavior across the nominal coverage level $1 - \\alpha$ for the primary model (test period, mean over five seeds). The class-1 quantile is "
           "feasible at every level (" + npos + " final-calibration positives).}\n\\label{tab:alpha}\n\\begin{tabular}{llccccc}\n\\toprule\n$1-\\alpha$ & Method & Cov. & Class-1 cov. & Set size & PPV & False alarms / 100 \\\\\n\\midrule\n"
           + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_alpha", tex, ev)


def table_lag_drift():
    ev = PD.EV("tab_lag_drift")
    rows = []
    for L in ("60", "240", "1000"):
        b = f"/runs/*/lag/{L}"
        rows.append(" & ".join([L, ev.reduce("mean", CF, b + "/eci_interval/coverage", fmt=".3f"), ev.reduce("mean", CF, b + "/eci_interval/width", fmt=".3f"),
                                ev.reduce("mean", CF, b + "/recal_w10k_interval/coverage", fmt=".3f"), ev.reduce("mean", CF, b + "/recal_w10k_interval/width", fmt=".3f"),
                                ev.reduce("mean", CF, b + "/mondrian_eci/coverage_class1", fmt=".3f"), ev.reduce("mean", CF, b + "/mondrian_eci/mean_set_size", fmt=".2f"),
                                ev.reduce("mean", CF, b + "/mondrian_recal_w10k/coverage_class1", fmt=".3f")]) + " \\\\")
    d = []
    for sc, label in (("shift+0.5", "logit shift $+0.5$"), ("shift+1.0", "logit shift $+1.0$"), ("casemix", "case-mix change")):
        cells = []
        for m in ("static", "recal_w10k", "eci_interval"):
            b = f"/runs/*/drift/{sc}/{m}"
            cells += [ev.reduce("mean", CF, b + "/cov_pre", fmt=".3f"), ev.reduce("mean", CF, b + "/cov_post", fmt=".3f")]
        d.append(" & ".join([label] + cells) + " \\\\")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Robustness of the primary model's calibration (test period, mean over five seeds). Left: labels delayed by $L_d$ arrivals; coverage and width of the score "
           "intervals, and class-1 coverage of the Mondrian sets. Right: coverage before and after an imposed drift at the stream midpoint (a stress test on real predictions, not a finding about the hospital).}\n"
           "\\label{tab:lag-drift}\n\\begin{tabular}{lccccccc}\n\\toprule\n$L_d$ & ECI cov. & ECI width & Recal. cov. & Recal. width & Mond.-ECI class-1 cov. & Mond.-ECI size & Mond.-recal. class-1 cov. \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n\\medskip\n\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{Static} & \\multicolumn{2}{c}{Recalibration $W=10$k} & \\multicolumn{2}{c}{ECI} \\\\\n"
           "Drift & before & after & before & after & before & after \\\\\n\\midrule\n" + "\n".join(d) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write("tab_lag_drift", tex, ev)


def macros_B():
    h2, tr = "/landmarks/h2", "/landmarks/triage"
    L = []
    for k, nm in (("logreg", "Lr"), ("xgboost_timing", "XgbT"), ("temporal_grud", "Tmp"), ("hybrid_frozen", "Frz"), ("hybrid_e2e", "EndToEnd")):
        L += [(f"Auroc{nm}", K, f"{h2}/models/{k}/auroc_seed_mean", ".3f"), (f"Sd{nm}", K, f"{h2}/models/{k}/auroc_seed_sd", ".3f"),
              (f"Lo{nm}", K, f"{h2}/models/{k}/auroc_bootstrap_ci95_of_seed_mean/0", ".3f"), (f"Hi{nm}", K, f"{h2}/models/{k}/auroc_bootstrap_ci95_of_seed_mean/1", ".3f"),
              (f"Auprc{nm}", K, f"{h2}/models/{k}/auprc_seed_mean", ".3f")]
    for k, nm in (("logreg", "TrLr"), ("xgboost", "TrXgb")):
        L += [(f"Auroc{nm}", K, f"{tr}/models/{k}/auroc_seed_mean", ".3f"), (f"Lo{nm}", K, f"{tr}/models/{k}/auroc_bootstrap_ci95_of_seed_mean/0", ".3f"),
              (f"Hi{nm}", K, f"{tr}/models/{k}/auroc_bootstrap_ci95_of_seed_mean/1", ".3f")]
    for cid, lm in (("C1", "h2"), ("C2", "h2"), ("C3", "h2"), ("C4", "h2"), ("C5", "triage")):
        nm = {"C1": "COne", "C2": "CTwo", "C3": "CThree", "C4": "CFour", "C5": "CFive"}[cid]
        b = f"/landmarks/{lm}/contrasts/{cid}"
        L += [(f"{nm}Delta", K, b + "/delta_seed_mean", "+.4f"), (f"{nm}Lo", K, b + "/delta_bootstrap_ci95_of_seed_mean/0", "+.4f"),
              (f"{nm}Hi", K, b + "/delta_bootstrap_ci95_of_seed_mean/1", "+.4f"), (f"{nm}Holm", K, b + "/holm_adjusted_p", ".4f")]
    L += [("TreesOverHybrid", "derived.json", "/trees_minus_hybrid_delta", ".4f"), ("TreesOverHybridLo", "derived.json", "/trees_minus_hybrid_lo", ".4f"),
          ("TreesOverHybridHi", "derived.json", "/trees_minus_hybrid_hi", ".4f"), ("FrozenOverEndToEnd", "derived.json", "/frozen_minus_e2e_delta", ".4f")]
    L += [("NewsHtwo", N, "/landmarks/h2/auroc", ".3f"), ("NewsHtwoLo", N, "/landmarks/h2/auroc_ci95/0", ".3f"), ("NewsHtwoHi", N, "/landmarks/h2/auroc_ci95/1", ".3f"),
          ("NewsTriage", N, "/landmarks/triage/auroc", ".3f"), ("NewsTriageComplete", N, "/landmarks/triage/complete_case_auroc", ".3f"),
          ("NewsHtwoComplete", N, "/landmarks/h2/complete_case_auroc", ".3f"), ("NewsTriageAllMissing", N, "/landmarks/triage/stays_with_all_five_missing", ",d")]
    return L


def table_tuning():
    """Supplementary table: every configuration of every tuned model with its validation AUROC (mean over the three tuning seeds) and the selected one."""
    ev = PD.EV("tab_tuning")
    CL = "ph2_classical.json"
    rows = []
    NL = " \\\\"

    def mark(flag):
        return "selected" if flag else ""
    for lm, title in (("h2", r"2\,h landmark"), ("triage", "Arrival landmark")):
        xgb_key = "xgboost_timing" if lm == "h2" else "xgboost"
        sel = ev.get(CL, f"/{lm}/logreg/selected/C", note="not displayed: marks the selected row")
        for i in range(4):
            c = ev.get(CL, f"/{lm}/logreg/tuning_trace/{i}/C", fmt="g")
            v = ev.get(CL, f"/{lm}/logreg/tuning_trace/{i}/val_auroc", fmt=".4f")
            rows.append(f"{title} & Logistic regression & $C = {c}$ & {v} & {mark(float(c) == float(sel))}" + NL)
        sd = ev.get(CL, f"/{lm}/{xgb_key}/selected/max_depth", note="not displayed: marks the selected row")
        sl = ev.get(CL, f"/{lm}/{xgb_key}/selected/learning_rate", note="not displayed: marks the selected row")
        for i in range(4):
            d = ev.get(CL, f"/{lm}/{xgb_key}/tuning_trace/{i}/max_depth", fmt="d")
            l = ev.get(CL, f"/{lm}/{xgb_key}/tuning_trace/{i}/learning_rate", fmt="g")
            v = ev.get(CL, f"/{lm}/{xgb_key}/tuning_trace/{i}/val_auroc_mean", fmt=".4f")
            name = "Timing-aware trees" if lm == "h2" else "Gradient-boosted trees"
            rows.append(f"{title} & {name} & depth {d}, rate {l} & {v} & {mark(int(d) == int(sd) and float(l) == float(sl))}" + NL)
    for art, name in (("ph2_sel_temporal.json", "Temporal-Only GRU-D"), ("ph2_sel_e2e.json", "Hybrid GRU-D, end to end")):
        si = ev.get(art, "/selected_index", note="not displayed: marks the selected row")
        for i in range(4):
            h = ev.get(art, f"/tuning_trace/{i}/config/0", fmt="d")
            lr = ev.get(art, f"/tuning_trace/{i}/config/1", fmt="g")
            v = ev.get(art, f"/tuning_trace/{i}/val_auroc_mean", fmt=".4f")
            rows.append(r"2\,h landmark" + f" & {name} & hidden {h}, rate {lr} & {v} & {mark(i == si)}" + NL)
    art = "ph2_sel_hybrid_frozen.json"
    si = ev.get(art, "/selected_index", note="not displayed: marks the selected row")
    for i in range(4):
        h = ev.get(art, f"/tuning_trace/{i}/config/0", fmt="d")
        dr = ev.get(art, f"/tuning_trace/{i}/config/1", fmt="g")
        v = ev.get(art, f"/tuning_trace/{i}/val_auroc_mean", fmt=".4f")
        rows.append(r"2\,h landmark" + f" & Hybrid GRU-D, frozen encoder & static width {h}, dropout {dr} & {v} & {mark(i == si)}" + NL)
    head = (r"\begin{table}[t]" "\n" r"\centering" "\n"
            r"\caption{Tuning budget and outcome: four configurations per model, three tuning seeds, validation AUROC on the primary outcome (mean over the tuning seeds). "
            r"Logistic regression is deterministic. The search spaces were written to file before any run.}" "\n"
            r"\label{tab:tuning}" "\n" r"\begin{tabular}{llccc}" "\n" r"\toprule" "\n"
            r"Landmark & Model & Configuration & Validation AUROC & " + NL + "\n" + r"\midrule" + "\n")
    tex = head + "\n".join(rows) + "\n" + r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n"
    write("tab_tuning", tex, ev)


def macros_C():
    """Narrative numbers for the Results, Discussion, and Abstract. Tuples: (suffix, artifact, pointer, fmt[, reduction])."""
    CFx = "ph2_conformal.json"
    CA = "ph2_calibration.json"
    L = []
    # ---- intervals at alpha = 0.10 (primary model): seed mean and SD
    for k, nm in (("static", "Static"), ("eci", "Eci"), ("recal_w10k", "RecalTen"), ("recal_w40k", "RecalForty")):
        b = f"/runs/*/intervals/{k}"
        L += [(f"IntCov{nm}", CFx, b + "/coverage", ".3f", "mean"), (f"IntCovSd{nm}", CFx, b + "/coverage", ".3f", "std"),
              (f"IntWid{nm}", CFx, b + "/width", ".3f", "mean"), (f"IntCovOne{nm}", CFx, b + "/coverage_class1", ".3f", "mean")]
    YN = ["First", "Second", "Third", "Fourth"]
    L += [(f"YearN{YN[i]}", CFx, f"/runs/0/year_block_n/{i}", ",d") for i in range(4)]
    L += [(f"YearCov{YN[i]}", CFx, f"/runs/*/static_interval_year_coverage/{i}", ".3f", "mean") for i in range(4)]
    L += [(f"YearCovEci{YN[i]}", CFx, f"/runs/*/eci_interval_year_coverage/{i}", ".3f", "mean") for i in range(4)]
    L += [("NCs", CFx, "/runs/0/n_cs", ",d"), ("NFc", CFx, "/runs/0/n_fc", ",d"), ("NTest", CFx, "/runs/0/n_test", ",d"),
          ("PosCs", CFx, "/runs/0/pos_cs", ",d"), ("PosFc", CFx, "/runs/0/pos_fc", ",d"), ("PosTest", CFx, "/runs/0/pos_test", ",d")]
    # ---- sets at alpha = 0.10
    for k, nm in (("static_lac", "StaticLac"), ("eci_sets", "EciSets"), ("static_mondrian_lac", "StaticMond"), ("mondrian_eci", "MondEci"), ("mondrian_recal_w10k", "MondRecal")):
        b = f"/runs/*/sweep/0.1/{k}"
        L += [(f"SetCov{nm}", CFx, b + "/metrics/coverage_marginal", ".3f", "mean"), (f"SetCovOne{nm}", CFx, b + "/metrics/coverage_class1", ".3f", "mean"),
              (f"SetCovZero{nm}", CFx, b + "/metrics/coverage_class0", ".3f", "mean"),
              (f"SetSize{nm}", CFx, b + "/metrics/mean_set_size", ".2f", "mean"), (f"SetSingle{nm}", CFx, b + "/metrics/rate_singleton_1", "pct", "mean"),
              (f"SetPpv{nm}", CFx, b + "/operating/ppv_only1", ".2f", "mean"), (f"SetSens{nm}", CFx, b + "/operating/sensitivity_only1", ".2f", "mean"),
              (f"SetFa{nm}", CFx, b + "/operating/false_alarms_per_100_patients", ".1f", "mean"), (f"SetAbstOne{nm}", CFx, b + "/operating/abstain_any_class1", "pct", "mean"),
              (f"SetFpr{nm}", CFx, b + "/operating/fpr_only1", ".3f", "mean")]
    # ---- alpha sweep highlights
    for a, an in (("0.2", "Twenty"), ("0.05", "Five"), ("0.01", "One")):
        for k, nm in (("static_lac", "StaticLac"), ("mondrian_eci", "MondEci")):
            b = f"/runs/*/sweep/{a}/{k}"
            L += [(f"A{an}CovOne{nm}", CFx, b + "/metrics/coverage_class1", ".3f", "mean"), (f"A{an}Ppv{nm}", CFx, b + "/operating/ppv_only1", ".2f", "mean"),
                  (f"A{an}Fa{nm}", CFx, b + "/operating/false_alarms_per_100_patients", ".1f", "mean"), (f"A{an}Size{nm}", CFx, b + "/metrics/mean_set_size", ".2f", "mean")]
    # ---- label delay (L = 60 and 1000) and drift
    for lag, ln in (("60", "Sixty"), ("1000", "Thousand")):
        b = f"/runs/*/lag/{lag}"
        L += [(f"Lag{ln}EciCov", CFx, b + "/eci_interval/coverage", ".3f", "mean"), (f"Lag{ln}EciWid", CFx, b + "/eci_interval/width", ".3f", "mean"),
              (f"Lag{ln}RecalCov", CFx, b + "/recal_w10k_interval/coverage", ".3f", "mean"), (f"Lag{ln}MondEciCovOne", CFx, b + "/mondrian_eci/coverage_class1", ".3f", "mean"),
              (f"Lag{ln}MondEciSize", CFx, b + "/mondrian_eci/mean_set_size", ".2f", "mean")]
    for sc, sn in (("shift+0.5", "ShiftHalf"), ("shift+1.0", "ShiftOne"), ("casemix", "Casemix")):
        for m, mn in (("static", "Static"), ("recal_w10k", "Recal"), ("eci_interval", "Eci")):
            b = f"/runs/*/drift/{sc}/{m}"
            L += [(f"Drift{sn}{mn}Pre", CFx, b + "/cov_pre", ".3f", "mean"), (f"Drift{sn}{mn}Post", CFx, b + "/cov_post", ".3f", "mean")]
    # ---- other models: static coverage (model dependence of the static result)
    L += [("OtherXgbStaticCov", "ph2_conformal_xgb.json", "/runs/*/intervals/static/coverage", ".3f", "mean"),
          ("OtherLrStaticCov", "ph2_conformal_lr.json", "/runs/*/intervals/static/coverage", ".3f", "mean"),
          ("OtherXgbEciCov", "ph2_conformal_xgb.json", "/runs/*/intervals/eci/coverage", ".3f", "mean"),
          ("OtherLrEciCov", "ph2_conformal_lr.json", "/runs/*/intervals/eci/coverage", ".3f", "mean")]
    # ---- calibration (primary model, test period) and subgroups
    for k, nm in (("hybrid_e2e", "EndToEnd"), ("xgboost_timing", "XgbT"), ("logreg", "Lr")):
        L += [(f"CalBrier{nm}", CA, f"/calibration/{k}/brier/mean", ".4f"), (f"CalEce{nm}", CA, f"/calibration/{k}/ece/mean", ".3f"),
              (f"CalSlope{nm}", CA, f"/calibration/{k}/slope/mean", ".2f"), (f"CalIntercept{nm}", CA, f"/calibration/{k}/intercept/mean", ".2f"),
              (f"CalMeanPred{nm}", CA, f"/calibration/{k}/mean_pred/mean", "pct")]
    L += [("CalObserved", CA, "/calibration/hybrid_e2e/observed_rate", "pct")]
    for g, gn in (("sex_female", "Female"), ("sex_male", "Male"), ("age_lt65", "AgeUnder"), ("age_ge65", "AgeOver"), ("acuity_1_2", "AcuityHigh"), ("acuity_3", "AcuityMid")):
        L += [(f"Sub{gn}Auroc", CA, f"/subgroups/hybrid_e2e/{g}/auroc_seed_mean", ".3f"), (f"Sub{gn}Lo", CA, f"/subgroups/hybrid_e2e/{g}/auroc_bootstrap_ci95_of_seed_mean/0", ".3f"),
              (f"Sub{gn}Hi", CA, f"/subgroups/hybrid_e2e/{g}/auroc_bootstrap_ci95_of_seed_mean/1", ".3f"), (f"Sub{gn}N", CA, f"/subgroups/hybrid_e2e/{g}/n", ",d"),
              (f"Sub{gn}Pos", CA, f"/subgroups/hybrid_e2e/{g}/positives", ",d")]
    L += [("SubAcuityLowN", CA, "/subgroups/hybrid_e2e/acuity_4_5/n", ",d"), ("SubAcuityLowPos", CA, "/subgroups/hybrid_e2e/acuity_4_5/positives", ",d")]
    RL = "ph2_reliability.json"
    L += [("ReliTopPred", RL, "/models/hybrid_e2e/mean_pred/9", "pct"), ("ReliTopObs", RL, "/models/hybrid_e2e/observed/9", "pct"),
          ("ReliLowPred", RL, "/models/hybrid_e2e/mean_pred/0", "pct"), ("ReliLowObs", RL, "/models/hybrid_e2e/observed/0", "pct"),
          ("ReliNinthPred", RL, "/models/hybrid_e2e/mean_pred/8", "pct"), ("ReliNinthObs", RL, "/models/hybrid_e2e/observed/8", "pct")]
    CH = "ph2_characteristics.json"
    for g, gn in (("train_period", "Train"), ("test", "Test")):
        L += [(f"Char{gn}ElixPct", CH, f"/groups/{g}/elix_ge1_pct", "pct"), (f"Char{gn}AgeMed", CH, f"/groups/{g}/age_q/1", ".0f"),
              (f"Char{gn}Patients", CH, f"/groups/{g}/patients", ",d"), (f"Char{gn}Stays", CH, f"/groups/{g}/stays", ",d"), (f"Char{gn}Primary", CH, f"/groups/{g}/primary_pct", "pct")]
    L += [("CharSmdElix", CH, "/smd/elix", ".2f"), ("CharSmdRows", CH, "/smd/rows", ".2f"), ("CharSmdRowsZero", CH, "/smd/rows_zero", ".2f"), ("CharSmdAge", CH, "/smd/age", ".2f"), ("CharSmdFemale", CH, "/smd/female", ".2f"),
          ("CharSmdAcuity", CH, "/smd/acuity_1_2", ".2f"), ("CharSmdPrimary", CH, "/smd/primary", ".2f")]
    FX = "ph2_fixed_cohort.json"
    L += [("FixN", FX, "/fixed_set_size", ",d")]
    for h, hn in (("1", "One"), ("2", "Two"), ("3", "Three"), ("4", "Four"), ("6", "Six")):
        L += [(f"FixTrees{hn}", FX, f"/landmarks/{h}/fixed_set/trees", ".3f"), (f"FixLr{hn}", FX, f"/landmarks/{h}/fixed_set/logreg", ".3f"),
              (f"FixPrev{hn}", FX, f"/landmarks/{h}/fixed_set/prevalence", "pct")]
    L += [("FixTreesOneLo", FX, "/landmarks/1/fixed_set/trees_ci95/0", ".3f"), ("FixTreesOneHi", FX, "/landmarks/1/fixed_set/trees_ci95/1", ".3f"),
          ("FixTreesSixLo", FX, "/landmarks/6/fixed_set/trees_ci95/0", ".3f"), ("FixTreesSixHi", FX, "/landmarks/6/fixed_set/trees_ci95/1", ".3f")]
    AC = "ph2_acuity_shortcut.json"
    for k, nm in (("W", "Work"), ("Aq", "Acu"), ("W+Aq", "WorkAcu"), ("hybrid_grud_end_of_stay", "Hyb")):
        L += [(f"AcuOverall{nm}", AC, f"/overall/{k}/auroc", ".3f")]
        for sk, sn in (("acuity_1_2", "High"), ("acuity_3", "Mid")):
            L += [(f"Acu{sn}{nm}", AC, f"/strata/{sk}/models/{k}/auroc", ".3f"), (f"Acu{sn}{nm}Lo", AC, f"/strata/{sk}/models/{k}/ci95/0", ".3f"), (f"Acu{sn}{nm}Hi", AC, f"/strata/{sk}/models/{k}/ci95/1", ".3f")]
    L += [("AcuHighN", AC, "/strata/acuity_1_2/n", ",d"), ("AcuHighPos", AC, "/strata/acuity_1_2/positives", ",d"), ("AcuMidN", AC, "/strata/acuity_3/n", ",d"), ("AcuMidPos", AC, "/strata/acuity_3/positives", ",d"),
          ("AcuLowN", AC, "/strata/acuity_4_5/n", ",d"), ("AcuLowPos", AC, "/strata/acuity_4_5/positives", ",d")]
    SR = "ph2_search.json"
    L += [("LitDate", SR, "/date", None)]
    for q, nm in (("Q1_workflow_shortcut_ed", "One"), ("Q2_conformal_ed_or_deterioration", "Two"), ("Q3_conformal_class_conditional_clinical", "Three"), ("Q4_online_conformal_clinical", "Four"), ("Q5_conformal_delayed_feedback", "Five"), ("Q6_monitoring_pattern_leakage_clinical_prediction", "Six"), ("Q7_ed_prediction_time_landmark", "Seven")):
        L += [(f"LitHits{nm}", SR, f"/queries/{q}/total_hits", ",d"), (f"LitRelated{nm}", SR, f"/queries/{q}/related", ",d"), (f"LitMeets{nm}", SR, f"/queries/{q}/meets", ",d")]
    YD = "ph2_years_decision.json"
    for k, nm in (("hybrid_e2e", "Hybrid"), ("xgboost_timing", "Trees"), ("logreg", "Lr")):
        L += [(f"YearAuroc{nm}Min", YD, f"/year_blocks/*/models/{k}/auroc_seed_mean", ".3f", "min"), (f"YearAuroc{nm}Max", YD, f"/year_blocks/*/models/{k}/auroc_seed_mean", ".3f", "max")]
    D = "/decision/reported"
    for i, tn in ((1, "Five"), (2, "Ten"), (5, "Thirty")):
        for k, nm in (("hybrid_e2e", "Hybrid"), ("xgboost_timing", "Trees"), ("logreg", "Lr")):
            L += [(f"Nb{nm}{tn}", YD, f"{D}/models/{k}/net_benefit/{i}", ".3f")]
        L += [(f"NbAll{tn}", YD, f"{D}/treat_all/{i}", ".3f"),
              (f"NbHybridGainLo{tn}", YD, f"{D}/models/hybrid_e2e/minus_treat_all_ci95/{i}/0", ".3f"), (f"NbHybridGainHi{tn}", YD, f"{D}/models/hybrid_e2e/minus_treat_all_ci95/{i}/1", ".3f"),
              (f"NbHybridTreesLo{tn}", YD, f"{D}/hybrid_minus_trees_ci95/{i}/0", "+.4f"), (f"NbHybridTreesHi{tn}", YD, f"{D}/hybrid_minus_trees_ci95/{i}/1", "+.4f")]
    # ---- end-of-stay diagnostics used in the Discussion
    L += [("DurQOneHybrid", "ph1_duration.json", "/models/Hybrid GRU-D v2/quartiles/1/mean", ".3f"), ("DurQFourHybrid", "ph1_duration.json", "/models/Hybrid GRU-D v2/quartiles/4/mean", ".3f"),
          ("DurQOnePrev", "ph1_duration.json", "/test_quartile_prevalence/0", "pct"), ("DurQFourPrev", "ph1_duration.json", "/test_quartile_prevalence/3", "pct"),
          ("SweepTreesOne", "ph2_sweep.json", "/landmarks/1/trees", ".3f"), ("SweepTreesSix", "ph2_sweep.json", "/landmarks/6/trees", ".3f"), ("SweepLrSix", "ph2_sweep.json", "/landmarks/6/logreg", ".3f"),
          ("SweepTreesThree", "ph2_sweep.json", "/landmarks/3/trees", ".3f"), ("SweepTreesFour", "ph2_sweep.json", "/landmarks/4/trees", ".3f"), ("SweepPrevSix", "ph2_sweep.json", "/landmarks/6/prevalence_test", "pct"),
          ("SweepEligSix", "ph2_sweep.json", "/landmarks/6/eligible/test", ",d"),
          ("ScoreNeeded", "ph2_scores.json", "/runs/*/risk_needed_for_label_1", ".2f", "mean"), ("ScoreMedEvents", "ph2_scores.json", "/runs/*/test/median_risk_events", ".2f", "mean"),
          ("ScoreQNinetyEvents", "ph2_scores.json", "/runs/*/test/q90_risk_events", ".2f", "mean"), ("ScoreMedNon", "ph2_scores.json", "/runs/*/test/median_risk_non_events", ".3f", "mean"),
          ("StreamPerm", "ph2_stream_order.json", "/n_perm", "d"),
          ("StreamEciMin", "ph2_stream_order.json", "/runs/*/eci_interval/cov/min", ".3f", "min"), ("StreamEciMax", "ph2_stream_order.json", "/runs/*/eci_interval/cov/max", ".3f", "max"),
          ("StreamRecalMin", "ph2_stream_order.json", "/runs/*/recal_w10k/cov/min", ".3f", "min"), ("StreamRecalMax", "ph2_stream_order.json", "/runs/*/recal_w10k/cov/max", ".3f", "max"),
          ("StreamEciWidMin", "ph2_stream_order.json", "/runs/*/eci_interval/width/min", ".3f", "min"), ("StreamEciWidMax", "ph2_stream_order.json", "/runs/*/eci_interval/width/max", ".3f", "max")]
    # ---- landmark cohort: ineligible shares and test-period timing
    L += [("HtwoTestEligible", "ph2_cohorts.json", "/landmarks/h2/splits/test/eligible_stays", ",d"), ("HtwoTestStays", "ph2_cohorts.json", "/landmarks/h2/splits/test/stays_in_split", ",d"),
          ("TriageTestPositives", "ph2_cohorts.json", "/landmarks/triage/splits/test/primary_positives", ",d"),
          ("HtwoTestPositives", "ph2_cohorts.json", "/landmarks/h2/splits/test/primary_positives", ",d"),
          ("HtwoTestSecondaryPositives", "ph2_cohorts.json", "/landmarks/h2/splits/test/secondary_positives", ",d"),
          ("HtwoTruncStays", "ph2_cohorts.json", "/h2_sequence_rows_within_2h/stays_truncated", ",d"),
          ("StaticVsMapie", "derived.json", "/static_vs_mapie_max_abs_coverage_diff", ".4f"),
          ("HtwoTestIneligiblePct", "derived.json", "/h2_test_ineligible_share", "pct"),
          ("HtwoTruncPct", "ph2_cohorts.json", "/h2_sequence_rows_within_2h/share_truncated", "pct")]
    return L


NL = " \\\\"


def _wrap(label, caption, cols, header, rows, star=False):
    env = "table*" if star else "table"
    return ("\\begin{" + env + "}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\begin{tabular}{" + cols + "}\n\\toprule\n" + header + NL + "\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{" + env + "}\n")


def table_calibration():
    ev = PD.EV("tab_calibration")
    CA = "ph2_calibration.json"
    rows = []
    for k, label in (("logreg", "Logistic regression"), ("xgboost_timing", "Timing-aware trees"), ("temporal_grud", "Temporal-Only GRU-D"), ("hybrid_frozen", "Hybrid GRU-D, frozen encoder"),
                     ("hybrid_e2e", "Hybrid GRU-D, end to end")):
        b = f"/calibration/{k}"
        rows.append(" & ".join([label, ev.get(CA, b + "/brier/mean", fmt=".4f"), ev.get(CA, b + "/ece/mean", fmt=".3f"), ev.get(CA, b + "/slope/mean", fmt=".2f"),
                                ev.get(CA, b + "/intercept/mean", fmt=".2f"), ev.get(CA, b + "/mean_pred/mean", fmt="pct")]) + NL)
    obs = ev.get(CA, "/calibration/hybrid_e2e/observed_rate", fmt="pct")
    tex1 = _wrap("tab:calibration", "Probability calibration on the test period (primary outcome; mean over seeds). Observed event rate " + obs + ". Slope and intercept are from a logistic regression of the outcome on the logit of the score.",
                 "lccccc", "Model & Brier & ECE & Slope & Intercept & Mean predicted risk", rows)
    rows = []
    for g, label in (("sex_female", "Women"), ("sex_male", "Men"), ("age_lt65", "Age below 65 years"), ("age_ge65", "Age 65 years or older"), ("acuity_1_2", "Triage acuity 1 or 2"), ("acuity_3", "Triage acuity 3")):
        for k, mname in (("hybrid_e2e", "Hybrid GRU-D"), ("xgboost_timing", "Timing-aware trees")):
            b = f"/subgroups/{k}/{g}"
            rows.append(" & ".join([label, mname, ev.get(CA, b + "/n", fmt=",d"), ev.get(CA, b + "/positives", fmt=",d"), ev.get(CA, b + "/auroc_seed_mean", fmt=".3f"),
                                    "[" + ev.get(CA, b + "/auroc_bootstrap_ci95_of_seed_mean/0", fmt=".3f") + ", " + ev.get(CA, b + "/auroc_bootstrap_ci95_of_seed_mean/1", fmt=".3f") + "]"]) + NL)
    tex2 = _wrap("tab:subgroups", "Discrimination by subgroup on the test period (AUROC, seed mean; 95\\% patient-cluster bootstrap interval of the seed mean). Triage acuity 4 or 5 is omitted: "
                 + ev.get(CA, "/subgroups/hybrid_e2e/acuity_4_5/n", fmt=",d") + " stays and " + ev.get(CA, "/subgroups/hybrid_e2e/acuity_4_5/positives", fmt="d") + " event.",
                 "llrrcc", "Subgroup & Model & Stays & Events & AUROC & 95\\% CI", rows)
    write("tab_calibration", tex1 + "\n" + tex2, ev)


def table_elixhauser():
    ev = PD.EV("tab_elixhauser")
    EL = "ph1_elixhauser.json"
    rows = []
    for s in (42, 123, 7, 456, 789):
        b = f"/hybrid/{s}"
        rows.append(" & ".join([str(s), ev.get(EL, b + "/auroc_with", fmt=".4f"), ev.get(EL, b + "/auroc_without", fmt=".4f"), ev.get(EL, b + "/delta_auroc", fmt="+.4f"),
                                "[" + ev.get(EL, b + "/delta_auroc_ci95/0", fmt="+.4f") + ", " + ev.get(EL, b + "/delta_auroc_ci95/1", fmt="+.4f") + "]"]) + NL)
    rows.append(" & ".join(["Trees", ev.get(EL, "/xgboost/auroc_with", fmt=".4f"), ev.get(EL, "/xgboost/auroc_without", fmt=".4f"), ev.get(EL, "/xgboost/delta_auroc", fmt="+.4f"),
                            "[" + ev.get(EL, "/xgboost/delta_auroc_ci95/0", fmt="+.4f") + ", " + ev.get(EL, "/xgboost/delta_auroc_ci95/1", fmt="+.4f") + "]"]) + NL)
    tex = _wrap("tab:elixhauser", "Effect of removing the comorbidity count on the end-of-stay task (test period). Rows 42 to 789: Hybrid GRU-D per seed with the frozen encoder of that seed; the last row is the tree ensemble. "
                "Interval: paired patient-cluster bootstrap, 1{,}000 resamples.", "lcccc", "Seed or model & AUROC with count & AUROC without & Difference & 95\\% CI", rows)
    write("tab_elixhauser", tex, ev)


def table_duration():
    ev = PD.EV("tab_duration")
    DU = "ph1_duration.json"
    rows = []
    for k, label in (("Static-Only MLP v2", "Static features only"), ("mTAN v2", "mTAN"), ("Temporal-Only GRU-D v2", "Temporal-Only GRU-D"), ("XGBoost-Matched v2", "Gradient-boosted trees"),
                     ("Hybrid GRU-D v2", "Hybrid GRU-D")):
        b = f"/models/{k}"
        cells = [label, ev.get(DU, b + "/overall_auroc/mean", fmt=".3f")]
        for q in (1, 2, 3, 4):
            cells.append(ev.get(DU, b + f"/quartiles/{q}/mean", fmt=".3f"))
        rows.append(" & ".join(cells) + NL)
    prev = [ev.get(DU, f"/test_quartile_prevalence/{q}", fmt="pct") for q in range(4)]
    n = [ev.get(DU, f"/test_quartile_n/{q}", fmt=",d") for q in range(4)]
    rows.append("\\midrule")
    rows.append(" & ".join(["Stays, event rate", ""] + [f"{n[q]}, {prev[q]}" for q in range(4)]) + NL)
    tex = _wrap("tab:duration", "End-of-stay task: AUROC within quartiles of ED duration (cut points from the training period; test period; seed mean). Q1 is the shortest quartile.", "lccccc",
                "Model & Overall & Q1 & Q2 & Q3 & Q4", rows)
    write("tab_duration", tex, ev)


def table_stream():
    ev = PD.EV("tab_stream")
    SO = "ph2_stream_order.json"
    rows = []
    for k, label, u in (("eci_interval", "ECI interval", "width"), ("recal_w10k", "Recalibration, $W = 10$k", "width")):
        b = f"/runs/*/{k}"
        rows.append(" & ".join([label, ev.reduce("min", SO, b + "/cov/min", fmt=".3f"), ev.reduce("mean", SO, b + "/cov/mean", fmt=".3f"), ev.reduce("max", SO, b + "/cov/max", fmt=".3f"),
                                ev.reduce("min", SO, b + f"/{u}/min", fmt=".3f"), ev.reduce("max", SO, b + f"/{u}/max", fmt=".3f")]) + NL)
    tex = _wrap("tab:stream", "Stream-order robustness for the primary model at the landmark: coverage and width over " + ev.get(SO, "/n_perm", fmt="d")
                + " permutations of the order within each year block, five seeds (minimum, mean, and maximum across permutations and seeds).", "lccccc",
                "Method & Cov. min & Cov. mean & Cov. max & Width min & Width max", rows)
    write("tab_stream", tex, ev)


def table_other_models():
    ev = PD.EV("tab_other_models")
    rows = []
    for art, label in (("ph2_conformal.json", "Hybrid GRU-D, end to end"), ("ph2_conformal_xgb.json", "Timing-aware trees"), ("ph2_conformal_lr.json", "Logistic regression")):
        for k, ml in (("static", "Static"), ("eci", "ECI"), ("recal_w10k", "Recalibration $W = 10$k")):
            b = f"/runs/*/intervals/{k}"
            rows.append(" & ".join([label, ml, ev.reduce("mean", art, b + "/coverage", fmt=".3f"), ev.reduce("std", art, b + "/coverage", fmt=".3f"), ev.reduce("mean", art, b + "/width", fmt=".3f")]) + NL)
    tex = _wrap("tab:other-models", "Score intervals at $\\alpha = 0.10$ for three models of the landmark task (test period; mean and SD over five seeds; logistic regression is deterministic).", "llccc",
                "Model & Method & Coverage & SD & Width", rows)
    write("tab_other_models", tex, ev)


def table_sweep():
    ev = PD.EV("tab_sweep")
    SW, CO = "ph2_sweep.json", "ph2_cohorts.json"
    rows = []
    for h in ("0", "1", "2", "3", "4", "6"):
        if h in ("0", "2"):
            lm = "triage" if h == "0" else "h2"
            n = ev.get(CO, f"/landmarks/{lm}/splits/test/eligible_stays", fmt=",d")
            prev = ev.get(CO, f"/landmarks/{lm}/splits/test/primary_prevalence", fmt="pct")
        else:
            n = ev.get(SW, f"/landmarks/{h}/eligible/test", fmt=",d")
            prev = ev.get(SW, f"/landmarks/{h}/prevalence_test", fmt="pct")
        cells = [h if h != "0" else "0 (arrival)", n, prev]
        for k in ("logreg", "trees"):
            cells.append(ev.get(SW, f"/landmarks/{h}/{k}", fmt=".3f") + " [" + ev.get(SW, f"/landmarks/{h}/{k}_ci95/0", fmt=".3f") + ", " + ev.get(SW, f"/landmarks/{h}/{k}_ci95/1", fmt=".3f") + "]")
        rows.append(" & ".join(cells) + NL)
    tex = _wrap("tab:sweep", "Discrimination at landmarks from 0 to 6 hours after arrival (test period; primary outcome in the following 24 hours; AUROC with the patient-cluster bootstrap 95\\% interval of the seed mean). "
                "The eligible population differs between landmarks: stays that have ended, and stays with an earlier event, drop out as the landmark moves later.", "lccll",
                "Landmark (h) & Eligible test stays & Event rate & Logistic regression & Timing-aware trees", rows)
    write("tab_sweep", tex, ev)

def table_years():
    ev = PD.EV("tab_years")
    YD = "ph2_years_decision.json"
    rows = []
    for yr in ("2017", "2018", "2019", "2020"):
        b = f"/year_blocks/{yr}"
        cells = [yr, ev.get(YD, b + "/n", fmt=",d"), ev.get(YD, b + "/positives", fmt=",d")]
        for k in ("logreg", "xgboost_timing", "hybrid_e2e"):
            cells.append(ev.get(YD, b + f"/models/{k}/auroc_seed_mean", fmt=".3f") + " [" + ev.get(YD, b + f"/models/{k}/auroc_bootstrap_ci95/0", fmt=".3f") + ", "
                         + ev.get(YD, b + f"/models/{k}/auroc_bootstrap_ci95/1", fmt=".3f") + "]")
        rows.append(" & ".join(cells) + NL)
    tex = _wrap("tab:years", "AUROC by approximate calendar-year block of the test period (primary outcome at the two-hour landmark; seed mean with the patient-cluster bootstrap 95\\% interval). "
                "The blocks differ strongly in size, and the last holds few events.", "lrrccc", "Block & Stays & Events & Logistic regression & Timing-aware trees & Hybrid GRU-D", rows)
    write("tab_years", tex, ev)


def table_search():
    ev = PD.EV("tab_search")
    SR = "ph2_search.json"
    spec = [("Q1_workflow_shortcut_ed", "Q1", "ED prediction models and workflow shortcuts or leakage (measurement frequency, missingness patterns)"),
            ("Q2_conformal_ed_or_deterioration", "Q2", "Conformal prediction with ED, triage, deterioration or intensive care"),
            ("Q3_conformal_class_conditional_clinical", "Q3", "Class-conditional or Mondrian conformal prediction in clinical data"),
            ("Q4_online_conformal_clinical", "Q4", "Adaptive or online conformal prediction under shift in clinical data"),
            ("Q5_conformal_delayed_feedback", "Q5", "Conformal prediction with delayed feedback or labels"),
            ("Q6_monitoring_pattern_leakage_clinical_prediction", "Q6", "Informative presence, monitoring frequency, or label leakage in clinical prediction (all settings; added after review)"),
            ("Q7_ed_prediction_time_landmark", "Q7", "ED prediction models with a stated prediction time or restricted early inputs (added after review)")]
    rows = []
    for k, qid, label in spec:
        b = f"/queries/{k}"
        rows.append(" & ".join([qid, label, ev.get(SR, b + "/total_hits", fmt=",d"), ev.get(SR, b + "/screened", fmt=",d"), ev.get(SR, b + "/related", fmt=",d"), ev.get(SR, b + "/meets", fmt=",d")]) + NL)
    tex = _wrap("tab:search", "Recorded literature search (OpenAlex, title and abstract index, English, 2015 onward, run on " + ev.get(SR, "/date") + "). Hits are all records matching a query; all records up to 60 per query, ranked by relevance, were screened by title (and abstract where available). "
                "Related: same clinical or methodological area as the claim tested. Meets: reports the analysis that the manuscript states it did not find. The query strings are in the code repository.", "lp{7cm}rrrr",
                "ID & Topic of the query & Hits & Screened & Related & Meets", rows)
    write("tab_search", tex, ev)


def table_acuity_shortcut():
    ev = PD.EV("tab_acuity_shortcut")
    AC = "ph2_acuity_shortcut.json"
    names = [("W", "Workflow quantities"), ("Aq", "Triage acuity alone"), ("W+Aq", "Workflow quantities and acuity"), ("hybrid_grud_end_of_stay", "Hybrid GRU-D (whole-stay vital signs)")]
    rows = []
    for k, label in names:
        cells = [label, ev.get(AC, f"/overall/{k}/auroc", fmt=".3f")]
        for sk in ("acuity_1_2", "acuity_3"):
            cells.append(ev.get(AC, f"/strata/{sk}/models/{k}/auroc", fmt=".3f") + " [" + ev.get(AC, f"/strata/{sk}/models/{k}/ci95/0", fmt=".3f") + ", " + ev.get(AC, f"/strata/{sk}/models/{k}/ci95/1", fmt=".3f") + "]")
        rows.append(" & ".join(cells) + NL)
    tex = _wrap("tab:acuity-shortcut", "Shortcut diagnostic within triage-acuity strata on the end-of-stay task (test period; AUROC with the patient-cluster bootstrap 95\\% interval). Acuity 1 or 2: "
                + ev.get(AC, "/strata/acuity_1_2/n", fmt=",d") + " stays, " + ev.get(AC, "/strata/acuity_1_2/positives", fmt=",d") + " events; acuity 3: " + ev.get(AC, "/strata/acuity_3/n", fmt=",d") + " stays, "
                + ev.get(AC, "/strata/acuity_3/positives", fmt=",d") + " events. Acuity 4 or 5 (" + ev.get(AC, "/strata/acuity_4_5/n", fmt=",d") + " stays, " + ev.get(AC, "/strata/acuity_4_5/positives", fmt=",d")
                + " event) cannot be evaluated. Within a stratum, acuity alone is constant and cannot discriminate. The scores of Hybrid GRU-D are averaged over the five seeds before scoring, so its overall value can differ from the seed mean of the main table in the third decimal.", "lccc", "Inputs & Overall & Acuity 1 or 2 & Acuity 3", rows)
    write("tab_acuity_shortcut", tex, ev)


def table_fixed_cohort():
    ev = PD.EV("tab_fixed_cohort")
    FX = "ph2_fixed_cohort.json"
    rows = []
    for h in ("1", "2", "3", "4", "6"):
        b = f"/landmarks/{h}"
        cells = [h, ev.get(FX, b + "/fixed_set/positives", fmt=",d"), ev.get(FX, b + "/fixed_set/prevalence", fmt="pct")]
        for k in ("logreg", "trees"):
            cells.append(ev.get(FX, b + f"/fixed_set/{k}", fmt=".3f") + " [" + ev.get(FX, b + f"/fixed_set/{k}_ci95/0", fmt=".3f") + ", " + ev.get(FX, b + f"/fixed_set/{k}_ci95/1", fmt=".3f") + "]")
        cells.append(ev.get(FX, b + "/own_cohort/trees", fmt=".3f"))
        rows.append(" & ".join(cells) + NL)
    tex = _wrap("tab:fixed-cohort", "Discrimination at each landmark on one fixed set of " + ev.get(FX, "/fixed_set_size", fmt=",d") + " test stays (those still in the ED and event-free at 6 hours, hence eligible at every earlier landmark); "
                "each landmark is scored on its own outcome window $(L, L + 24$\\,h$]$ (AUROC with the patient-cluster bootstrap 95\\% interval). The last column gives the trees on the full eligible cohort of the landmark, as in the main sweep.",
                "lrrccc", "Landmark (h) & Events & Rate & Logistic regression & Timing-aware trees & Trees, full cohort", rows)
    write("tab_fixed_cohort", tex, ev)


def table_characteristics():
    ev = PD.EV("tab_characteristics")
    CH = "ph2_characteristics.json"

    def iqr(g, k, f=".0f"):
        return ev.get(CH, f"/groups/{g}/{k}/1", fmt=f) + " (" + ev.get(CH, f"/groups/{g}/{k}/0", fmt=f) + "--" + ev.get(CH, f"/groups/{g}/{k}/2", fmt=f) + ")"

    def pc(g, k):
        return ev.get(CH, f"/groups/{g}/{k}", fmt="pct")

    G = ("train_period", "test")
    rows = [
        " & ".join(["Stays", ev.get(CH, "/groups/train_period/stays", fmt=",d"), ev.get(CH, "/groups/test/stays", fmt=",d"), ""]) + NL,
        " & ".join(["Patients", ev.get(CH, "/groups/train_period/patients", fmt=",d"), ev.get(CH, "/groups/test/patients", fmt=",d"), ""]) + NL,
        " & ".join(["Age, years, median (IQR)"] + [iqr(g, "age_q") for g in G] + [ev.get(CH, "/smd/age", fmt="+.2f")]) + NL,
        " & ".join(["Female"] + [pc(g, "female_pct") for g in G] + [ev.get(CH, "/smd/female", fmt="+.2f")]) + NL,
        " & ".join(["Triage acuity 1"] + [pc(g, "acuity_1_pct") for g in G] + [""]) + NL,
        " & ".join(["Triage acuity 2"] + [pc(g, "acuity_2_pct") for g in G] + [ev.get(CH, "/smd/acuity_1_2", fmt="+.2f") + "$^a$"]) + NL,
        " & ".join(["Triage acuity 3"] + [pc(g, "acuity_3_pct") for g in G] + [""]) + NL,
        " & ".join(["Triage acuity 4 or 5"] + [pc(g, "acuity_4_5_pct") for g in G] + [""]) + NL,
        " & ".join(["Comorbidity count $\\geq 1$"] + [pc(g, "elix_ge1_pct") for g in G] + [ev.get(CH, "/smd/elix", fmt="+.2f") + "$^b$"]) + NL,
        " & ".join(["All triage vital signs missing"] + [pc(g, "triage_all_missing_pct") for g in G] + [""]) + NL,
        " & ".join(["ED vital-sign rows by $L$, median (IQR)"] + [iqr(g, "rows_q") for g in G] + [ev.get(CH, "/smd/rows", fmt="+.2f")]) + NL,
        " & ".join(["No ED vital sign by $L$"] + [pc(g, "rows_zero_pct") for g in G] + [ev.get(CH, "/smd/rows_zero", fmt="+.2f")]) + NL,
        " & ".join(["ED length of stay, h, median (IQR)$^c$"] + [iqr(g, "ed_los_h_q", ".1f") for g in G] + [ev.get(CH, "/smd/ed_los", fmt="+.2f")]) + NL,
        " & ".join(["Primary outcome"] + [pc(g, "primary_pct") for g in G] + [ev.get(CH, "/smd/primary", fmt="+.2f")]) + NL,
        " & ".join(["Secondary outcome"] + [pc(g, "secondary_pct") for g in G] + [""]) + NL,
    ]
    tex = _wrap("tab:characteristics", "Characteristics of the stays eligible at the two-hour landmark, by period (MIMIC-IV-ED, retrospective single-center proxy). SMD: standardized "
                "difference of the test period relative to the training period. $^a$Acuity 1 or 2 against 3 to 5. $^b$Mean count. $^c$Description only; not an input of any landmark model.",
                "lrrr", "Characteristic & Training period & Test period & SMD", rows)
    write("tab_characteristics", tex, ev)
