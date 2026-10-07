"""Part D of the asset generator: the compact tables of the submitted (short) version of the manuscript. The full tables of parts A to C stay unchanged for the
extended version. New files: tab_cohort_merged (cohort + key characteristics + recording), tab_shortcuts_arrival (shortcut diagnostic + arrival increment),
tab_discrimination_cal (discrimination + calibration slope and intercept). tab_timing_slim is hand-written text (no numbers) in floats/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data_v2 as PD  # noqa: E402
from gen_assets_v2_b import _wrap, write, NL, SD, FLOATS, K, N, H2, TR  # noqa: E402

CH, RC_, AI = "ph2_characteristics.json", "ph2_recording.json", "ph2_arrival_increment.json"
E, S, CA = "ph2_cohorts.json", "ph1_shortcuts.json", "ph2_calibration.json"


def ci(ev, art, b, f=".3f"):
    return ev.get(art, b + "/0", fmt=f) + ", " + ev.get(art, b + "/1", fmt=f)


def table_cohort_merged():
    ev = PD.EV("tab_cohort_merged")
    rows = []
    for k, label in (("fit", "Training"), ("val", "Validation"), ("cs", "Calibration selection"), ("fc", "Final calibration"), ("test", "Test period")):
        b = f"/landmarks/h2/splits/{k}"
        rows.append(" & ".join([label, ev.get(E, b + "/stays_in_split", fmt=",d"), ev.get(E, b + "/eligible_stays", fmt=",d"),
                                ev.get(E, b + "/primary_positives", fmt=",d") + " (" + ev.get(E, b + "/primary_prevalence", fmt="pct") + ")",
                                ev.get(E, b + "/secondary_positives", fmt=",d") + " (" + ev.get(E, b + "/secondary_prevalence", fmt="pct") + ")"]) + NL)
    t = "/landmarks/h2/totals"
    rows.append("\\midrule")
    rows.append(" & ".join(["All", ev.get(E, t + "/stays_in_split", fmt=",d"), ev.get(E, t + "/eligible_stays", fmt=",d"), ev.get(E, t + "/primary_positives", fmt=",d"),
                            ev.get(E, t + "/secondary_positives", fmt=",d")]) + NL)
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Eligible stays by period}}" + NL)
    rows.append("\\multicolumn{1}{l}{Characteristic} & \\multicolumn{2}{c}{Training period} & \\multicolumn{2}{c}{Test period}" + NL)

    def iqr(g, k, f=".0f"):
        return ev.get(CH, f"/groups/{g}/{k}/1", fmt=f) + " (" + ev.get(CH, f"/groups/{g}/{k}/0", fmt=f) + "--" + ev.get(CH, f"/groups/{g}/{k}/2", fmt=f) + ")"

    def two(label, a, b):
        return label + " & \\multicolumn{2}{c}{" + a + "} & \\multicolumn{2}{c}{" + b + "}" + NL

    G = ("train_period", "test")
    rows.append(two("Stays", ev.get(CH, "/groups/train_period/stays", fmt=",d"), ev.get(CH, "/groups/test/stays", fmt=",d")))
    rows.append(two("Age, years, median (IQR)", iqr(G[0], "age_q"), iqr(G[1], "age_q")))
    rows.append(two("Comorbidity count $\\geq 1$", ev.get(CH, "/groups/train_period/elix_ge1_pct", fmt="pct"), ev.get(CH, "/groups/test/elix_ge1_pct", fmt="pct")))
    rows.append(two("No ED vital sign by $L$", ev.get(CH, "/groups/train_period/rows_zero_pct", fmt="pct"), ev.get(CH, "/groups/test/rows_zero_pct", fmt="pct")))
    rows.append(two("First ED vital-sign row repeats triage$^a$", ev.get(RC_, "/by_period/train/first_row_equals_triage_share_of_comparable", fmt="pct"),
                    ev.get(RC_, "/by_period/test/first_row_equals_triage_share_of_comparable", fmt="pct")))
    rows.append(two("Primary outcome", ev.get(CH, "/groups/train_period/primary_pct", fmt="pct"), ev.get(CH, "/groups/test/primary_pct", fmt="pct")))
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Cohort at the 2-hour landmark and characteristics of the eligible stays by period (MIMIC-IV-ED, retrospective single-center proxy). "
           "Eligible stays are in the ED at $L$ and event-free at $L$. Primary outcome: ICU admission or in-hospital death in $(L, L+24\\,\\mathrm{h}]$; secondary outcome: the benchmark composite "
           "restricted to events after $L$. Training period: the four training-period subsets together. $^a$Among stays with at least three vital signs recorded in both tables.}\n\\label{tab:cohort}\n\\footnotesize\n"
           "\\begin{tabular}{lrrrr}\n\\toprule\nSubset & Stays & Eligible & Primary events (rate) & Secondary events (rate) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write("tab_cohort_merged", tex, ev)


def table_shortcuts_arrival():
    ev = PD.EV("tab_shortcuts_arrival")
    rows = ["\\multicolumn{4}{l}{\\emph{Logistic regression, end-of-stay task}}" + NL]
    b = "/baselines/duration_alone"
    rows.append(" & ".join(["ED duration alone", ev.get(S, b + "/test_auroc", fmt=".3f"), "[" + ci(ev, S, b + "/test_auroc_ci95") + "]", ev.get(S, b + "/test_auprc", fmt=".3f")]) + NL)
    b = "/baselines/A_only"
    rows.append(" & ".join(["Workflow quantities", ev.get(S, b + "/test_auroc", fmt=".3f"), "[" + ci(ev, S, b + "/test_auroc_ci95") + "]", ev.get(S, b + "/test_auprc", fmt=".3f")]) + NL)
    for k, label in (("Arr", "Arrival information"), ("Arr+duration", "Arrival information and ED duration"), ("Arr+W", "Arrival information and workflow quantities")):
        rows.append(" & ".join([label, ev.get(AI, f"/models/{k}/auroc", fmt=".3f"), "[" + ci(ev, AI, f"/models/{k}/auroc_ci95") + "]", "--"]) + NL)
    for k, label in (("Arr+duration minus Arr", "ED duration added to arrival information"), ("Arr+W minus Arr", "Workflow quantities added to arrival information")):
        rows.append(" & ".join([label + " (difference)", ev.get(AI, f"/differences/{k}/delta", fmt="+.3f"), "[" + ci(ev, AI, f"/differences/{k}/ci95", "+.3f") + "]", "--"]) + NL)
    rows.append("\\midrule\n\\multicolumn{4}{l}{\\emph{Models that read the last ten vital-sign rows of the whole stay}}" + NL)
    for k, label in (("Static-Only MLP v2", "Static features, neural network"), ("XGBoost-Matched v2", "Trees, whole stay"), ("Hybrid GRU-D v2", "Hybrid GRU-D, whole stay")):
        b = f"/models/{k}"
        rows.append(" & ".join([label, ev.get(S, b + "/auroc_seed_summary/mean", fmt=".3f"), "seed SD " + ev.get(S, b + "/auroc_seed_summary/sd", fmt=".3f"),
                                ev.get(S, b + "/auprc_seed_summary/mean", fmt=".3f")]) + NL)
    tex = _wrap("tab:shortcuts", "Shortcut diagnostic and information at arrival on the end-of-stay task of the original analysis (test period; outcome: death or ICU transfer within 12 h of ED departure). "
                "Workflow quantities describe how a stay was monitored: ED duration, number of retained measurements, and per-vital measurement counts. Arrival information: the triage vital signs and "
                "their missingness, age, sex, triage acuity, and comorbidity count. Upper block: patient-cluster bootstrap 95\\% interval (paired for differences). Lower block: standard deviation over five seeds.",
                "lccc", "Inputs & AUROC & Interval & AUPRC", rows)
    write("tab_shortcuts_arrival", tex.replace("\\centering\n", "\\centering\n\\footnotesize\n", 1), ev)


def table_discrimination_cal():
    ev = PD.EV("tab_discrimination_cal")
    rows = []

    def block(lm, models, news_key, cal):
        for k, label in models:
            b = f"/landmarks/{lm}/models/{k}"
            sd = f"{SD}{ev.get(K, b + '/auroc_seed_sd', fmt='.4f')}" if k not in ("logreg",) else ""
            sec = ev.get(K, f"/landmarks/{lm}/secondary_outcome/models/{k}/auroc_seed_mean", fmt=".3f")
            sl, ic = ("--", "--") if not cal else (ev.get(CA, f"/calibration/{k}/slope/mean", fmt=".2f"), ev.get(CA, f"/calibration/{k}/intercept/mean", fmt=".2f"))
            rows.append(f"{label} & {ev.get(K, b + '/auroc_seed_mean', fmt='.3f')}{sd} & [{ev.get(K, b + '/auroc_bootstrap_ci95_of_seed_mean/0', fmt='.3f')}, "
                        f"{ev.get(K, b + '/auroc_bootstrap_ci95_of_seed_mean/1', fmt='.3f')}] & {ev.get(K, b + '/auprc_seed_mean', fmt='.3f')} & {sec} & {sl} & {ic} \\\\")
        rows.append(f"NEWS2 (five parameters) & {ev.get(N, f'/landmarks/{news_key}/auroc', fmt='.3f')} & [{ev.get(N, f'/landmarks/{news_key}/auroc_ci95/0', fmt='.3f')}, "
                    f"{ev.get(N, f'/landmarks/{news_key}/auroc_ci95/1', fmt='.3f')}] & -- & -- & -- & -- \\\\")
    rows.append("\\multicolumn{7}{l}{\\emph{2-hour landmark}} \\\\")
    block("h2", H2, "h2", True)
    rows.append("\\midrule\n\\multicolumn{7}{l}{\\emph{Arrival (triage) landmark}} \\\\")
    block("triage", TR, "triage", False)
    obs = ev.get(CA, "/calibration/hybrid_e2e/observed_rate", fmt="pct")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Discrimination and calibration on the test period for the primary outcome $Y_L$ (ICU admission or in-hospital death within 24\\,h of the landmark; "
           "MIMIC-IV-ED, retrospective single-center proxy). AUROC is the mean over five seeds with the standard deviation across seeds; the interval is the patient-cluster "
           "bootstrap 95\\% interval of the seed-averaged AUROC. Sec.: AUROC for the secondary outcome (benchmark composite after $L$). Slope and Int.: slope and intercept of a logistic regression of the "
           "outcome on the logit of the score (mean over seeds; observed event rate " + obs + "). The primary model is Hybrid GRU-D trained end to end.}\n\\label{tab:discrimination}\n"
           "\\begin{tabular}{lcccccc}\n\\toprule\nModel & AUROC & 95\\% CI & AUPRC & Sec. & Slope & Int. \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write("tab_discrimination_cal", tex, ev)


def table_timing_slim():
    """Text only (no numbers): the audit of every input against the landmark, one line per input group."""
    ev = PD.EV("tab_timing_slim")
    T = "\\texttt"
    rows = [
        ["Triage vital signs, age, sex, triage acuity", T + "{ed.triage} (no timestamp; arrival, " + T + "{edstays.intime}, is the anchor), " + T + "{patients}", "(a)", "all models; NEWS2"],
        ["Elixhauser comorbidity count", "earlier admissions in " + T + "{hosp.diagnoses\\_icd}, dated by " + T + "{admissions.dischtime} (documentation time)", "(a); (d) for recent admissions",
         "all models except Temporal-Only GRU-D"],
        ["ED vital signs, counts and intervals up to $L$", T + "{ed.vitalsign.charttime} $\\le L$ (documentation time)", "(b)",
         "2\\,h models; timing features of the trees; mask and elapsed-time inputs of GRU-D"],
        ["ED vital signs after $L$, ED length of stay, measurements in the whole stay", T + "{ed.vitalsign.charttime} $> L$; " + T + "{edstays.outtime}", "(c)",
         "end-of-stay models of the original analysis only"],
        ["Disposition, admission, ICU transfer and death times", T + "{edstays}, " + T + "{admissions}, " + T + "{icustays}", "(c)", "outcome definition only"],
    ]
    body = "\n".join(" & ".join(r) + NL for r in rows)
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{Timing audit of the model inputs against the landmark $L$. Classes: (a) available at or before arrival; (b) charted between arrival and $L$; "
           "(c) available only after $L$; (d) ambiguous. Documentation time is when a value was entered, which can be later than when it was measured. MIMIC-IV-ED, retrospective single-center proxy.}\n"
           "\\label{tab:timing}\n\\footnotesize\n\\begin{tabular}{p{4.2cm}p{6.0cm}p{1.3cm}p{4.2cm}}\n\\toprule\nInput & Source and timestamp & Class & Used by \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    write("tab_timing_slim", tex, ev)
