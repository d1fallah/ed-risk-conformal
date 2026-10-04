#!/usr/bin/env python3
"""
Generate booktabs LaTeX tables from M2's real result artifacts, mirroring module 1's
src/generate_latex_tables.py pattern (CLAUDE.md Golden Rule). Every cell goes through
src/paper_data.py's Evidence accessor -- see that module's docstring for why.

This REPLACES the three untraced original generators (Benchmark_scripts/m2_generate_tables_v2.py,
m2_generate_table_g7.py, m2_generate_table_lag.py), which wrote paper_assets/tables/*.tex directly
from pandas with no provenance sidecar and no independent re-verification path. Every table's
content (captions, labels, row/column structure, which CSV/JSON columns feed which cell) is
preserved EXACTLY from those originals -- confirmed correct by inspection and by this module's own
smoke-test cross-checks against the existing .tex output before this rewrite -- only the *access
path* changes, from raw pandas reads to Evidence.get_csv()/reduce_csv()/get(), so every number now
carries a provenance sidecar and can be independently replayed by verify_paper_assets.py.

Output: paper_assets/tables/*.tex + paper_assets/provenance/*.json (one sidecar per table)
Usage: python generate_latex_tables.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_data import Evidence  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "paper_assets", "tables")

PROXY = "Proxy cohort (MIMIC-IV-ED-derived)"
SEEDS = [42, 123, 7, 456, 789]


def write(name: str, content: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"wrote {path}")


def finish(filename: str, lines: list[str], ev: Evidence) -> None:
    content = "\n".join(lines) + "\n"
    write(filename, content)
    sidecar_path = ev.write_provenance()
    print(f"  provenance -> {sidecar_path}")


def pm(ev: Evidence, artifact: str, filters: dict, column: str, digits: int = 3, note: str = "") -> str:
    """'mean $\\pm$ std' over every row matching filters -- the standard 5-seed aggregate cell."""
    mean = ev.reduce_csv("mean", artifact, filters, column, fmt=f".{digits}f", note=note)
    std = ev.reduce_csv("std", artifact, filters, column, fmt=f".{digits}f", note=note)
    return f"{mean} $\\pm$ {std}"


def bf(cell: str) -> str:
    """Bold a formatted cell (supervisor rule: the best result in every table is bold)."""
    return "\\textbf{" + cell + "}"


def best_idx(values, higher: bool = True) -> set:
    """Indices of the best value(s); None entries are ignored, ties are all bold."""
    vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not vals:
        return set()
    target = max(v for _, v in vals) if higher else min(v for _, v in vals)
    return {i for i, v in vals if abs(v - target) < 1e-12}


# ---------------------------------------------------------------------------------------------
# T1 -- Headline discrimination/calibration, all 6 v2 models
# Source: data/m2_v2_results.csv. Was Benchmark_scripts/m2_generate_tables_v2.py:table_headline_discrimination.
# ---------------------------------------------------------------------------------------------
MODEL_ORDER = ["Static-Only MLP v2", "mTAN v2", "Temporal-Only GRU-D v2",
               "Frozen-Temporal Head Control v2", "XGBoost-Matched v2", "Hybrid GRU-D v2"]
MODEL_LABEL = {
    "Static-Only MLP v2": "Static-Only MLP",
    "mTAN v2": "mTAN (tuned)",
    "Temporal-Only GRU-D v2": "Temporal-Only GRU-D",
    "Frozen-Temporal Head Control v2": "Frozen-Temporal Head Control (tuned)",
    "XGBoost-Matched v2": "XGBoost-Matched (tuned)",
    "Hybrid GRU-D v2": "Hybrid GRU-D (primary, tuned)",
}


def table_headline_discrimination() -> None:
    """The "vs. T=2 AUROC" column was REMOVED 2026-09-22 on supervisor review. It compared each
    model's AUROC here against its value in an earlier pipeline that differed in TWO ways at once
    (a fixed triage-plus-last-observation representation AND a random rather than chronological
    split), so no difference in that column could be attributed to either change alone -- a
    confounded comparison presented as a result column. The historical values are not lost: they
    remain in the module's own findings dossier. Removing the column also removes this table's only
    ev.literal() records, so every cell here is now a replayable artifact-sourced value."""
    ev = Evidence("headline_discrimination_v2")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Discrimination and calibration, all 6 models, primary cohort, "
        r"chronological split. Three models (XGBoost-Matched, Frozen-Temporal "
        r"Head Control, Hybrid GRU-D) are reported at their final tuned configurations. "
        r"Mean $\pm$ SD over five training seeds. $^{\ddagger}$XGBoost-Matched is deterministic "
        r"(identical predictions for every seed), so its cells give the point value and the 95\% "
        r"patient-cluster bootstrap interval ("
        + ev.get("m2_v2_rev_s2_xgb_bootstrap.json", "/n_boot", fmt=",.0f") + r" resamples), a measure of population "
        r"uncertainty rather than seed variability. Best value per column in bold (highest AUROC "
        r"and AUPRC, lowest Brier score and ECE). "
        + PROXY + r".}",
        r"\label{tab:m2-v2-headline}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & AUROC & AUPRC & Brier & ECE \\",
        r"\midrule",
    ]
    s2 = "m2_v2_rev_s2_xgb_bootstrap.json"
    rows, raws = [], []
    for m in MODEL_ORDER:
        filt = {"model": m}
        if m == "XGBoost-Matched v2":
            # S2 (revision): deterministic model, identical across seeds -> patient-cluster bootstrap CI
            auroc, auprc = (
                f"{ev.get(s2, f'/{k}/point', fmt='.4f')} "
                f"[{ev.get(s2, f'/{k}/ci_lo', fmt='.4f')}, {ev.get(s2, f'/{k}/ci_hi', fmt='.4f')}]$^{{\\ddagger}}$"
                for k in ("auroc", "auprc"))
            r_auroc = ev.get(s2, "/auroc/point", note="raw, bold-best selection only")
            r_auprc = ev.get(s2, "/auprc/point", note="raw, bold-best selection only")
        else:
            auroc = pm(ev, "m2_v2_results.csv", filt, "auroc", 4)
            auprc = pm(ev, "m2_v2_results.csv", filt, "auprc", 4)
            r_auroc = ev.reduce_csv("mean", "m2_v2_results.csv", filt, "auroc", note="raw, bold-best selection only")
            r_auprc = ev.reduce_csv("mean", "m2_v2_results.csv", filt, "auprc", note="raw, bold-best selection only")
        brier = ev.reduce_csv("mean", "m2_v2_results.csv", filt, "brier", fmt=".4f")
        ece = ev.reduce_csv("mean", "m2_v2_results.csv", filt, "ece", fmt=".4f")
        r_brier = ev.reduce_csv("mean", "m2_v2_results.csv", filt, "brier", note="raw, bold-best selection only")
        r_ece = ev.reduce_csv("mean", "m2_v2_results.csv", filt, "ece", note="raw, bold-best selection only")
        rows.append([MODEL_LABEL[m], auroc, auprc, brier, ece])
        raws.append([r_auroc, r_auprc, r_brier, r_ece])
    for col, higher in ((0, True), (1, True), (2, False), (3, False)):
        for i in best_idx([r[col] for r in raws], higher):
            rows[i][col + 1] = bf(rows[i][col + 1])
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("headline_discrimination_v2.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T2 -- The 5 v2 discrimination deltas (bootstrap CIs, Holm-corrected)
# Source: data/m2_v2_deltas.json. Was table_deltas().
# ---------------------------------------------------------------------------------------------
DELTA_LABEL = {
    "Delta_representation": r"Hybrid GRU-D $-$ XGBoost-Matched (primary comparison)",
    "Delta_static_contribution": r"Hybrid GRU-D $-$ Frozen-Temporal Head Control (static fusion)",
    "Delta_temporal_alone": r"Temporal-Only GRU-D $-$ XGBoost-Matched",
    "Delta_static_only": r"Static-Only MLP $-$ Temporal-Only GRU-D",
    "Delta_modern_comparator": r"Hybrid GRU-D $-$ mTAN",
}
DELTA_ORDER = ["Delta_representation", "Delta_static_contribution", "Delta_temporal_alone",
               "Delta_static_only", "Delta_modern_comparator"]


def table_deltas() -> None:
    ev = Evidence("deltas_v2")
    alpha = ev.literal("0.05", reason="declared significance threshold alpha, not a measured "
                        "value -- matches holm_significant_at_0.05's own field name in the source "
                        "JSON")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{The five pre-specified pairwise differences in AUROC (first model minus second), "
        r"patient-cluster bootstrap 95\% CIs, Holm-corrected for multiplicity. All values use the "
        rf"final tuned configurations. Every difference is significant at $\alpha={alpha}$. The "
        r"difference between Temporal-Only GRU-D and XGBoost-Matched does not, on its own, isolate "
        r"temporal representation, because only XGBoost-Matched receives the static features "
        r"(Sec.~\ref{sec:results-discrimination}); on the composite outcome it shows tuned "
        r"XGBoost-Matched significantly ahead of Temporal-Only GRU-D alone. Significant differences "
        r"in bold. " + PROXY + r".}",
        r"\label{tab:m2-v2-deltas}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & Mean & SD (seeds) & Holm-corrected $p$ & Significant \\",
        r"\midrule",
    ]
    for name in DELTA_ORDER:
        mean = ev.get("m2_v2_deltas.json", f"/{name}/mean", fmt="+.4f")
        sd = ev.get("m2_v2_deltas.json", f"/{name}/sd", fmt=".4f")
        p_raw = ev.get("m2_v2_deltas.json", f"/{name}/holm_corrected_p",
                        note="raw p, for the <0.001 display-threshold decision only")
        sig = ev.get("m2_v2_deltas.json", f"/{name}/holm_significant_at_0.05")
        p_str = ev.literal("$<$0.001", reason="display threshold for p<0.0005, applied to the "
                            "already-recorded raw p above") if p_raw < 0.0005 else f"{p_raw:.4f}"
        lines.append(f"{DELTA_LABEL[name]} & {bf(mean) if sig else mean} & {sd} & {p_str} & {'Yes' if sig else 'No'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("deltas_v2.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T3 -- Marginal vs. class-1 coverage, MAPIE vs. ECI vs. Mondrian-LAC (headline conformal result)
# Source: data/m2_v2_conformal_summary.csv. Was table_conformal_summary().
# ---------------------------------------------------------------------------------------------
CONF_MODELS = ["Temporal-Only GRU-D v2", "Frozen-Temporal Head Control v2", "Hybrid GRU-D v2"]


def table_conformal_summary() -> None:
    ev = Evidence("conformal_summary_v2")
    valid_thr = ev.literal("0.89", reason=VALID_THRESHOLD_REASON)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Marginal and class-1 (deterioration) coverage of MAPIE (split-conformal), ECI "
        r"(adaptive) and Mondrian-LAC (label-conditional) prediction sets on the chronological "
        r"primary split; the nominal level is 90\%. Entries are five-seed means, with the standard "
        r"deviation shown for Mondrian-LAC. MAPIE here is the primary classification-set run (MAPIE "
        r"library); the static split-conformal column of Table~\ref{tab:m2-eci-vs-recal} reports a "
        r"separately recomputed interval construction of the same quantile (see its caption). "
        rf"Class-1 coverage values at or above the validity threshold of {valid_thr} are in bold. "
        + PROXY + r".}",
        r"\label{tab:m2-v2-conformal}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & MAPIE (marginal / class-1) & ECI (marginal / class-1) & Mondrian-LAC (class-1) \\",
        r"\midrule",
    ]
    art = "m2_v2_conformal_summary.csv"
    for m in CONF_MODELS:
        filt = {"model": m}
        mapie = (f"{ev.reduce_csv('mean', art, filt, 'mapie_coverage', fmt='.4f')} / "
                 f"{ev.reduce_csv('mean', art, filt, 'mapie_coverage_class1', fmt='.4f')}")
        eci = (f"{ev.reduce_csv('mean', art, filt, 'eci_coverage', fmt='.4f')} / "
               f"{ev.reduce_csv('mean', art, filt, 'eci_coverage_class1', fmt='.4f')}")
        mondrian = pm(ev, art, filt, "mondrian_lac_coverage_class1", 4)
        r_mond = ev.reduce_csv("mean", art, filt, "mondrian_lac_coverage_class1", note="raw, bold rule only")
        if r_mond >= 0.89:
            mondrian = bf(mondrian)
        label = "Hybrid GRU-D (primary)" if m == "Hybrid GRU-D v2" else m.replace(" v2", "")
        lines.append(f"{label} & {mapie} & {eci} & {mondrian} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("conformal_summary_v2.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T4 -- NEWS2 six-arm sensitivity analysis
# Source: data/m2_news2_sensitivity.csv (one row per arm, no seed column). Was table_news2_sensitivity().
# ---------------------------------------------------------------------------------------------
ARM_ORDER = ["A_composite_critical", "B_mortality_alone", "C_icu_transfer_alone",
             "D_mortality_24h_from_edout", "E_complete_case_composite", "F_last_ed_vitals_composite"]
ARM_LABEL = {
    "A_composite_critical": "A: triage, composite outcome",
    "B_mortality_alone": "B: triage, in-hospital death",
    "C_icu_transfer_alone": "C: triage, ICU transfer",
    "D_mortality_24h_from_edout": "D: triage, death within 24 h of ED departure",
    "E_complete_case_composite": "E: triage, complete cases, composite",
    "F_last_ed_vitals_composite": "F: last-recorded ED vitals, composite",
}


def table_news2_sensitivity() -> None:
    ev = Evidence("news2_sensitivity")
    news2_range = ev.literal("0.71-0.86", reason="published ED-validation NEWS2 AUROC range -- "
                              "now independently verified (Phase 2 bib-verification pass, "
                              "2026-09-22) against three primary sources cited in "
                              "paper/refs/m2.bib: gerry2020ews (BMJ 2020), mhdyunin2025news2 "
                              "(PLOS ONE 2025), guan2022ewsmeta (PLOS ONE 2022); see "
                              "paper/sections/02_related_work.tex")
    # Chronological test period (m2_news2_sensitivity_v2.csv, from test_v2.csv) -- the same population
    # as every other reported result. The earlier random-split run (m2_news2_sensitivity.csv) is not
    # tabulated: it used a different test population with upstream-imputed triage vitals.
    art = "m2_news2_sensitivity_v2.csv"
    n_all = ev.get_csv(art, {"arm": "A_composite_critical"}, "n",
                       note="raw, for the excluded-stay derivation in the caption only")
    n_cc = ev.get_csv(art, {"arm": "E_complete_case_composite"}, "n",
                      note="raw, for the excluded-stay derivation in the caption only")
    n_excluded = ev.literal(f"{int(n_all - n_cc):,}", reason="derived as n(arm A) - n(arm E), both traced "
                            "above: stays with at least one of the five scored triage parameters missing")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{NEWS2 sensitivity analysis on the chronological test period. NEWS2 is computed "
        r"from 5 of its 7 parameters (consciousness level and supplemental oxygen are not available); "
        r"a missing parameter scores 0 points. Scored at triage (Arm A), it falls well below the AUROC "
        rf"range reported in published ED validations ({news2_range}). Arms B--D change the outcome "
        rf"definition. Arm E excludes the {n_excluded} stays with a missing triage parameter; Arm F "
        r"rescores on the last-recorded ED vitals and excludes stays with none. Highest AUROC in "
        r"bold. " + PROXY + r".}",
        r"\label{tab:m2-news2-sensitivity}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Arm & $n$ (positives) & AUROC & 95\% CI \\",
        r"\midrule",
    ]
    news_rows, news_raw = [], []
    for arm in ARM_ORDER:
        filt = {"arm": arm}
        n = ev.get_csv(art, filt, "n", fmt=",.0f")
        pos_rate = ev.get_csv(art, filt, "pos_rate",
                               note="used only to derive the displayed positive count below")
        n_raw = ev.get_csv(art, filt, "n", note="raw, for the positive-count derivation only")
        n_pos = ev.literal(round(n_raw * pos_rate), reason="derived as round(n * pos_rate), both "
                            "already traced above -- not an independent measurement")
        auroc = ev.get_csv(art, filt, "auroc", fmt=".4f")
        ci_lo = ev.get_csv(art, filt, "auroc_ci_lo", fmt=".4f")
        ci_hi = ev.get_csv(art, filt, "auroc_ci_hi", fmt=".4f")
        news_rows.append([ARM_LABEL[arm], f"{n} ({n_pos})", auroc, f"[{ci_lo}, {ci_hi}]"])
        news_raw.append(ev.get_csv(art, filt, "auroc", note="raw, bold-best selection only"))
    for i in best_idx(news_raw, True):
        news_rows[i][2] = bf(news_rows[i][2])
    for r in news_rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    finish("news2_sensitivity.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T5 -- Pre-specified decisive comparators for the online label-conditional construction
# Source: data/m2_v2_mondrian_eci_comparators.csv, Hybrid GRU-D v2 only. Was table_mondrian_comparators().
# ---------------------------------------------------------------------------------------------
CMP_ORDER = ["A", "B", "C1", "C2", "D", "E"]
CMP_LABEL = {
    "A": "A: static Mondrian-LAC", "B": "B: Mondrian-ECI (online, label-cond.)",
    "C1": "C1: static Mondrian + recal.\\ ($W{=}10$k)", "C2": "C2: static Mondrian + recal.\\ ($W{=}40$k)",
    "D": "D: static score-bin Mondrian", "E": "E: online score-bin ECI",
}


VALID_THRESHOLD_REASON = ("declared validity threshold (both-class coverage >= this value), not "
                           "a measured value -- matches the module's alpha=0.1 conformal target "
                           "(1 - 0.1 = 0.90, with a small pre-specified slack to 0.89)")


def table_mondrian_comparators() -> None:
    ev = Evidence("mondrian_comparators")
    valid_thr = ev.literal("0.89", reason=VALID_THRESHOLD_REASON)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Pre-specified comparators for the online label-conditional construction (B), "
        r"Hybrid GRU-D (primary model). A method is \emph{valid} if coverage of both classes is "
        rf"$\geq {valid_thr}$. A: static Mondrian-LAC; B: Mondrian-ECI; C1 and C2: static Mondrian "
        r"with periodic recalibration over the two window lengths shown; D and E: conditioning on "
        r"bins of the model's own risk score instead of the true label, static (D) and online (E). "
        r"Labels are assumed immediate for all methods; recalibration windows are in arrivals, not "
        r"calendar time. The valid method with the smallest mean set size is in bold. " + PROXY + r".}",
        r"\label{tab:m2-mondrian-comparators}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Class-0 cov. & Class-1 cov. & Mean $|C|$ & "
        r"Informative $P(C{=}\{1\}\mid y{=}1)$ & Valid \\",
        r"\midrule",
    ]
    art = "m2_v2_mondrian_eci_comparators.csv"
    cmp_rows, cmp_size = [], []
    for m in CMP_ORDER:
        filt = {"model": "Hybrid GRU-D v2", "method": m}
        c0 = pm(ev, art, filt, "coverage_class0", 3)
        c1 = pm(ev, art, filt, "coverage_class1", 3)
        size = pm(ev, art, filt, "mean_set_size", 3)
        info = pm(ev, art, filt, "informative_class1", 3)
        c0_mean = ev.reduce_csv("mean", art, filt, "coverage_class0", note="validity check only")
        c1_mean = ev.reduce_csv("mean", art, filt, "coverage_class1", note="validity check only")
        valid = "yes" if (c0_mean >= float(valid_thr) and c1_mean >= float(valid_thr)) else "no"
        cmp_rows.append([CMP_LABEL[m], c0, c1, size, info, valid])
        cmp_size.append(ev.reduce_csv("mean", art, filt, "mean_set_size", note="raw, bold rule only")
                        if valid == "yes" else None)
    for i in best_idx(cmp_size, higher=False):
        cmp_rows[i] = [cmp_rows[i][0]] + [bf(c) for c in cmp_rows[i][1:5]] + [cmp_rows[i][5]]
    for r in cmp_rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("mondrian_comparators.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T6 -- ECI versus periodic recalibration (the module's headline negative-result table)
# Source: data/m2_v2_eci_vs_recalibration.csv. Was table_eci_vs_recal().
# ---------------------------------------------------------------------------------------------
ER_ORDER = ["M1", "M2", "M3", "M4"]
ER_LABEL = {"M1": "static split-conformal", "M2": "ECI",
            "M3": r"recal.\ ($W{=}10$k)", "M4": r"recal.\ ($W{=}40$k)"}
ER_MODELS = [("Hybrid GRU-D v2", "Hybrid GRU-D (primary)"),
             ("Temporal-Only GRU-D v2", "Temporal-Only GRU-D"),
             ("Frozen-Temporal Head Control v2", "Frozen-Temporal Head Control")]


def table_eci_vs_recal() -> None:
    ev = Evidence("eci_vs_recalibration")
    valid_thr = ev.literal("0.89", reason=VALID_THRESHOLD_REASON)
    cross_check_tol = ev.literal("0.003", reason="declared cross-check tolerance in "
                                  "Benchmark_scripts/m2_eci_vs_recalibration.py (asserted at runtime: "
                                  "'T6 FAILED {model} seed {seed}: M1 marginal coverage ... vs MAPIE "
                                  "...' if exceeded) between this table's locally re-implemented static "
                                  "split-conformal interval construction and Table VI's MAPIE "
                                  "prediction-set coverage -- a declared design tolerance the pipeline "
                                  "checks against, not a measured value itself")
    # Computed (not hand-copied) so it stays correct if the underlying CSVs are ever regenerated:
    # re-reads both tables' own already-recorded mean coverage for the static/MAPIE construction and
    # takes the max absolute gap across the three GRU-D-family models, for the caption's cross-table
    # traceability note below.
    _mapie_art, _static_art = "m2_v2_conformal_summary.csv", "m2_v2_eci_vs_recalibration.csv"
    _gaps = []
    for _key, _ in ER_MODELS:
        _mapie_mean = ev.reduce_csv("mean", _mapie_art, {"model": _key}, "mapie_coverage",
                                     note="same value as Table VI's own MAPIE marginal-coverage cell "
                                          "for this model; re-read here only to compute the cross-table "
                                          "gap fact below")
        _static_mean = ev.reduce_csv("mean", _static_art, {"model": _key, "method": "M1"}, "coverage",
                                      note="same value as this table's own static split-conformal "
                                           "Cov. cell for this model")
        _gaps.append(abs(_mapie_mean - _static_mean))
    observed_max_gap = ev.literal(f"{max(_gaps):.4f}", reason="max |this table's static "
                                   "split-conformal mean coverage - Table VI's MAPIE mean coverage| "
                                   "across the three GRU-D-family models, computed above from the two "
                                   "tables' own recorded cells -- a derived display fact, not an "
                                   "independent measurement")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Static split-conformal, ECI and periodic recalibration of split-conformal: "
        r"marginal coverage and mean clipped interval width (mean $\pm$ SD over seeds) on the "
        rf"chronological test period. Valid means marginal coverage $\geq {valid_thr}$. ECI is run "
        r"as implemented in the authors' reference code. Labels are assumed immediate for all "
        r"methods. The static column is an independently recomputed interval "
        r"$[\hat{p}-q,\hat{p}+q]\cap[0,1]$ built from the same LAC quantile as the MAPIE "
        r"classification-set run of Table~\ref{tab:m2-v2-conformal}, so that its width is "
        r"comparable with the other three columns. The pipeline requires the two to agree in "
        rf"marginal coverage to within {cross_check_tol} on every model and seed (largest gap "
        rf"between the unrounded means: {observed_max_gap}). For each model, the valid method with "
        r"the narrowest interval is in bold. " + PROXY + r".}",
        r"\label{tab:m2-eci-vs-recal}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "cc" * len(ER_ORDER) + "}",
        r"\toprule",
        r"Model & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ER_LABEL[m]}}}" for m in ER_ORDER) + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        " & " + " & ".join("Cov. & Width" for _ in ER_ORDER) + r" \\",
        r"\midrule",
    ]
    art = "m2_v2_eci_vs_recalibration.csv"
    for key, label in ER_MODELS:
        cells, widths = [], []
        for m in ER_ORDER:
            filt = {"model": key, "method": m}
            cells.append(pm(ev, art, filt, "coverage", 4))
            cells.append(pm(ev, art, filt, "width", 3))
            cov_raw = ev.reduce_csv("mean", art, filt, "coverage", note="raw, bold rule only")
            widths.append(ev.reduce_csv("mean", art, filt, "width", note="raw, bold rule only")
                          if cov_raw >= float(valid_thr) else None)
        for i in best_idx(widths, higher=False):
            cells[2 * i], cells[2 * i + 1] = bf(cells[2 * i]), bf(cells[2 * i + 1])
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("eci_vs_recalibration.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T7 -- Horizon ablation (1h/6h/12h)
# Source: data/m2_v2_horizon_summary.csv (one row per horizon x model) + m2_v2_horizon_deltas.json.
# Was table_horizon_ablation().
# ---------------------------------------------------------------------------------------------
HZ_MODELS = [("Hybrid GRU-D v2", "Hybrid"), ("Temporal-Only GRU-D v2", "Temporal-Only"),
             ("XGBoost-Matched v2", "XGBoost")]
HZ_ROWS = [("1h", r"1h$^{\dagger}$"), ("6h", "6h"), ("12h", "12h")]


def table_horizon_ablation() -> None:
    ev = Evidence("horizon_ablation")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Sensitivity to the outcome definition: the same models and the same full per-stay "
        r"input sequence, evaluated against three time-anchored label definitions $y_h$, "
        r"$h\in\{1\mathrm{h}, 6\mathrm{h}, 12\mathrm{h}\}$, on the chronological split. The input "
        r"is not truncated at $h$, so the rows compare outcome definitions, not predictions made "
        r"$h$ hours ahead. Encoders were trained per definition with the hyperparameters tuned for "
        r"the composite outcome; 3 of 6 models. ``Sig.'' requires Holm significance and all 5 "
        r"per-seed confidence intervals excluding zero. $^{\dagger}$Underpowered (few test and "
        r"validation positives); not interpreted. Highest AUROC per definition and significant "
        r"differences in bold. "
        + PROXY + r".}",
        r"\label{tab:m2-horizon}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrccccc}",
        r"\toprule",
        r"Label definition & Test pos. & Hybrid & Temporal-Only & XGBoost & "
        r"Hybrid $-$ XGB (95\% CI; sig.) & Temporal $-$ XGB (95\% CI; sig.) \\",
        r"\midrule",
    ]
    csv_art, json_art = "m2_v2_horizon_summary.csv", "m2_v2_horizon_deltas.json"
    for h, label in HZ_ROWS:
        cells, hz_raw = [], []
        for key, _ in HZ_MODELS:
            filt = {"horizon": h, "model": key}
            mean = ev.get_csv(csv_art, filt, "auroc_mean", fmt=".4f")
            sd = ev.get_csv(csv_art, filt, "auroc_sd", fmt=".4f")
            cells.append(f"{mean} $\\pm$ {sd}")
            hz_raw.append(ev.get_csv(csv_art, filt, "auroc_mean", note="raw, bold-best selection only"))
        for i in best_idx(hz_raw, True):
            cells[i] = bf(cells[i])
        n_pos = ev.get_csv(csv_art, {"horizon": h, "model": HZ_MODELS[0][0]}, "n_pos", fmt=",.0f")
        dcells = []
        for dn in ("D_repr", "D_temporal"):
            mean = ev.get(json_art, f"/{h}/deltas/{dn}/mean", fmt="+.4f")
            ci_lo = ev.get(json_art, f"/{h}/deltas/{dn}/ci_lo_mean", fmt="+.4f")
            ci_hi = ev.get(json_art, f"/{h}/deltas/{dn}/ci_hi_mean", fmt="+.4f")
            sig = ev.get(json_art, f"/{h}/deltas/{dn}/called_significant")
            dcell = f"{mean} [{ci_lo}, {ci_hi}]; {'yes' if sig else 'no'}"
            dcells.append(bf(dcell) if sig else dcell)
        lines.append(f"{label} & {n_pos} & " + " & ".join(cells) + " & " + " & ".join(dcells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("horizon_ablation.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T8 -- G7 clinician-facing conformal decision table
# Source: data/m2_v2_conformal_summary.csv. Was m2_generate_table_g7.py.
# ---------------------------------------------------------------------------------------------
G7_MODELS = [("Temporal-Only GRU-D v2", "Temporal-Only GRU-D"),
             ("Frozen-Temporal Head Control v2", "Frozen-Temporal Head Control"),
             ("Hybrid GRU-D v2", "Hybrid GRU-D (primary)")]
# (method label, coverage_class1 col, empty_set_rate_class1 col or None, is_mapie)
G7_METHODS = [
    ("MAPIE (split-conformal)", "mapie_coverage_class1", "mapie_empty_set_rate_class1", True),
    ("ECI (adaptive)", "eci_coverage_class1", None, False),
    ("Mondrian-LAC (label-conditional)", "mondrian_lac_coverage_class1", "mondrian_lac_empty_set_rate_class1", False),
]


def table_g7_clinician_decision() -> None:
    """A real, confirmed discrepancy versus the pre-rewrite g7_clinician_decision.tex: the original
    Benchmark_scripts/m2_generate_table_g7.py computed std via `sub_model[col].values.std()` --
    converting to a numpy array before calling .std(), which defaults to ddof=0 (population std) --
    while every OTHER table in this module (headline_discrimination_v2, conformal_summary_v2,
    mondrian_comparators, eci_vs_recalibration) calls pandas' own Series.std() directly, which
    defaults to ddof=1 (sample std). This Evidence-based rewrite uses reduce_csv("std", ...)
    consistently for every table, i.e. ddof=1 everywhere -- matching the manuscript-wide convention
    the other nine tables already use, not the one-table-only ddof=0 the original G7 script
    happened to compute. Confirmed numerically: for Temporal-Only GRU-D's MAPIE class-1 coverage
    (5 seeds), ddof=0 gives 0.501 (displays as the old table's "0.5"), ddof=1 gives 0.560 (displays
    as "0.6") -- a real, reproducible difference, not a formatting artifact. Recorded here rather
    than silently reproducing the inconsistency."""
    ev = Evidence("g7_clinician_decision")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Clinician-facing view of the coverage results in Table~\ref{tab:m2-v2-conformal}. "
        r"Among patients who deteriorate (class 1): how often each method's 90\%-target prediction "
        r"set or interval \emph{fails} to cover the outcome (miss rate, the falsely reassuring "
        r"case), how often it abstains (empty-set rate), and the average interval width or set "
        r"size. MAPIE's width and set-size cell is left blank (--) because its interval width was "
        r"not persisted per class. Entries are five-seed means $\pm$ SD. For each model, the method "
        r"with the highest class-1 coverage (lowest miss rate) is in bold. " + PROXY + r".}",
        r"\label{tab:m2-g7-clinician-decision}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Method & Coverage (class 1, \%) & Miss rate (class 1, \%) & "
        r"Empty-set rate (class 1, \%) & Width / set size \\",
        r"\midrule",
    ]
    art = "m2_v2_conformal_summary.csv"
    for key, model_label in G7_MODELS:
        first = True
        g7_rows, g7_raw = [], []
        for method_label, cov_col, empty_col, is_mapie in G7_METHODS:
            filt = {"model": key}
            # Raw (unformatted) mean/std here, on purpose: the displayed percentage must be
            # scaled (x100) BEFORE rounding to 1dp, not rounded to 3dp in fraction-space first and
            # then scaled (which would round twice and drift from the original script's output --
            # found by diffing against the pre-rewrite .tex, e.g. 0.6 vs the original's 0.5).
            cov_mean = ev.reduce_csv("mean", art, filt, cov_col,
                                      note="raw; formatted at percent scale below, not here")
            cov_std = ev.reduce_csv("std", art, filt, cov_col,
                                     note="raw; formatted at percent scale below, not here")
            derived_reason = ("displayed as (raw value already traced above) x 100, rounded to "
                               "1dp -- not an independent measurement")
            cov_cell = ev.literal(f"{cov_mean * 100:.1f} $\\pm$ {cov_std * 100:.1f}", reason=derived_reason)
            miss_cell = ev.literal(f"{(1 - cov_mean) * 100:.1f} $\\pm$ {cov_std * 100:.1f}", reason=derived_reason)
            if empty_col is not None:
                empty_mean = ev.reduce_csv("mean", art, filt, empty_col, note="raw; percent-scaled below")
                empty_std = ev.reduce_csv("std", art, filt, empty_col, note="raw; percent-scaled below")
                empty_cell = ev.literal(f"{empty_mean * 100:.1f} $\\pm$ {empty_std * 100:.1f}", reason=derived_reason)
            else:
                empty_cell = ev.literal("--", reason="ECI produces a continuous interval, no "
                                         "empty-set concept -- never estimated")
            if is_mapie:
                width_cell = ev.literal("--", reason="MAPIE's own interval width was not "
                                         "separately persisted per class in this artifact")
            else:
                width_cell = pm(ev, art, filt, "eci_width" if "ECI" in method_label else "mondrian_mean_set_size", 3)
            model_cell = model_label if first else ""
            first = False
            g7_rows.append([model_cell, method_label, cov_cell, miss_cell, empty_cell, width_cell])
            g7_raw.append(cov_mean)
        for i in best_idx(g7_raw, True):
            g7_rows[i][2], g7_rows[i][3] = bf(g7_rows[i][2]), bf(g7_rows[i][3])
        for r in g7_rows:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("g7_clinician_decision.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T9 -- Label lag (two tables: marginal, class-conditional)
# Source: data/m2_v2_label_lag.csv, Hybrid GRU-D v2 only. Was m2_generate_table_lag.py:main().
# ---------------------------------------------------------------------------------------------
LAGS = ["0", "60", "240", "1000", "4000", "16000", "inf"]
HOURS = {"0": "0", "60": r"$\approx$12\,h", "240": r"$\approx$2\,d", "1000": r"$\approx$8\,d",
         "4000": r"$\approx$33\,d", "16000": r"$\approx$4.4\,mo", "inf": "never"}


def _lagcell(ev: Evidence, lag: str) -> str:
    if lag == "inf":
        return ev.literal(r"$\infty$", reason="the label-lag sweep's 'never observed' arm, a "
                           "declared sentinel row, not a measured lag value")
    return ev.literal(f"{int(lag):,}".replace(",", "{,}"),
                       reason="LaTeX thousands-separator formatting of the declared lag value "
                       "(the sweep's own experimental design, not a measurement)")


def _hours(ev: Evidence, lag: str) -> str:
    return ev.literal(HOURS[lag], reason="coarse hour-equivalent conversion at ~5 arrivals/hour "
                       "(the sweep's own declared, documented approximation, not a measurement -- "
                       "see m2_generate_table_lag.py's original docstring, +/-50% uncertain)")


def table_label_lag() -> None:
    ev = Evidence("label_lag")
    valid_thr = ev.literal("0.89", reason=VALID_THRESHOLD_REASON)
    slack_thr = ev.literal("0.02", reason="declared ECI-not-needed slack threshold (a valid "
                            "recalibration counts as 'not needing' ECI if its width is at most "
                            "this much wider) -- a design decision, not a measured value")
    art = "m2_v2_label_lag.csv"
    a_c0 = ev.reduce_csv("mean", art, {"model": "Hybrid GRU-D v2", "family": "class", "lag": "0", "method": "A"}, "coverage_class0", fmt=".3f")
    a_c1 = ev.reduce_csv("mean", art, {"model": "Hybrid GRU-D v2", "family": "class", "lag": "0", "method": "A"}, "coverage_class1", fmt=".3f")
    a_size = ev.reduce_csv("mean", art, {"model": "Hybrid GRU-D v2", "family": "class", "lag": "0", "method": "A"}, "size", fmt=".3f")
    a_txt = f"{a_c0} / {a_c1} / {a_size}"
    a_validity = "valid" if min(float(a_c0), float(a_c1)) >= float(valid_thr) else "not valid"
    lag_1000_plain = ev.literal(1000, reason="the realistic pre-specified lag value 1000, written "
                                 "without the thousands-separator braces here since it's inside a "
                                 "prose set notation ($L\\in\\{...\\}$) rather than a table cell -- "
                                 "same declared experimental-design value as the '1{,}000' row "
                                 "label elsewhere in this table")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Label lag (pre-specified), "
        r"Hybrid GRU-D (primary model). " + PROXY + r". Lag $L$ is in arrivals; hour equivalents assume "
        r"$\sim$5 arrivals/hour and are approximate ($\pm$50\%). \emph{Marginal:} coverage / mean "
        r"clipped interval width for ECI and for static split-conformal with periodic "
        rf"recalibration ($W$=10k and $W$=40k). Valid means coverage $\geq$ {valid_thr}; ECI is not "
        rf"needed under these conditions if a valid recalibration is at most {slack_thr} wider. "
        r"For each lag, the valid method with the narrowest interval is in bold. "
        r"$L=\infty$ is the setting in which no label is ever observed, so recalibration reduces to "
        r"the static interval. Realistic band "
        rf"(pre-specified): $L\in\{{60, 240, {lag_1000_plain}\}}$. Labels are otherwise assumed uniform-lag; ECI's "
        r"learning rate is re-selected per lag on the calibration-selection fold.}",
        r"\label{tab:m2-label-lag-marginal}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"$L$ (arrivals) & $\approx$ time & ECI & recal.\ $W$=10k & recal.\ $W$=40k \\",
        r"\midrule",
    ]
    for lag in LAGS:
        cells, lag_w = [], []
        for m in ("M2", "M3", "M4"):
            filt = {"model": "Hybrid GRU-D v2", "family": "marginal", "lag": lag, "method": m}
            cov = ev.reduce_csv("mean", art, filt, "coverage", fmt=".4f")
            size = ev.reduce_csv("mean", art, filt, "size", fmt=".3f")
            cells.append(f"{cov} / {size}")
            c_raw = ev.reduce_csv("mean", art, filt, "coverage", note="raw, bold rule only")
            lag_w.append(ev.reduce_csv("mean", art, filt, "size", note="raw, bold rule only")
                         if c_raw >= float(valid_thr) else None)
        for i in best_idx(lag_w, higher=False):
            cells[i] = bf(cells[i])
        lines.append(f"{_lagcell(ev, lag)} & {_hours(ev, lag)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]

    lines += [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Label lag, class-conditional (Mondrian) sets, Hybrid GRU-D (primary model). " + PROXY + r". "
        r"Class-0 / class-1 coverage, mean $|C|$ and the informative rate "
        r"$P(C{=}\{1\}\mid y{=}1)$ for Mondrian-ECI (B) and static Mondrian with periodic "
        rf"recalibration (C1: $W$=10k). Valid means both coverages $\geq$ {valid_thr}. The static "
        r"anchor (A) uses no label feedback and so does not depend on lag; its class-0 / class-1 "
        r"coverage and mean $|C|$ are " + a_txt + rf" ({a_validity}). For each lag, the valid "
        r"method with the smaller mean set size is in bold.}",
        r"\label{tab:m2-label-lag-class}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"$L$ (arrivals) & $\approx$ time & B: cov$_0$ / cov$_1$ & B: $|C|$ & B: informative & "
        r"C1: cov$_0$ / cov$_1$ / $|C|$ / informative \\",
        r"\midrule",
    ]
    for lag in LAGS:
        filt_b = {"model": "Hybrid GRU-D v2", "family": "class", "lag": lag, "method": "B"}
        filt_c1 = {"model": "Hybrid GRU-D v2", "family": "class", "lag": lag, "method": "C1"}
        b_c0 = ev.reduce_csv("mean", art, filt_b, "coverage_class0", fmt=".3f")
        b_c1 = ev.reduce_csv("mean", art, filt_b, "coverage_class1", fmt=".3f")
        b_size = ev.reduce_csv("mean", art, filt_b, "size", fmt=".3f")
        b_info = ev.reduce_csv("mean", art, filt_b, "informative_class1", fmt=".3f")
        c_c0 = ev.reduce_csv("mean", art, filt_c1, "coverage_class0", fmt=".3f")
        c_c1 = ev.reduce_csv("mean", art, filt_c1, "coverage_class1", fmt=".3f")
        c_size = ev.reduce_csv("mean", art, filt_c1, "size", fmt=".3f")
        c_info = ev.reduce_csv("mean", art, filt_c1, "informative_class1", fmt=".3f")
        b_cells = [f"{b_c0} / {b_c1}", b_size, b_info]
        c_cell = f"{c_c0} / {c_c1} / {c_size} / {c_info}"
        cand = [float(b_size) if min(float(b_c0), float(b_c1)) >= float(valid_thr) else None,
                float(c_size) if min(float(c_c0), float(c_c1)) >= float(valid_thr) else None]
        win = best_idx(cand, higher=False)
        if 0 in win:
            b_cells = [bf(c) for c in b_cells]
        if 1 in win:
            c_cell = bf(c_cell)
        lines.append(f"{_lagcell(ev, lag)} & {_hours(ev, lag)} & " + " & ".join(b_cells) + f" & {c_cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("label_lag.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T10 -- Small required-ablation re-runs: logit-vs-clipping + streaming latency
# Source: data/m2_v2_logit_vs_clipping.csv + data/m2_v2_latency.json. Was small_ablations().
# ---------------------------------------------------------------------------------------------
SA_MODELS = ["Frozen-Temporal Head Control v2", "Hybrid GRU-D v2", "Temporal-Only GRU-D v2"]
SA_BOUNDINGS = ["clipping", "logit"]


def table_small_ablations() -> None:
    ev = Evidence("small_ablations")
    art = "m2_v2_logit_vs_clipping.csv"
    hw_gpu = ev.get("m2_v2_latency.json", "/hardware/gpu")
    hw_cpu = ev.get("m2_v2_latency.json", "/hardware/cpu_count")
    n_timed = ev.get("m2_v2_latency.json", "/methodology/n_timed", fmt=",.0f")
    budget_ms = ev.get("m2_v2_latency.json", "/methodology/budget_ms", fmt=",.0f")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Interval bounding and streaming latency, chronological split (" + PROXY + r"). "
        r"\emph{Logit vs.\ clipping:} marginal coverage / class-1 coverage / mean width of the "
        r"same ECI radius under each bounding, 5-seed means. The logit-bounded interval lies "
        r"strictly inside $(0,1)$ and so cannot contain a hard label: its coverage is 0 by "
        rf"construction. \emph{{Latency:}} batch size 1, {n_timed} timed calls, Hybrid GRU-D "
        rf"(primary model) seed 42, {hw_gpu} and {hw_cpu} logical CPUs; hardware- and load-dependent, this machine "
        rf"only; the production deployment path is not measured here. Budget: {budget_ms}\,ms per "
        r"patient update. For each model, the bounding with the higher coverage is in bold.}",
        r"\label{tab:m2-small-ablations}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & bounding & coverage / class-1 coverage & width \\",
        r"\midrule",
    ]
    for m in SA_MODELS:
        sa_rows, sa_raw = [], []
        for bnd in SA_BOUNDINGS:
            filt = {"model": m, "bounding": bnd}
            cov = ev.reduce_csv("mean", art, filt, "coverage", fmt=".4f")
            cov1 = ev.reduce_csv("mean", art, filt, "coverage_class1", fmt=".4f")
            width = ev.reduce_csv("mean", art, filt, "width", fmt=".4f")
            sa_rows.append([m.replace(' v2', ''), bnd, f"{cov} / {cov1}", width])
            sa_raw.append(ev.reduce_csv("mean", art, filt, "coverage", note="raw, bold rule only"))
        for i in best_idx(sa_raw, True):
            sa_rows[i][2] = bf(sa_rows[i][2])
        for r in sa_rows:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\midrule",
              r"Latency path & \multicolumn{3}{c}{median / p95 / p99 / max (ms) or amortized per arrival} \\",
              r"\midrule"]
    lat = "m2_v2_latency.json"
    for key, name in (("forward_gpu", "forward pass, GPU"), ("forward_cpu", "forward pass, CPU"),
                       ("end_to_end_gpu_forward_plus_eci", "end to end: GPU forward + ECI step + set")):
        med = ev.get(lat, f"/paths/{key}/median_ms", fmt=".2f")
        p95 = ev.get(lat, f"/paths/{key}/p95_ms", fmt=".2f")
        p99 = ev.get(lat, f"/paths/{key}/p99_ms", fmt=".2f")
        mx = ev.get(lat, f"/paths/{key}/max_ms", fmt=".1f")
        lines.append(f"{name} & \\multicolumn{{3}}{{c}}{{{med} / {p95} / {p99} / {mx}}} \\\\")
    for key, name in (("split_conformal_set", "split-conformal set"),
                       ("eci_step_plus_set", "ECI step + set"),
                       ("mondrian_eci_step_plus_set", "Mondrian-ECI step + set")):
        amort = ev.get(lat, f"/paths/{key}/amortized_ms_per_arrival", fmt=".4f")
        lines.append(f"{name} & \\multicolumn{{3}}{{c}}{{{amort} per arrival}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("small_ablations.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T11 -- Conformal/coverage parameter consolidation (venue_style_notes.md Sec E: consolidate every
# locked parameter in one place, matching what M1's Table 3 already did for CKKS parameters).
# Every value here is a DECLARED DESIGN CONSTANT (the module's own experimental protocol), not a
# measurement -- ev.literal() throughout, each reason naming which table(s) already use this same
# constant so a reader can cross-check.
# ---------------------------------------------------------------------------------------------
def table_parameter_summary() -> None:
    ev = Evidence("parameter_summary")
    alpha = ev.literal("0.10", reason="declared target miscoverage (90% target coverage), the "
                        "module's mandatory-core conformal target throughout T3/T5/T6/T8/T9 -- not "
                        "the unrelated alpha=0.05 Holm-correction significance threshold used for "
                        "the discrimination deltas in T2/T7")
    valid_thr = ev.literal("0.89", reason=VALID_THRESHOLD_REASON)
    horizons = ev.literal("1h, 6h, 12h", reason="the three pre-specified time-anchored label "
                           "definitions (CLAUDE.md M2 required ablation), same set T7 uses -- these "
                           "vary the outcome definition, not the prediction lead time")
    lag_band = ev.literal("60, 240, 1{,}000", reason="pre-specified realistic label-lag band "
                           "(arrivals), M2_LabelLag_PreRegistration.md -- same values T9's "
                           "(label_lag) table rows use")
    windows = ev.literal("10{,}000, 40{,}000", reason="the two periodic-recalibration windows W "
                          "(arrivals) tested throughout T6/T9, M3/M4 in their own column labels")
    cadence = ev.literal("2{,}000", reason="recalibration refresh cadence R (arrivals) -- the "
                          "threshold is recomputed every R arrivals from the trailing W-arrival "
                          "window, same constant used to generate T6/T9's recal. columns")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Conformal/coverage parameters used throughout this paper's results, "
        r"consolidated in one place (all are declared design constants of the module's "
        r"experimental protocol, not measurements). " + PROXY + r".}",
        r"\label{tab:parameter-summary}",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Parameter & Value \\",
        r"\midrule",
        rf"Target miscoverage $\alpha$ & {alpha} (90\% target coverage) \\",
        rf"Validity threshold (coverage $\geq$) & {valid_thr} \\",
        rf"Time-anchored label definitions $h$ & {horizons} \\",
        r"Realistic label-lag band $L$ (arrivals) & " + lag_band + r" \\",
        r"Recalibration windows $W$ (arrivals) & " + windows + r" \\",
        rf"Recalibration cadence $R$ (arrivals) & {cadence} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    finish("parameter_summary.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T12 -- Complete model hyperparameters (supplementary). Added 2026-09-22 on supervisor review:
# "equations essentielles dans Methods + hyperparametres complets dans supplementary material".
# The four TUNED models' configurations are read from their committed tuned-config JSON artifacts
# (so they replay); the two FIXED-configuration models (Temporal-Only GRU-D encoder, Static-Only
# MLP) are hardcoded in their training scripts rather than persisted as artifacts, so they are
# ev.literal() with the exact script and constant named in the reason.
# ---------------------------------------------------------------------------------------------
def table_model_hyperparameters() -> None:
    ev = Evidence("model_hyperparameters")

    def lit(value, script, names):
        return ev.literal(value, reason=f"fixed (not tuned) configuration, hardcoded as {names} in "
                                         f"Benchmark_scripts/{script}; no tuned-config artifact exists "
                                         f"for this model because no search was run over it")

    hy = "m2_v2_hybrid_head_tuned_config.json"
    fc = "m2_v2_frozen_control_tuned_config.json"
    xg = "m2_v2_xgboost_tuned_config.json"
    mt = "m2_v2_mtan_tuned_config_matched.json"
    note = "Proxy cohort (MIMIC-IV-ED-derived)."

    rows = [
        (r"Temporal-Only GRU-D (encoder; fixed)",
         "hidden 32, classifier hidden 32, dropout 0.2, weight decay $10^{-4}$, lr 0.01, "
         "batch 128, max epochs 100, patience 10",
         lit("hidden 32, classifier hidden 32, dropout 0.2, weight decay 1e-4, lr 0.01, batch 128, "
             "max epochs 100, patience 10",
             "m2_temporal_only_grud_v2.py", "HIDDEN/DROPOUT/WEIGHT_DECAY/LR/BATCH_SIZE")),
        (r"Static-Only MLP (fixed)",
         "hidden 32, dropout 0.2, lr $10^{-3}$, max epochs 3{,}000, patience 10",
         lit("hidden 32, dropout 0.2, lr 1e-3, max epochs 3000, patience 10",
             "m2_static_only_mlp_v2.py", "HIDDEN/DROPOUT/LR/MAX_EPOCHS/PATIENCE")),
    ]
    for label, artifact, fields in (
        ("Frozen-Temporal Head Control (tuned)", fc,
         [("fusion dropout", "/fusion_dropout"), ("lr", "/lr"), ("batch", "/batch_size"),
          ("max epochs", "/max_epochs"), ("patience", "/patience")]),
        ("Hybrid GRU-D (primary, tuned)", hy,
         [("static hidden", "/static_hidden"), ("static dropout", "/static_dropout"),
          ("fusion dropout", "/fusion_dropout"), ("lr", "/lr"), ("batch", "/batch_size"),
          ("max epochs", "/max_epochs"), ("patience", "/patience")]),
        ("XGBoost-Matched (tuned)", xg,
         [("n estimators", "/effective/n_estimators"), ("lr", "/effective/learning_rate"),
          ("max depth", "/effective/max_depth"), ("subsample", "/effective/subsample"),
          ("colsample", "/effective/colsample_bytree"),
          ("min child weight", "/effective/min_child_weight"),
          ("reg lambda", "/effective/reg_lambda"),
          ("scale pos weight", "/effective/scale_pos_weight")]),
        ("mTAN (tuned)", mt,
         [("nhidden", "/nhidden"), ("embed time", "/embed_time"),
          ("ref points", "/num_ref_points"), ("heads", "/num_heads"),
          ("lr", "/lr"), ("weight decay", "/weight_decay")]),
    ):
        cells = [f"{name} {ev.get(artifact, ptr, note=note)}" for name, ptr in fields]
        rows.append((label, ", ".join(cells), None))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Complete model hyperparameters. The four tuned models' values are read directly "
        r"from their committed tuned-configuration artifacts; the two fixed-configuration models "
        r"(no search was run over them) are reported from the constants in their training scripts. "
        r"Conformal parameters are in Table~I of the main paper. " + PROXY + r".}",
        r"\label{tab:m2-hyperparameters}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Model & Configuration \\",
        r"\midrule",
    ]
    for label, display, _ in rows:
        lines.append(f"{label} & {display} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("model_hyperparameters.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# Revision E3 -- adaptive set-valued baselines on the marginal LAC score (supplementary)
# Source: data/m2_v2_rev_e3_adaptive_sets.csv (Benchmark_scripts/m2_rev_e3_adaptive_sets.py).
# ---------------------------------------------------------------------------------------------
E3_ROWS = [("static_lac", "0", "static split-conformal (LAC)"), ("aci_lac", "0", "ACI"),
           ("eci_lac", "0", "ECI"), ("aci_lac", "240", "ACI, delay {d}"),
           ("eci_lac", "240", "ECI, delay {d}"), ("static_mondrian_lac", "0", "static Mondrian-LAC")]


def table_rev_adaptive_sets() -> None:
    ev = Evidence("rev_adaptive_sets")
    valid_thr = ev.literal("0.89", reason=VALID_THRESHOLD_REASON)
    delay = ev.literal(240, reason="pre-specified label delay in arrivals for E3 (middle of the "
                       "pre-specified realistic band), M2_Revision_PreSpecification.md")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Adaptive conformal prediction sets on the marginal LAC score, against static "
        r"split-conformal sets: marginal and class-conditional coverage, mean set size, and the share "
        r"of deteriorating patients whose set is $\{1\}$ (mean $\pm$ SD over seeds). ACI and ECI use "
        r"the reference implementations and the same learning-rate selection and warm-up as the "
        rf"interval ECI; labels are immediate unless a delay of {delay} arrivals is shown. Static "
        r"Mondrian-LAC is the class-conditional reference. Pre-specified rule: an adaptive method "
        rf"improves on static only if marginal coverage $\geq {valid_thr}$ and its mean set size is "
        r"smaller than static on all three models. For each model, the valid method with the smallest "
        r"mean set size is in bold. " + PROXY + r".}",
        r"\label{tab:m2-rev-adaptive-sets}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Method & Marginal cov. & Class-0 cov. & Class-1 cov. & Mean $|C|$ & "
        r"$P(C{=}\{1\}\mid y{=}1)$ \\",
        r"\midrule",
    ]
    art = "m2_v2_rev_e3_adaptive_sets.csv"
    for i, (key, label) in enumerate(ER_MODELS):
        e3_rows, e3_size = [], []
        for j, (method, lag, mlabel) in enumerate(E3_ROWS):
            f = {"model": key, "method": method, "lag": lag}
            cells = [pm(ev, art, f, c, 3) for c in
                     ("coverage_marginal", "coverage_class0", "coverage_class1", "mean_set_size", "informative_class1")]
            e3_rows.append([label if j == 0 else '', mlabel.format(d=delay)] + cells)
            cov_raw = ev.reduce_csv("mean", art, f, "coverage_marginal", note="raw, bold rule only")
            e3_size.append(ev.reduce_csv("mean", art, f, "mean_set_size", note="raw, bold rule only")
                           if cov_raw >= float(valid_thr) else None)
        for k in best_idx(e3_size, higher=False):
            e3_rows[k][5] = bf(e3_rows[k][5])
        for r in e3_rows:
            lines.append(" & ".join(r) + r" \\")
        if i < len(ER_MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("rev_adaptive_sets.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# Revision S1 -- significance of the interval-width differences (supplementary)
# Source: data/m2_v2_rev_s1_width_bootstrap.json (Benchmark_scripts/m2_rev_s1_width_bootstrap.py).
# ---------------------------------------------------------------------------------------------
S1_CONTRASTS = [("ECI - static", r"ECI $-$ static"), ("recal(10k) - static", r"recal.\ ($W{=}10$k) $-$ static"),
                ("recal(40k) - static", r"recal.\ ($W{=}40$k) $-$ static")]


def _pfmt(p: float) -> str:
    return r"$<0.001$" if p < 0.001 else f"{p:.3f}"


def table_rev_width_tests() -> None:
    ev = Evidence("rev_width_tests")
    art = "m2_v2_rev_s1_width_bootstrap.json"
    n_boot = ev.get(art, "/n_boot", fmt=",.0f")
    fam = ev.get(art, "/family_size")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Differences in mean interval width against static split-conformal: mean over seeds "
        r"of the per-seed difference, SD over seeds, the number of seeds whose 95\% patient-cluster "
        rf"bootstrap interval ({n_boot} resamples) excludes zero, and the Holm-adjusted $p$ over the "
        rf"{fam} contrasts. A difference is significant only if it is Holm-significant and all five "
        r"per-seed intervals exclude zero. The intervals are held at the values each method produced "
        r"on the observed stream, so the test concerns patient sampling, not the variability of the "
        r"online procedure. " + PROXY + r".}",
        r"\label{tab:m2-rev-width-tests}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Contrast & Mean diff. & SD & CIs excl.\ 0 & Holm $p$ \\",
        r"\midrule",
    ]
    for i, (key, label) in enumerate(ER_MODELS):
        for j, (c, clabel) in enumerate(S1_CONTRASTS):
            base = f"/contrasts/{key} | {c}"
            mean = ev.get(art, base + "/mean", fmt="+.4f")
            sd = ev.get(art, base + "/sd", fmt=".4f")
            n_ex = ev.get(art, base + "/excludes_zero_n_of_5")
            p = _pfmt(ev.get(art, base + "/holm_p"))
            sig = ev.get(art, base + "/significant_by_paper_rule")
            mark = r"$^{*}$" if sig else ""
            mcell = bf(f"{mean}{mark}") if sig else f"{mean}{mark}"
            lines.append(f"{label if j == 0 else ''} & {clabel} & {mcell} & {sd} & {n_ex}/5 & {p} \\\\")
        if i < len(ER_MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\multicolumn{6}{l}{\footnotesize $^{*}$Significant under the rule above (also in bold).}",
              r"\end{tabular}", r"\end{table*}"]
    finish("rev_width_tests.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# Revision E4 -- controlled within-period drift, stress test (supplementary/secondary only)
# Source: data/m2_v2_rev_e4_drift_stress.csv (Benchmark_scripts/m2_rev_e4_drift_stress.py).
# ---------------------------------------------------------------------------------------------
E4_SCEN = [("shift+0.5", "logit shift $+{a}$"), ("shift+1.0", "logit shift $+{b}$"),
           ("casemix", "case-mix change")]
E4_METH = [("static", "static split-conformal", "width"), ("recal_10k", r"recal.\ ($W{=}10$k)", "width"),
           ("eci_interval", "ECI (interval)", "width"), ("aci_lac", "ACI (set)", "set size"),
           ("eci_lac", "ECI (set)", "set size")]


def table_rev_drift_stress() -> None:
    ev = Evidence("rev_drift_stress")
    valid_thr = ev.literal("0.89", reason=VALID_THRESHOLD_REASON)
    art = "m2_v2_rev_e4_drift_stress.csv"
    sh_a = ev.literal("0.5", reason="pre-specified logit-shift size (E4, M2_Revision_PreSpecification.md)")
    sh_b = ev.literal("1.0", reason="pre-specified logit-shift size (E4, M2_Revision_PreSpecification.md)")
    roll = ev.literal("2{,}000", reason="pre-specified trailing-window length in arrivals (E4)")
    noise_sd = ev.literal("0.007", reason="binomial standard deviation of a coverage estimate over a "
                          "2,000-arrival window at coverage 0.90, sqrt(0.9*0.1/2000) = 0.0067 -- a "
                          "derived display fact explaining why the recovery-time measure is not reported")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Stress test with synthetic drift imposed on the real predictions of the primary "
        r"model (Hybrid GRU-D) at the midpoint of the test stream; model, calibration and labels are "
        r"unchanged. Logit shift: $\mathrm{logit}\,\hat{p}$ raised by the stated amount after the "
        r"change point. Case-mix change: the stream re-ordered so that the more severe half of stays "
        r"by triage acuity follows the change point (chronological order kept within each half). "
        r"Coverage before and after the change, class-1 coverage after it, and mean size (interval "
        r"width or set size) before and after, mean $\pm$ SD over seeds. The pre-specified "
        rf"time-to-recovery measure (return of the trailing {roll}-arrival coverage to within 0.01 "
        rf"of target) is not reported: the sampling SD of that coverage is about {noise_sd}, so every "
        rf"method crossed the band by chance. Coverage after the change at or above {valid_thr} is in bold. "
        r"Synthetic drift; not a finding about the hospital. "
        + PROXY + r".}",
        r"\label{tab:m2-rev-drift}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Drift & Method & Cov.\ before & Cov.\ after & Class-1 cov.\ after & Size before & Size after \\",
        r"\midrule",
    ]
    for i, (sc, slabel) in enumerate(E4_SCEN):
        for j, (m, mlabel, unit) in enumerate(E4_METH):
            f = {"model": "Hybrid GRU-D v2", "scenario": sc, "method": m}
            cells = [pm(ev, art, f, "cov_pre", 3), pm(ev, art, f, "cov_post", 3), pm(ev, art, f, "cov1_post", 3),
                     pm(ev, art, f, "size_pre", 3), pm(ev, art, f, "size_post", 3)]
            if ev.reduce_csv("mean", art, f, "cov_post", note="raw, bold rule only") >= 0.89:
                cells[1] = bf(cells[1])
            lines.append(f"{slabel.format(a=sh_a, b=sh_b) if j == 0 else ''} & {mlabel} ({unit}) & " + " & ".join(cells) + r" \\")
        if i < len(E4_SCEN) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    finish("rev_drift_stress.tex", lines, ev)


if __name__ == "__main__":
    table_parameter_summary()
    table_model_hyperparameters()
    table_headline_discrimination()
    table_deltas()
    table_conformal_summary()
    table_news2_sensitivity()
    table_mondrian_comparators()
    table_eci_vs_recal()
    table_horizon_ablation()
    table_g7_clinician_decision()
    table_label_lag()
    table_small_ablations()
    table_rev_adaptive_sets()
    table_rev_width_tests()
    table_rev_drift_stress()
