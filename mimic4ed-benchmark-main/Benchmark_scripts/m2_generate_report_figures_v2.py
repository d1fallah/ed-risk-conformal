"""
Generates the v2 (irregular-series, chronological-primary-split) figure
set, reading only already-saved result files -- no number is computed or
estimated here, matching this project's Golden Rule. Same conventions as
m2_generate_report_figures.py (the T=2 counterpart, unchanged and
reproducible from data/archive_pre_irregular/): matplotlib Agg backend,
pdf.fonttype=42, 300 DPI, saved as both .pdf and .png.

Five figures, per Phase 4 of the session's "Clearing the Runway to the
Manuscript" plan:
  fig1_headline_auroc_v2       -- all 6 v2 models, with 95% patient-cluster
                                   bootstrap CIs where available (falls
                                   back to seed-SD error bars and a printed
                                   warning if m2_v2_deltas.json does not
                                   yet exist -- never fabricates a CI)
  fig2_calibration_v2          -- reliability diagrams, all 6 models
  fig3_conformal_coverage_v2   -- MAPIE vs ECI vs Mondrian-LAC, marginal
                                   and class-1, on the chronological-
                                   PRIMARY split -- the module's central
                                   figure (Sec 24.3)
  fig4_deltas_forest_v2        -- the 5 v2 deltas with bootstrap CIs
                                   (requires data/m2_v2_deltas.json)
  fig5_news2_sensitivity       -- 6-arm NEWS2 sensitivity with bootstrap
                                   CIs (shared with the T=2 cohort; not
                                   v2-specific, generated here since no
                                   prior figure covers it)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "paper_assets", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS_V2 = ["Static-Only MLP v2", "mTAN v2", "Temporal-Only GRU-D v2",
             "Frozen-Temporal Head Control v2", "XGBoost-Matched v2", "Hybrid GRU-D v2"]
SHORT = {"Static-Only MLP v2": "Static-Only", "mTAN v2": "mTAN",
         "Temporal-Only GRU-D v2": "Temporal-Only", "Frozen-Temporal Head Control v2": "Frozen-Control",
         "XGBoost-Matched v2": "XGBoost-Matched", "Hybrid GRU-D v2": "Hybrid (primary)"}
COLORS = {"Static-Only MLP v2": "#55A868", "mTAN v2": "#937860",
          "Temporal-Only GRU-D v2": "#4C72B0", "Frozen-Temporal Head Control v2": "#8172B2",
          "XGBoost-Matched v2": "#DD8452", "Hybrid GRU-D v2": "#C44E52"}
GRUD_FAMILY_V2 = ["Temporal-Only GRU-D v2", "Frozen-Temporal Head Control v2", "Hybrid GRU-D v2"]


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


# ── Fig 1: Headline AUROC, all 6 v2 models, with real CIs where available ──
def fig_headline_auroc_v2():
    r = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_results.csv"))
    means = [r[r["model"] == m]["auroc"].mean() for m in MODELS_V2]
    stds = [r[r["model"] == m]["auroc"].std() for m in MODELS_V2]

    deltas_path = os.path.join(DATA_DIR, "m2_v2_deltas.json")
    has_ci = os.path.exists(deltas_path)
    if not has_ci:
        print("  WARNING: data/m2_v2_deltas.json not found -- fig1 uses seed-SD error bars only, "
              "not bootstrap CIs. Re-run after m2_v2_analyze.py completes for the CI version.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(MODELS_V2))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[m] for m in MODELS_V2],
                  edgecolor="black", linewidth=0.8, width=0.62)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.006, f"{m:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS_V2], rotation=15, ha="right")
    ax.set_ylabel("AUROC (mean $\\pm$ SD across 5 seeds)" + ("" if has_ci else "  [seed-SD only -- see console warning]"))
    ax.set_ylim(0.75, 0.97)
    ax.set_title("Primary cohort, chronological-primary split: 6-model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig1_headline_auroc_v2")


# ── Fig 2: Calibration (reliability diagrams), all 6 v2 models ─────────────
def fig_calibration_v2():
    preds = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_predictions.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODELS_V2):
        sub = preds[preds["model"] == model]
        # pool all 5 seeds for the reliability diagram (illustrative; per-seed ECE is in results csv)
        y, p = sub["y_true"].values, sub["y_prob"].values
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins[1:-1], right=True)
        conf, acc, n = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum() == 0:
                continue
            conf.append(p[m].mean()); acc.append(y[m].mean()); n.append(m.sum())
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.plot(conf, acc, "o-", color=COLORS[model], markersize=4)
        ax.set_title(SHORT[model], fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Mean predicted probability (bin)")
    fig.supylabel("Observed frequency (bin)")
    fig.suptitle("Calibration, primary cohort, all seeds pooled per model", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_calibration_v2")


# ── Fig 3: Conformal coverage -- MAPIE vs ECI vs Mondrian-LAC (the central figure) ──
def fig_conformal_coverage_v2():
    conf = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_conformal_summary.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(GRUD_FAMILY_V2))
    width = 0.25

    # Panel 1: marginal coverage
    mapie_m = [conf[conf.model == m]["mapie_coverage"].mean() for m in GRUD_FAMILY_V2]
    mapie_s = [conf[conf.model == m]["mapie_coverage"].std() for m in GRUD_FAMILY_V2]
    eci_m = [conf[conf.model == m]["eci_coverage"].mean() for m in GRUD_FAMILY_V2]
    eci_s = [conf[conf.model == m]["eci_coverage"].std() for m in GRUD_FAMILY_V2]
    ax1.bar(x - width/2, mapie_m, width, yerr=mapie_s, capsize=3, label="MAPIE", color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax1.bar(x + width/2, eci_m, width, yerr=eci_s, capsize=3, label="ECI", color="#C44E52", edgecolor="black", linewidth=0.6)
    ax1.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% target")
    ax1.set_xticks(x); ax1.set_xticklabels([SHORT[m] for m in GRUD_FAMILY_V2], rotation=15, ha="right")
    ax1.set_ylabel("Marginal coverage"); ax1.set_ylim(0.82, 0.94)
    ax1.set_title("Marginal coverage, chronological test period")
    ax1.legend(fontsize=8); ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: class-1 coverage, all three constructions
    mapie_c1 = [conf[conf.model == m]["mapie_coverage_class1"].mean() for m in GRUD_FAMILY_V2]
    eci_c1 = [conf[conf.model == m]["eci_coverage_class1"].mean() for m in GRUD_FAMILY_V2]
    # FIX (found during Phase 0 provenance-sidecar work, 2026-09-22): this used to read
    # data/m2_mondrian_conformal.csv, filtered on model names WITHOUT the "v2" suffix
    # ("Hybrid GRU-D") against GRUD_FAMILY_V2 names WITH it ("Hybrid GRU-D v2") -- a mismatch that
    # matched 0 rows and silently produced NaN bars (warnings.filterwarnings("ignore") above
    # suppressed pandas' "Mean of empty slice" warning), so this panel's Mondrian-LAC bars were
    # blank on every previously-saved copy of this "central figure". m2_v2_conformal_summary.csv
    # already carries the correct, current mondrian_lac_coverage_class1 column -- the same column
    # src/generate_latex_tables.py:table_conformal_summary() (T3) already sources and
    # src/verify_paper_assets.py already independently verifies -- so this now reads from there
    # instead of the separate, differently-named, and staler m2_mondrian_conformal.csv.
    lac_c1 = [conf[conf.model == m]["mondrian_lac_coverage_class1"].mean() for m in GRUD_FAMILY_V2]
    w2 = 0.22
    ax2.bar(x - w2, mapie_c1, w2, label="MAPIE (marginal)", color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax2.bar(x, eci_c1, w2, label="ECI (marginal)", color="#C44E52", edgecolor="black", linewidth=0.6)
    ax2.bar(x + w2, lac_c1, w2, label="Mondrian-LAC (label-cond.)", color="#55A868", edgecolor="black", linewidth=0.6)
    ax2.axhline(0.90, color="gray", linestyle="--", linewidth=1)
    ax2.set_xticks(x); ax2.set_xticklabels([SHORT[m] for m in GRUD_FAMILY_V2], rotation=15, ha="right")
    ax2.set_ylabel("Class-1 (deteriorating-patient) coverage"); ax2.set_ylim(0, 1.0)
    ax2.set_title("Coverage within the deteriorating class (class 1)")
    ax2.legend(fontsize=8); ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "fig3_conformal_coverage_v2")


# ── Fig 4: The 5 v2 deltas, forest plot (requires m2_v2_deltas.json) ───────
def fig_deltas_forest_v2():
    path = os.path.join(DATA_DIR, "m2_v2_deltas.json")
    if not os.path.exists(path):
        print("  SKIPPED fig4_deltas_forest_v2: data/m2_v2_deltas.json does not exist yet "
              "(run m2_v2_analyze.py first). Not fabricating this figure from partial data.")
        return
    with open(path) as f:
        deltas = json.load(f)

    names = list(deltas.keys())
    means = [deltas[n]["mean"] for n in names]
    ci_los = [np.mean([r["ci_lo"] for r in deltas[n]["per_seed"]]) for n in names]
    ci_his = [np.mean([r["ci_hi"] for r in deltas[n]["per_seed"]]) for n in names]
    sig = [deltas[n].get("holm_significant_at_0.05", False) for n in names]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.5))
    y = np.arange(len(names))
    for yi, m, lo, hi, s in zip(y, means, ci_los, ci_his, sig):
        color = "#C44E52" if s else "#4C72B0"
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2)
        ax.plot(m, yi, "o", color=color, markersize=7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("AUROC delta (bootstrap mean of per-seed 95% CIs)")
    ax.set_title("Discrimination deltas: patient-cluster bootstrap CIs, Holm-corrected\n(red = significant at $\\alpha$=0.05)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig4_deltas_forest_v2")


# ── Fig 5: NEWS2 six-arm sensitivity ────────────────────────────────────────
def fig_news2_sensitivity():
    # Chronological test period (test_v2.csv), the population every other reported result uses.
    # The earlier random-split run (m2_news2_sensitivity.csv) is kept on disk but not plotted.
    path = os.path.join(DATA_DIR, "m2_news2_sensitivity_v2.csv")
    if not os.path.exists(path):
        print("  SKIPPED fig5_news2_sensitivity: data/m2_news2_sensitivity_v2.csv not found.")
        return
    df = pd.read_csv(path)
    arm_labels = {
        "A_composite_critical": "A: triage, composite outcome",
        "B_mortality_alone": "B: triage, in-hospital death",
        "C_icu_transfer_alone": "C: triage, ICU transfer",
        "D_mortality_24h_from_edout": "D: triage, death <=24 h after ED",
        "E_complete_case_composite": "E: triage, complete cases",
        "F_last_ed_vitals_composite": "F: last ED vitals, composite",
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(df))
    ax.errorbar(df["auroc"], y, xerr=[df["auroc"] - df["auroc_ci_lo"], df["auroc_ci_hi"] - df["auroc"]],
                fmt="o", color="#4C72B0", capsize=4)
    ax.axvspan(0.71, 0.86, color="gray", alpha=0.15, label="Published ED-validation range (0.71-0.86)")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([arm_labels.get(a, a) for a in df["arm"]])
    ax.set_xlabel("AUROC (95% patient-cluster bootstrap CI)")
    ax.set_title("NEWS2 (5 of 7 parameters) on the chronological test period")
    # Upper right (top rows, inside the shaded band) is the only empty corner: lower right covers
    # Arm F's point and upper left covers Arm A's.
    ax.legend(fontsize=8, loc="upper right")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig5_news2_sensitivity")


# ── Fig 6: ECI vs. periodic recalibration -- coverage/width efficiency frontier ────────────
# Added 2026-09-22 (Phase 2 of the manuscript plan): backbone item 4 (is the adaptive conformal
# machinery actually needed?) previously had a table (T6, m2_v2_eci_vs_recalibration.csv) but no
# paired figure -- every other backbone item does. Visualizes exactly what T6 tabulates: among the
# methods that reach the coverage target, which has the narrowest (most efficient) interval.
ER_METHOD_LABEL = {"M1": "static split-conformal", "M2": "ECI",
                    "M3": "recal. ($W$=10k)", "M4": "recal. ($W$=40k)"}
ER_METHOD_COLOR = {"M1": "#8172B2", "M2": "#C44E52", "M3": "#4C72B0", "M4": "#55A868"}
ER_METHOD_ORDER = ["M1", "M2", "M3", "M4"]
ER_MODEL_MARKER = {"Hybrid GRU-D v2": "o", "Temporal-Only GRU-D v2": "^", "Frozen-Temporal Head Control v2": "s"}
ER_MODEL_SHORT = {"Hybrid GRU-D v2": "Hybrid (primary)", "Temporal-Only GRU-D v2": "Temporal-Only",
                   "Frozen-Temporal Head Control v2": "Frozen-Control"}


def fig_eci_vs_recal_v2():
    path = os.path.join(DATA_DIR, "m2_v2_eci_vs_recalibration.csv")
    if not os.path.exists(path):
        print("  SKIPPED fig6_eci_vs_recalibration_v2: data/m2_v2_eci_vs_recalibration.csv not found.")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in ER_METHOD_ORDER:
        for model in ER_MODEL_MARKER:
            sub = df[(df.method == method) & (df.model == model)]
            if sub.empty:
                continue
            w_mean, w_std = sub["width"].mean(), sub["width"].std()
            c_mean, c_std = sub["coverage"].mean(), sub["coverage"].std()
            ax.errorbar(w_mean, c_mean, xerr=w_std, yerr=c_std, fmt=ER_MODEL_MARKER[model],
                        color=ER_METHOD_COLOR[method], markersize=8, capsize=3,
                        markeredgecolor="black", markeredgewidth=0.6)
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% target")
    method_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ER_METHOD_COLOR[m],
                                 markeredgecolor="black", markersize=8, label=ER_METHOD_LABEL[m])
                       for m in ER_METHOD_ORDER]
    model_handles = [plt.Line2D([0], [0], marker=ER_MODEL_MARKER[mo], color="gray", linestyle="",
                                markersize=8, label=ER_MODEL_SHORT[mo]) for mo in ER_MODEL_MARKER]
    leg1 = ax.legend(handles=method_handles, title="Method (color)", fontsize=8, loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model (marker)", fontsize=8, loc="upper right")
    ax.set_xlabel("Mean clipped interval width (narrower = more efficient)")
    ax.set_ylabel("Marginal coverage")
    ax.axhline(0.89, color="lightgray", linestyle=":", linewidth=1)  # validity threshold
    ax.set_ylim(0.889, 0.909)  # fixed window: auto-scaling zoomed into a 0.0035 range
    ax.set_title("Static split-conformal, ECI and periodic recalibration:\ncoverage and interval width")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig6_eci_vs_recalibration_v2")


def main():
    print("Generating v2 report figures...")
    fig_headline_auroc_v2()
    fig_calibration_v2()
    fig_conformal_coverage_v2()
    fig_deltas_forest_v2()
    fig_news2_sensitivity()
    fig_eci_vs_recal_v2()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
