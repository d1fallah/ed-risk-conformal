"""
Generates every new figure for the M2 supervisor report (LaTeX), reading
only already-saved result files -- no number is computed or estimated
here, matching this project's Golden Rule. Follows the established M2/
CLAUDE.md figure conventions: matplotlib Agg backend, pdf.fonttype=42
(embeds fonts as real text, not curves, in the PDF -- required for
publication-quality vector output), 300 DPI, saved as both .pdf (for
LaTeX \\includegraphics) and .png (for quick preview).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "paper_assets", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = ["Temporal-Only GRU-D", "Frozen-Temporal Head Control", "Static-Only MLP",
          "Hybrid GRU-D", "XGBoost-Matched"]
SHORT = {"Temporal-Only GRU-D": "Temporal-Only", "Frozen-Temporal Head Control": "Frozen-Control",
         "Static-Only MLP": "Static-Only", "Hybrid GRU-D": "Hybrid", "XGBoost-Matched": "XGBoost-Matched"}
COLORS = {"Temporal-Only GRU-D": "#4C72B0", "Frozen-Temporal Head Control": "#8172B2",
          "Static-Only MLP": "#55A868", "Hybrid GRU-D": "#C44E52", "XGBoost-Matched": "#DD8452"}


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


# ── Fig 1: Headline AUROC comparison, all 5 models ──────────────────────────
def fig_headline_auroc():
    r = pd.read_csv(os.path.join(DATA_DIR, "results_hybrid.csv"))
    means = [r[r["model"] == m]["auroc"].mean() for m in MODELS]
    stds = [r[r["model"] == m]["auroc"].std() for m in MODELS]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(MODELS))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[m] for m in MODELS],
                  edgecolor="black", linewidth=0.8, width=0.62)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.008, f"{m:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(0.9184, color="gray", linestyle="--", linewidth=1, label="XGBoost-Rich (historical ceiling, 0.9184)")
    ax.axhline(0.9009, color="gray", linestyle=":", linewidth=1, label="Legacy GRU-D Tuned (historical, confounded, 0.9009)")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("AUROC (mean $\\pm$ SD across 5 seeds)")
    ax.set_ylim(0.70, 0.95)
    ax.set_title("Core five-model comparison, real-missingness data (post-fix)")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig1_headline_auroc")


# ── Fig 2: The missingness bug, illustrated -- training curves pre/post ────
def fig_missingness_bug():
    with open(os.path.join(DATA_DIR, "m2_temporal_only_grud_convergence_diagnostic_fullbatch_seed42.json")) as f:
        pre = json.load(f)
    with open(os.path.join(DATA_DIR, "m2_temporal_only_grud_convergence_diagnostic_realmissing_seed42.json")) as f:
        post = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ylabel, ylim in [(axes[0], "val_loss", "Validation loss", (0.14, 0.35)),
                                   (axes[1], "val_auroc", "Validation AUROC", (0.45, 0.90))]:
        pre_x = [h["epoch"] for h in pre]
        pre_y = [h[key] for h in pre]
        post_x = [h["epoch"] for h in post if h["epoch"] <= 2000]
        post_y = [h[key] for h in post if h["epoch"] <= 2000]
        ax.plot(pre_x, pre_y, color="#C44E52", linewidth=1.1, label="Before fix (imputed vitals, $M$=all-ones)")
        ax.plot(post_x, post_y, color="#4C72B0", linewidth=1.1, label="After fix (real missingness recovered)")
        ax.set_xlabel("Epoch (full-batch)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].axvline(268, color="#C44E52", linestyle=":", linewidth=0.8)
    axes[0].axvline(1611, color="#4C72B0", linestyle=":", linewidth=0.8)
    axes[1].axvline(268, color="#C44E52", linestyle=":", linewidth=0.8)
    axes[1].axvline(1611, color="#4C72B0", linestyle=":", linewidth=0.8)
    fig.suptitle("Temporal-Only GRU-D training dynamics, before vs. after the missingness fix (seed 42)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig2_missingness_bug_training_curves")


# ── Fig 3: Calibration (ECE) across all 5 models ────────────────────────────
def fig_calibration():
    r = pd.read_csv(os.path.join(DATA_DIR, "results_hybrid.csv"))
    means = [r[r["model"] == m]["ece"].mean() for m in MODELS]
    stds = [r[r["model"] == m]["ece"].std() for m in MODELS]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(MODELS))
    ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_title("Probabilistic calibration, all 5 models")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_calibration_ece")


# ── Fig 4: Conformal coverage/width, 3 GRU-D-family models ──────────────────
def fig_conformal():
    c = pd.read_csv(os.path.join(DATA_DIR, "m2_hybrid_conformal_summary.csv"))
    c = c.set_index("model")
    gru_models = ["Temporal-Only GRU-D", "Frozen-Temporal Head Control", "Hybrid GRU-D"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(gru_models))
    w = 0.35
    axes[0].bar(x - w / 2, [c.loc[m, "coverage_mapie_mean"] * 100 for m in gru_models], w,
                yerr=[c.loc[m, "coverage_mapie_sd"] * 100 for m in gru_models], capsize=3,
                label="MAPIE", color="#4C72B0", edgecolor="black", linewidth=0.7)
    axes[0].bar(x + w / 2, [c.loc[m, "coverage_eci_mean"] * 100 for m in gru_models], w,
                yerr=[c.loc[m, "coverage_eci_sd"] * 100 for m in gru_models], capsize=3,
                label="ECI", color="#DD8452", edgecolor="black", linewidth=0.7)
    axes[0].axhline(90, color="red", linestyle="--", linewidth=1, label="Target 90%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SHORT[m] for m in gru_models], rotation=15, ha="right")
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_ylim(85, 95)
    axes[0].set_title("A) Conformal coverage")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - w / 2, [c.loc[m, "set_size_mapie_mean"] for m in gru_models], w,
                yerr=[c.loc[m, "set_size_mapie_sd"] for m in gru_models], capsize=3,
                label="MAPIE set size", color="#4C72B0", edgecolor="black", linewidth=0.7)
    axes[1].bar(x + w / 2, [c.loc[m, "width_eci_mean"] for m in gru_models], w,
                yerr=[c.loc[m, "width_eci_sd"] for m in gru_models], capsize=3,
                label="ECI width", color="#DD8452", edgecolor="black", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SHORT[m] for m in gru_models], rotation=15, ha="right")
    axes[1].set_ylabel("Mean set size / interval width")
    axes[1].set_title("B) Conformal set size / width")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Conformal prediction, 3 GRU-D-family models (post-fix)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_conformal")


# ── Fig 5: The 5 deltas, forest plot, pre-fix vs post-fix ───────────────────
def fig_deltas_forest():
    with open(os.path.join(DATA_DIR, "m2_hybrid_deltas.json")) as f:
        post = json.load(f)
    # Pre-fix values transcribed from the actual pre-fix m2_hybrid_analyze.py
    # run's console output (recorded verbatim in this session's transcript
    # and in M2_Findings_Dossier.md Sec 18's pre-fix column) -- the raw
    # data/m2_hybrid_deltas.json was overwritten by the post-fix rebuild, so
    # this is the only remaining record. (mean, min-of-5-seeds'-CI-los,
    # max-of-5-seeds'-CI-his), matching the same min/max-across-seeds
    # convention used for the post-fix side below.
    pre_fix = {
        "Delta_representation": (-0.0222, -0.0259, -0.0179),
        "Delta_static_contribution": (0.1249, 0.1025, 0.1471),
        "Delta_temporal_alone": (-0.1481, -0.1706, -0.1234),
        "Delta_static_only": (0.0913, 0.0653, 0.1144),
        "Delta_rich": (0.0350, 0.0304, 0.0391),
    }
    order = ["Delta_representation", "Delta_static_contribution", "Delta_temporal_alone",
             "Delta_static_only", "Delta_rich"]
    labels = [r"$\Delta_{\mathrm{representation}}$ (primary)", r"$\Delta_{\mathrm{static\ contribution}}$",
              r"$\Delta_{\mathrm{temporal\ alone}}$", r"$\Delta_{\mathrm{static\ only}}$", r"$\Delta_{\mathrm{rich}}$"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(order))
    for i, key in enumerate(order):
        pm, plo, phi = pre_fix[key]
        ax.errorbar(pm, y[i] + 0.15, xerr=[[pm - plo], [phi - pm]], fmt="s", color="#C44E52",
                    capsize=4, markersize=6, label="Pre-fix (mean across seeds' CIs)" if i == 0 else None)

        seeds = post[key]["per_seed"]
        pts = np.array([s["point_estimate"] for s in seeds])
        los = np.array([s["ci_lo"] for s in seeds])
        his = np.array([s["ci_hi"] for s in seeds])
        m, lo, hi = pts.mean(), los.min(), his.max()
        ax.errorbar(m, y[i] - 0.15, xerr=[[m - lo], [hi - m]], fmt="o", color="#4C72B0",
                    capsize=4, markersize=6, label="Post-fix (range across seeds' CIs)" if i == 0 else None)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$AUROC (patient-cluster bootstrap 95% CI)")
    ax.set_title("The five primary comparisons: pre-fix vs. post-fix")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig5_deltas_forest")


# ── Fig 6: Subgroup interaction test, per seed, pre-fix vs post-fix ─────────
def fig_subgroup_interaction():
    with open(os.path.join(DATA_DIR, "m2_hybrid_subgroup_results.json")) as f:
        post = json.load(f)
    pre_fix_did = [
        {"seed": 42, "did": -0.0001, "ci_lo": -0.0069, "ci_hi": 0.0065},
        {"seed": 123, "did": 0.0004, "ci_lo": -0.0067, "ci_hi": 0.0066},
        {"seed": 7, "did": 0.0009, "ci_lo": -0.0055, "ci_hi": 0.0076},
        {"seed": 456, "did": -0.0035, "ci_lo": -0.0096, "ci_hi": 0.0030},
        {"seed": 789, "did": 0.0023, "ci_lo": -0.0041, "ci_hi": 0.0082},
    ]
    post_did = post["primary_interaction_test"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    seeds = [d["seed"] for d in pre_fix_did]
    y = np.arange(len(seeds))
    for i, (pre, pos) in enumerate(zip(pre_fix_did, post_did)):
        ax.errorbar(pre["did"], y[i] + 0.15, xerr=[[pre["did"] - pre["ci_lo"]], [pre["ci_hi"] - pre["did"]]],
                    fmt="s", color="#C44E52", capsize=4, markersize=6,
                    label="Pre-fix" if i == 0 else None)
        ax.errorbar(pos["did"], y[i] - 0.15, xerr=[[pos["did"] - pos["ci_lo"]], [pos["ci_hi"] - pos["did"]]],
                    fmt="o", color="#4C72B0", capsize=4, markersize=6,
                    label="Post-fix" if i == 0 else None)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.invert_yaxis()
    ax.set_xlabel("DiD = (AUROC$_{Hybrid}$$-$AUROC$_{XGBoost}$)$_{Sparse}$ $-$ (same)$_{Dense}$")
    ax.set_title("Primary subgroup hypothesis: interaction test, pre-fix vs. post-fix")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig6_subgroup_interaction")


if __name__ == "__main__":
    print("Generating M2 report figures...")
    fig_headline_auroc()
    fig_missingness_bug()
    fig_calibration()
    fig_conformal()
    fig_deltas_forest()
    fig_subgroup_interaction()
    print(f"\nAll figures saved to {FIG_DIR}")
