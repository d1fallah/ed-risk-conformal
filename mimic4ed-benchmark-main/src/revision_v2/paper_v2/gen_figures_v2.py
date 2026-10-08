"""Figures of the revision manuscript, drawn from the frozen evidence through the provenance accessor (every plotted value is a recorded reduction)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data_v2 as PD  # noqa: E402

OUT = os.path.join(PD.ASSET_DIR, "figures")
os.makedirs(OUT, exist_ok=True)
CF, K = "ph2_conformal.json", "ph2_contrasts.json"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.fontsize": 7.5})
# Okabe-Ito colour-blind-safe palette; every series also has its own marker and, where lines overlap, its own line style.
BLACK, BLUE, VERM, GRAY = "#000000", "#0072B2", "#D55E00", "#999999"


def fig_alpha():
    ev = PD.EV("fig_alpha")
    alphas = ["0.2", "0.1", "0.05", "0.01"]
    cov = [1 - float(a) for a in alphas]
    meths = [("static_lac", "Static LAC", BLACK, "o", "-"), ("static_mondrian_lac", "Static Mondrian-LAC", BLUE, "s", "--"), ("mondrian_eci", "Mondrian-ECI", VERM, "^", "-.")]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for k, label, col, mk, ls in meths:
        c1 = [float(ev.reduce("mean", CF, f"/runs/*/sweep/{a}/{k}/metrics/coverage_class1")) for a in alphas]
        ppv = [float(ev.reduce("mean", CF, f"/runs/*/sweep/{a}/{k}/operating/ppv_only1")) for a in alphas]
        fa = [float(ev.reduce("mean", CF, f"/runs/*/sweep/{a}/{k}/operating/false_alarms_per_100_patients")) for a in alphas]
        ax[0].plot(cov, c1, marker=mk, ls=ls, color=col, label=label)
        ax[1].plot(cov, ppv, marker=mk, ls=ls, color=col)
        ax[2].plot(cov, fa, marker=mk, ls=ls, color=col)
    ax[0].plot([0.78, 1], [0.78, 1], "k:", lw=0.8)
    ax[0].set_ylabel("class-1 coverage"); ax[1].set_ylabel("PPV of $\\{1\\}$"); ax[2].set_ylabel("false alarms per 100 patients")
    for a_ in ax:
        a_.set_xlabel("nominal coverage $1-\\alpha$")
    ax[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.45))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_alpha.pdf")); fig.savefig(os.path.join(OUT, "fig_alpha.png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


def fig_alpha_col(panels=("class1", "fa"), name="fig_alpha_col", size=(3.45, 2.0)):
    """One-column version of the coverage-level figure for the submitted (short) manuscript; same evidence as fig_alpha, panels chosen by `panels`."""
    ev = PD.EV(name)
    alphas = ["0.2", "0.1", "0.05", "0.01"]
    cov = [1 - float(a) for a in alphas]
    meths = [("static_lac", "Static LAC", BLACK, "o", "-"), ("static_mondrian_lac", "Static Mondrian-LAC", BLUE, "s", "--"), ("mondrian_eci", "Mondrian-ECI", VERM, "^", "-.")]
    key = {"class1": ("coverage_class1", "metrics", "class-1 coverage"), "ppv": ("ppv_only1", "operating", "PPV of $\\{1\\}$"), "fa": ("false_alarms_per_100_patients", "operating", "false alarms per 100 patients")}
    fig, ax = plt.subplots(1, len(panels), figsize=size)
    ax = [ax] if len(panels) == 1 else list(ax)
    for k, label, col, mk, ls in meths:
        for a_, p in zip(ax, panels):
            m, grp, _ = key[p]
            y = [float(ev.reduce("mean", CF, f"/runs/*/sweep/{a}/{k}/{grp}/{m}")) for a in alphas]
            a_.plot(cov, y, marker=mk, ms=3, lw=1.0, ls=ls, color=col, label=label if p == panels[0] else None)
    for a_, p in zip(ax, panels):
        a_.set_ylabel(key[p][2], fontsize=7.5); a_.set_xlabel("nominal coverage $1-\\alpha$", fontsize=7.5); a_.tick_params(labelsize=7)
        if p == "class1":
            a_.plot([0.78, 1], [0.78, 1], "k:", lw=0.8)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=6.3, handlelength=1.8, columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


def fig_discrimination():
    ev = PD.EV("fig_discrimination")
    names = [("logreg", "Logistic regression"), ("xgboost_timing", "Timing-aware trees"), ("temporal_grud", "Temporal-Only GRU-D"), ("hybrid_frozen", "Hybrid, frozen encoder"),
             ("hybrid_e2e", "Hybrid, end to end")]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for i, (k, label) in enumerate(names):
        b = f"/landmarks/h2/models/{k}"
        m = float(ev.get(K, b + "/auroc_seed_mean")); lo = float(ev.get(K, b + "/auroc_bootstrap_ci95_of_seed_mean/0")); hi = float(ev.get(K, b + "/auroc_bootstrap_ci95_of_seed_mean/1"))
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color="tab:blue", capsize=2)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([n[1] for n in names]); ax.invert_yaxis()
    ax.set_xlabel("AUROC, primary outcome (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_discrimination.pdf")); fig.savefig(os.path.join(OUT, "fig_discrimination.png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


def fig_hist():
    ev = PD.EV("fig_hist")
    H = "ph2_hist.json"
    import numpy as np
    edges = np.array([float(x) for x in ev.series(H, "/bin_edges/*")])
    ce = np.array([float(x) for x in ev.series(H, "/counts_events/*")])
    cn = np.array([float(x) for x in ev.series(H, "/counts_non_events/*")])
    t_m = float(ev.get(H, "/marginal_threshold_risk_for_label_1"))
    t_1 = float(ev.get(H, "/mondrian_class1_threshold_risk_for_label_1"))
    t_0 = float(ev.get(H, "/mondrian_class0_max_risk_for_label_0"))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = edges[1] - edges[0]
    # no outcome as filled bars, outcome as an outline, so neither hides the other where they overlap
    ax.bar(edges[:-1], cn, width=w, align="edge", color=GRAY, alpha=0.6, label="no outcome")
    ax.stairs(ce, edges, color=VERM, lw=1.4, label="outcome")
    ax.set_yscale("log")
    ax.axvline(t_m, color="k", ls="--", lw=1)
    ax.axvline(t_1, color=BLUE, ls=":", lw=1.2)
    ax.axvline(t_0, color=BLUE, ls=":", lw=1.2)
    ax.text(t_m, ax.get_ylim()[1], " marginal\n threshold", va="top", ha="left", fontsize=7)
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("test stays (log scale)")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_hist.pdf")); fig.savefig(os.path.join(OUT, "fig_hist.png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


def fig_sweep():
    ev = PD.EV("fig_sweep")
    SW = "ph2_sweep.json"
    hs = ["0", "1", "2", "3", "4", "6"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for k, label, col, off in (("logreg", "Logistic regression", BLACK, -0.06), ("trees", "Timing-aware trees", BLUE, 0.06)):
        m = [float(ev.get(SW, f"/landmarks/{h}/{k}")) for h in hs]
        lo = [float(ev.get(SW, f"/landmarks/{h}/{k}_ci95/0")) for h in hs]
        hi = [float(ev.get(SW, f"/landmarks/{h}/{k}_ci95/1")) for h in hs]
        x = [float(h) + off for h in hs]
        ax.errorbar(x, m, yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]], fmt="o-", ms=3, lw=1, capsize=2, color=col, label=label)
    FX = "ph2_fixed_cohort.json"
    fh = ["1", "2", "3", "4", "6"]
    for k, col in (("logreg", BLACK), ("trees", BLUE)):
        ax.plot([float(h) for h in fh], [float(ev.get(FX, f"/landmarks/{h}/fixed_set/{k}")) for h in fh], ls="--", marker="o", ms=2.5, lw=1, color=col, mfc="white")
    ax.plot([], [], ls="--", color="k", lw=1, label="fixed set of stays")
    ax.set_xlabel("landmark (hours after arrival)")
    ax.set_ylabel("AUROC (95% bootstrap CI)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sweep.pdf")); fig.savefig(os.path.join(OUT, "fig_sweep.png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


def fig_forest():
    ev = PD.EV("fig_forest")
    rows = [("C1", "h2", "C1 Hybrid $-$ trees"), ("C2", "h2", "C2 Hybrid $-$ logistic reg."), ("C3", "h2", "C3 Hybrid $-$ Temporal-Only"),
            ("C4", "h2", "C4 Hybrid $-$ frozen enc."), ("C5", "triage", "C5 Trees $-$ log. reg. (arrival)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for i, (cid, lm, label) in enumerate(rows):
        b = f"/landmarks/{lm}/contrasts/{cid}"
        d = float(ev.get(K, b + "/delta_seed_mean")); lo = float(ev.get(K, b + "/delta_bootstrap_ci95_of_seed_mean/0")); hi = float(ev.get(K, b + "/delta_bootstrap_ci95_of_seed_mean/1"))
        sig = bool(ev.get(K, b + "/significant"))
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o" if sig else "s", color=BLUE if sig else BLACK, mfc=BLUE if sig else "white", capsize=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("AUROC difference (95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_forest.pdf")); fig.savefig(os.path.join(OUT, "fig_forest.png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


def fig_reliability():
    ev = PD.EV("fig_reliability")
    R, YD = "ph2_reliability.json", "ph2_years_decision.json"
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    sty = (("logreg", "Logistic regression", BLACK, "o", "-"), ("xgboost_timing", "Timing-aware trees", BLUE, "s", "--"), ("hybrid_e2e", "Hybrid GRU-D (end to end)", VERM, "^", "-."))
    for j, (k, label, col, mk, ls) in enumerate(sty):
        x = [float(v) for v in ev.series(R, f"/models/{k}/mean_pred/*")]
        y = [float(v) for v in ev.series(R, f"/models/{k}/observed/*")]
        lo = [float(v) for v in ev.series(R, f"/models/{k}/observed_ci95/*/0")]
        hi = [float(v) for v in ev.series(R, f"/models/{k}/observed_ci95/*/1")]
        # patient-cluster bootstrap 95% interval of the observed rate; the three models are offset by 7% on the log axis so the intervals stay apart.
        # A lower bound of 0 cannot be drawn on a log axis and is drawn at 1e-4 (bottom of the axis).
        xs = [v * (1.07 ** (j - 1)) for v in x]
        yerr = [[a - max(b, 1e-4) for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]]
        ax[0].errorbar(xs, y, yerr=yerr, marker=mk, ms=3, lw=1, ls=ls, capsize=1.5, elinewidth=0.8, color=col, label=label)
    ax[0].plot([0.0005, 0.6], [0.0005, 0.6], "k:", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("mean predicted risk (decile of predicted risk)"); ax[0].set_ylabel("observed event rate (95% CI)")
    ax[0].legend(frameon=False, loc="lower right")
    ax[0].set_title("(a) reliability", fontsize=8.5)
    grid = [float(v) for v in ev.series(YD, "/decision/grid/*")]
    for k, label, col, mk, ls in sty:
        ax[1].plot(grid, [float(v) for v in ev.series(YD, f"/decision/models/{k}/net_benefit/*")], color=col, ls=ls, lw=1.2)
    ax[1].plot(grid, [float(v) for v in ev.series(YD, "/decision/treat_all/*")], color=GRAY, ls=":", lw=1.4, label="treat all")
    ax[1].axhline(0, color="k", lw=0.8, label="treat none")
    ax[1].set_ylim(-0.05, 0.09)
    ax[1].set_xlabel("threshold probability"); ax[1].set_ylabel("net benefit")
    ax[1].legend(frameon=False, loc="upper right")
    ax[1].set_title("(b) decision curve", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_reliability.pdf")); fig.savefig(os.path.join(OUT, "fig_reliability.png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


def fig_pr():
    ev = PD.EV("fig_pr")
    PRJ = "ph2_pr_points.json"
    rec = [float(v) for v in ev.series(PRJ, "/curve/recall/*")]
    prec = [float(v) for v in ev.series(PRJ, "/curve/precision/*")]
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    ax.plot(rec, prec, color=BLACK, lw=1.2, label="precision-recall curve of the score")
    pts = (("static_lac", "Static LAC", BLACK, "o"), ("static_mondrian_lac", "Static Mondrian-LAC", BLUE, "s"),
           ("mondrian_eci", "Mondrian-ECI", VERM, "^"), ("mondrian_recal_w10k", "Mondrian recalibration", GRAY, "D"))
    for k, label, col, mk in pts:
        b = f"/operating_points/{k}"
        ax.plot([float(ev.get(PRJ, b + "/sensitivity"))], [float(ev.get(PRJ, b + "/ppv"))], marker=mk, ls="none", color=col, mfc="white" if k == "mondrian_recal_w10k" else col, ms={"static_lac": 6, "static_mondrian_lac": 8, "mondrian_eci": 4.5, "mondrian_recal_w10k": 10}[k], mew=1.2, label=label + r" $\{1\}$")
    ax.axhline(float(ev.get(PRJ, "/prevalence")), color=GRAY, ls=":", lw=1)
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision (PPV)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5, markerscale=0.6, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pr.pdf")); fig.savefig(os.path.join(OUT, "fig_pr.png"), dpi=300)
    plt.close(fig)
    ev.write_provenance()


if __name__ == "__main__":
    fig_discrimination()
    fig_pr()
    fig_alpha()
    fig_alpha_col(("class1", "fa"), "fig_alpha_col", (3.45, 2.0))
    fig_hist()
    fig_sweep()
    fig_forest()
    fig_reliability()
    print("figures written")
