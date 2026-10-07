"""
Phase 1.4 -- duration-stratified discrimination (all six stored models).
  ED-duration quartile cut points are computed on the TRAINING period (all 288,208 training-period stays), then applied to the test
  period. Per model and quartile: n, prevalence, within-quartile AUROC (seed mean +- SD over 5 seeds), and a patient-cluster
  bootstrap 95% CI (1,000 resamples) of the seed-averaged within-quartile AUROC; overall AUROC for comparison.
Phase 1.5 -- score distributions of the three GRU-D-family models, final calibration versus test: quantiles (0.5, 0.9, 0.95) of
  p_hat | y = 0, p_hat | y = 1 and the LAC score 1 - p_hat_y, ECDFs on a 101-point grid, and the static q_hat against the empirical
  0.9 quantile of the test LAC score (marginal and per class). Descriptive; no p-value.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import m2_label_lag as ll  # noqa: E402
from m2_mondrian_eci_comparators import _class_threshold  # noqa: E402

QS = [0.5, 0.9, 0.95]


def dur_quartiles():
    fr = C.load_frames()
    tr = fr["train_period"].copy()
    d = (pd.to_datetime(tr["outtime"]) - pd.to_datetime(tr["intime"])).dt.total_seconds() / 3600.0
    cuts = np.quantile(d, [0.25, 0.5, 0.75])
    return fr, cuts


def part_1_4():
    fr, cuts = dur_quartiles()
    te = fr["test"]
    q_id = np.searchsorted(cuts, te["duration_h"].values, side="right")        # 0..3
    y = te["outcome_critical"].astype(int).values
    boot = C.ClusterBoot(te["subject_id"].values, B=1000)
    preds = pd.read_csv(os.path.join(C.DATA, "m2_v2_predictions.csv"))
    pos = pd.Series(np.arange(len(te)), index=te["stay_id"].values)
    res = {"cut_points_hours_training_period": cuts.tolist(), "n_training_period": int(len(fr["train_period"])),
           "test_quartile_n": [int((q_id == k).sum()) for k in range(4)],
           "test_quartile_prevalence": [float(y[q_id == k].mean()) for k in range(4)],
           "test_quartile_positives": [int(y[q_id == k].sum()) for k in range(4)],
           "bootstrap": "patient-cluster, 1,000 resamples, 95% percentile CI of the seed-averaged AUROC", "models": {}}
    for model, g in preds.groupby("model"):
        scorers = {}
        for seed, gs in g.groupby("seed"):
            sc = np.empty(len(te)); sc[pos.loc[gs["stay_id"].values].values] = gs["y_prob"].values
            scorers[int(seed)] = sc
        seeds = sorted(scorers)
        sub = {k: [C.Scorer(y[q_id == k], scorers[s][q_id == k]) for s in seeds] for k in range(4)}
        allsc = [C.Scorer(y, scorers[s]) for s in seeds]
        per_q = {k: [sc.auroc() for sc in sub[k]] for k in range(4)}
        overall = [sc.auroc() for sc in allsc]
        draws_q = {k: [] for k in range(4)}; draws_all = []
        for w in boot.weights():
            for k in range(4):
                wk = w[q_id == k]
                draws_q[k].append(np.mean([sc.auroc(wk) for sc in sub[k]]))
            draws_all.append(np.mean([sc.auroc(w) for sc in allsc]))
        res["models"][model] = {
            "overall_auroc": {**C.seed_summary(overall), "bootstrap_ci95_of_seed_mean": C.pct_ci(draws_all)},
            "quartiles": {str(k + 1): {**C.seed_summary(per_q[k]), "bootstrap_ci95_of_seed_mean": C.pct_ci(draws_q[k])} for k in range(4)}}
        print(model, "overall", round(np.mean(overall), 4), "Q1..Q4", [round(np.mean(per_q[k]), 4) for k in range(4)], flush=True)
    C.dump(res, "phase1", "p1_4_duration_strata.json")


def ecdf_grid(x, grid):
    x = np.sort(np.asarray(x, float))
    return (np.searchsorted(x, grid, side="right") / max(len(x), 1)).tolist()


def part_1_5():
    preds, y, _year = ll.get_all_predictions()
    grid = np.linspace(0, 1, 101)
    res = {"quantile_levels": QS, "ecdf_grid": grid.tolist(), "alpha": C.ALPHA, "models": {}}
    for model in ("Temporal-Only GRU-D v2", "Frozen-Temporal Head Control v2", "Hybrid GRU-D v2"):
        per_seed = {}
        for seed in C.SEEDS:
            P = preds[(model, seed)]
            rec = {}
            for fold, lab in (("fc", "final_calibration"), ("te", "test")):
                p, yy = np.asarray(P[fold], float), np.asarray(y[fold], int)
                lac = np.where(yy == 1, 1.0 - p, p)                               # 1 - p_hat_y
                rec[lab] = {"n": int(len(p)), "n_pos": int(yy.sum()),
                            "p_given_y0_q": {str(q): float(np.quantile(p[yy == 0], q)) for q in QS},
                            "p_given_y1_q": {str(q): float(np.quantile(p[yy == 1], q)) for q in QS},
                            "lac_q": {str(q): float(np.quantile(lac, q)) for q in QS},
                            "lac_class0_q": {str(q): float(np.quantile(lac[yy == 0], q)) for q in QS},
                            "lac_class1_q": {str(q): float(np.quantile(lac[yy == 1], q)) for q in QS},
                            "ecdf_p_y0": ecdf_grid(p[yy == 0], grid), "ecdf_p_y1": ecdf_grid(p[yy == 1], grid),
                            "ecdf_lac": ecdf_grid(lac, grid)}
            pf, yf = np.asarray(P["fc"], float), np.asarray(y["fc"], int)
            pt, yt = np.asarray(P["te"], float), np.asarray(y["te"], int)
            lac_t = np.where(yt == 1, 1.0 - pt, pt)
            q_static = _class_threshold(np.where(yf == 1, 1.0 - pf, pf), C.ALPHA)
            q0 = _class_threshold(pf[yf == 0], C.ALPHA)
            q1 = _class_threshold(1.0 - pf[yf == 1], C.ALPHA)
            rec["static_qhat_marginal"] = q_static
            rec["test_empirical_0.9_quantile_lac"] = float(np.quantile(lac_t, 0.9))
            rec["static_qhat_class0"] = q0
            rec["static_qhat_class1"] = q1
            rec["test_empirical_0.9_quantile_lac_class0"] = float(np.quantile(lac_t[yt == 0], 0.9))
            rec["test_empirical_0.9_quantile_lac_class1"] = float(np.quantile(lac_t[yt == 1], 0.9))
            rec["marginal_coverage_at_static_qhat_on_test"] = float((lac_t <= q_static).mean())
            per_seed[seed] = rec
        res["models"][model] = per_seed
        r = per_seed[42]
        print(model, "seed42 static q", round(r["static_qhat_marginal"], 4), "test emp q0.9", round(r["test_empirical_0.9_quantile_lac"], 4),
              "class1 q", round(r["static_qhat_class1"], 4), round(r["test_empirical_0.9_quantile_lac_class1"], 4), flush=True)
    C.dump(res, "phase1", "p1_5_score_distributions.json")


if __name__ == "__main__":
    part_1_4()
    part_1_5()
