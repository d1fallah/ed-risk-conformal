"""
Phase 2 supplementary analyses on the test period (2 h landmark, primary outcome Y_L); nothing here selects a parameter on test data.
  * AUROC by approximate calendar-year block of the test period for logistic regression, the timing-aware trees and the primary Hybrid GRU-D
    (seed-averaged, patient-cluster bootstrap 95% CI, 1,000 resamples);
  * decision-curve net benefit, NB(t) = TP/N - FP/N * t/(1-t), of the same three models and of treating every stay as positive, with the score
    used directly as the probability (the models are not recalibrated), on a threshold grid 0.01..0.40, and for six reported thresholds the
    patient-cluster bootstrap 95% CI of the net-benefit difference (model minus treat-all and Hybrid minus trees).
Counts and aggregates only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402

MODELS = [("logreg", "h2_logreg", [42]), ("xgboost_timing", "h2_xgboost_timing", C.SEEDS), ("hybrid_e2e", "h2_hybrid_e2e", C.SEEDS)]
GRID = [round(0.01 * i, 2) for i in range(1, 41)]
REPORT = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def nb(y, p, t, w=None):
    w = np.ones(len(y)) if w is None else w
    n = w.sum()
    sel = p >= t
    tp = (w * y * sel).sum()
    fp = (w * (1 - y) * sel).sum()
    return float(tp / n - fp / n * t / (1 - t))


def nb_all(y, t, w=None):
    w = np.ones(len(y)) if w is None else w
    n = w.sum()
    prev = (w * y).sum() / n
    return float(prev - (1 - prev) * t / (1 - t))


def main():
    df = P.load_cohort("h2", "primary")
    te = P.role(df, "test")
    y = te["y"].values.astype(float)
    pid = te["subject_id"].values
    year = te["approx_real_year"].values
    S = {}
    for name, key, seeds in MODELS:
        S[name] = []
        for s in seeds:
            sid, sc = P.load_preds(f"{key}_test", s)
            assert np.array_equal(sid, te["stay_id"].values)
            S[name].append(np.asarray(sc, float))
    res = {"n_test": int(len(y)), "positives": int(y.sum()), "year_blocks": {}, "decision": {"grid": GRID, "report_thresholds": REPORT, "models": {}, "treat_all": []}}
    # ---- AUROC by year block
    for yr in sorted(set(year)):
        m = year == yr
        yy = y[m]
        rec = {"n": int(m.sum()), "positives": int(yy.sum()), "prevalence": float(yy.mean()), "models": {}}
        boot = C.ClusterBoot(pid[m], B=1000)
        W = list(boot.weights())
        for name, arrs in S.items():
            scs = [C.Scorer(yy, p[m]) for p in arrs]
            pt = float(np.mean([sc.auroc() for sc in scs]))
            dr = [np.mean([sc.auroc(w) for sc in scs]) for w in W]
            rec["models"][name] = {"auroc_seed_mean": pt, "auroc_bootstrap_ci95": C.pct_ci(dr)}
        res["year_blocks"][str(int(yr))] = rec
        print(int(yr), rec["n"], rec["positives"], {k: round(v["auroc_seed_mean"], 3) for k, v in rec["models"].items()}, flush=True)
    # ---- decision curve
    res["decision"]["treat_all"] = [nb_all(y, t) for t in GRID]
    for name, arrs in S.items():
        res["decision"]["models"][name] = {"net_benefit": [float(np.mean([nb(y, p, t) for p in arrs])) for t in GRID]}
    boot = C.ClusterBoot(pid, B=1000)
    W = list(boot.weights())
    d = {name: {t: [] for t in REPORT} for name in S}
    dall = {t: [] for t in REPORT}
    for w in W:
        for t in REPORT:
            dall[t].append(nb_all(y, t, w))
        for name, arrs in S.items():
            for t in REPORT:
                d[name][t].append(float(np.mean([nb(y, p, t, w) for p in arrs])))
    rep = {"thresholds": REPORT, "models": {}}
    for name in S:
        rec = {"net_benefit": [], "minus_treat_all": [], "minus_treat_all_ci95": []}
        for t in REPORT:
            pt = float(np.mean([nb(y, p, t) for p in S[name]]))
            diff = np.array(d[name][t]) - np.array(dall[t])
            rec["net_benefit"].append(pt)
            rec["minus_treat_all"].append(pt - nb_all(y, t))
            rec["minus_treat_all_ci95"].append(C.pct_ci(diff))
        rep["models"][name] = rec
    rep["treat_all"] = [nb_all(y, t) for t in REPORT]
    dh = {t: np.array(d["hybrid_e2e"][t]) - np.array(d["xgboost_timing"][t]) for t in REPORT}
    rep["hybrid_minus_trees"] = [float(np.mean([nb(y, p, t) for p in S["hybrid_e2e"]]) - np.mean([nb(y, p, t) for p in S["xgboost_timing"]])) for t in REPORT]
    rep["hybrid_minus_trees_ci95"] = [C.pct_ci(dh[t]) for t in REPORT]
    res["decision"]["reported"] = rep
    print({n: [round(v, 4) for v in r["net_benefit"]] for n, r in rep["models"].items()}, [round(v, 4) for v in rep["treat_all"]])
    C.dump(res, "phase2", "p2_13_year_blocks_decision.json")


if __name__ == "__main__":
    main()
