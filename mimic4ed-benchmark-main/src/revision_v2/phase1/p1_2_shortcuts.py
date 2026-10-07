"""
Phase 1.2 -- shortcut / leakage diagnostic baselines (not candidate models). Reported whatever they show.

Logistic regression fitted on the development `training` subset, regularisation tuned on `validation` (AUROC), evaluated once on
the test period with 1,000 patient-cluster bootstrap resamples (95% percentile CI). Primary target: outcome_critical.
  Group A (post-decision workflow): ED duration (outtime - intime) and log1p of it, sequence length (retained ED rows), per-vital
           ED measurement counts (7).
  Group B (triage-time missingness): per-vital triage-missing indicators (7), all-triage-missing flag.
Rows reported: A only, B only, A + B, duration alone. For reference, the six stored models' test AUROC is added (seed mean +- SD
across the 5 seeds, reported separately from the bootstrap CI of each seed's AUROC; rule 10).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402

VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def build(df, split_name):
    values, obs, pad, delta, th = C.sequences(split_name, df["stay_id"].values)
    f = {}
    dur = df["duration_h"].values
    f["duration_h"] = dur
    f["log1p_duration_h"] = np.log1p(np.clip(dur, 0, None))
    f["seq_len_ed"] = pad[:, 1:].sum(axis=1)
    for i, v in enumerate(VITALS):
        f[f"count_{v}"] = obs[:, 1:, i].sum(axis=1)
    for i, v in enumerate(VITALS):
        f[f"triage_missing_{v}"] = 1.0 - obs[:, 0, i]
    f["triage_all_missing"] = (obs[:, 0, :].sum(axis=1) == 0).astype(float)
    return pd.DataFrame(f, index=df.index)


GROUPS = {
    "A_only": ["duration_h", "log1p_duration_h", "seq_len_ed"] + [f"count_{v}" for v in VITALS],
    "B_only": [f"triage_missing_{v}" for v in VITALS] + ["triage_all_missing"],
    "duration_alone": ["duration_h"],
}
GROUPS["A_plus_B"] = GROUPS["A_only"] + GROUPS["B_only"]


def fit_eval(Xf, yf, Xv, yv, Xt):
    sc = StandardScaler().fit(Xf)
    best = None
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(Xf), yf)
        a = C.Scorer(yv, m.decision_function(sc.transform(Xv))).auroc()
        if best is None or a > best[0]:
            best = (a, c, m)
    a, c, m = best
    return m.decision_function(sc.transform(Xt)), {"selected_C": c, "validation_auroc": a}


def main():
    fr = C.load_frames()
    fit, val, te = fr["fit"], fr["val"], fr["test"]
    Ff, Fv, Ft = build(fit, "train_v2"), build(val, "train_v2"), build(te, "test_v2")
    yf, yv, yt = fit["outcome_critical"].astype(int).values, val["outcome_critical"].astype(int).values, te["outcome_critical"].astype(int).values
    boot = C.ClusterBoot(te["subject_id"].values, B=1000)
    res = {"target": "outcome_critical", "n_fit": len(fit), "n_val": len(val), "n_test": len(te),
           "n_test_positive": int(yt.sum()), "bootstrap": "patient-cluster, 1,000 resamples, 95% percentile CI", "baselines": {}, "models": {}}
    for name, cols in GROUPS.items():
        score, info = fit_eval(Ff[cols].values, yf, Fv[cols].values, yv, Ft[cols].values)
        pa, da = C.boot_metric(yt, score, boot, "auroc")
        pp, dp = C.boot_metric(yt, score, boot, "auprc")
        res["baselines"][name] = {"features": cols, **info, "test_auroc": pa, "test_auroc_ci95": C.pct_ci(da),
                                  "test_auprc": pp, "test_auprc_ci95": C.pct_ci(dp)}
        print(name, round(pa, 4), C.pct_ci(da), "AUPRC", round(pp, 4))
    # reference: stored model predictions, aligned to the test frame by stay_id
    preds = pd.read_csv(os.path.join(C.DATA, "m2_v2_predictions.csv"))
    pos = pd.Series(np.arange(len(te)), index=te["stay_id"].values)
    for model, g in preds.groupby("model"):
        per = {}
        for seed, gs in g.groupby("seed"):
            idx = pos.loc[gs["stay_id"].values].values
            assert len(gs) == len(te)
            sc_ = np.empty(len(te)); sc_[idx] = gs["y_prob"].values
            pa, da = C.boot_metric(yt, sc_, boot, "auroc")
            pp, _ = C.boot_metric(yt, sc_, boot, "auprc") if False else (C.Scorer(yt, sc_).auprc(), None)
            per[int(seed)] = {"auroc": pa, "auroc_ci95": C.pct_ci(da), "auprc": pp}
        res["models"][model] = {"auroc_seed_summary": C.seed_summary([v["auroc"] for v in per.values()]),
                                "auprc_seed_summary": C.seed_summary([v["auprc"] for v in per.values()]),
                                "per_seed_bootstrap_ci95_auroc": {k: v["auroc_ci95"] for k, v in per.items()}}
        print(model, round(res["models"][model]["auroc_seed_summary"]["mean"], 4))
    C.dump(res, "phase1", "p1_2_shortcuts.json")


if __name__ == "__main__":
    main()
