"""
Phase 2 sensitivity analyses for the classical landmark models (reviewer C1 and M3; run before any GPU re-run), 2 h landmark, primary outcome, same stays, labels, split roles,
tuning budget (4 configurations, tuning seeds {42, 123, 7}, validation AUROC) and five final seeds as p2_2_classical.py. Nothing stored by the primary analysis is overwritten:
predictions go to data/revision_v2/preds under new names (h2dd_*, h2nt_*, h2vo_*).
Variants:
  dd : de-duplicated. For every stay whose first ED vital-sign row repeats the triage measurement (at least three of heart rate, respiratory rate, oxygen saturation, systolic
       and diastolic pressure recorded in both, all equal), that row is removed and the later rows are shifted up. All features of the primary analysis, rebuilt.
  nt : no timing features (aggregate vital-sign features and static profile; the per-vital counts stay).
  vo : values only (mean, min, max, last of each vital sign and the static profile; no counts, no timing features).
Hypothesis stated before running: if the advantage of the timing-aware trees over logistic regression, and their AUROC, depend on the copied row (which changes the row counts and
timings in the later years), then after de-duplication both fall and the trees-minus-logistic-regression difference changes; if the timing features carry the ranking, removing them
(nt) closes the gap between trees and logistic regression.
Test AUROC with 1,000 patient-cluster bootstrap resamples (paired draws); the trees are scored on the seed-averaged prediction. Aggregates only.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402
import p2_2_classical as P2  # noqa: E402

CMP = [1, 2, 3, 4, 5]          # heart rate, respiratory rate, oxygen saturation, systolic, diastolic: indices into P.VITALS
VARIANTS = ["dd", "nt", "vo"]


def drop_copied_first_row(values, obs, pad, delta, th):
    n, T, V = values.shape
    both = (obs[:, 0, :][:, CMP] > 0) & (obs[:, 1, :][:, CMP] > 0) & (pad[:, 1] > 0)[:, None]
    eq = np.isclose(values[:, 0, :][:, CMP], values[:, 1, :][:, CMP], rtol=0, atol=1e-6)
    ncmp = both.sum(1)
    flag = (ncmp >= 3) & (eq | ~both).all(1)
    out = []
    for a in (values, obs, pad, delta, th):
        b = a.copy()
        idx = np.where(flag)[0]
        b[idx, 1:T - 1] = a[idx, 2:T]
        b[idx, T - 1] = 0
        out.append(b)
    return (*out, flag)


def features(df, variant):
    values, obs, pad, delta, th = P.h2_arrays(df)
    flag = np.zeros(len(df), bool)
    if variant == "dd":
        values, obs, pad, delta, th, flag = drop_copied_first_row(values, obs, pad, delta, th)
    agg = P.aggregate_features(values, obs, pad)
    if variant == "vo":
        agg = agg[[c for c in agg.columns if not c.endswith("_count")]]
    parts = [agg]
    if variant in ("dd",):
        parts.append(P.timing_features(values, obs, pad, th))
    parts.append(pd.DataFrame(P.static_matrix(df), columns=P.STATIC))
    return pd.concat(parts, axis=1), flag


def fit_variant(variant):
    tag = f"h2{variant}"
    df = P.load_cohort("h2", "primary")
    R = {r: P.role(df, r) for r in P.ROLES}
    F = {r: features(R[r], variant) for r in P.ROLES}
    X = {r: F[r][0] for r in P.ROLES}
    flagged = {r: float(F[r][1].mean()) for r in P.ROLES}
    y = {r: R[r]["y"].values for r in P.ROLES}
    rec = {"n_features": int(X["fit"].shape[1]), "flagged_share_by_role": flagged}
    trace = []
    for c in P2.LR_GRID:
        m = P2.lr_model(c).fit(X["fit"], y["fit"])
        trace.append({"C": c, "val_auroc": float(roc_auc_score(y["val"], m.predict_proba(X["val"])[:, 1]))})
    best = max(trace, key=lambda t: t["val_auroc"])
    m = P2.lr_model(best["C"]).fit(X["fit"], y["fit"])
    for r in P.ROLES:
        P.save_preds(f"{tag}_logreg_{r}", 42, R[r], m.predict_proba(X[r])[:, 1])
    rec["logreg_selected"] = best
    trace = []
    for depth, lr in P2.XGB_GRID:
        aucs = []
        for s in C.TUNE_SEEDS:
            mm = P2.xgb_model(depth, lr, s)
            mm.fit(X["fit"], y["fit"], eval_set=[(X["val"], y["val"])], verbose=False)
            aucs.append(float(roc_auc_score(y["val"], mm.predict_proba(X["val"])[:, 1])))
        trace.append({"max_depth": depth, "learning_rate": lr, "val_auroc_mean": float(np.mean(aucs))})
    bx = max(trace, key=lambda t: t["val_auroc_mean"])
    for s in C.SEEDS:
        mm = P2.xgb_model(bx["max_depth"], bx["learning_rate"], s)
        mm.fit(X["fit"], y["fit"], eval_set=[(X["val"], y["val"])], verbose=False)
        for r in P.ROLES:
            P.save_preds(f"{tag}_xgboost_{r}", s, R[r], mm.predict_proba(X[r])[:, 1])
    rec["xgboost_selected"] = {k: bx[k] for k in ("max_depth", "learning_rate")}
    print(variant, rec["n_features"], "features; flagged", {k: round(v, 3) for k, v in flagged.items()}, "LR C", best["C"], "XGB", rec["xgboost_selected"], flush=True)
    return rec


def mean_score(name, seeds, te):
    acc = []
    for s in seeds:
        sid, sc = P.load_preds(name, s)
        assert np.array_equal(sid, te["stay_id"].values)
        acc.append(np.asarray(sc, float))
    return np.mean(acc, axis=0)


def main():
    t0 = time.time()
    res = {"variants": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    for v in VARIANTS:
        res["variants"][v] = fit_variant(v)
    df = P.load_cohort("h2", "primary")
    te = P.role(df, "test")
    y = te["y"].values.astype(int)
    boot = C.ClusterBoot(te["subject_id"].values, B=1000)
    sc = {"original_logreg": mean_score("h2_logreg_test", [42], te), "original_trees": mean_score("h2_xgboost_timing_test", C.SEEDS, te)}
    for v in VARIANTS:
        sc[f"{v}_logreg"] = mean_score(f"h2{v}_logreg_test", [42], te)
        sc[f"{v}_trees"] = mean_score(f"h2{v}_xgboost_test", C.SEEDS, te)
    draws = {}
    res["auroc"] = {}
    for k, s in sc.items():
        pt, d = C.boot_metric(y, s, boot, "auroc")
        draws[k] = (pt, d)
        res["auroc"][k] = {"auroc": pt, "ci95": C.pct_ci(d)}
        print(k, round(pt, 4), [round(x, 4) for x in C.pct_ci(d)], flush=True)
    res["differences"] = {}
    pairs = [("original_trees", "original_logreg")] + [(f"{v}_trees", f"{v}_logreg") for v in VARIANTS]
    pairs += [(f"{v}_{m}", f"original_{m}") for v in VARIANTS for m in ("trees", "logreg")]
    for a, b in pairs:
        diff = draws[a][1] - draws[b][1]
        res["differences"][f"{a} minus {b}"] = {"delta": draws[a][0] - draws[b][0], "ci95": C.pct_ci(diff)}
        print(a, "minus", b, round(draws[a][0] - draws[b][0], 4), [round(x, 4) for x in C.pct_ci(diff)], flush=True)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["n_test"] = int(len(y)); res["positives"] = int(y.sum())
    C.dump(res, "phase2", "p2_26_dedup_and_ablation_classical.json")
    print("minutes", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
