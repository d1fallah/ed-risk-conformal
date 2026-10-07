"""
Phase 2 diagnostic (reviewer M5): arrival information versus information at two hours on THE SAME stays and THE SAME outcome.
The arrival-landmark models (logistic regression, gradient-boosted trees; trained on the arrival cohort, outcome in (0, 24 h]) are scored on the test stays that are
eligible at the 2 h landmark, against the 2 h outcome (events in (2 h, 26 h]), and compared with the 2 h landmark models on the identical stays and labels.
Hypothesis stated before running: the 2 h models exceed the arrival models on the common stays by less than or about as much as the headline comparison
(0.887 on the arrival cohort against 0.905 on the 2 h cohort), because the headline comparison also changes the population.
Seed-averaged scores, 1,000 patient-cluster bootstrap resamples (paired draws), aggregates only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402


def mean_scores(key, seeds):
    sid0, acc = None, []
    for s in seeds:
        sid, sc = P.load_preds(key, s)
        sid0 = sid if sid0 is None else sid0
        assert np.array_equal(sid, sid0)
        acc.append(np.asarray(sc, float))
    return sid0, np.mean(acc, axis=0)


def main():
    h2 = P.role(P.load_cohort("h2", "primary"), "test")
    ids, y, pid = h2["stay_id"].values, h2["y"].values.astype(int), h2["subject_id"].values
    boot = C.ClusterBoot(pid, B=1000)
    res = {"n_common": int(len(y)), "positives_h2_outcome": int(y.sum()), "models": {}, "differences": {}}
    pos = {}
    for name, key, seeds in (("arrival_logreg", "triage_logreg_test", [42]), ("arrival_trees", "triage_xgboost_test", C.SEEDS),
                             ("h2_logreg", "h2_logreg_test", [42]), ("h2_trees", "h2_xgboost_timing_test", C.SEEDS), ("h2_hybrid", "h2_hybrid_e2e_test", C.SEEDS)):
        sid, sc = mean_scores(key, seeds)
        idx = pd_index(sid, ids)
        s = sc[idx]
        pt, d = C.boot_metric(y, s, boot, "auroc")
        pos[name] = (pt, d)
        res["models"][name] = {"auroc": pt, "auroc_ci95": C.pct_ci(d)}
        print(name, round(pt, 4), C.pct_ci(d), flush=True)
    for a, b in (("h2_trees", "arrival_trees"), ("h2_logreg", "arrival_logreg"), ("h2_hybrid", "arrival_trees")):
        diff = pos[a][1] - pos[b][1]
        res["differences"][f"{a} minus {b}"] = {"delta": pos[a][0] - pos[b][0], "ci95": C.pct_ci(diff)}
        print(a, "minus", b, round(pos[a][0] - pos[b][0], 4), C.pct_ci(diff))
    C.dump(res, "phase2", "p2_21_arrival_on_h2_population.json")


def pd_index(sid, ids):
    import pandas as pd
    return pd.Series(np.arange(len(sid)), index=np.asarray(sid)).loc[np.asarray(ids)].values


if __name__ == "__main__":
    main()
