"""
Phase 2 supplementary analysis: reliability (calibration) curve of the 2 h landmark primary outcome on the test period.
Ten equal-frequency bins of the predicted risk, pooled over the stays; per bin the seed-averaged mean predicted risk and observed event rate, with a
patient-cluster bootstrap 95% CI (1,000 resamples) of the observed rate in the seed-averaged score ranking. Counts and aggregates only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402

MODELS = [("logreg", "h2_logreg", [42]), ("xgboost_timing", "h2_xgboost_timing", C.SEEDS), ("hybrid_e2e", "h2_hybrid_e2e", C.SEEDS)]
BINS = 10


def main():
    df = P.load_cohort("h2", "primary")
    te = P.role(df, "test")
    y = te["y"].values.astype(int)
    pid = te["subject_id"].values
    boot = C.ClusterBoot(pid, B=1000)
    W = list(boot.weights())
    res = {"n_test": int(len(y)), "positives": int(y.sum()), "bins": BINS, "binning": "equal-frequency on the seed-averaged predicted risk", "models": {}}
    for name, key, seeds in MODELS:
        arrs = []
        for s in seeds:
            sid, sc = P.load_preds(f"{key}_test", s)
            assert np.array_equal(sid, te["stay_id"].values)
            arrs.append(np.asarray(sc, float))
        p = np.mean(arrs, axis=0)
        order = np.argsort(p, kind="stable")
        chunks = np.array_split(order, BINS)
        rec = {"mean_pred": [], "observed": [], "observed_ci95": [], "n": [], "positives": []}
        for c in chunks:
            rec["mean_pred"].append(float(p[c].mean()))
            rec["observed"].append(float(y[c].mean()))
            rec["n"].append(int(len(c)))
            rec["positives"].append(int(y[c].sum()))
            ds = []
            for w in W:
                ww = w[c]
                ds.append(float((ww * y[c]).sum() / ww.sum()) if ww.sum() > 0 else float("nan"))
            rec["observed_ci95"].append(C.pct_ci(np.array(ds)[~np.isnan(ds)]))
        res["models"][name] = rec
        print(name, [round(v, 3) for v in rec["mean_pred"]], [round(v, 3) for v in rec["observed"]], flush=True)
    C.dump(res, "phase2", "p2_12_reliability.json")


if __name__ == "__main__":
    main()
