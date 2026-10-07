"""
Phase 2 diagnostic (reviewer M6): sampling uncertainty of the coverage of the STATIC conformal methods on the test period. A patient-cluster bootstrap is valid for
static methods (a fixed threshold applied to independent patients), unlike for the online methods, so it is reported here with a 95% percentile interval and the
share of resamples that reach the pre-specified validity threshold of 0.89.
Hypothesis stated before running: with 78,492 test stays the 95% interval of marginal coverage is about +-0.002, so the primary model (0.892) is not
distinguishable from 0.89 and the tree ensemble (0.888) is not distinguishable from it either; the class-1 coverage interval (6,088 events) is about +-0.01.
Per model: static marginal LAC sets (alpha = 0.10; calibrated on the final-calibration subset), their class-1 coverage, and static Mondrian-LAC class-1 coverage.
Seed-averaged coverage per resample for the models with five seeds. Aggregates only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import p2_common as P  # noqa: E402

MODELS = [("hybrid_e2e", "h2_hybrid_e2e", C.SEEDS), ("xgboost_timing", "h2_xgboost_timing", C.SEEDS), ("logreg", "h2_logreg", [42])]
ALPHA = 0.10
VALID = 0.89


def main():
    df = P.load_cohort("h2", "primary")
    te, fc = P.role(df, "test"), P.role(df, "fc")
    y_te, y_fc = te["y"].values.astype(int), fc["y"].values.astype(int)
    boot = C.ClusterBoot(te["subject_id"].values, B=1000)
    W = np.array(list(boot.weights()))                     # (B, n)
    res = {"n_test": int(len(y_te)), "positives": int(y_te.sum()), "alpha": ALPHA, "validity_threshold": VALID, "bootstrap_resamples": int(len(W)), "models": {}}
    for name, key, seeds in MODELS:
        cov_marg, c1_lac, c1_mond = [], [], []
        for s in seeds:
            sid_te, p_te = P.load_preds(f"{key}_test", s)
            sid_fc, p_fc = P.load_preds(f"{key}_fc", s)
            assert np.array_equal(sid_te, te["stay_id"].values) and np.array_equal(sid_fc, fc["stay_id"].values)
            p_te, p_fc = np.asarray(p_te, float), np.asarray(p_fc, float)
            sets, q = RC.static_lac_sets(p_fc, y_fc, p_te, ALPHA)
            covered = sets[np.arange(len(y_te)), y_te]
            cov_marg.append(covered.astype(float))
            c1_lac.append(covered.astype(float))
            msets, _q = RC.mondrian_static_sets(p_fc, y_fc, p_te, ALPHA)
            c1_mond.append(msets[np.arange(len(y_te)), y_te].astype(float))
        cm, cl, cmo = (np.mean(a, axis=0) for a in (cov_marg, c1_lac, c1_mond))     # per-stay coverage indicator, seed-averaged
        m1 = y_te == 1
        point = {"marginal": float(cm.mean()), "class1_static_lac": float(cl[m1].mean()), "class1_static_mondrian": float(cmo[m1].mean())}
        d_marg = (W @ cm) / W.sum(1)
        w1 = W[:, m1]
        d_c1 = (w1 @ cl[m1]) / w1.sum(1)
        d_mo = (w1 @ cmo[m1]) / w1.sum(1)
        res["models"][name] = {
            "seeds": len(seeds), "point": point,
            "marginal_ci95": C.pct_ci(d_marg), "marginal_share_resamples_at_or_above_threshold": float((d_marg >= VALID).mean()),
            "class1_static_lac_ci95": C.pct_ci(d_c1), "class1_static_mondrian_ci95": C.pct_ci(d_mo),
            "class1_static_mondrian_share_resamples_at_or_above_threshold": float((d_mo >= VALID).mean()),
            "per_seed_marginal": [float(a.mean()) for a in cov_marg],
        }
        print(name, {k: round(v, 4) for k, v in point.items()}, "marginal CI", [round(v, 4) for v in C.pct_ci(d_marg)],
              "share>=0.89", res["models"][name]["marginal_share_resamples_at_or_above_threshold"],
              "| c1 mondrian CI", [round(v, 4) for v in C.pct_ci(d_mo)], flush=True)
    C.dump(res, "phase2", "p2_22_static_coverage_bootstrap.json")


if __name__ == "__main__":
    main()
