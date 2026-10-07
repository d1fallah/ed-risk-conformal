"""
Phase 2 supplementary analyses on the test period (primary outcome Y_L, 2 h landmark):
  * probabilistic calibration of every model: Brier score, expected calibration error (10 equal-frequency bins), calibration slope and intercept
    (logistic regression of the outcome on the logit score), mean predicted risk against observed rate;
  * subgroup discrimination of the primary model (and of the timing-aware trees, as the strongest baseline): AUROC by sex, age band (< 65, >= 65 years)
    and triage-acuity group (acuity 1 to 2, 3, 4 to 5), with a patient-cluster bootstrap 95% CI (1,000 resamples) of the seed-averaged AUROC.
No test information is used to select anything.
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402

MODELS = [("logreg", "h2_logreg", [42]), ("xgboost_timing", "h2_xgboost_timing", C.SEEDS), ("temporal_grud", "h2_temporal_grud", C.SEEDS),
          ("hybrid_frozen", "h2_hybrid_frozen", C.SEEDS), ("hybrid_e2e", "h2_hybrid_e2e", C.SEEDS)]


def ece(y, p, bins=10):
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) * abs(y[c].mean() - p[c].mean()) for c in chunks) / len(y))


def slope_intercept(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(penalty=None, max_iter=1000).fit(z, y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def main():
    df = P.load_cohort("h2", "primary")
    te = P.role(df, "test")
    y = te["y"].values.astype(int)
    res = {"n_test": int(len(y)), "positives": int(y.sum()), "calibration": {}, "subgroups": {}}
    S = {}
    for name, key, seeds in MODELS:
        S[name] = []
        for s in seeds:
            sid, sc = P.load_preds(f"{key}_test", s)
            assert np.array_equal(sid, te["stay_id"].values)
            S[name].append(np.asarray(sc, float))
    for name, arrs in S.items():
        rec = []
        for p in arrs:
            sl, ic = slope_intercept(y, p)
            rec.append({"brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p), "slope": sl, "intercept": ic, "mean_pred": float(p.mean())})
        res["calibration"][name] = {k: {"mean": float(np.mean([r[k] for r in rec])), "sd": float(np.std([r[k] for r in rec], ddof=1)) if len(rec) > 1 else 0.0} for k in rec[0]}
        res["calibration"][name]["observed_rate"] = float(y.mean())
    age = te["age"].values
    groups = {"sex_female": te["sex"].values == 0, "sex_male": te["sex"].values == 1, "age_lt65": age < 65, "age_ge65": age >= 65,
              "acuity_1_2": te["triage_acuity"].values <= 2, "acuity_3": te["triage_acuity"].values == 3, "acuity_4_5": te["triage_acuity"].values >= 4}
    for name in ("hybrid_e2e", "xgboost_timing"):
        res["subgroups"][name] = {}
        for g, m in groups.items():
            yy = y[m]
            if yy.sum() < 20:
                res["subgroups"][name][g] = {"n": int(m.sum()), "positives": int(yy.sum()), "note": "fewer than 20 positives"}
                continue
            boot = C.ClusterBoot(te["subject_id"].values[m], B=1000)
            scs = [C.Scorer(yy, p[m]) for p in S[name]]
            pt = [sc.auroc() for sc in scs]
            dr = [np.mean([sc.auroc(w) for sc in scs]) for w in boot.weights()]
            res["subgroups"][name][g] = {"n": int(m.sum()), "positives": int(yy.sum()), "prevalence": float(yy.mean()), "auroc_seed_mean": float(np.mean(pt)),
                                         "auroc_seed_sd": float(np.std(pt, ddof=1)), "auroc_bootstrap_ci95_of_seed_mean": C.pct_ci(dr)}
        print(name, {g: round(v.get("auroc_seed_mean", float("nan")), 3) for g, v in res["subgroups"][name].items()}, flush=True)
    C.dump(res, "phase2", "p2_7_calibration_subgroups.json")
    print({n: {k: round(v["mean"], 4) for k, v in r.items() if isinstance(v, dict)} for n, r in res["calibration"].items()})


if __name__ == "__main__":
    main()
