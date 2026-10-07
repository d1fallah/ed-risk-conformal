"""
Phase 2 diagnostic (reviewer M7): is the positive predictive value of the singleton set {1} a property of the model's precision-recall curve?
For the static sets of the primary model (seed 42, alpha = 0.10) the set {1} is issued exactly for the stays whose score exceeds one threshold; this is verified
numerically (the {1} indicator equals "score > t" for the smallest t that reproduces it). The precision-recall curve of the model is then computed on the test
period together with the operating points of the static marginal LAC set, the static Mondrian-LAC set, and, from the stored suite (seed 42), Mondrian-ECI and
periodic Mondrian recalibration, whose threshold moves with the stream. Aggregates only (curve on a fixed grid of 200 score quantiles).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import p2_common as P  # noqa: E402

ALPHA = 0.10


def main():
    df = P.load_cohort("h2", "primary")
    te, fc = P.role(df, "test"), P.role(df, "fc")
    y, y_fc = te["y"].values.astype(int), fc["y"].values.astype(int)
    sid, p = P.load_preds("h2_hybrid_e2e_test", 42)
    sid_fc, p_fc = P.load_preds("h2_hybrid_e2e_fc", 42)
    assert np.array_equal(sid, te["stay_id"].values)
    p, p_fc = np.asarray(p, float), np.asarray(p_fc, float)
    res = {"seed": 42, "alpha": ALPHA, "n_test": int(len(y)), "prevalence": float(y.mean())}
    # precision-recall curve on a grid of score quantiles (higher score = positive)
    qs = np.unique(np.quantile(p, np.linspace(0.0, 0.999, 200)))
    rec, prec = [], []
    for t in qs:
        sel = p > t
        tp = int((sel & (y == 1)).sum())
        rec.append(tp / int(y.sum()))
        prec.append(tp / max(int(sel.sum()), 1))
    res["curve"] = {"recall": rec, "precision": prec}
    pts = {}
    for name, (sets, _q) in (("static_lac", RC.static_lac_sets(p_fc, y_fc, p, ALPHA)), ("static_mondrian_lac", RC.mondrian_static_sets(p_fc, y_fc, p, ALPHA))):
        only1 = sets[:, 1] & ~sets[:, 0]
        # exact threshold rule: the smallest score issued {1}, and no stay above it is not issued {1}
        t = float(p[only1].min()) if only1.any() else None
        rule = bool(only1.any() and np.array_equal(only1, p >= t))
        op = RC.operating_characteristics(sets, y)
        pts[name] = {"threshold_rule_exact": rule, "min_score_issued_1": t, "issued_rate": float(only1.mean()), "ppv": op["ppv_only1"], "sensitivity": op["sensitivity_only1"],
                     "false_alarms_per_100": op["false_alarms_per_100_patients"]}
        # a plain classifier threshold with the same sensitivity
        order = np.argsort(-p)
        tp = np.cumsum(y[order])
        k = int(np.searchsorted(tp, pts[name]["sensitivity"] * y.sum() - 1e-9)) + 1
        pts[name]["same_sensitivity_threshold_ppv"] = float(tp[k - 1] / k)
        print(name, pts[name], flush=True)
    stored = json.load(open(os.path.join(C.RES, "phase2", "p2_5_conformal.json")))
    run0 = [r for r in stored["runs"] if r["seed"] == 42][0]
    for name in ("mondrian_eci", "mondrian_recal_w10k"):
        op = run0["sweep"][str(ALPHA)][name]["operating"]
        pts[name] = {"threshold_rule_exact": False, "ppv": op["ppv_only1"], "sensitivity": op["sensitivity_only1"], "false_alarms_per_100": op["false_alarms_per_100_patients"],
                     "issued_rate": (op["cross"]["only1"]["class0"] + op["cross"]["only1"]["class1"]) / op["n"]}
        order = np.argsort(-p)
        tp = np.cumsum(y[order])
        k = int(np.searchsorted(tp, pts[name]["sensitivity"] * y.sum() - 1e-9)) + 1
        pts[name]["same_sensitivity_threshold_ppv"] = float(tp[k - 1] / k)
        print(name, pts[name], flush=True)
    res["operating_points"] = pts
    C.dump(res, "phase2", "p2_23_pr_operating_points.json")


if __name__ == "__main__":
    main()
