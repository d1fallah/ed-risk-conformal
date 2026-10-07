"""
Phase 2 supplementary analysis (tests the explanation offered for the shortcut): does the discrimination of workflow quantities persist within levels of triage acuity?
Hypothesis stated before running: if monitoring intensity only reflects the triage acuity, then within an acuity stratum the workflow model's AUROC falls toward 0.5 and
adding it to acuity gives no gain; if it also carries clinical judgment made after triage, it keeps discriminating inside strata.
Models (end-of-stay task of the original analysis, outcome_critical, same fit / validation / test frames and the same logistic-regression protocol as p1_2):
  W   : workflow quantities (ED duration, its log, retained rows, per-vital counts)            -- group A of p1_2
  Aq  : triage acuity alone (one-hot of levels 1 to 5)
  W+Aq: both
Test AUROC overall and within acuity strata (1 to 2, 3, 4 to 5; a stratum with fewer than 20 positives is reported as not estimable), patient-cluster bootstrap 95% CI
(1,000 resamples). The stored Hybrid GRU-D predictions (seed mean) are scored in the same strata for reference. Counts and aggregates only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1"))
import rv2_common as C  # noqa: E402
import p1_2_shortcuts as S  # noqa: E402

STRATA = {"acuity_1_2": lambda a: a <= 2, "acuity_3": lambda a: a == 3, "acuity_4_5": lambda a: a >= 4}


def onehot(df):
    a = df["triage_acuity"].fillna(3).astype(int).clip(1, 5)
    return pd.DataFrame({f"acuity_{k}": (a == k).astype(float) for k in range(1, 6)}, index=df.index)


def main():
    fr = C.load_frames()
    fit, val, te = fr["fit"], fr["val"], fr["test"]
    Ff, Fv, Ft = S.build(fit, "train_v2"), S.build(val, "train_v2"), S.build(te, "test_v2")
    A = S.GROUPS["A_only"]
    X = {"W": (Ff[A], Fv[A], Ft[A]), "Aq": (onehot(fit), onehot(val), onehot(te)),
         "W+Aq": (pd.concat([Ff[A], onehot(fit)], axis=1), pd.concat([Fv[A], onehot(val)], axis=1), pd.concat([Ft[A], onehot(te)], axis=1))}
    yf, yv, yt = fit["outcome_critical"].astype(int).values, val["outcome_critical"].astype(int).values, te["outcome_critical"].astype(int).values
    ac = te["triage_acuity"].fillna(3).values
    scores = {}
    for k, (a, b, c) in X.items():
        scores[k], _ = S.fit_eval(a.values, yf, b.values, yv, c.values)
    preds = pd.read_csv(os.path.join(C.DATA, "m2_v2_predictions.csv"))
    pos = pd.Series(np.arange(len(te)), index=te["stay_id"].values)
    g = preds[preds["model"] == "Hybrid GRU-D v2"]
    arr = []
    for seed, gs in g.groupby("seed"):
        s_ = np.empty(len(te)); s_[pos.loc[gs["stay_id"].values].values] = gs["y_prob"].values
        arr.append(s_)
    scores["hybrid_grud_end_of_stay"] = np.mean(arr, axis=0)
    res = {"n_test": int(len(yt)), "positives": int(yt.sum()), "overall": {}, "strata": {}}
    boot_all = C.ClusterBoot(te["subject_id"].values, B=1000)
    for k, s_ in scores.items():
        p, d = C.boot_metric(yt, s_, boot_all, "auroc")
        res["overall"][k] = {"auroc": p, "ci95": C.pct_ci(d)}
    for sname, f in STRATA.items():
        m = f(ac)
        yy = yt[m]
        rec = {"n": int(m.sum()), "positives": int(yy.sum()), "prevalence": float(yy.mean())}
        if yy.sum() < 20 or (1 - yy).sum() < 20:
            rec["note"] = "fewer than 20 events or non-events"
        else:
            boot = C.ClusterBoot(te["subject_id"].values[m], B=1000)
            rec["models"] = {}
            for k, s_ in scores.items():
                p, d = C.boot_metric(yy, s_[m], boot, "auroc")
                rec["models"][k] = {"auroc": p, "ci95": C.pct_ci(d)}
        res["strata"][sname] = rec
        print(sname, rec["n"], rec["positives"], {k: round(v["auroc"], 3) for k, v in rec.get("models", {}).items()}, flush=True)
    print("overall", {k: round(v["auroc"], 3) for k, v in res["overall"].items()})
    C.dump(res, "phase2", "p2_16_shortcut_by_acuity.json")


if __name__ == "__main__":
    main()
