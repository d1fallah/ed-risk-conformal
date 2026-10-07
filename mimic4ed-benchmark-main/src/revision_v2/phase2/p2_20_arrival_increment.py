"""
Phase 2 diagnostic (reviewer M2): how much does monitoring add to what is known at arrival, on the end-of-stay task?
Hypothesis stated before running: the workflow quantities add a clear increment over arrival information (triage vital signs and their missingness, age, sex,
triage acuity, comorbidity count); if they only restated triage severity, the increment would be near zero.
Same fit / validation / test frames and the same logistic-regression protocol as p1_2 (regularization on validation AUROC, one evaluation on the test period,
1,000 patient-cluster bootstrap resamples; the draws are paired across models, so differences carry a CI).
  Arr      : arrival information (7 triage vital values with missingness indicators, age, sex, one-hot triage acuity, comorbidity count)
  W        : workflow quantities (group A of p1_2)
  Arr+W    : both
  Arr+dur  : arrival information and ED duration alone
Aggregates only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1"))
import rv2_common as C  # noqa: E402
import p1_2_shortcuts as S  # noqa: E402


def arrival(df, split_name):
    values, obs, pad, delta, th = C.sequences(split_name, df["stay_id"].values)
    f = {}
    for i, v in enumerate(S.VITALS):
        x = values[:, 0, i].astype(float)
        f[f"triage_{v}"] = np.where(obs[:, 0, i] > 0, x, 0.0)
        f[f"triage_missing_{v}"] = 1.0 - obs[:, 0, i]
    a = df["triage_acuity"].fillna(3).astype(int).clip(1, 5)
    for k in range(1, 6):
        f[f"acuity_{k}"] = (a == k).astype(float).values
    f["age"] = df["age"].astype(float).values
    f["female"] = (df["sex"] == 0).astype(float).values
    f["comorbidity"] = df["comorbidity_elixhauser_count"].astype(float).values
    return pd.DataFrame(f, index=df.index)


def main():
    fr = C.load_frames()
    fit, val, te = fr["fit"], fr["val"], fr["test"]
    W = {k: S.build(d, n) for k, d, n in (("fit", fit, "train_v2"), ("val", val, "train_v2"), ("test", te, "test_v2"))}
    A = {k: arrival(d, n) for k, d, n in (("fit", fit, "train_v2"), ("val", val, "train_v2"), ("test", te, "test_v2"))}
    wcols = S.GROUPS["A_only"]
    sets = {
        "Arr": lambda k: A[k],
        "W": lambda k: W[k][wcols],
        "Arr+W": lambda k: pd.concat([A[k], W[k][wcols]], axis=1),
        "Arr+duration": lambda k: pd.concat([A[k], W[k][["duration_h", "log1p_duration_h"]]], axis=1),
    }
    yf, yv, yt = (d["outcome_critical"].astype(int).values for d in (fit, val, te))
    boot = C.ClusterBoot(te["subject_id"].values, B=1000)
    res = {"n_test": int(len(yt)), "positives": int(yt.sum()), "models": {}, "differences": {}}
    draws = {}
    for name, f in sets.items():
        sc, info = S.fit_eval(f("fit").values, yf, f("val").values, yv, f("test").values)
        pt, d = C.boot_metric(yt, sc, boot, "auroc")
        draws[name] = (pt, d)
        res["models"][name] = {"n_features": int(f("fit").shape[1]), **info, "auroc": pt, "auroc_ci95": C.pct_ci(d)}
        print(name, round(pt, 4), C.pct_ci(d), flush=True)
    for a, b in (("Arr+W", "Arr"), ("Arr+duration", "Arr"), ("Arr+W", "Arr+duration")):
        diff = draws[a][1] - draws[b][1]
        res["differences"][f"{a} minus {b}"] = {"delta": draws[a][0] - draws[b][0], "ci95": C.pct_ci(diff)}
        print(a, "minus", b, round(draws[a][0] - draws[b][0], 4), C.pct_ci(diff))
    C.dump(res, "phase2", "p2_20_arrival_increment.json")


if __name__ == "__main__":
    main()
