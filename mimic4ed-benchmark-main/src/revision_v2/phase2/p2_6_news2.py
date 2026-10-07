"""
Phase 2 comparator -- five-parameter NEWS2 (the repository's `helpers.add_score_NEWS2`; level of consciousness and supplemental oxygen are not in the
extract) scored on the same landmark cohorts and test period as the models:
  triage landmark: triage values;
  2 h landmark: the last value of each parameter recorded by the landmark (triage or ED row), the information-matched comparator.
A missing parameter contributes 0 points (as in the repository). AUROC with a 1,000-resample patient-cluster bootstrap; also on stays with all five
parameters recorded at the landmark (complete cases). Primary outcome Y_L.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402
from helpers import add_score_NEWS2  # noqa: E402

PARAMS = {"resprate": "triage_resprate", "temperature": "triage_temperature", "sbp": "triage_sbp", "heartrate": "triage_heartrate", "o2sat": "triage_o2sat"}


def news2_frame(vals):
    d = pd.DataFrame({PARAMS[k]: vals[k] for k in PARAMS})
    add_score_NEWS2(d)
    return d


def main():
    res = {"bootstrap": "patient-cluster, 1,000 resamples, 95% percentile CI", "landmarks": {}}
    for lm in ("triage", "h2"):
        df = P.load_cohort(lm, "primary")
        te = P.role(df, "test")
        y = te["y"].values.astype(int)
        if lm == "triage":
            v, o = P.triage_arrays(te)
            vals = {k: np.where(o[:, P.VITALS.index(k)] > 0, v[:, P.VITALS.index(k)], np.nan) for k in PARAMS}
        else:
            X = P.h2_feature_frame(te, timing=False)
            vals = {k: X[f"{k}_last"].values for k in PARAMS}
        nf = news2_frame(vals)
        score = nf["score_NEWS2"].values.astype(float)
        complete = np.all([~np.isnan(vals[k]) for k in PARAMS], axis=0)
        boot = C.ClusterBoot(te["subject_id"].values, B=1000)
        pt, d = C.boot_metric(y, score, boot, "auroc")
        rec = {"n_test": int(len(y)), "positives": int(y.sum()), "auroc": pt, "auroc_ci95": C.pct_ci(d),
               "stays_with_all_five_parameters": int(complete.sum()), "share_complete": float(complete.mean()),
               "stays_with_all_five_missing": int((~np.any([~np.isnan(vals[k]) for k in PARAMS], axis=0)).sum())}
        yc, sc = y[complete], score[complete]
        boot_c = C.ClusterBoot(te["subject_id"].values[complete], B=1000)
        ptc, dc = C.boot_metric(yc, sc, boot_c, "auroc")
        rec.update({"complete_case_auroc": ptc, "complete_case_auroc_ci95": C.pct_ci(dc), "complete_case_positives": int(yc.sum())})
        res["landmarks"][lm] = rec
        print(lm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "auroc_ci95"}, flush=True)
    C.dump(res, "phase2", "p2_6_news2.json")


if __name__ == "__main__":
    main()
