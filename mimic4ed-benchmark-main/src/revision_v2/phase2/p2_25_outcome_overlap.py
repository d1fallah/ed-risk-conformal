"""
Phase 2 count (reviewer minor): how the primary landmark outcome relates to the original benchmark endpoint among the stays eligible at the 2 h landmark.
Counts of landmark-eligible stays by (primary outcome, original endpoint) and, among primary positives that are not original positives, the type of event
(ICU admission after the original 12-hour window, or in-hospital death without the original ICU criterion). Counts only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402


def main():
    import p2_1_cohorts as P1
    fr = C.load_frames()
    d = pd.concat([P1.label_frame(fr["train_period"], 2.0), P1.label_frame(fr["test"], 2.0)])
    d = d[d["lm_eligible"]].copy()
    prim, orig = d["y_primary"].astype(bool), d["outcome_critical"].astype(bool)
    res = {"eligible": int(len(d)), "primary_positive": int(prim.sum()), "original_positive_among_eligible": int(orig.sum()),
           "primary_and_original": int((prim & orig).sum()), "primary_only": int((prim & ~orig).sum()), "original_only": int((~prim & orig).sum())}
    po = d[prim & ~orig]
    L = po["intime"] + pd.Timedelta(hours=2)
    icu_in = po["intime_icu"].notna() & (po["intime_icu"] > L) & (po["intime_icu"] <= L + pd.Timedelta(hours=24))
    res["primary_only_icu_admission_in_window"] = int(icu_in.sum())
    res["primary_only_death_in_window_without_icu"] = int((~icu_in).sum())
    res["primary_only_icu_later_than_12h_after_departure"] = int((icu_in & (po["intime_icu"] > po["outtime"] + pd.Timedelta(hours=12))).sum())
    print(res)
    C.dump(res, "phase2", "p2_25_outcome_overlap.json")


if __name__ == "__main__":
    main()
