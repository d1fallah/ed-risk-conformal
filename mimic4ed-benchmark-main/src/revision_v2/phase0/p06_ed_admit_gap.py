"""
Phase 0, task 0.6 -- distribution of (edouttime - admittime) for stays linked to a hospital admission (aggregates only).

edouttime is the ED departure time recorded on the admission row; admittime is the admission time. A negative difference means the
admission is stamped before the ED stay ended. Also reports edstays.outtime - admittime for the same stays, for comparison,
overall and within the analysed cohort (train_v2 and test_v2).
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "revision_v2", "phase0")
QS = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def summarize(d_min):
    d = pd.Series(d_min).dropna()
    return {"n": int(len(d)), "quantiles_minutes": {str(p): float(np.quantile(d, p)) for p in QS},
            "frac_negative": float((d < 0).mean()), "frac_zero": float((d == 0).mean()), "frac_positive": float((d > 0).mean()),
            "n_negative": int((d < 0).sum()), "n_zero": int((d == 0).sum()), "n_positive": int((d > 0).sum()),
            "frac_abs_le_1min": float((d.abs() <= 1).mean()), "frac_abs_le_60min": float((d.abs() <= 60).mean())}


def main():
    os.makedirs(OUT, exist_ok=True)
    ed = pd.read_csv(os.path.join(ROOT, "mimic4_full", "ed", "edstays.csv"), usecols=["subject_id", "stay_id", "hadm_id", "intime", "outtime"],
                     parse_dates=["intime", "outtime"])
    adm = pd.read_csv(os.path.join(ROOT, "mimic4_full", "hosp", "admissions.csv"),
                      usecols=["subject_id", "hadm_id", "admittime", "edregtime", "edouttime"],
                      parse_dates=["admittime", "edregtime", "edouttime"])
    lk = ed.dropna(subset=["hadm_id"]).merge(adm, on=["subject_id", "hadm_id"], how="inner")
    lk["edout_minus_admit_min"] = (lk["edouttime"] - lk["admittime"]).dt.total_seconds() / 60.0
    lk["outtime_minus_admit_min"] = (lk["outtime"] - lk["admittime"]).dt.total_seconds() / 60.0
    lk["admit_minus_intime_min"] = (lk["admittime"] - lk["intime"]).dt.total_seconds() / 60.0
    res = {"edstays_rows": int(len(ed)), "edstays_linked_to_hadm": int(ed["hadm_id"].notna().sum()), "linked_rows_with_admission_row": int(len(lk)),
           "edouttime_missing": int(lk["edouttime"].isna().sum()),
           "all_linked_stays": {"edouttime_minus_admittime": summarize(lk["edout_minus_admit_min"]),
                                "edstays_outtime_minus_admittime": summarize(lk["outtime_minus_admit_min"]),
                                "admittime_minus_edstays_intime": summarize(lk["admit_minus_intime_min"])},
           "edouttime_equals_edstays_outtime_share": float((lk["edouttime"] == lk["outtime"]).mean())}
    for split in ("train_v2", "test_v2"):
        ids = pd.read_csv(os.path.join(ROOT, "data", f"{split}.csv"), usecols=["stay_id"])["stay_id"]
        sub = lk[lk["stay_id"].isin(ids)]
        res[split] = {"linked_stays": int(len(sub)), "edouttime_minus_admittime": summarize(sub["edout_minus_admit_min"]),
                      "edstays_outtime_minus_admittime": summarize(sub["outtime_minus_admit_min"])}
    json.dump(res, open(os.path.join(OUT, "p06_ed_admit_gap.json"), "w"), indent=2)
    for k in ("all_linked_stays", "train_v2", "test_v2"):
        s = res[k]["edouttime_minus_admittime"] if k != "all_linked_stays" else res[k]["edouttime_minus_admittime"]
        print(k, "n", s["n"], "neg/zero/pos", s["frac_negative"], s["frac_zero"], s["frac_positive"], "q05/50/95 (min)",
              s["quantiles_minutes"]["0.05"], s["quantiles_minutes"]["0.5"], s["quantiles_minutes"]["0.95"])


if __name__ == "__main__":
    main()
