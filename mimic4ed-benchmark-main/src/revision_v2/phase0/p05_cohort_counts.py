"""
Phase 0, task 0.5 -- reproduce the cohort counts and report stays and positives for every subset (counts only).

Subsets: the four development-pool patient subsets of the training period (training, validation, calibration_selection,
final_conformal_calibration), the test period, and the returning-patient stays that were excluded because the patient straddled
the 2016 cut (kept out of evaluation). Positives are for the primary composite target outcome_critical and, for the horizon
ablation, y_1h, y_6h, y_12h among horizon-eligible stays. Patient identifiers are read only to count.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "revision_v2", "phase0")
EXPECTED = {"after_filter_stays": 418025, "train_stays": 288208, "train_patients": 139371, "test_stays": 84241,
            "test_patients": 61843, "excluded_stays": 45576, "excluded_patients": 18968,
            "dev_patients": {"training": 108400, "validation": 15486, "calibration_selection": 7743, "final_conformal_calibration": 7742},
            "master_stays_plan": 425088}


def tally(df, label):
    out = {"stays": int(len(df)), "patients": int(df["subject_id"].nunique()),
           "positives_outcome_critical": int(df["outcome_critical"].astype(bool).sum()),
           "prevalence_outcome_critical": float(df["outcome_critical"].astype(bool).mean()) if len(df) else None}
    for h in ("1h", "6h", "12h"):
        el = df[df["horizon_eligible"].astype(bool)] if "horizon_eligible" in df else df.iloc[0:0]
        out[f"eligible_{h}"] = int(len(el))
        out[f"positives_y_{h}"] = int(el[f"y_{h}"].sum()) if len(el) else 0
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    master = pd.read_csv(os.path.join(ROOT, "data", "master_dataset.csv"), usecols=["subject_id", "stay_id", "age", "triage_acuity", "outcome_critical"])
    res = {"expected_by_plan": EXPECTED, "master_rows": int(len(master)), "master_patients": int(master.subject_id.nunique())}
    filt = master[(master["age"] >= 18) & master["triage_acuity"].notnull()]
    res["after_age_ge18_and_acuity_notnull_stays"] = int(len(filt)); res["after_filter_patients"] = int(filt.subject_id.nunique())
    tr = pd.read_csv(os.path.join(ROOT, "data", "train_v2.csv"), usecols=["subject_id", "stay_id", "outcome_critical", "horizon_eligible", "y_1h", "y_6h", "y_12h"])
    te = pd.read_csv(os.path.join(ROOT, "data", "test_v2.csv"), usecols=["subject_id", "stay_id", "outcome_critical", "horizon_eligible", "y_1h", "y_6h", "y_12h"])
    res["train_period"] = tally(tr, "train"); res["test_period"] = tally(te, "test")
    kept = set(tr["stay_id"]) | set(te["stay_id"])
    excl = filt[~filt["stay_id"].isin(kept)]
    res["excluded_returning_patient_stays"] = {"stays": int(len(excl)), "patients": int(excl["subject_id"].nunique()),
                                                 "positives_outcome_critical": int(excl["outcome_critical"].astype(bool).sum()),
                                                 "prevalence_outcome_critical": float(excl["outcome_critical"].astype(bool).mean())}
    res["train_test_patient_overlap"] = int(len(set(tr.subject_id) & set(te.subject_id)))
    res["train_test_stay_overlap"] = int(len(set(tr.stay_id) & set(te.stay_id)))
    # development-pool subsets
    dev = json.load(open(os.path.join(ROOT, "data", "m2_v2_dev_pool_split.json")))
    res["dev_subsets"] = {}
    seen = set()
    for name in ("training", "validation", "calibration_selection", "final_conformal_calibration"):
        ids = set(dev[name]["subject_ids"])
        sub = tr[tr["subject_id"].isin(ids)]
        res["dev_subsets"][name] = {"patients_in_split_file": len(ids), **tally(sub, name),
                                    "overlap_with_earlier_subsets": int(len(ids & seen))}
        seen |= ids
    res["dev_subsets_cover_train_patients"] = bool(seen == set(tr.subject_id))
    res["dev_subsets_stays_total"] = int(sum(v["stays"] for v in res["dev_subsets"].values()))
    # which of the plan's numbers differ
    diffs = {}
    if res["after_age_ge18_and_acuity_notnull_stays"] != EXPECTED["after_filter_stays"]:
        diffs["after_filter_stays"] = (res["after_age_ge18_and_acuity_notnull_stays"], EXPECTED["after_filter_stays"])
    if res["master_rows"] != EXPECTED["master_stays_plan"]:
        diffs["master_rows"] = (res["master_rows"], EXPECTED["master_stays_plan"])
    for k, got in (("train_stays", res["train_period"]["stays"]), ("train_patients", res["train_period"]["patients"]),
                   ("test_stays", res["test_period"]["stays"]), ("test_patients", res["test_period"]["patients"]),
                   ("excluded_stays", res["excluded_returning_patient_stays"]["stays"]),
                   ("excluded_patients", res["excluded_returning_patient_stays"]["patients"])):
        if got != EXPECTED[k]:
            diffs[k] = (got, EXPECTED[k])
    for k, v in EXPECTED["dev_patients"].items():
        if res["dev_subsets"][k]["patients_in_split_file"] != v:
            diffs[f"dev_{k}"] = (res["dev_subsets"][k]["patients_in_split_file"], v)
    res["differences_from_plan_found_vs_expected"] = diffs
    json.dump(res, open(os.path.join(OUT, "p05_cohort_counts.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "expected_by_plan"}, indent=1))


if __name__ == "__main__":
    main()
