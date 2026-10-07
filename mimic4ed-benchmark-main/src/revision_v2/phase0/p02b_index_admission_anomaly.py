"""
Phase 0, task 0.2 (follow-up) -- stays whose own linked admission passes the extraction rule (counts and aggregates only).

p02 found 42 stay-admission links where the admission linked to the ED stay (same hadm_id) is accepted as a 'prior' admission
because its dischtime precedes the stay's intime, and 41 links where admittime is not before intime. This script counts the
stays involved, how many of them are in the analysis cohort and the test period, how many would have a different comorbidity
count without that admission, and how the timestamps relate. No identifiers are printed or stored.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Benchmark_scripts"))
from medcodes import elixhauser  # noqa: E402
from medcode_utils import eci_var_map  # noqa: E402

RAW = os.path.join(ROOT, "mimic4_full")
OUT = os.path.join(ROOT, "results", "revision_v2", "phase0")
TIMERANGE_DAYS = 356 * 5


def main():
    master = pd.read_csv(os.path.join(ROOT, "data", "master_dataset.csv"), usecols=["subject_id", "stay_id", "hadm_id", "intime", "outtime"],
                         parse_dates=["intime", "outtime"])
    adm = pd.read_csv(os.path.join(RAW, "hosp", "admissions.csv"), usecols=["subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "edouttime"],
                      parse_dates=["admittime", "dischtime", "edregtime", "edouttime"])
    dx = pd.read_csv(os.path.join(RAW, "hosp", "diagnoses_icd.csv"), usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    pairs = master.rename(columns={"hadm_id": "own_hadm"}).merge(adm.rename(columns={"hadm_id": "src_hadm"}), on="subject_id")
    gap = pairs["intime"] - pairs["dischtime"]
    src = pairs[(pairs["intime"] > pairs["dischtime"]) & (gap <= pd.Timedelta(days=TIMERANGE_DAYS))]
    own = src[src["src_hadm"] == src["own_hadm"]]
    res = {"links_own_admission_accepted": int(len(own)), "stays": int(own["stay_id"].nunique())}
    res["own_admittime_ge_intime"] = int((own["admittime"] >= own["intime"]).sum())
    res["own_dischtime_lt_admittime"] = int((own["dischtime"] < own["admittime"]).sum())
    res["own_edregtime_missing"] = int(own["edregtime"].isna().sum())
    res["own_intime_minus_dischtime_days_quantiles"] = {
        k: float(v) for k, v in ((own["intime"] - own["dischtime"]).dt.total_seconds() / 86400).describe()[["min", "50%", "max"]].items()}
    res["own_admittime_minus_intime_days_quantiles"] = {
        k: float(v) for k, v in ((own["admittime"] - own["intime"]).dt.total_seconds() / 86400).describe()[["min", "50%", "max"]].items()}
    # does the own admission change the count? compare flags with and without it
    codes = dx[["icd_code", "icd_version"]].drop_duplicates()
    rows = []
    for c, v in zip(codes["icd_code"], codes["icd_version"]):
        for name in elixhauser(c, int(v)) or []:
            rows.append((c, int(v), eci_var_map[name]))
    cmap = pd.DataFrame(rows, columns=["icd_code", "icd_version", "flag"])
    dx2 = dx.merge(cmap, on=["icd_code", "icd_version"])[["hadm_id", "flag"]].drop_duplicates()
    stays = own["stay_id"].unique()
    s_all = src[src["stay_id"].isin(stays)][["stay_id", "src_hadm"]]
    f_all = s_all.merge(dx2, left_on="src_hadm", right_on="hadm_id")[["stay_id", "flag"]].drop_duplicates().groupby("stay_id").size()
    s_wo = src[src["stay_id"].isin(stays) & (src["src_hadm"] != src["own_hadm"])][["stay_id", "src_hadm"]]
    f_wo = s_wo.merge(dx2, left_on="src_hadm", right_on="hadm_id")[["stay_id", "flag"]].drop_duplicates().groupby("stay_id").size()
    f_all = f_all.reindex(stays, fill_value=0); f_wo = f_wo.reindex(stays, fill_value=0)
    res["stays_whose_count_changes_if_own_admission_removed"] = int((f_all != f_wo).sum())
    res["max_count_change"] = int((f_all - f_wo).max())
    res["mean_count_change_over_these_stays"] = float((f_all - f_wo).mean())
    # cohort membership
    for name in ("train_v2", "test_v2"):
        p = os.path.join(ROOT, "data", f"{name}.csv")
        ids = pd.read_csv(p, usecols=["stay_id"])["stay_id"]
        res[f"{name}_n_stays"] = int(len(ids))
        res[f"{name}_stays_with_own_admission_accepted"] = int(np.isin(stays, ids.values).sum())
        res[f"{name}_stays_count_changes"] = int(np.isin(stays[(f_all != f_wo).values], ids.values).sum())
    # other timestamp inconsistencies of the same kind: linked admission discharged before ED arrival at all
    linked = master.dropna(subset=["hadm_id"]).merge(adm.rename(columns={"hadm_id": "hadm_id"}), on=["subject_id", "hadm_id"], how="left")
    res["stays_linked_to_admission"] = int(len(linked))
    res["linked_dischtime_before_intime"] = int((linked["dischtime"] < linked["intime"]).sum())
    res["linked_admittime_before_intime_minus_1d"] = int((linked["admittime"] < linked["intime"] - pd.Timedelta(days=1)).sum())
    json.dump(res, open(os.path.join(OUT, "p02b_index_admission_anomaly.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
