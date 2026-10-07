"""
Phase 2 feature-builder fix for the 42-stay Elixhauser anomaly (DEVIATIONS.md P0-5; AUDIT_REPORT.md 0.2).

For every ED stay, the comorbidity count is recomputed from prior admissions only, rejecting an otherwise accepted admission if its hadm_id
equals the stay's own linked hadm_id or its admittime is not before the stay's arrival (an admission that starts at or after arrival is not a prior admission). Only stays whose count would change are written (stay_id and
the corrected count) to data/revision_v2/elix_fix.csv; the loader overrides the stored count for them. Aggregates only are printed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
from medcodes import elixhauser  # noqa: E402
from medcode_utils import eci_var_map  # noqa: E402

RAW = os.path.join(C.ROOT, "mimic4_full")
TIMERANGE = 356 * 5


def main():
    master = pd.read_csv(os.path.join(C.DATA, "master_dataset.csv"), usecols=lambda c: c in ("subject_id", "stay_id", "hadm_id", "intime") or c.startswith("eci_"),
                         parse_dates=["intime"])
    adm = pd.read_csv(os.path.join(RAW, "hosp", "admissions.csv"), usecols=["subject_id", "hadm_id", "admittime", "dischtime"], parse_dates=["admittime", "dischtime"])
    dx = pd.read_csv(os.path.join(RAW, "hosp", "diagnoses_icd.csv"), usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    pairs = master[["subject_id", "stay_id", "hadm_id", "intime"]].rename(columns={"hadm_id": "own"}).merge(adm.rename(columns={"hadm_id": "src"}), on="subject_id")
    gap = pairs["intime"] - pairs["dischtime"]
    acc = pairs[(pairs["intime"] > pairs["dischtime"]) & (gap <= pd.Timedelta(days=TIMERANGE))]
    bad = (acc["src"] == acc["own"]) | (acc["admittime"] >= acc["intime"])
    affected = acc.loc[bad, "stay_id"].unique()
    good = acc[~bad & acc["stay_id"].isin(affected)]
    rows = []
    codes = dx[["icd_code", "icd_version"]].drop_duplicates()
    for c, v in zip(codes["icd_code"], codes["icd_version"]):
        for name in elixhauser(c, int(v)) or []:
            rows.append((c, int(v), eci_var_map[name]))
    cmap = pd.DataFrame(rows, columns=["icd_code", "icd_version", "flag"])
    dx2 = dx.merge(cmap, on=["icd_code", "icd_version"])[["hadm_id", "flag"]].drop_duplicates()
    f = good[["stay_id", "src"]].merge(dx2, left_on="src", right_on="hadm_id")[["stay_id", "flag"]].drop_duplicates().groupby("stay_id").size()
    fixed = f.reindex(affected, fill_value=0)
    eci_cols = [c for c in master.columns if c.startswith("eci_")]
    stored = master.set_index("stay_id").loc[affected, eci_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    out = pd.DataFrame({"stay_id": affected, "count_fixed": fixed.values, "count_stored": stored.values})
    chg = out[out["count_fixed"] != out["count_stored"]]
    chg[["stay_id", "count_fixed"]].to_csv(os.path.join(C.DATA, "revision_v2", "elix_fix.csv"), index=False)
    print({"accepted_links_rejected": int(bad.sum()), "stays_affected": int(len(affected)), "stays_count_changes": int(len(chg)),
           "mean_change": float((chg["count_stored"] - chg["count_fixed"]).mean()) if len(chg) else 0.0})


if __name__ == "__main__":
    os.makedirs(os.path.join(C.DATA, "revision_v2"), exist_ok=True)
    main()
