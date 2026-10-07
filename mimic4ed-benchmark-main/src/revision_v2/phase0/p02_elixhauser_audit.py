"""
Phase 0, task 0.2 -- Elixhauser timing audit (counts and aggregates only; nothing patient-level is printed or written).

Question (R1): can the comorbidity count used as a static feature (comorbidity_elixhauser_count, the sum of the 31 eci_* flags
written by Benchmark_scripts/medcode_utils.commorbidity) contain diagnoses coded during or after the index ED visit?

Method: independently re-derive, from the raw MIMIC-IV hosp tables, the set of source admissions that the extraction rule admits for
every ED stay, then check (i) no source admission is discharged at or after the stay's arrival, (ii) the stay's own linked
admission is never a source, (iii) the distribution of arrival minus discharge time of the source admissions; finally recompute
the 31 flags from the source admissions and compare them with the stored columns of data/master_dataset.csv.

Outputs: results/revision_v2/phase0/p02_elixhauser_audit.json (aggregates only).
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
TIMERANGE_DAYS = 356 * 5            # as written in extract_master_dataset.py:99
QS = [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]


def q(a):
    a = np.asarray(a, dtype=float)
    return {str(p): float(np.quantile(a, p)) for p in QS} if len(a) else {}


def main():
    os.makedirs(OUT, exist_ok=True)
    master = pd.read_csv(os.path.join(ROOT, "data", "master_dataset.csv"),
                         usecols=lambda c: c in ("subject_id", "stay_id", "hadm_id", "intime") or c.startswith("eci_"))
    master["intime"] = pd.to_datetime(master["intime"])
    eci_cols = [c for c in master.columns if c.startswith("eci_")]
    res = {"n_stays": int(len(master)), "n_subjects": int(master.subject_id.nunique()),
           "timerange_days_in_code": TIMERANGE_DAYS, "timerange_days_if_365x5": 365 * 5, "n_eci_cols": len(eci_cols),
           "master_sorted_by_subject_id": bool(master["subject_id"].is_monotonic_increasing),
           "master_subject_blocks_contiguous": bool(master["subject_id"].ne(master["subject_id"].shift()).sum()
                                                    == master["subject_id"].nunique())}

    adm = pd.read_csv(os.path.join(RAW, "hosp", "admissions.csv"), usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                      parse_dates=["admittime", "dischtime"])
    dx = pd.read_csv(os.path.join(RAW, "hosp", "diagnoses_icd.csv"), usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
                     dtype={"icd_code": str})
    res["n_admissions"] = int(len(adm)); res["n_diagnosis_rows"] = int(len(dx))

    # candidate pairs (stay, admission of the same patient)
    pairs = master[["subject_id", "stay_id", "hadm_id", "intime"]].rename(columns={"hadm_id": "own_hadm"}).merge(
        adm.rename(columns={"hadm_id": "src_hadm"}), on="subject_id", how="inner")
    gap = pairs["intime"] - pairs["dischtime"]
    rule = (pairs["intime"] > pairs["dischtime"]) & (gap <= pd.Timedelta(days=TIMERANGE_DAYS))   # the extraction rule
    src = pairs[rule].copy()
    src["gap_days"] = (src["intime"] - src["dischtime"]).dt.total_seconds() / 86400.0
    res["n_source_admission_links"] = int(len(src))
    res["stays_with_at_least_one_source_admission"] = int(src["stay_id"].nunique())
    res["stays_with_no_source_admission"] = int(len(master) - src["stay_id"].nunique())

    # (i) any source admission discharged at or after arrival, or admitted at or after arrival
    res["check_i_source_dischtime_ge_intime"] = int((src["dischtime"] >= src["intime"]).sum())
    res["check_i_source_admittime_ge_intime"] = int((src["admittime"] >= src["intime"]).sum())
    res["check_i_source_dischtime_missing"] = int(src["dischtime"].isna().sum())
    # (ii) own linked admission among sources
    res["check_ii_own_admission_is_source"] = int((src["src_hadm"] == src["own_hadm"]).sum())
    # independent check of the rule on the excluded side: admissions of the same patient that overlap or follow the stay
    res["n_pairs_same_patient"] = int(len(pairs))
    res["n_pairs_dischtime_ge_intime_excluded"] = int((pairs["dischtime"] >= pairs["intime"]).sum())
    res["n_pairs_older_than_window_excluded"] = int((gap > pd.Timedelta(days=TIMERANGE_DAYS)).sum())
    # (iii) distribution of arrival minus discharge
    res["gap_days_all_source_links"] = q(src["gap_days"])
    last = src.sort_values("gap_days").groupby("stay_id", as_index=False).first()          # most recent source admission per stay
    res["gap_days_most_recent_source_per_stay"] = q(last["gap_days"])
    for d in (1, 7, 30, 90, 365):
        res[f"stays_whose_most_recent_source_ended_within_{d}d"] = int((last["gap_days"] <= d).sum())
    res["source_links_with_gap_ge_1825d_in_window"] = int((src["gap_days"] > 365 * 5).sum())

    # recompute the 31 flags from the source admissions and compare with the stored columns
    codes = dx[["icd_code", "icd_version"]].drop_duplicates()
    cats = {}
    for c, v in zip(codes["icd_code"], codes["icd_version"]):
        out = elixhauser(c, int(v))
        if out:
            cats[(c, int(v))] = out
    cat_names = list(eci_var_map.keys())
    col_of = {k: eci_var_map[k] for k in cat_names}
    rows = []
    for (c, v), lst in cats.items():
        for name in lst:
            rows.append((c, v, col_of[name]))
    cmap = pd.DataFrame(rows, columns=["icd_code", "icd_version", "flag"])
    dx2 = dx.merge(cmap, on=["icd_code", "icd_version"], how="inner")[["hadm_id", "flag"]].drop_duplicates()
    sf = src[["stay_id", "src_hadm"]].merge(dx2, left_on="src_hadm", right_on="hadm_id", how="inner")[["stay_id", "flag"]].drop_duplicates()
    wide = sf.assign(v=1).pivot_table(index="stay_id", columns="flag", values="v", aggfunc="max", fill_value=0)
    wide = wide.reindex(index=master["stay_id"].values, columns=eci_cols, fill_value=0).fillna(0).astype(int)
    stored = master[eci_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    stored.index = master["stay_id"].values
    diff_cells = (wide.values != stored.values)
    stays_diff = diff_cells.any(axis=1)
    res["recompute_flags_n_stays_identical_to_stored"] = int((~stays_diff).sum())
    res["recompute_flags_n_stays_different"] = int(stays_diff.sum())
    res["recompute_flags_n_cells_different"] = int(diff_cells.sum())
    cnt_new, cnt_old = wide.sum(axis=1).values, stored.sum(axis=1).values
    res["recompute_count_exact_match_stays"] = int((cnt_new == cnt_old).sum())
    res["stored_count_distribution"] = {str(k): int(v) for k, v in pd.Series(cnt_old).value_counts().sort_index().items() if k <= 12}
    res["stored_count_mean"] = float(cnt_old.mean())
    res["stored_count_zero_share"] = float((cnt_old == 0).mean())
    # where flags differ, which side has more (a stored flag absent from the rule would mean other-time sources)
    res["cells_stored1_recomputed0"] = int(((stored.values == 1) & (wide.values == 0)).sum())
    res["cells_stored0_recomputed1"] = int(((stored.values == 0) & (wide.values == 1)).sum())
    # ordering caveat of the extraction loop (pointer-based, assumes subject-sorted tables)
    json.dump(res, open(os.path.join(OUT, "p02_elixhauser_audit.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    print("gap quantiles (days), most recent source per stay:", res["gap_days_most_recent_source_per_stay"])


if __name__ == "__main__":
    main()
