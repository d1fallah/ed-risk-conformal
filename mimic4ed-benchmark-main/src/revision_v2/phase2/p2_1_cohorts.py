"""
Phase 2.1 -- landmark cohorts on the SAME patient split assignments (R2, R5; addendum version of 2.1).

Landmarks: triage = intime (operational anchor; the triage table has no timestamp); 2 h = intime + 2 h (primary).
Eligibility (R5): present in the ED at L (outtime > L) and event-free at L (no ICU admission or in-hospital death at or before L).
PRIMARY outcome Y_L = 1 if ICU admission (icustays.intime of the first ICU stay of the admission) or in-hospital death
(admissions.deathtime) occurs in (L, L + 24 h]. SECONDARY outcome = the original composite (in-hospital death, or ICU transfer
within 12 h of ED departure) restricted to eligible stays with t_event > L (t_event from the cohort build's `event_time`).
Outcomes are ascertained in hospital only.

Inputs at the 2 h landmark: vital signs with intime <= charttime <= L (most recent T_ED rows kept, the same streaming-buffer rule as the
original builder), triage vitals assigned time intime (slot 0), static features that pass the Phase 0 audit. The sequence tensors are
built with the repository's own builder functions (cleaning, outlier rules, Che et al. delta construction), with the number of ED slots
reduced from 10 to T_ED for the 2 h landmark; T_ED is the smallest value covering at least 99.5% of eligible stays and the number of
truncated stays is reported.

Writes row-level cohort frames and tensors to data/revision_v2/ (untracked, never printed) and aggregates only to
results/revision_v2/phase2/p2_1_cohorts.json.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import m2_build_irregular_cohort as bic  # noqa: E402
from helpers import (convert_temp_to_celcius, remove_outliers, convert_str_to_float, read_triage_table,  # noqa: E402
                     merge_vitalsign_sequence_on_edstay)
from m2_hybrid_common import VITALS_VALID_RANGE  # noqa: E402

OUTD = os.path.join(C.DATA, "revision_v2")
LANDMARK_H = {"triage": 0.0, "h2": 2.0}
HORIZON_H = 24.0
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]


def label_frame(df, L_h):
    """Adds landmark eligibility and outcomes. Times are datetimes; L = intime + L_h hours."""
    d = df.copy()
    for c in ("intime", "outtime", "intime_icu", "deathtime", "event_time"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    L = d["intime"] + pd.Timedelta(hours=L_h)
    H = L + pd.Timedelta(hours=HORIZON_H)
    present = d["outtime"] > L if L_h > 0 else pd.Series(True, index=d.index)
    icu_before = d["intime_icu"].notna() & (d["intime_icu"] <= L)
    dead_before = d["deathtime"].notna() & (d["deathtime"] <= L)
    event_free = ~(icu_before | dead_before)
    icu_in = d["intime_icu"].notna() & (d["intime_icu"] > L) & (d["intime_icu"] <= H)
    dead_in = d["deathtime"].notna() & (d["deathtime"] > L) & (d["deathtime"] <= H)
    d["lm_present"] = present
    d["lm_event_free"] = event_free
    d["lm_eligible"] = present & event_free
    d["y_primary"] = (icu_in | dead_in).astype(int)
    comp = d["outcome_critical"].astype(bool)
    has_t = d["event_time"].notna()
    sec_event_after = comp & has_t & (d["event_time"] > L)
    sec_excluded = comp & (~has_t | (d["event_time"] <= L))          # composite positive with no recoverable time, or event at/before L
    d["lm_eligible_secondary"] = d["lm_eligible"] & ~sec_excluded
    d["y_secondary"] = sec_event_after.astype(int)
    return d


def tally(d, label):
    el = d[d["lm_eligible"]]
    el2 = d[d["lm_eligible_secondary"]]
    return {"stays_in_split": int(len(d)), "patients_in_split": int(d["subject_id"].nunique()),
            "present_at_L": int(d["lm_present"].sum()), "event_free_at_L": int((d["lm_present"] & d["lm_event_free"]).sum()),
            "eligible_stays": int(len(el)), "eligible_patients": int(el["subject_id"].nunique()),
            "primary_positives": int(el["y_primary"].sum()), "primary_prevalence": float(el["y_primary"].mean()) if len(el) else None,
            "secondary_eligible_stays": int(len(el2)), "secondary_positives": int(el2["y_secondary"].sum()),
            "secondary_prevalence": float(el2["y_secondary"].mean()) if len(el2) else None,
            "original_composite_positives_in_split": int(d["outcome_critical"].astype(bool).sum()),
            "original_positives_retained_primary": int((d["outcome_critical"].astype(bool) & d["lm_eligible"] & (d["y_primary"] == 1)).sum()),
            "original_positives_retained_secondary": int((d["lm_eligible_secondary"] & (d["y_secondary"] == 1)).sum()),
            "original_positives_dropped_ineligible": int((d["outcome_critical"].astype(bool) & ~d["lm_eligible"]).sum()),
            "primary_positives_not_original_positive": int((d["lm_eligible"] & (d["y_primary"] == 1) & ~d["outcome_critical"].astype(bool)).sum())}


def build_h2_sequences(df, split_name, vitalsign_clean, triage_raw, t_ed):
    """Sequences for the eligible 2 h stays of df, using the repository builder with T_ED slots."""
    bic.T_MAX_ED = t_ed
    bic.T_MAX_TOTAL = 1 + t_ed
    keep = vitalsign_clean[vitalsign_clean["stay_id"].isin(df["stay_id"])].merge(
        df[["stay_id", "intime"]], on="stay_id")
    keep["charttime"] = pd.to_datetime(keep["charttime"])
    keep = keep[(keep["charttime"] >= keep["intime"]) & (keep["charttime"] <= keep["intime"] + pd.Timedelta(hours=2))]
    n_rows = keep.groupby("stay_id").size()
    seq_by_stay = merge_vitalsign_sequence_on_edstay(keep.drop(columns=["intime"]), t_max=t_ed)
    values, obs, pad, delta = bic.build_irregular_sequences(df.reset_index(drop=True), triage_raw, seq_by_stay)
    t_hours = np.zeros(pad.shape, dtype=np.float32)
    intime = pd.to_datetime(df["intime"]).values.astype("datetime64[ns]").astype(np.int64)
    sid = df["stay_id"].values
    for i in range(len(df)):
        seq = seq_by_stay.get(sid[i])
        if seq is None or len(seq) == 0:
            continue
        times = seq["charttime"].values.astype("datetime64[ns]").astype(np.int64)
        k = min(len(times), t_ed)
        t_hours[i, 1:1 + k] = (times[:k].astype(np.float64) - intime[i]) / 3.6e12
    return values, obs, pad, delta, t_hours, n_rows


def main():
    os.makedirs(OUTD, exist_ok=True)
    fr = C.load_frames()
    tr, te = fr["train_period"], fr["test"]
    for d in (tr, te):
        d["split_role"] = ""
    for k in ("fit", "val", "cs", "fc"):
        tr.loc[tr["stay_id"].isin(fr[k]["stay_id"]), "split_role"] = k
    te["split_role"] = "test"
    res = {"landmarks": {}, "horizon_h": HORIZON_H}

    # raw tables for the 2 h tensors, cleaned exactly as in the original builder
    triage_raw = remove_outliers(convert_temp_to_celcius(read_triage_table(os.path.join(bic.MIMIC_ED_PATH, "triage.csv"))), VITALS_VALID_RANGE)
    vs = pd.read_csv(os.path.join(bic.MIMIC_ED_PATH, "vitalsign.csv"))
    ids = np.concatenate([tr["stay_id"].values, te["stay_id"].values])
    vs = vs[vs["stay_id"].isin(ids)].copy()
    vs["pain"] = vs["pain"].apply(convert_str_to_float).astype(float)
    vs = vs.rename(columns={c: f"vs_{c}" for c in VITALS})
    vs = remove_outliers(convert_temp_to_celcius(vs), VITALS_VALID_RANGE)
    vs = vs.rename(columns={f"vs_{c}": c for c in VITALS})

    for lm, L_h in LANDMARK_H.items():
        tr_l, te_l = label_frame(tr, L_h), label_frame(te, L_h)
        rec = {"L_hours_after_intime": L_h, "splits": {}}
        for role, d in (("fit", tr_l[tr_l.split_role == "fit"]), ("val", tr_l[tr_l.split_role == "val"]), ("cs", tr_l[tr_l.split_role == "cs"]),
                        ("fc", tr_l[tr_l.split_role == "fc"]), ("test", te_l)):
            rec["splits"][role] = tally(d, role)
        rec["totals"] = {k: sum(rec["splits"][r][k] for r in rec["splits"]) for k in
                         ("stays_in_split", "eligible_stays", "primary_positives", "secondary_positives", "original_composite_positives_in_split",
                          "original_positives_retained_primary", "original_positives_retained_secondary", "original_positives_dropped_ineligible")}
        res["landmarks"][lm] = rec
        cols = ["stay_id", "subject_id", "split_role", "intime", "outtime", "approx_real_year", "age", "sex", "triage_acuity",
                "comorbidity_elixhauser_count", "lm_present", "lm_event_free", "lm_eligible", "lm_eligible_secondary", "y_primary", "y_secondary",
                "outcome_critical", "duration_h"]
        keep = pd.concat([tr_l, te_l])
        keep["duration_h"] = (keep["outtime"] - keep["intime"]).dt.total_seconds() / 3600.0
        keep[cols].to_csv(os.path.join(OUTD, f"{lm}_cohort.csv"), index=False)
        print(lm, {r: (v["eligible_stays"], v["primary_positives"], v["secondary_positives"]) for r, v in rec["splits"].items()}, flush=True)

    # 2 h tensors: choose T_ED from the eligible training-period stays
    h2 = pd.read_csv(os.path.join(OUTD, "h2_cohort.csv"), parse_dates=["intime", "outtime"])
    el = h2[h2["lm_eligible"]]
    v2 = vs.merge(el[["stay_id", "intime"]], on="stay_id")
    v2["charttime"] = pd.to_datetime(v2["charttime"])
    v2 = v2[(v2["charttime"] >= v2["intime"]) & (v2["charttime"] <= v2["intime"] + pd.Timedelta(hours=2))]
    cnt = v2.groupby("stay_id").size().reindex(el["stay_id"].values, fill_value=0)
    dist = {"quantiles": {str(q): float(np.quantile(cnt, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)},
            "stays_zero_rows_within_2h": int((cnt == 0).sum()), "eligible_stays": int(len(cnt))}
    t_ed = int(max(2, np.quantile(cnt, 0.995)))
    dist["T_ED_chosen"] = t_ed
    dist["stays_truncated"] = int((cnt > t_ed).sum())
    dist["share_truncated"] = float((cnt > t_ed).mean())
    res["h2_sequence_rows_within_2h"] = dist
    print("rows within 2h:", dist, flush=True)
    for split_name, roles in (("train_v2", ["fit", "val", "cs", "fc"]), ("test_v2", ["test"])):
        sub = h2[h2["split_role"].isin(roles) & h2["lm_eligible"]]
        src = fr["train_period"] if split_name == "train_v2" else fr["test"]
        dfx = src.set_index("stay_id").loc[sub["stay_id"].values].reset_index()
        dfx["intime"] = pd.to_datetime(dfx["intime"])
        values, obs, pad, delta, t_hours, n_rows = build_h2_sequences(dfx, split_name, vs, triage_raw, t_ed)
        np.savez_compressed(os.path.join(OUTD, f"h2_sequences_{split_name}.npz"), values=values, obs_mask=obs, pad_mask=pad,
                            delta_hours=delta, t_hours=t_hours, stay_id=dfx["stay_id"].values)
        res.setdefault("h2_tensors", {})[split_name] = {"rows": int(len(dfx)), "slots": int(pad.shape[1]), "max_t_hours": float(t_hours.max()),
                                                        "max_t_hours_le_2": bool(t_hours.max() <= 2.0 + 1e-6),
                                                        "triage_slot_all_present": bool((pad[:, 0] == 1).all()),
                                                        "stays_with_zero_ed_rows": int((pad[:, 1:].sum(axis=1) == 0).sum())}
        print(split_name, res["h2_tensors"][split_name], flush=True)
    C.dump(res, "phase2", "p2_1_cohorts.json")


if __name__ == "__main__":
    main()
