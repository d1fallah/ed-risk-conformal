"""
Phase 0, tasks 0.3 (aggregates) and 0.4 -- sequence truncation and measurement timing relative to landmarks.
Counts and aggregates only; no identifiers or rows are printed or written.

Facts measured:
  * raw number of ED vital-sign rows per analysed stay and how many exceed T_MAX_ED = 10 (the build script's own counter is
    always 0 because merge_vitalsign_sequence_on_edstay already keeps the last 10 rows);
  * whether the triage slot (index 0) is ever missing or overwritten in the stored tensors;
  * for truncated stays, how many lose measurements charted at or before intime + 2 h (relevant to a 2 h landmark);
  * how many stays have retained measurements charted after 1 h, 2 h, 6 h, 12 h from arrival (information that would not be
    available at a prediction origin at those times), overall and by split.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "revision_v2", "phase0")
T_MAX = 10
QS = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def q(a):
    a = np.asarray(a, dtype=float)
    return {str(p): float(np.quantile(a, p)) for p in QS}


def main():
    os.makedirs(OUT, exist_ok=True)
    vs = pd.read_csv(os.path.join(ROOT, "mimic4_full", "ed", "vitalsign.csv"), usecols=["stay_id", "charttime"], parse_dates=["charttime"])
    res = {"t_max_ed": T_MAX}
    for split in ("train_v2", "test_v2"):
        df = pd.read_csv(os.path.join(ROOT, "data", f"{split}.csv"), usecols=["stay_id", "intime", "outtime"], parse_dates=["intime", "outtime"])
        v = vs[vs["stay_id"].isin(df["stay_id"])].merge(df, on="stay_id")
        v["t_h"] = (v["charttime"] - v["intime"]).dt.total_seconds() / 3600.0
        v = v.sort_values(["stay_id", "charttime"])
        n_rows = v.groupby("stay_id").size().reindex(df["stay_id"].values, fill_value=0)
        v["rank_from_end"] = v.groupby("stay_id").cumcount(ascending=False)       # 0 = last measurement
        dropped = v[v["rank_from_end"] >= T_MAX]
        r = {"n_stays": int(len(df)), "stays_zero_ed_rows": int((n_rows == 0).sum()), "stays_ge1_ed_rows": int((n_rows > 0).sum()),
             "ed_rows_per_stay_quantiles": q(n_rows.values), "ed_rows_mean": float(n_rows.mean()),
             "stays_with_more_than_10_rows_truncated": int((n_rows > T_MAX).sum()),
             "share_stays_truncated": float((n_rows > T_MAX).mean()),
             "total_rows_dropped": int(len(dropped)),
             "dropped_rows_per_truncated_stay_quantiles": q((n_rows[n_rows > T_MAX] - T_MAX).values) if (n_rows > T_MAX).any() else {},
             "truncated_stays_losing_a_row_charted_within_2h_of_arrival": int(dropped.loc[dropped["t_h"] <= 2, "stay_id"].nunique()),
             "truncated_stays_losing_a_row_charted_within_1h_of_arrival": int(dropped.loc[dropped["t_h"] <= 1, "stay_id"].nunique()),
             "truncated_stays_losing_a_row_charted_at_or_before_arrival": int(dropped.loc[dropped["t_h"] <= 0, "stay_id"].nunique()),
             "rows_charted_before_arrival_total": int((v["t_h"] < 0).sum()),
             "rows_charted_after_departure_total": int((v["charttime"] > v["outtime"]).sum()),
             "stays_with_row_after_departure": int(v.loc[v["charttime"] > v["outtime"], "stay_id"].nunique()),
             "retained_rows_charted_after_departure": int(((v["rank_from_end"] < T_MAX) & (v["charttime"] > v["outtime"])).sum()),
             "stays_with_retained_row_after_departure": int(v.loc[(v["rank_from_end"] < T_MAX) & (v["charttime"] > v["outtime"]), "stay_id"].nunique()),
             "retained_rows_charted_before_arrival": int(((v["rank_from_end"] < T_MAX) & (v["t_h"] < 0)).sum())}
        # stored tensors
        with np.load(os.path.join(ROOT, "data", f"m2_irregular_sequences_{split}.npz")) as z:
            pad, th, obs, sid = z["pad_mask"], z["t_hours"], z["obs_mask"], z["stay_id"]
        r["tensor_rows"] = int(len(pad))
        r["tensor_triage_slot_pad_equals_1_all"] = bool((pad[:, 0] == 1).all())
        r["tensor_triage_slot_t_hours_all_zero"] = bool((th[:, 0] == 0).all())
        r["tensor_slots_used_quantiles"] = q(pad.sum(axis=1))
        r["tensor_stays_with_full_11_slots"] = int((pad.sum(axis=1) == 11).sum())
        r["tensor_stays_with_triage_slot_no_observed_value"] = int((obs[:, 0, :].sum(axis=1) == 0).sum())
        # retained measurements after landmark / horizon hours (slots 1..10 only)
        ed_t = np.where(pad[:, 1:] == 1, th[:, 1:], -np.inf)
        last_t = ed_t.max(axis=1)
        last_t = np.where(np.isfinite(last_t), last_t, np.nan)
        r["retained_last_measurement_time_h_quantiles_among_stays_with_rows"] = q(last_t[~np.isnan(last_t)])
        for h in (1, 2, 6, 12):
            n_after = int((np.nan_to_num(last_t, nan=-1) > h).sum())
            r[f"stays_with_retained_measurement_after_{h}h"] = n_after
            r[f"share_stays_with_retained_measurement_after_{h}h"] = float(n_after / len(pad))
            slots_after = int(((pad[:, 1:] == 1) & (th[:, 1:] > h)).sum())
            r[f"retained_measurements_after_{h}h"] = slots_after
        r["retained_measurements_total"] = int((pad[:, 1:] == 1).sum())
        # stays whose entire retained window lies after 2 h (no information from the first 2 h except triage)
        first_t = np.where(pad[:, 1:] == 1, th[:, 1:], np.inf).min(axis=1)
        r["stays_whose_first_retained_ed_row_is_after_2h"] = int(((first_t > 2) & np.isfinite(first_t)).sum())
        r["stays_whose_first_retained_ed_row_is_after_6h"] = int(((first_t > 6) & np.isfinite(first_t)).sum())
        # tensor and raw-table counts agree on the number of retained rows?
        r["raw_rows_retained_equals_tensor_slots"] = bool(int(np.minimum(n_rows.values, T_MAX).sum()) == int((pad[:, 1:] == 1).sum()))
        res[split] = r
    json.dump(res, open(os.path.join(OUT, "p04_truncation_timing.json"), "w"), indent=2)
    for s in ("train_v2", "test_v2"):
        r = res[s]
        print(s, {k: r[k] for k in ("n_stays", "stays_zero_ed_rows", "stays_with_more_than_10_rows_truncated", "share_stays_truncated",
                                     "truncated_stays_losing_a_row_charted_within_2h_of_arrival",
                                     "tensor_triage_slot_pad_equals_1_all", "tensor_triage_slot_t_hours_all_zero",
                                     "stays_with_retained_measurement_after_1h", "stays_with_retained_measurement_after_2h",
                                     "stays_with_retained_measurement_after_6h", "stays_with_retained_measurement_after_12h",
                                     "raw_rows_retained_equals_tensor_slots")})


if __name__ == "__main__":
    main()
