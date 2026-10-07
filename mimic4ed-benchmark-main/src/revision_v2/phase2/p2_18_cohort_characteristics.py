"""
Phase 2 descriptive table: characteristics of the eligible stays at the 2 h landmark, training period (fit, validation, calibration-selection and final-calibration
subsets) against the test period, with the standardized difference between periods. Aggregates only (medians, quartiles, percentages, counts).
ED length of stay is reported for description only; it is not an input of any landmark model.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402


def smd_cont(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / s) if s > 0 else 0.0


def smd_bin(a, b):
    p, q = float(np.mean(a)), float(np.mean(b))
    s = np.sqrt((p * (1 - p) + q * (1 - q)) / 2)
    return float((q - p) / s) if s > 0 else 0.0


def q(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))]


def main():
    df = P.load_cohort("h2", "primary")
    sec = P.load_cohort("h2", "secondary").set_index("stay_id")["y"]
    _, obs, pad, _, _ = P.h2_arrays(df)
    df["rows_by_L"] = pad[:, 1:].sum(axis=1)
    df["triage_all_missing"] = (obs[:, 0, :].sum(axis=1) == 0).astype(float)
    df["y_sec"] = sec.reindex(df["stay_id"]).values
    groups = {"train_period": df[df["split_role"] != "test"], "test": df[df["split_role"] == "test"]}
    res = {"groups": {}, "smd": {}}
    for g, d in groups.items():
        ac = d["triage_acuity"].fillna(-1)
        res["groups"][g] = {
            "stays": int(len(d)), "patients": int(d["subject_id"].nunique()),
            "age_q": q(d["age"]), "female_pct": float((d["sex"] == 0).mean()),
            "acuity_1_pct": float((ac == 1).mean()), "acuity_2_pct": float((ac == 2).mean()), "acuity_3_pct": float((ac == 3).mean()),
            "acuity_4_5_pct": float((ac >= 4).mean()), "acuity_missing_pct": float((ac < 0).mean()),
            "elix_q": q(d["comorbidity_elixhauser_count"]), "elix_ge1_pct": float((d["comorbidity_elixhauser_count"] >= 1).mean()),
            "rows_q": q(d["rows_by_L"]), "rows_zero_pct": float((d["rows_by_L"] == 0).mean()),
            "triage_all_missing_pct": float(d["triage_all_missing"].mean()),
            "ed_los_h_q": q(d["duration_h"]),
            "primary_pct": float(d["y"].mean()), "secondary_pct": float(np.nanmean(d["y_sec"])),
        }
    a, b = groups["train_period"], groups["test"]
    res["smd"] = {"age": smd_cont(a["age"], b["age"]), "female": smd_bin(a["sex"] == 0, b["sex"] == 0),
                  "acuity_1_2": smd_bin(a["triage_acuity"] <= 2, b["triage_acuity"] <= 2),
                  "elix": smd_cont(a["comorbidity_elixhauser_count"], b["comorbidity_elixhauser_count"]),
                  "rows": smd_cont(a["rows_by_L"], b["rows_by_L"]), "rows_zero": smd_bin(a["rows_by_L"] == 0, b["rows_by_L"] == 0),
                  "ed_los": smd_cont(np.log1p(a["duration_h"].clip(lower=0)), np.log1p(b["duration_h"].clip(lower=0))),
                  "primary": smd_bin(a["y"], b["y"])}
    for g, r in res["groups"].items():
        print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("smd", {k: round(v, 3) for k, v in res["smd"].items()})
    C.dump(res, "phase2", "p2_18_cohort_characteristics.json")


if __name__ == "__main__":
    main()
