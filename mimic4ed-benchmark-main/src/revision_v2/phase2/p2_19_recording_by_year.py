"""
Phase 2 diagnostic (reviewer C1): is the fall in stays with no ED vital sign by the 2 h landmark (training 41.8%, test 0.1%) a step at one point in time,
which would point to a documentation or EHR change, or a gradual change in care? Stays eligible at the 2 h landmark (primary outcome cohort), by approximate
calendar year and period. Per stay: ED vital-sign rows charted in [intime, intime + 2 h], all ED vital-sign rows of the stay, and the delay from arrival to the
first charted ED vital sign. Hypothesis stated before running: a documentation artifact shows as an abrupt change between two adjacent years, a change in care as
a gradual trend. Counts, shares, and quantiles only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402
import m2_build_irregular_cohort as bic  # noqa: E402


def q(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, p)) for p in (10, 25, 50, 75, 90)] if len(x) else None


def main():
    df = P.load_cohort("h2", "primary")[["stay_id", "split_role", "intime", "outtime", "approx_real_year"]].copy()
    vs = pd.read_csv(os.path.join(bic.MIMIC_ED_PATH, "vitalsign.csv"), usecols=["stay_id", "charttime"])
    vs = vs[vs["stay_id"].isin(df["stay_id"])].copy()
    vs["charttime"] = pd.to_datetime(vs["charttime"])
    vs = vs.merge(df[["stay_id", "intime"]], on="stay_id")
    vs["delay_h"] = (vs["charttime"] - vs["intime"]).dt.total_seconds() / 3600.0
    g = vs.groupby("stay_id")
    per = pd.DataFrame({
        "rows_total": g.size(),
        "rows_by_L": g["delay_h"].apply(lambda d: int(((d >= 0) & (d <= 2.0)).sum())),
        "first_delay_h": g["delay_h"].apply(lambda d: float(d[d >= 0].min()) if (d >= 0).any() else np.nan),
        "any_before_arrival": g["delay_h"].apply(lambda d: bool((d < 0).any())),
    })
    # Does the first ED vital-sign row repeat the triage measurement? Raw values (before any cleaning) of the five vital signs present in both tables;
    # a stay counts as "identical" when every vital recorded in both is equal.
    VIT5 = ["heartrate", "resprate", "o2sat", "sbp", "dbp"]
    raw = pd.read_csv(os.path.join(bic.MIMIC_ED_PATH, "vitalsign.csv"), usecols=["stay_id", "charttime"] + VIT5)
    raw = raw[raw["stay_id"].isin(df["stay_id"])].copy()
    raw["charttime"] = pd.to_datetime(raw["charttime"])
    raw = raw.sort_values(["stay_id", "charttime"]).groupby("stay_id", as_index=False).first()
    tri = pd.read_csv(os.path.join(bic.MIMIC_ED_PATH, "triage.csv"), usecols=["stay_id"] + VIT5)
    m = raw.merge(tri, on="stay_id", suffixes=("_ed", "_tri"))
    both = np.column_stack([(m[f"{v}_ed"].notna() & m[f"{v}_tri"].notna()).values for v in VIT5])
    eq = np.column_stack([(m[f"{v}_ed"] == m[f"{v}_tri"]).values for v in VIT5])
    m["n_compared"] = both.sum(1)
    m["first_row_equals_triage"] = (m["n_compared"] >= 3) & ((eq | ~both).all(1))
    per = per.join(m.set_index("stay_id")[["n_compared", "first_row_equals_triage"]], how="left")
    df = df.merge(per, left_on="stay_id", right_index=True, how="left")
    df["rows_total"] = df["rows_total"].fillna(0)
    df["rows_by_L"] = df["rows_by_L"].fillna(0)
    df["any_before_arrival"] = df["any_before_arrival"].fillna(False).astype(bool)
    df["period"] = np.where(df["split_role"] == "test", "test", "train")
    df["ed_los_h"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600.0
    res = {"n_stays": int(len(df)), "by_year": {}, "by_period": {}}

    def summ(d):
        return {"stays": int(len(d)), "period": sorted(set(d["period"])),
                "zero_rows_by_L_share": float((d["rows_by_L"] == 0).mean()),
                "no_ed_vital_in_stay_share": float((d["rows_total"] == 0).mean()),
                "rows_by_L_q": q(d["rows_by_L"]), "rows_total_q": q(d["rows_total"]),
                "first_delay_h_q": q(d["first_delay_h"]), "first_delay_over_2h_share_of_stays_with_rows":
                    float((d.loc[d["rows_total"] > 0, "first_delay_h"] > 2.0).mean()),
                "rows_charted_before_arrival_share": float(d["any_before_arrival"].mean()),
                "first_row_equals_triage_share_of_comparable": float(d.loc[d["n_compared"].fillna(0) >= 3, "first_row_equals_triage"].astype(bool).mean())
                if (d["n_compared"].fillna(0) >= 3).any() else None,
                "comparable_stays": int((d["n_compared"].fillna(0) >= 3).sum()),
                "ed_los_h_q": q(d["ed_los_h"])}
    for yr, d in df.groupby("approx_real_year"):
        res["by_year"][str(int(yr))] = summ(d)
    for per_, d in df.groupby("period"):
        res["by_period"][per_] = summ(d)
    print(f"{'year':>5} {'period':>8} {'stays':>7} {'zeroByL':>8} {'noVitals':>8} {'firstDelay med (IQR) h':>24} {'>2h':>6} {'rowsByL med':>11} {'rowsTot med':>11}")
    C.dump(res, "phase2", "p2_19_recording_by_year.json")
    for yr, r in list(res["by_year"].items()) + [(k, v) for k, v in res["by_period"].items()]:
        fd = r["first_delay_h_q"]
        fds = f"{fd[2]:>8.2f} ({fd[1]:.2f}-{fd[3]:.2f})" if fd else f"{'none':>24}"
        print(f"{yr:>5} {','.join(r['period']):>8} {r['stays']:>7} {r['zero_rows_by_L_share']:>8.3f} {r['no_ed_vital_in_stay_share']:>8.3f} "
              f"{fds} {r['first_delay_over_2h_share_of_stays_with_rows']:>9.3f} {r['rows_by_L_q'][2]:>11.0f} {r['rows_total_q'][2]:>11.0f}"
              f"  firstRow==triage {r['first_row_equals_triage_share_of_comparable'] if r['first_row_equals_triage_share_of_comparable'] is None else round(r['first_row_equals_triage_share_of_comparable'], 3)}"
              f" (n={r['comparable_stays']})")


if __name__ == "__main__":
    main()
