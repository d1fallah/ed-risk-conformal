"""Cohort description for the manuscript: size, age, sex, triage acuity and outcome prevalence per period.

Reads the chronological split files used by every model (data/train_v2.csv, data/test_v2.csv) and writes
data/m2_cohort_demographics.json, from which generate_paper_macros.py reads the values quoted in the text.
Descriptive only; no model is involved. Usage: python src/cohort_demographics.py
"""
import json
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "m2_cohort_demographics.json")
COLS = ["subject_id", "age", "gender", "triage_acuity", "outcome_critical"]


def describe(df):
    q1, med, q3 = df["age"].quantile([0.25, 0.5, 0.75])
    acuity = df["triage_acuity"].value_counts(normalize=True)
    return {
        "n_stays": int(len(df)),
        "n_patients": int(df["subject_id"].nunique()),
        "age_median": float(med),
        "age_q1": float(q1),
        "age_q3": float(q3),
        "female_share": float((df["gender"] == "F").mean()),
        "female_pct": float(100 * (df["gender"] == "F").mean()),
        "outcome_events": int(df["outcome_critical"].sum()),
        "outcome_prevalence": float(df["outcome_critical"].mean()),
        "outcome_pct": float(100 * df["outcome_critical"].mean()),
        "acuity_missing_share": float(df["triage_acuity"].isna().mean()),
        "acuity_levels_share": {str(int(k)): float(v) for k, v in acuity.sort_index().items()},
    }


def main():
    out = {}
    for name, fn in (("train_period", "train_v2.csv"), ("test_period", "test_v2.csv")):
        df = pd.read_csv(os.path.join(DATA, fn), usecols=COLS)
        out[name] = describe(df)
        out[name]["source_file"] = f"data/{fn}"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
