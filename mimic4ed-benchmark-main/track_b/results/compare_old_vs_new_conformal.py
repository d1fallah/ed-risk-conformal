"""Old (misaligned, archived) vs new (aligned) M2 conformal results. Read-only; prints a summary."""
import json
import os

import numpy as np
import pandas as pd

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
A = os.path.join(D, "archive_pre_alignment_fix_2026-10-01")


def agg(df, cols, by="model"):
    return df.groupby(by)[cols].mean().round(4)


print("=== 1. MAPIE (static) and ECI, mean over 5 seeds per model (m2_v2_conformal_summary.csv)")
cols = ["mapie_coverage", "mapie_coverage_class1", "eci_coverage", "eci_coverage_class1", "eci_width"]
old, new = pd.read_csv(os.path.join(A, "m2_v2_conformal_summary.csv")), pd.read_csv(os.path.join(D, "m2_v2_conformal_summary.csv"))
print("OLD (misaligned)\n", agg(old, cols).to_string(), "\nNEW (aligned)\n", agg(new, cols).to_string())
for lab, d in (("OLD", old), ("NEW", new)):
    print(f"  {lab} Mondrian-LAC class-1 / class-0 coverage (mean):",
          d.groupby("model")[["mondrian_lac_coverage_class1", "mondrian_lac_coverage_class0"]].mean().round(4).to_dict("index"))

print("\n=== 2. ECI vs periodic recalibration (m2_v2_eci_vs_recalibration.csv)")
cols = ["coverage", "coverage_class1", "width"]
old, new = pd.read_csv(os.path.join(A, "m2_v2_eci_vs_recalibration.csv")), pd.read_csv(os.path.join(D, "m2_v2_eci_vs_recalibration.csv"))
for lab, d in (("OLD", old), ("NEW", new)):
    print(lab, "\n", d.groupby(["model", "method"])[cols].mean().round(4).to_string())
for lab, path in (("OLD", A), ("NEW", D)):
    v = json.load(open(os.path.join(path, "m2_v2_eci_vs_recalibration_verdict.json")))
    print(lab, "verdicts:", {m: (x if isinstance(x, str) else x.get("verdict")) for m, x in v.items() if isinstance(x, (str, dict))})

print("\n=== 3. Mondrian-ECI verdicts")
for f in ("m2_v2_mondrian_eci_pilot_verdict.json", "m2_v2_mondrian_eci_confirmatory_verdict.json"):
    for lab, path in (("OLD", A), ("NEW", D)):
        j = json.load(open(os.path.join(path, f)))
        print(f"{f[:38]:38s} {lab}: verdict={j.get('verdict')} cov1={j.get('coverage_class1'):.3f} informative1={j.get('informative_class1'):.3f} "
              f"set_size={j.get('mean_set_size'):.3f} sat={j.get('q1_saturated_frac'):.3f}")
for lab, path in (("OLD", A), ("NEW", D)):
    j = json.load(open(os.path.join(path, "m2_v2_mondrian_eci_comparators_verdict.json")))
    print(f"comparators {lab}:", {m: v.get("verdict") for m, v in j.items() if isinstance(v, dict)})

print("\n=== 4. Label lag verdicts (marginal, per lag)")
for lab, path in (("OLD", A), ("NEW", D)):
    j = json.load(open(os.path.join(path, "m2_v2_label_lag_verdict.json")))
    for m, v in j["models"].items():
        print(f"{lab} {m[:26]:26s}", {lag: x["verdict"] for lag, x in v["marginal"].items()})
