"""
M2 -- Required ablation: logit-transform vs. clipping for ECI's bounded formulation, on the v2
chronological primary split (small robustness ablation; the T=2-era single-seed version is in
m2_ablations.py::run_eci_logit).

Same ECI radius q (module's established implementation, lr chosen by the lag-0 two-stage rule on
`calibration_selection`), same score |y - p|. Only the final bounding differs:
  clipping : interval [p - q, p + q] clipped to [0, 1]         (used throughout the module)
  logit    : interval [sigmoid(logit(p) - q), sigmoid(logit(p) + q)]   (roadmap's stated formulation)
The label is never logit-transformed (logit(0), logit(1) are infinite), matching m2_ablations.py.

Expected by construction: sigmoid(.) is strictly inside (0, 1), so a logit interval can never
contain a hard label y in {0, 1} and its coverage is exactly 0 for every q. This script measures it
rather than assuming it, on 3 models x 5 seeds, and reports it once; no section is built around it.
Inference-only on saved predictions; the m2_label_lag prediction cache is reused.
"""
import os, sys, json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import m2_label_lag as ll

DATA_DIR = ll.DATA_DIR
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_logit_vs_clipping.csv")
EPS = 1e-6


def main():
    preds, y, _year = ll.get_all_predictions()
    yt = y["te"].astype(float)
    rows = []
    for model in ll.SEED_DIRS_V2:
        for seed in ll.FINAL_SEEDS:
            p = preds[(model, seed)]
            sp = np.concatenate([p["fc"], p["te"]]); sy = np.concatenate([y["fc"], y["te"]]).astype(int)
            n_cal = len(p["fc"])
            lr = ll.select_lr_marginal(p["cs"], y["cs"], 0)
            q = ll.eci_lag_sim(np.abs(sy - sp), np.zeros(len(sp), int), 1, 0, n_cal, lr)[n_cal:, 0]
            pt = p["te"].astype(float)
            lo_c, hi_c = ll.marginal_interval(pt, q)
            pc = np.clip(pt, EPS, 1 - EPS); lg = np.log(pc / (1 - pc))
            lo_l, hi_l = ll._sorted_bounds(lg - q, lg + q)
            lo_l, hi_l = 1 / (1 + np.exp(-lo_l)), 1 / (1 + np.exp(-hi_l))
            for name, lo, hi in (("clipping", lo_c, hi_c), ("logit", lo_l, hi_l)):
                cov = (yt >= lo) & (yt <= hi)
                rows.append({"model": model, "seed": seed, "bounding": name, "eci_lr": lr,
                             "coverage": float(cov.mean()), "coverage_class1": float(cov[yt == 1].mean()),
                             "width": float((hi - lo).mean())})
        print(f"  {model} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    g = df.groupby(["model", "bounding"])[["coverage", "coverage_class1", "width"]].agg(["mean", "std"]).round(4)
    print(g.to_string())
    # the by-construction property, checked directly rather than assumed
    logit_cov = df[df.bounding == "logit"].coverage.max()
    print(f"\nmax logit-interval coverage over all 15 runs: {logit_cov:.6f}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
