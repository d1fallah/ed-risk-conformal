"""
Phase 2 robustness -- stream-order permutations on the 2 h landmark primary model (as Phase 1.8, on the end-to-end Hybrid GRU-D predictions).
20 random permutations of the order within each calendar-year block of the test stream per seed; the final-calibration warm-up and the learning
rates selected in the original order are kept. Reports coverage and width of the ECI interval and of recalibration (W = 10k), and the ECI set size.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import p2_5_conformal as P5  # noqa: E402
from phase1.p1_8_stream_order import within_year_perm, evaluate, summarize  # noqa: E402

N_PERM = 20


def main():
    res = {"n_perm": N_PERM, "scheme": "permute within each calendar-year block of the test stream; final-calibration warm-up and learning rates fixed", "runs": {}}
    for seed in C.SEEDS:
        S = P5.load_streams(seed)
        p_cs, y_cs, p_fc, y_fc, p_te, y_te = S["cs"]["p"], S["cs"]["y"], S["fc"]["p"], S["fc"]["y"], S["test"]["p"], S["test"]["y"]
        year = np.asarray(S["test"]["df"]["approx_real_year"].values)
        lr_int = RC.select_lr_marginal_interval(p_cs, y_cs, C.ALPHA)
        lr_set = RC.select_lr_marginal_sets(p_cs, y_cs, C.ALPHA)
        ref = evaluate(p_fc, y_fc, p_te, y_te, lr_int, lr_set)
        rng = np.random.default_rng(C.BOOT_SEED + seed)
        perms = [evaluate(p_fc, y_fc, p_te[ix], y_te[ix], lr_int, lr_set) for ix in (within_year_perm(year, rng) for _ in range(N_PERM))]
        d = {"original_order": ref}
        for meth, keys in (("eci_interval", ("cov", "cov1", "width")), ("recal_w10k", ("cov", "cov1", "width")), ("eci_set", ("cov", "cov1", "size"))):
            d[meth] = {k: summarize([r[meth][k] for r in perms]) for k in keys}
        d["valid_all_perms"] = {m: bool(all(r[m]["cov"] >= 0.89 for r in perms)) for m in ("eci_interval", "recal_w10k", "eci_set")}
        res["runs"][str(seed)] = d
        print(seed, {m: (round(d[m]["cov"]["min"], 4), round(d[m]["cov"]["max"], 4)) for m in ("eci_interval", "recal_w10k")}, flush=True)
    C.dump(res, "phase2", "p2_8_stream_order.json")


if __name__ == "__main__":
    main()
