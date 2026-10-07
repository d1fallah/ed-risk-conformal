"""
Phase 1.8 -- stream-order robustness. 20 random permutations of the order WITHIN each calendar-year block of the test stream (the
final-calibration warm-up keeps its order), per seed and model, for ECI (interval and set) and recalibration with W = 10k
(R = 2000). The learning rates are those selected on calibration_selection in the original order (selection never sees the permuted
stream). A robustness diagnostic, not a full characterisation of stream-order effects: the order inside a year block is a
deterministic tiebreak on shifted timestamps, so real within-year order is unknown.
Outputs per run: coverage (marginal, class 1), mean width or set size; the original-order values are included as reference, and the
distribution over permutations (mean, SD, min, max) plus whether the validity criterion (marginal coverage >= 0.89) and the
ECI-versus-recalibration width ordering hold in every permutation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import m2_label_lag as ll  # noqa: E402

N_PERM = 20
MODELS = ["Temporal-Only GRU-D v2", "Frozen-Temporal Head Control v2", "Hybrid GRU-D v2"]
VALID = 0.89


def within_year_perm(year, rng):
    idx = np.arange(len(year))
    out = idx.copy()
    for yv in np.unique(year):
        m = np.where(year == yv)[0]
        out[m] = rng.permutation(m)
    return out


def metrics_interval(lo, hi, y):
    cov = (y >= lo) & (y <= hi)
    return {"cov": float(cov.mean()), "cov1": float(cov[y == 1].mean()), "cov0": float(cov[y == 0].mean()), "width": float((hi - lo).mean())}


def evaluate(p_fc, y_fc, p_te, y_te, lr_int, lr_set, alpha=C.ALPHA):
    (lo, hi), _ = RC.eci_interval(p_fc, y_fc, p_te, y_te, alpha, lr_int)
    eci = metrics_interval(lo, hi, y_te)
    lo, hi = RC.recal_interval_w(p_fc, y_fc, p_te, y_te, alpha, 10000)
    rec = metrics_interval(lo, hi, y_te)
    sets, _q, _lr = RC.eci_marginal_sets(None, None, p_fc, y_fc, p_te, y_te, alpha, lr=lr_set)
    cov_set = sets[np.arange(len(y_te)), y_te]
    es = {"cov": float(cov_set.mean()), "cov1": float(cov_set[y_te == 1].mean()), "size": float(sets.sum(axis=1).mean())}
    return {"eci_interval": eci, "recal_w10k": rec, "eci_set": es}


def summarize(vals):
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def main():
    preds, y, year = ll.get_all_predictions()
    year = np.asarray(year)
    res = {"n_perm": N_PERM, "scheme": "permute within each calendar-year block of the test stream; fc fixed", "runs": {}}
    for model in MODELS:
        for seed in C.SEEDS:
            P = preds[(model, seed)]
            p_cs, p_fc, p_te = P["cs"], P["fc"], P["te"]
            y_cs, y_fc, y_te = y["cs"].astype(int), y["fc"].astype(int), y["te"].astype(int)
            lr_int = RC.select_lr_marginal_interval(p_cs, y_cs, C.ALPHA)
            lr_set = RC.select_lr_marginal_sets(p_cs, y_cs, C.ALPHA)
            ref = evaluate(p_fc, y_fc, p_te, y_te, lr_int, lr_set)
            rng = np.random.default_rng(C.BOOT_SEED + seed)
            perms = [evaluate(p_fc, y_fc, p_te[ix], y_te[ix], lr_int, lr_set) for ix in (within_year_perm(year, rng) for _ in range(N_PERM))]
            d = {"lr_interval": lr_int, "lr_set": lr_set, "original_order": ref}
            for meth, keys in (("eci_interval", ("cov", "cov1", "width")), ("recal_w10k", ("cov", "cov1", "width")), ("eci_set", ("cov", "cov1", "size"))):
                d[meth] = {k: summarize([r[meth][k] for r in perms]) for k in keys}
            d["valid_all_perms"] = {m: bool(all(r[m]["cov"] >= VALID for r in perms)) for m in ("eci_interval", "recal_w10k", "eci_set")}
            d["eci_narrower_than_recal_all_perms"] = bool(all(r["eci_interval"]["width"] < r["recal_w10k"]["width"] for r in perms))
            d["eci_narrower_than_recal_original"] = bool(ref["eci_interval"]["width"] < ref["recal_w10k"]["width"])
            res["runs"][f"{model}|{seed}"] = d
            print(model, seed, "ECI cov", round(d["eci_interval"]["cov"]["mean"], 4), "[", round(d["eci_interval"]["cov"]["min"], 4), round(d["eci_interval"]["cov"]["max"], 4),
                  "] width", round(d["eci_interval"]["width"]["mean"], 4), "recal cov", round(d["recal_w10k"]["cov"]["mean"], 4),
                  "width", round(d["recal_w10k"]["width"]["mean"], 4), "orig ECI cov", round(ref["eci_interval"]["cov"], 4), flush=True)
    C.dump(res, "phase1", "p1_8_stream_order.json")


if __name__ == "__main__":
    main()
