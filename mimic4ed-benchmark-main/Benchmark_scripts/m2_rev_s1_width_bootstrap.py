"""
M2 revision S1 -- significance of the interval-width differences
==============================================================================
Executes S1 of M2_Revision_PreSpecification.md (supervisor sign-off 2026-10-03).

Contrasts, per model: ECI - static, recal(W=10k) - static, recal(W=40k) - static, in mean interval
width over the test stream (methods M2, M3, M4 vs M1 of m2_eci_vs_recalibration.py).

The per-stay widths are recomputed with the functions of m2_eci_vs_recalibration.py on the cached
predictions; the reported per-seed means (m2_v2_eci_vs_recalibration.csv) must reproduce exactly
before any test is run (check T16).

Test (the paper's existing rule, m2_v2_analyze.py): per seed, a patient-cluster bootstrap of the
mean paired per-stay width difference (10,000 resamples, patients drawn with replacement with all
their stays, RNG seed 42 + seed); two-sided bootstrap p-proxy 2*min(P(d<=0), P(d>=0)); the family
p-value of a contrast is the largest of its five per-seed p-proxies; Holm across the 9 contrasts. A
contrast is called significant only if Holm-significant and all five per-seed 95% CIs exclude zero.

Interpretation: the intervals are held fixed at the values the online procedures produced on the
realized stream; the bootstrap measures how much the mean width difference depends on which
patients were sampled, not the variability of the online procedure itself.
"""
import os, sys, json, warnings
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_rev_s1_width_bootstrap.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_label_lag import get_all_predictions
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, run_eci_4way
from m2_v2_common import load_split_frame
from m2_eci_vs_recalibration import static_interval, recal_interval, RECAL_R, WINDOWS
from m2_conformal_v2 import SEED_DIRS_V2, FINAL_SEEDS

N_BOOT = 10000
RNG_SEED = 42
CONTRASTS = {"ECI - static": "M2", "recal(10k) - static": "M3", "recal(40k) - static": "M4"}


def cluster_bootstrap_mean(d, groups, n_boot, seed):
    """Mean of d over stays, resampling patients (groups) with replacement, all stays included."""
    codes, uniq = pd.factorize(groups)
    n_pat = len(uniq)
    s = np.bincount(codes, weights=d, minlength=n_pat)
    c = np.bincount(codes, minlength=n_pat).astype(float)
    rng = np.random.RandomState(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.randint(0, n_pat, size=n_pat), minlength=n_pat)
        out[b] = (w @ s) / (w @ c)
    return float(d.mean()), out


def main():
    preds, y, year = get_all_predictions()
    y_cs, y_fc, y_te = y["cs"], y["fc"], y["te"]
    df_te = sort_chronologically(add_hybrid_static_features(load_split_frame("test_v2")))
    assert len(df_te) == len(y_te) and np.array_equal(df_te["approx_real_year"].values, year), "test ordering mismatch"
    subj = df_te["subject_id"].values
    reported = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_eci_vs_recalibration.csv")).set_index(["model", "seed", "method"])

    results, pvals = {}, {}
    for model in SEED_DIRS_V2:
        per_contrast = {k: [] for k in CONTRASTS}
        for seed in FINAL_SEEDS:
            P = preds[(model, seed)]
            p_cs, p_fc, p_te = P["cs"], P["fc"], P["te"]
            sp = np.concatenate([p_fc, p_te]); sy = np.concatenate([y_fc, y_te]).astype(float); nc = len(p_fc)
            iv = {"M1": static_interval(p_fc, y_fc.astype(float), p_te)}
            _lr, _c, _w, lo_e, hi_e, _cands = run_eci_4way(p_cs, y_cs, p_fc, y_fc, p_te, y_te)
            iv["M2"] = (lo_e, hi_e)
            for mid, W in WINDOWS.items():
                iv[mid] = recal_interval(sp, sy, nc, W, RECAL_R)
            w = {m: hi - lo for m, (lo, hi) in iv.items()}
            for m in w:   # T16: exact reproduction of the reported means
                rep = reported.loc[(model, seed, m), "width"]
                assert abs(w[m].mean() - rep) < 1e-12, f"T16 FAILED {model} {seed} {m}: {w[m].mean()} vs {rep}"
            for name, m in CONTRASTS.items():
                point, boot = cluster_bootstrap_mean(w[m] - w["M1"], subj, N_BOOT, RNG_SEED + seed)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                p = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
                per_contrast[name].append({"seed": seed, "point": point, "ci_lo": float(lo), "ci_hi": float(hi),
                                           "excludes_zero": bool(lo > 0 or hi < 0), "p_proxy": p})
            print(f"{model} seed {seed}: T16 pass", flush=True)
        for name, rows in per_contrast.items():
            key = f"{model} | {name}"
            pts = np.array([r["point"] for r in rows])
            results[key] = {"model": model, "contrast": name, "per_seed": rows, "mean": float(pts.mean()),
                            "sd": float(pts.std(ddof=1)),
                            "excludes_zero_n_of_5": int(sum(r["excludes_zero"] for r in rows)),
                            "p_proxy_conservative": max(r["p_proxy"] for r in rows)}
            pvals[key] = results[key]["p_proxy_conservative"]

    keys = list(pvals)
    reject, p_holm, _, _ = multipletests([pvals[k] for k in keys], alpha=0.05, method="holm")
    for k, r, ph in zip(keys, reject, p_holm):
        results[k]["holm_p"] = float(ph)
        results[k]["holm_significant"] = bool(r)
        results[k]["significant_by_paper_rule"] = bool(r and results[k]["excludes_zero_n_of_5"] == 5)
        print(f"{k}: mean {results[k]['mean']:+.4f}  CIs excl. 0: {results[k]['excludes_zero_n_of_5']}/5  "
              f"Holm p {ph:.4g}  significant {results[k]['significant_by_paper_rule']}")
    with open(OUT_JSON, "w") as f:
        json.dump({"n_boot": N_BOOT, "rng_seed_base": RNG_SEED, "family_size": len(keys), "contrasts": results}, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
