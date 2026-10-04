"""
M2 -- ECI vs. periodic recalibration on MARGINAL coverage
==============================================================================
Executes M2_ECI_vs_Recalibration_PreRegistration.md, which was written before
this script existed. Read it first: comparators, the validity band (marginal
coverage >= 0.89), the efficiency metric (mean clipped interval width, compared
only among valid methods), the 0.02 tie tolerance and the decision rule are all
fixed there and are imported/restated here, not tuned.

  M1  static split-conformal (MAPIE-equivalent), never refreshed
  M2  ECI, the module's established implementation, UNCHANGED (run_eci_4way)
  M3  M1 + periodic recalibration, W = 10,000
  M4  M1 + periodic recalibration, W = 40,000      (R = 2,000 arrivals for both)

Question: does the cheap remedy match ECI's marginal validity? A yes means the
module's roadmap-stated primary contribution is not needed against it, and the
manuscript must say so.

Checks that must pass before any comparison is printed:
  T6  M1's marginal coverage matches the MAPIE classification-set marginal
      coverage already recorded in m2_v2_conformal_summary.csv (same score,
      same fold) to within 0.003 on every model and seed.
  T7  M3 with a window covering the whole calibration fold and no further
      recalibration reproduces M1 exactly.

No patient-cluster bootstrap (invalid for online/recalibrating methods);
uncertainty is across-seed SD plus descriptive stream-block (K=10) and
calendar-year coverage. Stream blocks are not calendar periods.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_eci_vs_recalibration.csv")
OUT_BLOCKS = os.path.join(DATA_DIR, "m2_v2_eci_vs_recalibration_blocks.csv")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_eci_vs_recalibration_verdict.json")
MAPIE_SUMMARY = os.path.join(DATA_DIR, "m2_v2_conformal_summary.csv")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, run_eci_4way, DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
)
from m2_conformal_v2 import get_predictions_v2, SEED_DIRS_V2, FINAL_SEEDS
from m2_mondrian_eci_pilot import PRIMARY_MODEL
from m2_mondrian_eci_comparators import _class_threshold  # same finite-sample ceil-quantile as everywhere else

ALPHA = 0.1
RECAL_R = 2000
WINDOWS = {"M3": 10000, "M4": 40000}
VALID_MIN_COVERAGE = 0.89
TIE_TOL = 0.02
N_STREAM_BLOCKS = 10
METHODS = ["M1", "M2", "M3", "M4"]
NAMES = {"M1": "static split-conformal (MAPIE-equiv.)", "M2": "ECI (established implementation)",
         "M3": "M1 + recalibration (W=10k)", "M4": "M1 + recalibration (W=40k)"}


def interval_from_radius(p, q):
    lo = np.clip(p - q, 0.0, 1.0); hi = np.clip(p + q, 0.0, 1.0)
    return lo, hi


def static_interval(p_cal, y_cal, p_test):
    q = _class_threshold(np.abs(y_cal - p_cal), ALPHA)
    return interval_from_radius(p_test, q)


def recal_interval(p_stream, y_stream, n_cal, W, R):
    """Marginal thresholds recomputed every R arrivals from the most recent W arrivals
    (strictly before the block they govern)."""
    n = len(p_stream)
    lo = np.zeros(n - n_cal); hi = np.zeros(n - n_cal)
    for start in range(n_cal, n, R):
        w = slice(max(0, start - W), start)
        q = _class_threshold(np.abs(y_stream[w] - p_stream[w]), ALPHA)
        end = min(start + R, n)
        lo[start - n_cal:end - n_cal], hi[start - n_cal:end - n_cal] = interval_from_radius(p_stream[start:end], q)
    return lo, hi


def summarize(lo, hi, y):
    cov = (y >= lo) & (y <= hi)
    return {"coverage": float(cov.mean()), "coverage_class0": float(cov[y == 0].mean()),
            "coverage_class1": float(cov[y == 1].mean()), "width": float((hi - lo).mean())}, cov, (hi - lo)


def block_rows(model, seed, method, cov, width, y, year, rows):
    n = len(y)
    schemes = {"stream10": np.minimum((np.arange(n) * N_STREAM_BLOCKS) // n, N_STREAM_BLOCKS - 1),
               "year": pd.factorize(year, sort=True)[0]}
    for scheme, b in schemes.items():
        for blk in np.unique(b):
            m = b == blk
            rows.append({"model": model, "seed": seed, "method": method, "scheme": scheme, "block": int(blk),
                         "n": int(m.sum()), "coverage": float(cov[m].mean()), "width": float(width[m].mean())})


def main():
    print("Loading v2 data...")
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev = load_dev_pool_split_v2()
    cs_ids = set(dev["calibration_selection"]["subject_ids"].tolist())
    fc_ids = set(dev["final_conformal_calibration"]["subject_ids"].tolist())
    df_cs = sort_chronologically(df_train[df_train["subject_id"].isin(cs_ids)])
    df_fc = sort_chronologically(df_train[df_train["subject_id"].isin(fc_ids)])
    df_te = sort_chronologically(df_test)
    year = df_te["approx_real_year"].values

    Xc, Mc, Dc, Pc, yc = prepare_temporal_tensors_v2(df_cs, "train_v2", device=DEVICE)
    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_fc, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_te, "test_v2", device=DEVICE)
    Sc = torch.tensor(prepare_static_features_v2(df_cs), device=DEVICE)
    Sf = torch.tensor(prepare_static_features_v2(df_fc), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_te), device=DEVICE)
    yc_, yf_, yt_ = (a.cpu().numpy().astype(int) for a in (yc, yf, yt))

    mapie = pd.read_csv(MAPIE_SUMMARY).set_index(["model", "seed"])
    rows, brows = [], []
    for model in SEED_DIRS_V2:
        print(f"\n{'='*60}\n{model}\n{'='*60}")
        for seed in FINAL_SEEDS:
            p_cs = get_predictions_v2(model, seed, Xc, Mc, Dc, Pc, S=Sc)
            p_fc = get_predictions_v2(model, seed, Xf, Mf, Df, Pf, S=Sf)
            p_te = get_predictions_v2(model, seed, Xt, Mt, Dt, Pt, S=St)
            sp = np.concatenate([p_fc, p_te]); sy = np.concatenate([yf_, yt_]).astype(float); nc = len(p_fc)

            iv = {}
            iv["M1"] = static_interval(p_fc, yf_.astype(float), p_te)
            lr_eci, _c, _w, lo_e, hi_e, _cands = run_eci_4way(p_cs, yc_, p_fc, yf_, p_te, yt_)
            iv["M2"] = (lo_e, hi_e)
            for mid, W in WINDOWS.items():
                iv[mid] = recal_interval(sp, sy, nc, W, RECAL_R)

            # ---- correctness checks -----------------------------------------
            m1, _, _ = summarize(*iv["M1"], yt_)
            ref = mapie.loc[(model, seed), "mapie_coverage"]
            assert abs(m1["coverage"] - ref) <= 0.003, (
                f"T6 FAILED {model} seed {seed}: M1 marginal coverage {m1['coverage']:.4f} vs MAPIE {ref:.4f}")
            lo7, hi7 = recal_interval(sp, sy, nc, W=nc, R=len(p_te) + 1)
            assert np.allclose(lo7, iv["M1"][0]) and np.allclose(hi7, iv["M1"][1]), "T7 FAILED"

            for mid in METHODS:
                met, cov, wid = summarize(*iv[mid], yt_)
                rows.append({"model": model, "seed": seed, "method": mid, "method_name": NAMES[mid],
                             "eci_lr": lr_eci if mid == "M2" else np.nan, **met})
                block_rows(model, seed, mid, cov, wid, yt_, year, brows)
            print(f"  seed {seed}: T6/T7 pass (M1 {m1['coverage']:.4f} vs MAPIE {ref:.4f}); ECI lr={lr_eci}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(brows).to_csv(OUT_BLOCKS, index=False)
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_BLOCKS}")

    # ---- pre-registered decision rule --------------------------------------
    def verdict_for(model):
        g = df[df.model == model].groupby("method")[["coverage", "coverage_class0", "coverage_class1", "width"]].mean()
        valid = {m: bool(g.loc[m, "coverage"] >= VALID_MIN_COVERAGE) for m in METHODS}
        w2 = g.loc["M2", "width"]
        if not valid["M2"]:
            return "ECI INVALID (fails the validity band)", g, valid
        cheaper = [m for m in ("M3", "M4") if valid[m] and g.loc[m, "width"] <= w2 + TIE_TOL]
        if cheaper:
            return f"ECI NOT NEEDED ({cheaper} valid with width <= ECI + {TIE_TOL})", g, valid
        if not valid["M3"] and not valid["M4"]:
            return "SUPPORTED - sole valid method", g, valid
        if all((not valid[m]) or g.loc[m, "width"] > w2 + TIE_TOL for m in ("M3", "M4")):
            return "SUPPORTED - efficient", g, valid
        return "INCONCLUSIVE", g, valid

    result = {}
    for model in SEED_DIRS_V2:
        v, g, valid = verdict_for(model)
        result[model] = {"verdict": v, "valid": valid, "table": g.round(4).to_dict("index")}
        print(f"\n{'='*70}\n{model}   [{'PRIMARY' if model == PRIMARY_MODEL else 'replication'}]\n{'='*70}")
        print(f"{'method':40s}{'marg cov':>9s}{'cov1':>8s}{'width':>8s}  valid")
        for m in METHODS:
            r = g.loc[m]
            print(f"{NAMES[m]:40s}{r['coverage']:>9.4f}{r['coverage_class1']:>8.4f}{r['width']:>8.4f}  {'YES' if valid[m] else 'no'}")
        print(f"  VERDICT: {v}")
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
