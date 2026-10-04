"""
M2 revision E4 -- controlled within-period drift (stress test; supplementary/secondary only)
==============================================================================
Executes E4 of M2_Revision_PreSpecification.md (supervisor sign-off 2026-10-03: run, report as
supplementary/secondary). Synthetic drift imposed on real predictions; it is not a finding about
the hospital and does not change the manuscript's headline.

Drifts, starting at the midpoint of each model/seed test stream (model, calibration and labels
untouched):
  shift+0.5, shift+1.0  logit(p) shifted by delta for every stay after the change point
  casemix               the test stream re-ordered so that the stays after the change point are the
                        more severe half by triage acuity (acuity 1 = most severe). Stays are ranked
                        by acuity (least severe first; ties keep chronological order); the first
                        half precedes the change point. Chronological order is kept within each half.
Methods: static split-conformal (M1), periodic recalibration W=10k R=2k (M3), ECI on intervals
(established run_eci_4way), ACI and ECI on LAC sets (as in E3, immediate labels).
Metrics: coverage and width/set size before and after the change point; coverage in trailing
windows of 2,000 arrivals; arrivals after the change point until the trailing coverage, having left
[0.89, 0.91], first returns to it.
Expected direction, written in advance: static calibration loses coverage after the change and the
adaptive methods recover it. Reported whatever the outcome.
"""
import os, sys, json, warnings
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_rev_e4_drift_stress.csv")
OUT_TRAJ = os.path.join(DATA_DIR, "m2_v2_rev_e4_drift_stress_trajectories.csv")

sys.path.insert(0, SCRIPT_DIR)
from m2_label_lag import get_all_predictions
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, run_eci_4way
from m2_v2_common import load_split_frame
from m2_eci_vs_recalibration import static_interval, recal_interval, RECAL_R
from m2_conformal_v2 import SEED_DIRS_V2, FINAL_SEEDS
from m2_rev_e3_adaptive_sets import lac_sets, online_q, select_lr
from m2_mondrian_eci_pilot import set_metrics

ROLL = 2000
BAND = (0.89, 0.91)
TRAJ_STEP = 500
SCENARIOS = ("shift+0.5", "shift+1.0", "casemix")


def logit_shift(p, delta):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + delta)))


def apply_drift(p_te, y_te, acuity, scenario):
    n = len(p_te); cp = n // 2
    if scenario.startswith("shift"):
        delta = float(scenario.split("+")[1])
        p = p_te.copy(); p[cp:] = logit_shift(p_te[cp:], delta)
        return p, y_te.copy(), cp
    order = np.lexsort((np.arange(n), -acuity))     # least severe (acuity 5) first, ties chronological
    first, second = np.sort(order[:cp]), np.sort(order[cp:])
    idx = np.concatenate([first, second])
    return p_te[idx], y_te[idx], cp


def recovery(cov, cp):
    """Arrivals after cp until trailing-ROLL coverage, having left BAND, first re-enters it."""
    c = np.cumsum(np.concatenate([[0], cov.astype(float)]))
    t = np.arange(ROLL, len(cov) + 1)
    trail = (c[t] - c[t - ROLL]) / ROLL            # trailing coverage ending at position t-1
    pos = t - 1
    inb = (trail >= BAND[0]) & (trail <= BAND[1])
    after = pos >= cp
    left = np.where(after & ~inb)[0]
    if len(left) == 0:
        return "never_left", np.nan
    back = np.where(after & inb & (np.arange(len(pos)) > left[0]))[0]
    if len(back) == 0:
        return "not_recovered", np.nan
    return "recovered", float(pos[back[0]] - cp)


def main():
    preds, y, year = get_all_predictions()
    y_cs, y_fc, y_te0 = y["cs"], y["fc"], y["te"]
    df_te = sort_chronologically(add_hybrid_static_features(load_split_frame("test_v2")))
    assert len(df_te) == len(y_te0) and np.array_equal(df_te["approx_real_year"].values, year), "test ordering mismatch"
    acuity = df_te["triage_acuity"].values.astype(float)
    assert np.isfinite(acuity).all() and acuity.min() >= 1 and acuity.max() <= 5

    rows, traj = [], []
    for model in SEED_DIRS_V2:
        for seed in FINAL_SEEDS:
            P = preds[(model, seed)]
            p_cs, p_fc = P["cs"], P["fc"]
            lrs = {m: select_lr(m, p_cs, y_cs, 0) for m in ("aci_lac", "eci_lac")}
            for sc in SCENARIOS:
                p_te, y_te, cp = apply_drift(P["te"], y_te0, acuity, sc)
                nc = len(p_fc)
                sp = np.concatenate([p_fc, p_te]); sy = np.concatenate([y_fc, y_te]).astype(float)
                out = {}
                lo, hi = static_interval(p_fc, y_fc.astype(float), p_te); out["static"] = (lo, hi)
                out["recal_10k"] = recal_interval(sp, sy, nc, 10000, RECAL_R)
                _lr, _c, _w, lo_e, hi_e, _ = run_eci_4way(p_cs, y_cs, p_fc, y_fc, p_te, y_te)
                out["eci_interval"] = (lo_e, hi_e)
                for m in ("aci_lac", "eci_lac"):
                    q = online_q(m, np.abs(sy - sp), lrs[m], 0, lag_start=nc)[nc:]
                    out[m] = q
                for m, v in out.items():
                    if isinstance(v, tuple):
                        lo, hi = v; cov = (y_te >= lo) & (y_te <= hi); size = hi - lo; unit = "width"
                    else:
                        S = lac_sets(p_te, v); cov = S[np.arange(len(y_te)), y_te.astype(int)]; size = S.sum(1); unit = "set_size"
                    status, rec = recovery(cov, cp)
                    rows.append({"model": model, "seed": seed, "scenario": sc, "method": m, "size_unit": unit,
                                 "cov_pre": float(cov[:cp].mean()), "cov_post": float(cov[cp:].mean()),
                                 "cov1_post": float(cov[cp:][y_te[cp:] == 1].mean()),
                                 "size_pre": float(size[:cp].mean()), "size_post": float(size[cp:].mean()),
                                 "recovery_status": status, "recovery_arrivals": rec,
                                 "base_rate_pre": float(y_te[:cp].mean()), "base_rate_post": float(y_te[cp:].mean())})
                    c = np.cumsum(np.concatenate([[0], cov.astype(float)]))
                    for t in range(ROLL, len(cov) + 1, TRAJ_STEP):
                        traj.append({"model": model, "seed": seed, "scenario": sc, "method": m,
                                     "t_minus_cp": t - 1 - cp, "trailing_cov": (c[t] - c[t - ROLL]) / ROLL})
                print(f"{model} seed {seed} {sc}: " + ", ".join(
                    f"{r['method']} {r['cov_pre']:.3f}->{r['cov_post']:.3f}" for r in rows[-len(out):]), flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(traj).to_csv(OUT_TRAJ, index=False)
    print(f"Wrote {OUT_CSV}\nWrote {OUT_TRAJ}")


if __name__ == "__main__":
    main()
