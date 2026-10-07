"""
Phase 2.4 -- conformal suite on the primary 2 h landmark model (Hybrid GRU-D trained end-to-end, R6), five seeds.

Methods: static LAC, static Mondrian-LAC, ECI (interval and set), recalibration W = 10k (R = 2,000), Mondrian-ECI; alpha sweep
{0.20, 0.10, 0.05, 0.01}; clinical operating characteristics of the prediction sets; label delay L in {60, 240, 1000}; controlled drift test
(logit shifts +0.5 / +1.0 and a case-mix change at the stream midpoint). The code is the repository's, re-run unchanged on the new
predictions (rv2_conformal.py; equality with the stored results at alpha = 0.10 was verified in Phase 1: max difference 2e-16).
Streams: calibration_selection (learning-rate selection only), final_conformal_calibration (warm-up), test; each sorted by
(approx_real_year, intime) exactly as in the original (`sort_chronologically`). No patient-cluster bootstrap for online methods (invalid for
a stream): uncertainty is the across-seed SD plus descriptive stream-block and calendar-year coverage.
"""
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import p2_common as P  # noqa: E402
import m2_label_lag as ll  # noqa: E402
import m2_rev_e4_drift_stress as e4  # noqa: E402
from m2_mondrian_eci_pilot import set_metrics  # noqa: E402
from m2_eci_vs_recalibration import static_interval  # noqa: E402

MODEL = os.environ.get("P2_CONF_MODEL", "h2_hybrid_e2e")
OUT_NAME = "p2_5_conformal.json" if MODEL == "h2_hybrid_e2e" else f"p2_5_conformal_{MODEL}.json"
ALPHAS = [0.20, 0.10, 0.05, 0.01]
LAGS = [60, 240, 1000]
VALID = 0.89


def stream(role_name, df):
    d = P.role(df, role_name)
    order = np.lexsort((d["intime"].values.astype("datetime64[ns]").astype(np.int64), d["approx_real_year"].values))
    return d.iloc[order].reset_index(drop=True), order


def load_streams(seed):
    df = P.load_cohort("h2", "primary")
    out = {}
    for r in ("cs", "fc", "test"):
        d, order = stream(r, df)
        sid, sc = P.load_preds(f"{MODEL}_{r}", seed)
        base = P.role(df, r)
        assert np.array_equal(sid, base["stay_id"].values)
        out[r] = {"df": d, "p": np.asarray(sc, float)[order], "y": d["y"].values.astype(int)}
    return out


def interval_metrics(lo, hi, y):
    cov = (y >= lo) & (y <= hi)
    return {"coverage": float(cov.mean()), "coverage_class0": float(cov[y == 0].mean()), "coverage_class1": float(cov[y == 1].mean()),
            "width": float((hi - lo).mean())}


def eci_sets_lag(S, alpha, L):
    """Marginal ECI sets with labels delayed by L arrivals (lag applied to the test stream; selection run under the same lag)."""
    p_cs, y_cs, p_fc, y_fc, p_te = S["cs"]["p"], S["cs"]["y"], S["fc"]["p"], S["fc"]["y"], S["test"]["p"]
    sc_cs = np.abs(y_cs.astype(float) - p_cs)
    cands = []
    for lr in RC.LR_GRID:
        q = ll.eci_lag_sim(sc_cs, np.zeros(len(sc_cs), int), 1, L, 0, lr, alpha=alpha)[:, 0]
        m = set_metrics(RC.lac_sets(p_cs, q), y_cs)
        cands.append((abs(m["coverage_marginal"] - (1 - alpha)), m["mean_set_size"], lr))
    cands.sort()
    lr = cands[0][2]
    ps = np.concatenate([p_fc, p_te])
    ss = np.abs(np.concatenate([y_fc, S["test"]["y"]]).astype(float) - ps)
    q = ll.eci_lag_sim(ss, np.zeros(len(ss), int), 1, L, len(p_fc), lr, alpha=alpha)[len(p_fc):, 0]
    return RC.lac_sets(p_te, q), lr


def run_alpha(S, alpha):
    p_cs, y_cs, p_fc, y_fc, p_te, y_te = S["cs"]["p"], S["cs"]["y"], S["fc"]["p"], S["fc"]["y"], S["test"]["p"], S["test"]["y"]
    out = {}
    s, q = RC.static_lac_sets(p_fc, y_fc, p_te, alpha); out["static_lac"] = s
    s, q = RC.mondrian_static_sets(p_fc, y_fc, p_te, alpha); out["static_mondrian_lac"] = s
    s, q, lr = RC.eci_marginal_sets(p_cs, y_cs, p_fc, y_fc, p_te, y_te, alpha); out["eci_sets"] = s
    s, q, lr, fell = RC.mondrian_eci_sets_run(p_cs, y_cs, p_fc, y_fc, p_te, y_te, alpha); out["mondrian_eci"] = s
    s, q = RC.mondrian_recal_sets(p_fc, y_fc, p_te, y_te, alpha, 10000); out["mondrian_recal_w10k"] = s
    return out


def main():
    res = {"model": MODEL, "alphas": ALPHAS, "lags": LAGS, "runs": []}
    for seed in C.SEEDS:
        S = load_streams(seed)
        p_cs, y_cs, p_fc, y_fc, p_te, y_te = S["cs"]["p"], S["cs"]["y"], S["fc"]["p"], S["fc"]["y"], S["test"]["p"], S["test"]["y"]
        year = S["test"]["df"]["approx_real_year"].values
        acuity = S["test"]["df"]["triage_acuity"].values.astype(float)
        n_cal = len(p_fc)
        base = {"seed": seed, "n_cs": len(p_cs), "n_fc": n_cal, "n_test": len(p_te), "pos_cs": int(y_cs.sum()), "pos_fc": int(y_fc.sum()), "pos_test": int(y_te.sum())}
        # ---- main (alpha = 0.10) intervals
        lr_int = RC.select_lr_marginal_interval(p_cs, y_cs, C.ALPHA)
        lo, hi = static_interval(p_fc, y_fc.astype(float), p_te)
        intervals = {"static": interval_metrics(lo, hi, y_te)}
        (lo, hi), _q = RC.eci_interval(p_fc, y_fc, p_te, y_te, C.ALPHA, lr_int)
        intervals["eci"] = interval_metrics(lo, hi, y_te)
        intervals["eci"]["lr"] = lr_int
        for W in (10000, 40000):
            lo, hi = RC.recal_interval_w(p_fc, y_fc, p_te, y_te, C.ALPHA, W)
            intervals[f"recal_w{W // 1000}k"] = interval_metrics(lo, hi, y_te)
        yr = pd.factorize(year, sort=True)[0]
        base["year_block_n"] = [int((yr == k).sum()) for k in np.unique(yr)]
        lo, hi = static_interval(p_fc, y_fc.astype(float), p_te)
        base["static_interval_year_coverage"] = [float(((y_te >= lo) & (y_te <= hi))[yr == k].mean()) for k in np.unique(yr)]
        (lo, hi), _q = RC.eci_interval(p_fc, y_fc, p_te, y_te, C.ALPHA, lr_int)
        base["eci_interval_year_coverage"] = [float(((y_te >= lo) & (y_te <= hi))[yr == k].mean()) for k in np.unique(yr)]
        # ---- alpha sweep with sets and operating characteristics
        sweep = {}
        for alpha in ALPHAS:
            sets = run_alpha(S, alpha)
            sweep[str(alpha)] = {m: {"metrics": set_metrics(s, y_te), "operating": RC.operating_characteristics(s, y_te)} for m, s in sets.items()}
        sweep["class1_calibration_positives"] = int(y_fc.sum())
        sweep["class1_quantile_feasible"] = {str(a): RC.feasibility(int(y_fc.sum()), a) for a in ALPHAS}
        # ---- label delay (alpha = 0.10): intervals and sets
        lag = {}
        for L in LAGS:
            d = {}
            lr_m = ll.select_lr_marginal(p_cs, y_cs, L)
            sp = np.concatenate([p_fc, p_te]); sy = np.concatenate([y_fc, y_te]).astype(int); sc = np.abs(sy - sp); z1 = np.zeros(len(sp), int)
            q = ll.eci_lag_sim(sc, z1, 1, L, n_cal, lr_m)[:, 0]
            lo, hi = ll.marginal_interval(sp, q)
            d["eci_interval"] = interval_metrics(lo[n_cal:], hi[n_cal:], y_te)
            lo, hi = ll.recal_interval_lag(sp, sy.astype(float), n_cal, 10000, ll.RECAL_R, L)
            d["recal_w10k_interval"] = interval_metrics(lo, hi, y_te)
            s, lr_s = eci_sets_lag(S, C.ALPHA, L)
            d["eci_sets"] = set_metrics(s, y_te)
            lr_mo, _fell = ll.select_lr_mondrian(p_cs, y_cs, L)
            q2 = ll.eci_lag_sim(np.abs(sy - sp), sy, 2, L, n_cal, lr_mo)[n_cal:]
            d["mondrian_eci"] = set_metrics(RC._sets_from_q2(p_te, q2), y_te, q2)
            s, _q = ll.recal_sets_lag(sp, sy, n_cal, 10000, ll.RECAL_R, L)
            d["mondrian_recal_w10k"] = set_metrics(np.asarray(s, bool), y_te)
            lag[str(L)] = d
        # ---- controlled drift
        drift = {}
        lr_set = RC.select_lr_marginal_sets(p_cs, y_cs, C.ALPHA)
        for scn in e4.SCENARIOS:
            pd_, yd_, cp = e4.apply_drift(p_te, y_te, acuity, scn)
            sp = np.concatenate([p_fc, pd_]); sy = np.concatenate([y_fc, yd_]).astype(float)
            outm = {}
            lo, hi = static_interval(p_fc, y_fc.astype(float), pd_); outm["static"] = (lo, hi)
            outm["recal_w10k"] = RC.recal_interval_w(p_fc, y_fc, pd_, yd_, C.ALPHA, 10000)
            (lo, hi), _q = RC.eci_interval(p_fc, y_fc, pd_, yd_, C.ALPHA, lr_int); outm["eci_interval"] = (lo, hi)
            drift[scn] = {}
            for m, (lo, hi) in outm.items():
                cov = (yd_ >= lo) & (yd_ <= hi)
                status, rec = e4.recovery(cov, cp)
                drift[scn][m] = {"cov_pre": float(cov[:cp].mean()), "cov_post": float(cov[cp:].mean()), "cov1_post": float(cov[cp:][yd_[cp:] == 1].mean()),
                                 "width_pre": float((hi - lo)[:cp].mean()), "width_post": float((hi - lo)[cp:].mean()), "recovery_status": status,
                                 "recovery_arrivals": None if np.isnan(rec) else rec}
        res["runs"].append({**base, "intervals": intervals, "sweep": sweep, "lag": lag, "drift": drift})
        print("seed", seed, {k: round(v["coverage"], 4) for k, v in intervals.items()}, {k: round(v["width"], 4) for k, v in intervals.items()}, flush=True)
    C.dump(res, "phase2", OUT_NAME)


if __name__ == "__main__":
    main()
