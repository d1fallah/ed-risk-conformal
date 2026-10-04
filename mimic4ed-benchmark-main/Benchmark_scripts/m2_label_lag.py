"""
M2 -- Label lag: ECI / Mondrian-ECI vs. periodic recalibration when outcomes arrive late
==============================================================================
Executes M2_LabelLag_PreRegistration.md, written before this script existed. Read it
first: the lag grid, the realistic band, the methods, the validity definitions, the
0.02 tie tolerance and the decision rule are fixed there and restated here, not tuned.

A lag of L arrivals means the label of arrival j becomes available immediately before
arrival j+L+1 is predicted. The final-conformal-calibration fold is historical, fully
resolved data; lag applies to the test stream. L = None means no label ever arrives
during deployment.

ECI's update recursion is the vendored one (Error-quantified-Conformal-Inference/
core/methods.py::ECI). `eci_lag_sim` reproduces it event by event so that (i) lag can
be applied to the test stream only, (ii) one stream per class is supported, and (iii)
the proportional learning-rate scale is computed from RESOLVED scores only (the vendored
`ahead` mode uses the last 5 scores including unresolved ones). It is verified against
the vendored code before any lagged result is printed (T8, T9, T13).

No patient-cluster bootstrap (invalid for online/recalibrating methods): uncertainty is
the across-seed SD plus descriptive stream-block (K=10) and calendar-year coverage.
"""
import os, sys, json, math, warnings
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CACHE_DIR = os.path.join(DATA_DIR, "_lag_cache")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_label_lag.csv")
OUT_BLOCKS = os.path.join(DATA_DIR, "m2_v2_label_lag_blocks.csv")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_label_lag_verdict.json")
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, DEVICE, _sorted_bounds
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
)
from m2_conformal_v2 import get_predictions_v2, SEED_DIRS_V2, FINAL_SEEDS
from m2_mondrian_conformal import mondrian_lac_sets, mondrian_eci_sets, eci_method
from m2_mondrian_eci_pilot import set_metrics, PRIMARY_MODEL, LR_GRID
from m2_mondrian_eci_comparators import recal_sets, _class_threshold
from m2_eci_vs_recalibration import interval_from_radius, static_interval, recal_interval, summarize

ALPHA = 0.1
RECAL_R = 2000
WINDOWS = {"10k": 10000, "40k": 40000}
LAGS = [0, 60, 240, 1000, 4000, 16000, None]        # None = labels never arrive (pre-registered grid)
REALISTIC = [60, 240, 1000]                          # pre-registered decision band
VALID_MIN = 0.89
TIE_TOL = 0.02
SAT_GUARD = 0.25
N_BLOCKS = 10
T_BURNIN = 5


def lag_name(L):
    return "inf" if L is None else str(L)


# --------------------------------------------------------------------------- simulator
def eci_lag_sim(scores, g, G, L, lag_start, lr, alpha=ALPHA, T_burnin=T_BURNIN, prop=True):
    """Event-driven ECI with delayed feedback. Returns q_at, shape (n, G): the radius of each
    stream IN FORCE when arrival t is predicted (before arrival t's label is known).

    Item j (stream g[j], score scores[j]) is resolved immediately for j < lag_start, and at
    the moment arrival j+L+1 is about to be predicted for j >= lag_start (never if L is None).
    The update is the vendored ECI recursion (c=1), applied to the stream of the resolved
    item, using the radius that was in force when that item arrived."""
    n = len(scores)
    s_l = [float(x) for x in scores]
    g_l = [int(x) for x in g]
    q = [0.0] * G
    qt = [0.0] * G
    hist = [[] for _ in range(G)]
    qarr = [0.0] * n
    q_at = np.zeros((n, G))
    one_m_a = 1.0 - alpha
    for t in range(n):
        if t <= lag_start:
            j = t - 1
        elif L is None:
            j = -1
        else:
            j = t - 1 - L
            if j < lag_start:
                j = -1
        if j >= 0:
            k = g_l[j]; s = s_l[j]; qj = qarr[j]
            h = hist[k]
            lr_t = lr * (max(h) - min(h)) if (prop and h) else lr
            grad = alpha if qj >= s else -one_m_a
            d = s - qj
            e = math.exp(-abs(d))       # sigma'(d) is even: identical to the vendored form, without overflow when the radius winds up
            eq = d * e / ((1.0 + e) ** 2)
            qt[k] -= lr_t * grad
            q[k] = qt[k] + lr_t * eq
            h.append(s)
            if len(h) > T_burnin:
                h.pop(0)
        q_at[t] = q
        qarr[t] = q[g_l[t]]
    return q_at


def marginal_interval(p, q):
    lo, hi = _sorted_bounds(p - q, p + q)
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def select_lr_marginal(p_cs, y_cs, L):
    """run_eci_4way stage 1, with the same lag applied inside the self-contained selection run."""
    y = np.asarray(y_cs, float); sc = np.abs(y - p_cs)
    cands = []
    for lr in (0.01, 0.05, 0.1):
        q = eci_lag_sim(sc, np.zeros(len(sc), int), 1, L, 0, lr)[:, 0]
        lo, hi = marginal_interval(p_cs, q)
        cands.append({"lr": lr, "gap": abs(float(np.mean((y >= lo) & (y <= hi))) - 0.9), "w": float(np.mean(hi - lo))})
    cands.sort(key=lambda c: (c["gap"], c["w"]))
    return cands[0]["lr"]


def sets_from_q(p, q2):
    return np.column_stack([np.abs(0 - p) <= q2[:, 0], np.abs(1 - p) <= q2[:, 1]])


def select_lr_mondrian(p_cs, y_cs, L):
    """m2_mondrian_eci_confirmatory.select_lr_guarded with lag applied to the evaluated half."""
    y = np.asarray(y_cs, int); half = len(p_cs) // 2
    sc = np.abs(y - p_cs); trace = []
    for lr in LR_GRID:
        q = eci_lag_sim(sc, y, 2, L, half, lr)
        s = sets_from_q(p_cs[half:], q[half:])
        m = set_metrics(s, y[half:], q[half:])
        trace.append({"lr": lr, "c1": m["coverage_class1"], "size": m["mean_set_size"], "rej": m["q1_saturated_frac"] > SAT_GUARD})
    surv = [t for t in trace if not t["rej"]]
    if not surv:
        return min(LR_GRID), True
    return min(surv, key=lambda t: (abs(t["c1"] - (1 - ALPHA)), t["size"]))["lr"], False


# --------------------------------------------------------------------------- recalibration under lag
def _bound(start, n_cal, L):
    return start if L == 0 else max(n_cal, start - L)


def recal_interval_lag(p, y, n_cal, W, R, L):
    """Marginal recalibration; window ends at the latest RESOLVED arrival. L=None -> caller uses static."""
    n = len(p)
    lo = np.zeros(n - n_cal); hi = np.zeros(n - n_cal)
    for start in range(n_cal, n, R):
        b = _bound(start, n_cal, L)
        w = slice(max(0, b - W), b)
        q = _class_threshold(np.abs(y[w] - p[w]), ALPHA)
        end = min(start + R, n)
        lo[start - n_cal:end - n_cal], hi[start - n_cal:end - n_cal] = interval_from_radius(p[start:end], q)
    return lo, hi


def recal_sets_lag(p, y, n_cal, W, R, L):
    p = np.asarray(p, float); y = np.asarray(y, int)   # same casts as recal_sets
    n = len(p)
    sets = np.zeros((n - n_cal, 2), bool); q_all = np.zeros((n - n_cal, 2))
    for start in range(n_cal, n, R):
        b = _bound(start, n_cal, L)
        lo_ = max(0, b - W); pw, yw = p[lo_:b], y[lo_:b]
        q0 = _class_threshold(pw[yw == 0], ALPHA); q1 = _class_threshold(1.0 - pw[yw == 1], ALPHA)
        end = min(start + R, n); seg = p[start:end]
        sets[start - n_cal:end - n_cal, 0] = seg <= q0
        sets[start - n_cal:end - n_cal, 1] = (1.0 - seg) <= q1
        q_all[start - n_cal:end - n_cal] = (q0, q1)
    return sets, q_all


# --------------------------------------------------------------------------- correctness tests
def run_tests(p_cs, y_cs, p_fc, y_fc, p_te, y_te):
    sp = np.concatenate([p_fc, p_te]); sy = np.concatenate([y_fc, y_te]).astype(int); n_cal = len(p_fc); n = len(sp)
    sc = np.abs(sy - sp); z = np.zeros(n, int)
    lr = 0.05
    # T8: L=0 marginal == vendored ECI
    v = np.asarray(eci_method(scores=sc, alpha=ALPHA, lr=lr, T_burnin=T_BURNIN, ahead=1, proportional_lr=True)["q"], float)
    m = eci_lag_sim(sc, z, 1, 0, n_cal, lr)[:, 0]
    assert np.max(np.abs(v - m)) < 1e-9, f"T8 FAILED max|dq|={np.max(np.abs(v - m)):.3e}"
    print(f"T8  marginal simulator at L=0 == vendored ECI (max|dq|={np.max(np.abs(v - m)):.2e}): PASS")
    # T9: L=0 per-class == mondrian_eci_sets radii
    _, q_ref, _ = mondrian_eci_sets(sp, sy, n_cal=n_cal, lr=lr)
    q_sim = eci_lag_sim(sc, sy, 2, 0, n_cal, lr)[n_cal:]
    # mondrian_eci_sets clips its class-radius index to n_k-1 (the vendored ECI loop never writes the
    # update from the final class-k point), so rows AFTER a class's last arrival reuse a radius built from
    # one fewer point. The simulator applies that update. Identity is asserted on every row up to and
    # including each class's last arrival; the excluded rows are counted and reported (Deviation 21).
    n_excl = 0
    for k in (0, 1):
        last = int(np.where(sy == k)[0].max()) - n_cal
        keep = np.arange(len(q_ref)) <= last
        assert np.max(np.abs(q_ref[keep, k] - q_sim[keep, k])) < 1e-9, f"T9 FAILED (class {k})"
        n_excl += int((~keep).sum())
    assert n_excl <= 5, f"T9: unexpectedly many tail rows excluded ({n_excl})"
    print(f"T9  per-class simulator at L=0 == mondrian_eci_sets on all rows up to each class's last arrival "
          f"({n_excl} tail row(s) of {len(q_ref)} excluded: known end-of-stream clip in mondrian_eci_sets): PASS")
    # T10: causality under lag -- flipping labels not yet resolved at time t leaves q up to t unchanged
    rng = np.random.RandomState(11)
    for L in (60, 1000):
        t = int(n_cal + rng.randint(2000, len(p_te) - 8000))
        y2 = sy.copy(); y2[t - L:] = 1 - y2[t - L:]         # arrivals t-L .. are unresolved when t is predicted
        a = eci_lag_sim(np.abs(sy - sp), sy, 2, L, n_cal, lr)
        b = eci_lag_sim(np.abs(y2 - sp), y2, 2, L, n_cal, lr)
        assert np.array_equal(a[:t + 1], b[:t + 1]), f"T10 FAILED (L={L}): q at t depends on an unresolved label"
        assert not np.array_equal(a[t + 5000:], b[t + 5000:]) or L == 1000, "T10 sanity: flip must eventually matter"
    print("T10 lagged radius uses only resolved labels (unresolved-label flip leaves q unchanged): PASS")
    # T11: recal at L=0 == existing; window over whole fold, no refresh == static
    a = recal_interval_lag(sp, sy.astype(float), n_cal, 10000, RECAL_R, 0)
    b = recal_interval(sp, sy.astype(float), n_cal, 10000, RECAL_R)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "T11a FAILED"
    c = recal_sets_lag(sp, sy, n_cal, 10000, RECAL_R, 0); d = recal_sets(sp, sy, n_cal, 10000, RECAL_R)
    assert np.array_equal(c[0], d[0]) and np.array_equal(c[1], d[1]), "T11b FAILED"
    e = recal_sets_lag(sp, sy, n_cal, n_cal, len(p_te) + 1, 240)[0]
    assert np.array_equal(e, mondrian_lac_sets(p_fc, y_fc, p_te)[0]), "T11c FAILED"
    lo, hi = recal_interval_lag(sp, sy.astype(float), n_cal, n_cal, len(p_te) + 1, 240)
    slo, shi = static_interval(p_fc, y_fc.astype(float), p_te)
    assert np.array_equal(lo, slo) and np.array_equal(hi, shi), "T11d FAILED"
    # lagged recal must not read unresolved labels: flip them, thresholds unchanged
    for L in (240, 4000):
        yb = sy.copy(); yb[n_cal:] = 1 - yb[n_cal:]          # flip ALL test labels
        s1 = recal_sets_lag(sp, sy, n_cal, 10000, RECAL_R, L)[1]; s2 = recal_sets_lag(sp, yb, n_cal, 10000, RECAL_R, L)[1]
        blocks_unchanged = L // RECAL_R + 1                  # refresh at s uses test labels only if s - L > n_cal, i.e. block index i*R > L
        assert np.array_equal(s1[:blocks_unchanged * RECAL_R], s2[:blocks_unchanged * RECAL_R]), "T11e FAILED"
    print("T11 lagged recalibration: L=0 identical to existing, no-refresh identical to static, no unresolved-label reads: PASS")
    # T12: L=None -> q constant across the test window
    q = eci_lag_sim(sc, sy, 2, None, n_cal, lr)[n_cal:]
    assert np.all(q == q[0]), "T12 FAILED"
    print("T12 L=inf: ECI radius frozen across the test window: PASS")
    # T13: proportional_lr off, lag over whole stream == vendored ahead=L+1
    for L in (60, 1000):
        v = np.asarray(eci_method(scores=sc, alpha=ALPHA, lr=lr, T_burnin=T_BURNIN, ahead=L + 1, proportional_lr=False)["q"], float)
        m = eci_lag_sim(sc, z, 1, L, 0, lr, prop=False)[:, 0]
        assert np.max(np.abs(v - m)) < 1e-9, f"T13 FAILED (L={L}) max|dq|={np.max(np.abs(v - m)):.3e}"
    print("T13 simulator == vendored ECI(ahead=L+1) when proportional_lr is off: PASS")


# --------------------------------------------------------------------------- predictions cache
def get_all_predictions():
    cache = os.path.join(CACHE_DIR, "preds.npz")
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev = load_dev_pool_split_v2()
    cs_ids = set(dev["calibration_selection"]["subject_ids"].tolist())
    fc_ids = set(dev["final_conformal_calibration"]["subject_ids"].tolist())
    df_cs = sort_chronologically(df_train[df_train["subject_id"].isin(cs_ids)])
    df_fc = sort_chronologically(df_train[df_train["subject_id"].isin(fc_ids)])
    df_te = sort_chronologically(df_test)
    year = df_te["approx_real_year"].values
    # The cache is only valid if no v2 checkpoint or tuned config is newer than it (a re-trained pipeline
    # must never be scored from stale predictions).
    import glob
    dep = [f for f in glob.glob(os.path.join(DATA_DIR, "checkpoints", "*_v2_seed*.pt")) if "_hz" not in os.path.basename(f)]
    dep += glob.glob(os.path.join(DATA_DIR, "m2_v2_*_tuned_config*.json"))
    stamp = max(os.path.getmtime(f) for f in dep) if dep else 0.0
    if os.path.exists(cache) and os.path.getmtime(cache) > stamp:
        z = np.load(cache, allow_pickle=True)
        return z["preds"].item(), z["y"].item(), year
    if os.path.exists(cache):
        print("prediction cache is older than the latest checkpoint/config: rebuilding", flush=True)
    Xc, Mc, Dc, Pc, yc = prepare_temporal_tensors_v2(df_cs, "train_v2", device=DEVICE)
    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_fc, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_te, "test_v2", device=DEVICE)
    Sc = torch.tensor(prepare_static_features_v2(df_cs), device=DEVICE)
    Sf = torch.tensor(prepare_static_features_v2(df_fc), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_te), device=DEVICE)
    y = {"cs": yc.cpu().numpy().astype(int), "fc": yf.cpu().numpy().astype(int), "te": yt.cpu().numpy().astype(int)}
    preds = {}
    for model in SEED_DIRS_V2:
        for seed in FINAL_SEEDS:
            preds[(model, seed)] = {
                "cs": get_predictions_v2(model, seed, Xc, Mc, Dc, Pc, S=Sc),
                "fc": get_predictions_v2(model, seed, Xf, Mf, Df, Pf, S=Sf),
                "te": get_predictions_v2(model, seed, Xt, Mt, Dt, Pt, S=St)}
    np.savez(cache, preds=np.array(preds, dtype=object), y=np.array(y, dtype=object))
    return preds, y, year


# --------------------------------------------------------------------------- experiment
def block_rows(model, seed, L, method, fam, cov, size, y, year, rows):
    n = len(y)
    for scheme, b in {"stream10": np.minimum((np.arange(n) * N_BLOCKS) // n, N_BLOCKS - 1),
                      "year": pd.factorize(year, sort=True)[0]}.items():
        for blk in np.unique(b):
            m = b == blk; pos = m & (y == 1); neg = m & (y == 0)
            rows.append({"model": model, "seed": seed, "lag": lag_name(L), "method": method, "family": fam,
                         "scheme": scheme, "block": int(blk), "n": int(m.sum()), "n_pos": int(pos.sum()),
                         "cov": float(cov[m].mean()), "cov0": float(cov[neg].mean()) if neg.any() else np.nan,
                         "cov1": float(cov[pos].mean()) if pos.any() else np.nan, "size": float(size[m].mean())})


def main():
    print("Loading predictions...")
    preds, y, year = get_all_predictions()
    yt = y["te"]; yf = y["fc"]; yc = y["cs"]
    m0 = (PRIMARY_MODEL, FINAL_SEEDS[0]); pr = preds[m0]
    print("Correctness tests (must all pass before any lagged result is printed):")
    run_tests(pr["cs"], yc, pr["fc"], yf, pr["te"], yt)

    rows, brows = [], []
    for model in SEED_DIRS_V2:
        print(f"\n{'='*60}\n{model}\n{'='*60}", flush=True)
        for seed in FINAL_SEEDS:
            p = preds[(model, seed)]
            p_cs, p_fc, p_te = p["cs"], p["fc"], p["te"]
            sp = np.concatenate([p_fc, p_te]); sy = np.concatenate([yf, yt]).astype(int); n_cal = len(p_fc)
            sc = np.abs(sy - sp); z1 = np.zeros(len(sp), int)
            lr0_m = select_lr_marginal(p_cs, yc, 0)
            # references not affected by lag
            iv_static = static_interval(p_fc, yf.astype(float), p_te)
            sets_A = mondrian_lac_sets(p_fc, yf, p_te)[0]
            for L in LAGS:
                tag = lag_name(L)
                out_m, out_c = {}, {}
                out_m["M1"] = (iv_static, np.nan)
                # L=inf: no feedback exists to select on either, so the lag-0 choice is used (frozen ECI)
                lr_m = lr0_m if L is None else select_lr_marginal(p_cs, yc, L)
                q = eci_lag_sim(sc, z1, 1, L, n_cal, lr_m)[:, 0]
                out_m["M2"] = (marginal_interval(sp, q)[0][n_cal:], marginal_interval(sp, q)[1][n_cal:]), lr_m
                q = eci_lag_sim(sc, z1, 1, L, n_cal, lr0_m)[:, 0]
                out_m["M2fix"] = (marginal_interval(sp, q)[0][n_cal:], marginal_interval(sp, q)[1][n_cal:]), lr0_m
                if L is not None:
                    va = np.asarray(eci_method(scores=sc, alpha=ALPHA, lr=lr_m, T_burnin=T_BURNIN, ahead=L + 1,
                                                proportional_lr=True)["q"], float)
                    lo, hi = marginal_interval(sp, va); out_m["M2vend"] = ((lo[n_cal:], hi[n_cal:]), lr_m)
                for wid, W in WINDOWS.items():
                    mid = "M3" if wid == "10k" else "M4"
                    out_m[mid] = ((iv_static, np.nan) if L is None else (recal_interval_lag(sp, sy.astype(float), n_cal, W, RECAL_R, L), np.nan))
                out_m = {k: (v[0], v[1]) if isinstance(v[0], tuple) else v for k, v in out_m.items()}

                lr_b, ft_b = select_lr_mondrian(p_cs, yc, 0 if L is None else L)
                qb = eci_lag_sim(sc, sy, 2, L, n_cal, lr_b)
                out_c["A"] = ((sets_A, None), np.nan)
                out_c["B"] = ((sets_from_q(p_te, qb[n_cal:]), qb[n_cal:]), lr_b)
                for wid, W in WINDOWS.items():
                    cid = "C1" if wid == "10k" else "C2"
                    out_c[cid] = ((sets_A, None), np.nan) if L is None else (recal_sets_lag(sp, sy, n_cal, W, RECAL_R, L), np.nan)

                for mid, (iv, lr_used) in out_m.items():
                    lo, hi = iv[0], iv[1]
                    met, cov, wid_ = summarize(lo, hi, yt.astype(float))
                    rows.append({"model": model, "seed": seed, "lag": tag, "family": "marginal", "method": mid, "lr": lr_used,
                                 "guard_fell_through": False, "coverage": met["coverage"], "coverage_class0": met["coverage_class0"],
                                 "coverage_class1": met["coverage_class1"], "size": met["width"],
                                 "informative_class1": np.nan, "q1_saturated_frac": np.nan})
                    block_rows(model, seed, L, mid, "marginal", cov, wid_, yt, year, brows)
                for cid, ((sets, q), lr_used) in out_c.items():
                    mm = set_metrics(sets, yt, q)
                    rows.append({"model": model, "seed": seed, "lag": tag, "family": "class", "method": cid, "lr": lr_used,
                                 "guard_fell_through": ft_b if cid == "B" else False, "coverage": mm["coverage_marginal"],
                                 "coverage_class0": mm["coverage_class0"], "coverage_class1": mm["coverage_class1"],
                                 "size": mm["mean_set_size"], "informative_class1": mm["informative_class1"],
                                 "q1_saturated_frac": mm.get("q1_saturated_frac", np.nan)})
                    block_rows(model, seed, L, cid, "class", sets[np.arange(len(yt)), yt], sets.sum(1), yt, year, brows)
            print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(brows).to_csv(OUT_BLOCKS, index=False)
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_BLOCKS}")

    # ------------------------------------------------------------------ pre-registered verdicts
    def verdict(g, fam, L):
        """g: 5-seed-mean table indexed by method for one (model, lag, family)."""
        if fam == "marginal":
            valid = lambda m: bool(g.loc[m, "coverage"] >= VALID_MIN)
            eci, recs = "M2", ["M3", "M4"]
        else:
            valid = lambda m: bool(g.loc[m, "coverage_class0"] >= VALID_MIN and g.loc[m, "coverage_class1"] >= VALID_MIN)
            eci, recs = "B", ["A", "C1", "C2"]
        sz = g["size"]
        vr = [m for m in recs if valid(m)]
        if not valid(eci):
            return ("ECI INVALID (recalibration valid)" if vr else "BOTH INVALID"), "against"
        if any(sz[m] <= sz[eci] + TIE_TOL for m in vr):
            return "ECI NOT NEEDED", "against"
        if not vr:
            return "ECI SUPPORTED - sole valid method", "for"
        return "ECI SUPPORTED - efficient", "for"

    result = {"lags": [lag_name(L) for L in LAGS], "realistic_band": REALISTIC, "models": {}}
    for model in SEED_DIRS_V2:
        result["models"][model] = {}
        for fam in ("marginal", "class"):
            result["models"][model][fam] = {}
            for L in LAGS:
                g = df[(df.model == model) & (df.family == fam) & (df.lag == lag_name(L))].groupby("method")[
                    ["coverage", "coverage_class0", "coverage_class1", "size", "informative_class1"]].mean()
                v, direction = verdict(g, fam, L)
                result["models"][model][fam][lag_name(L)] = {"verdict": v, "direction": direction,
                                                              "table": g.round(4).to_dict("index")}
    prim = result["models"][PRIMARY_MODEL]
    repl = [m for m in SEED_DIRS_V2 if m != PRIMARY_MODEL]
    changed = []
    for L in REALISTIC:
        for fam in ("marginal", "class"):
            if prim[fam][str(L)]["direction"] == "for" and any(result["models"][m][fam][str(L)]["direction"] == "for" for m in repl):
                changed.append((L, fam))
    if changed:
        overall = "THE NEGATIVE RESULT DOES NOT SURVIVE LABEL LAG at " + ", ".join(f"L={a} ({b})" for a, b in changed)
    elif any(prim[f][str(L)]["direction"] == "for" for L in REALISTIC for f in ("marginal", "class")):
        overall = "MIXED (ECI supported on the primary model only at some realistic lag; not replicated)"
    else:
        overall = "THE NEGATIVE RESULT IS ROBUST TO REALISTIC LAG"
    result["overall"] = overall
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    for model in SEED_DIRS_V2:
        print(f"\n{'='*78}\n{model}   [{'PRIMARY' if model == PRIMARY_MODEL else 'replication'}]\n{'='*78}")
        for fam, ecim, recs in (("marginal", "M2", ["M1", "M3", "M4"]), ("class", "B", ["A", "C1", "C2"])):
            print(f"-- {fam} family: {'coverage / width' if fam == 'marginal' else 'cov0 / cov1 / |C|'}  (ECI={ecim})")
            for L in LAGS:
                t = result["models"][model][fam][lag_name(L)]["table"]
                def cell(m):
                    r = t[m]
                    return (f"{r['coverage']:.4f}/{r['size']:.3f}" if fam == "marginal"
                            else f"{r['coverage_class0']:.3f}/{r['coverage_class1']:.3f}/{r['size']:.3f}")
                print(f"  L={lag_name(L):>6s}  {ecim}: {cell(ecim):22s}" + "  ".join(f"{m}: {cell(m):22s}" for m in recs)
                      + f"  -> {result['models'][model][fam][lag_name(L)]['verdict']}")
    print(f"\nOVERALL: {overall}\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
