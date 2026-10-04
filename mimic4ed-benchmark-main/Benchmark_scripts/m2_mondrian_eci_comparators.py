"""
M2 -- Decisive comparators for Mondrian-ECI (Part 4 of M2_MondrianECI_PreRegistration.md)
==============================================================================
Comparators, thresholds, the validity definition and the decision rule were
written down BEFORE this script was first run (pre-registration Part 4,
timestamped 2026-09-20 02:00). Read that document first; nothing here may be
tuned against its outcome.

  A   corrected static Mondrian-LAC                     (anchor)
  B   Mondrian-ECI, guarded lr                          (construction under test)
  C1/C2 static Mondrian + periodic recalibration        (R=2000 arrivals; W=10000 / 40000)
  D   static score-bin Mondrian-LAC                     (10 quantile bins of the model's own p)
  E   online score-bin ECI, guarded lr                  (one ECI stream per bin)

Every method is deployable: at position t it uses only labels that arrived
before t. Labels are assumed immediate (lag 0) for all methods alike.

Correctness tests run before any metric is computed:
  T3  C with a window covering the whole calibration fold and no further
      recalibration reproduces A bit-for-bit (isolates the recalibration logic).
  T4  E is causal: flipping every label at positions >= t leaves the radii in
      force at positions <= t unchanged.
  T5  D attains ~0.90 coverage on synthetic exchangeable data (sanity of the
      bin-threshold arithmetic, independent of this cohort).

Inference: NO patient-cluster bootstrap (invalid for online/recalibrating
methods). Uncertainty = across-seed SD plus contiguous stream blocks
(K=10) and the 4 real calendar years, with within-block paired differences
against B. Stream blocks are NOT calendar periods.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_comparators.csv")
OUT_BLOCKS = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_comparators_blocks.csv")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_comparators_verdict.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
)
from m2_ablations import eci_method
from m2_mondrian_conformal import mondrian_lac_sets, mondrian_eci_sets
from m2_conformal_v2 import get_predictions_v2, SEED_DIRS_V2, FINAL_SEEDS
from m2_mondrian_eci_pilot import set_metrics, PRIMARY_MODEL, LR_GRID, ALPHA
from m2_mondrian_eci_confirmatory import select_lr_guarded, SAT_GUARD

# ---- pre-registered constants (Part 4) ------------------------------------
RECAL_R = 2000
RECAL_WINDOWS = {"C1": 10000, "C2": 40000}
N_BINS = 10
VALID_MIN_COVERAGE = 0.89
TIE_TOL = 0.02
N_STREAM_BLOCKS = 10
METHODS = ["A", "B", "C1", "C2", "D", "E"]
METHOD_NAMES = {"A": "static Mondrian-LAC", "B": "Mondrian-ECI (online, label-cond.)",
                "C1": "static Mondrian + recal (W=10k)", "C2": "static Mondrian + recal (W=40k)",
                "D": "static score-bin Mondrian", "E": "online score-bin ECI"}


def _class_threshold(scores_k, alpha):
    n_k = len(scores_k)
    if n_k == 0:
        return 1.0  # fail-open, matching mondrian_lac_sets
    level = min(1.0, np.ceil((n_k + 1) * (1 - alpha)) / n_k)
    return float(np.quantile(scores_k, level, method="higher"))


def recal_sets(p_stream, y_stream, n_cal, W, R, alpha=ALPHA):
    """C: static label-conditional thresholds recomputed every R arrivals from the
    most recent W labeled arrivals. Scores are the corrected LAC scores
    (s_1 = 1-p, s_0 = p). Returns (sets_test, q_test) where q_test is (n_test, 2)."""
    p = np.asarray(p_stream, float); y = np.asarray(y_stream, int)
    n = len(p)
    sets_all = np.zeros((n - n_cal, 2), dtype=bool)
    q_all = np.zeros((n - n_cal, 2), dtype=float)
    for start in range(n_cal, n, R):
        lo = max(0, start - W)
        pw, yw = p[lo:start], y[lo:start]          # strictly BEFORE `start`
        q0 = _class_threshold(pw[yw == 0], alpha)              # s_0 = p on class-0 points
        q1 = _class_threshold(1.0 - pw[yw == 1], alpha)        # s_1 = 1-p on class-1 points
        end = min(start + R, n)
        seg = p[start:end]
        sets_all[start - n_cal:end - n_cal, 0] = seg <= q0
        sets_all[start - n_cal:end - n_cal, 1] = (1.0 - seg) <= q1
        q_all[start - n_cal:end - n_cal] = (q0, q1)
    return sets_all, q_all


def bin_edges(p_ref, n_bins=N_BINS):
    """Interior quantile edges from a reference (non-test) fold."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(p_ref, qs))


def static_bin_sets(p_cal, y_cal, p_test, edges, alpha=ALPHA):
    """D: per-bin threshold on |y-p| from the calibration fold; set = {k : |k-p| <= q_g}."""
    g_cal = np.searchsorted(edges, p_cal, side="right")
    g_test = np.searchsorted(edges, p_test, side="right")
    s_cal = np.abs(y_cal - p_cal)
    q = np.array([_class_threshold(s_cal[g_cal == g], alpha) for g in range(len(edges) + 1)])
    q_t = q[g_test]
    return np.column_stack([np.abs(0 - p_test) <= q_t, np.abs(1 - p_test) <= q_t]), np.column_stack([q_t, q_t])


def online_bin_sets(p_stream, y_stream, n_cal, edges, lr, alpha=ALPHA, T_burnin=5):
    """E: one ECI stream per score bin. The bin of point t is a function of p_t alone
    (known at prediction time). q at t is stream g(t)'s radius built from the group-g
    points strictly before t (exclusive cumulative count, same causal mapping as
    mondrian_eci_sets)."""
    p = np.asarray(p_stream, float); y = np.asarray(y_stream, float)
    g = np.searchsorted(edges, p, side="right")
    scores = np.abs(y - p)
    q_t = np.ones(len(p), dtype=float)
    for grp in range(len(edges) + 1):
        mask = g == grp
        n_g = int(mask.sum())
        if n_g < 2:
            continue
        idx = np.where(mask)[0]
        qs = np.asarray(eci_method(scores=scores[idx], alpha=alpha, lr=lr, T_burnin=T_burnin,
                                   ahead=1, proportional_lr=True)["q"], float)
        j = np.clip(np.cumsum(mask) - mask.astype(int), 0, n_g - 1)
        q_t[idx] = qs[j[idx]]
    sets = np.column_stack([np.abs(0 - p) <= q_t, np.abs(1 - p) <= q_t])
    return sets[n_cal:], np.column_stack([q_t, q_t])[n_cal:], q_t


def select_lr_bin_guarded(p_cal_sel, y_cal_sel):
    half = len(p_cal_sel) // 2
    edges = bin_edges(p_cal_sel[:half])
    trace = []
    for lr in LR_GRID:
        s, q, _ = online_bin_sets(p_cal_sel, y_cal_sel, half, edges, lr)
        m = set_metrics(s, y_cal_sel[half:], q)
        trace.append({"lr": lr, "cov1": m["coverage_class1"], "size": m["mean_set_size"],
                      "sat": m["q1_saturated_frac"], "rej": m["q1_saturated_frac"] > SAT_GUARD})
    surv = [t for t in trace if not t["rej"]]
    if not surv:
        return min(LR_GRID), True
    return min(surv, key=lambda t: (abs(t["cov1"] - (1 - ALPHA)), t["size"]))["lr"], False


# ---- correctness tests -----------------------------------------------------
def run_tests(p_final, y_final, p_test, y_test):
    stream_p = np.concatenate([p_final, p_test]); stream_y = np.concatenate([y_final, y_test])
    n_cal = len(p_final)
    # T3: recalibration that never fires again == static anchor
    a_sets, _ = mondrian_lac_sets(p_final, y_final, p_test)
    c_sets, _ = recal_sets(stream_p, stream_y, n_cal, W=n_cal, R=len(p_test) + 1)
    assert np.array_equal(a_sets, c_sets), "T3 FAILED: recalibration logic diverges from the static anchor"
    print("T3 recal(no further recalibration) == static anchor: PASS")
    # T4: causality of E
    edges = bin_edges(p_final)
    n = len(stream_p); t = int(n * 0.6)
    _, _, qa = online_bin_sets(stream_p, stream_y, 0, edges, 0.05)
    y2 = stream_y.copy(); y2[t:] = 1 - y2[t:]
    _, _, qb = online_bin_sets(stream_p, y2, 0, edges, 0.05)
    assert np.array_equal(qa[:t + 1], qb[:t + 1]) and not np.array_equal(qa[t + 1:], qb[t + 1:]), "T4 FAILED"
    print("T4 online score-bin ECI is causal: PASS")
    # T5: D on synthetic exchangeable data
    rng = np.random.RandomState(3)
    def draw(m):
        yy = (rng.rand(m) < 0.08).astype(int)
        return np.clip(rng.beta(2, 25, m) + 0.45 * yy, 0.001, 0.999), yy
    pc, yc = draw(60000); pt, yt = draw(60000)
    s, _ = static_bin_sets(pc, yc, pt, bin_edges(pc))
    cov = s[np.arange(len(yt)), yt].mean()
    assert abs(cov - 0.90) < 0.01, f"T5 FAILED: static score-bin marginal coverage {cov:.4f}"
    print(f"T5 static score-bin marginal coverage on exchangeable data = {cov:.4f}: PASS")


def block_rows(model, seed, method, sets, y, year, rows):
    n = len(y); covered = sets[np.arange(n), y]; size = sets.sum(1)
    single1 = sets[:, 1] & ~sets[:, 0]
    schemes = {"stream10": np.minimum((np.arange(n) * N_STREAM_BLOCKS) // n, N_STREAM_BLOCKS - 1),
               "year": pd.factorize(year, sort=True)[0]}
    for scheme, b in schemes.items():
        for blk in np.unique(b):
            m = b == blk; pos = m & (y == 1); neg = m & (y == 0)
            rows.append({"model": model, "seed": seed, "method": method, "scheme": scheme, "block": int(blk),
                         "n": int(m.sum()), "n_pos": int(pos.sum()),
                         "cov0": float(covered[neg].mean()) if neg.any() else np.nan,
                         "cov1": float(covered[pos].mean()) if pos.any() else np.nan,
                         "mean_C": float(size[m].mean()),
                         "informative1": float(single1[pos].mean()) if pos.any() else np.nan})


def main():
    print("Loading v2 data...")
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev = load_dev_pool_split_v2()
    cal_sel_ids = set(dev["calibration_selection"]["subject_ids"].tolist())
    final_ids = set(dev["final_conformal_calibration"]["subject_ids"].tolist())
    df_cs = sort_chronologically(df_train[df_train["subject_id"].isin(cal_sel_ids)])
    df_fc = sort_chronologically(df_train[df_train["subject_id"].isin(final_ids)])
    df_te = sort_chronologically(df_test)
    year = df_te["approx_real_year"].values

    Xc, Mc, Dc, Pc, yc = prepare_temporal_tensors_v2(df_cs, "train_v2", device=DEVICE)
    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_fc, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_te, "test_v2", device=DEVICE)
    Sc = torch.tensor(prepare_static_features_v2(df_cs), device=DEVICE)
    Sf = torch.tensor(prepare_static_features_v2(df_fc), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_te), device=DEVICE)
    yc_, yf_, yt_ = (a.cpu().numpy().astype(int) for a in (yc, yf, yt))

    rows, brows = [], []
    tests_done = False
    for model in SEED_DIRS_V2:
        print(f"\n{'='*60}\n{model}\n{'='*60}")
        for seed in FINAL_SEEDS:
            p_cs = get_predictions_v2(model, seed, Xc, Mc, Dc, Pc, S=Sc)
            p_fc = get_predictions_v2(model, seed, Xf, Mf, Df, Pf, S=Sf)
            p_te = get_predictions_v2(model, seed, Xt, Mt, Dt, Pt, S=St)
            if not tests_done:
                run_tests(p_fc, yf_, p_te, yt_); tests_done = True

            stream_p = np.concatenate([p_fc, p_te]); stream_y = np.concatenate([yf_, yt_]); n_cal = len(p_fc)
            edges = bin_edges(p_fc)
            lr_B, ft_B, _ = select_lr_guarded(p_cs, yc_)
            lr_E, ft_E = select_lr_bin_guarded(p_cs, yc_)

            out = {}
            out["A"] = (mondrian_lac_sets(p_fc, yf_, p_te)[0], None)
            out["B"] = mondrian_eci_sets(stream_p, stream_y, n_cal=n_cal, lr=lr_B)[:2]
            for cid, W in RECAL_WINDOWS.items():
                out[cid] = recal_sets(stream_p, stream_y, n_cal, W=W, R=RECAL_R)
            out["D"] = static_bin_sets(p_fc, yf_, p_te, edges)
            s_E, q_E, _ = online_bin_sets(stream_p, stream_y, n_cal, edges, lr_E)
            out["E"] = (s_E, q_E)

            for m_id in METHODS:
                sets, q = out[m_id]
                met = set_metrics(sets, yt_, q)
                rows.append({"model": model, "seed": seed, "method": m_id, "method_name": METHOD_NAMES[m_id],
                             "lr": {"B": lr_B, "E": lr_E}.get(m_id, np.nan),
                             "guard_fell_through": {"B": ft_B, "E": ft_E}.get(m_id, False), **met})
                block_rows(model, seed, m_id, sets, yt_, year, brows)
            print(f"  seed {seed}: lr_B={lr_B} lr_E={lr_E}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(brows).to_csv(OUT_BLOCKS, index=False)
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_BLOCKS}")

    # ---- pre-registered verdict ------------------------------------------
    def verdict_for(model):
        g = df[df.model == model].groupby("method")[["coverage_class0", "coverage_class1", "mean_set_size",
                                                     "informative_class1"]].mean()
        valid = {m: bool(g.loc[m, "coverage_class0"] >= VALID_MIN_COVERAGE and g.loc[m, "coverage_class1"] >= VALID_MIN_COVERAGE)
                 for m in METHODS}
        B_size = g.loc["B", "mean_set_size"]
        cheaper = [m for m in ("A", "C1", "C2", "D") if valid[m] and g.loc[m, "mean_set_size"] <= B_size + TIE_TOL]
        if not valid["B"]:
            v = "B NOT VALID (fails the validity definition)"
        elif cheaper:
            v = f"NOT NEEDED (valid with |C| <= B+{TIE_TOL}: {cheaper})"
        elif valid["E"] and g.loc["E", "mean_set_size"] < B_size - TIE_TOL:
            v = "DOMINATED by online score-bin ECI (E)"
        elif not any(valid[m] for m in METHODS if m != "B"):
            v = "SUPPORTED - sole valid method"
        elif all((not valid[m]) or g.loc[m, "mean_set_size"] > B_size + TIE_TOL for m in METHODS if m != "B"):
            v = "SUPPORTED - efficient (every other valid method has larger sets)"
        else:
            v = "INCONCLUSIVE"
        return v, g, valid

    result = {}
    for model in SEED_DIRS_V2:
        v, g, valid = verdict_for(model)
        result[model] = {"verdict": v, "valid": valid, "table": g.round(4).to_dict("index")}
        print(f"\n{'='*66}\n{model}   [{'PRIMARY' if model == PRIMARY_MODEL else 'replication'}]\n{'='*66}")
        print(f"{'method':38s}{'cov0':>7s}{'cov1':>7s}{'|C|':>8s}{'inform1':>9s}  valid")
        for m in METHODS:
            r = g.loc[m]
            print(f"{METHOD_NAMES[m]:38s}{r['coverage_class0']:>7.4f}{r['coverage_class1']:>7.4f}{r['mean_set_size']:>8.4f}"
                  f"{r['informative_class1']:>9.4f}  {'YES' if valid[m] else 'no'}")
        print(f"  VERDICT: {v}")
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
