"""
Conformal suite for the revision (Phases 1.5 to 1.8 and 2.4): one place where the established constructions are re-run with the
level alpha as an argument. Built from the repository's own verified pieces:
  * `m2_label_lag.eci_lag_sim`  (event-driven ECI, equal to the vendored core.methods.ECI at L = 0 -- tests T8/T9/T13),
  * `m2_mondrian_conformal.mondrian_lac_sets` (static label-conditional sets),
  * `m2_mondrian_eci_comparators._class_threshold` / `recal_sets`.
Nothing in the algorithms is changed; only the level and, for permutation runs, the order of the test stream.

Set conventions: column 0 = label 0 in set, column 1 = label 1 in set; LAC score of label k is |k - p|.
"""
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np

import rv2_common as C  # noqa: F401  (puts Benchmark_scripts on sys.path)
from m2_label_lag import eci_lag_sim, marginal_interval  # noqa: E402
from m2_mondrian_conformal import mondrian_lac_sets  # noqa: E402
from m2_mondrian_eci_comparators import _class_threshold, recal_sets  # noqa: E402
from m2_mondrian_eci_pilot import set_metrics  # noqa: E402

LR_GRID = (0.01, 0.05, 0.1)
SAT_GUARD = 0.25
RECAL_R = 2000
T_BURNIN = 5


def lac_sets(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return np.column_stack([p <= q, (1.0 - p) <= q])


# ------------------------------------------------------------------------------------------ marginal constructions
def static_lac_sets(p_fc, y_fc, p_te, alpha):
    q = _class_threshold(np.abs(np.asarray(y_fc, float) - p_fc), alpha)
    return lac_sets(p_te, np.full(len(p_te), q)), q


def select_lr_marginal_sets(p_cs, y_cs, alpha):
    """lr for the marginal ECI set: coverage closest to 1 - alpha, then smallest mean set size (as E3), on calibration_selection only."""
    scores = np.abs(np.asarray(y_cs, float) - p_cs)
    cands = []
    for lr in LR_GRID:
        q = eci_lag_sim(scores, np.zeros(len(scores), int), 1, 0, 0, lr, alpha=alpha)[:, 0]
        m = set_metrics(lac_sets(p_cs, q), np.asarray(y_cs, int))
        cands.append((abs(m["coverage_marginal"] - (1 - alpha)), m["mean_set_size"], lr))
    cands.sort()
    return cands[0][2]


def eci_marginal_sets(p_cs, y_cs, p_fc, y_fc, p_te, y_te, alpha, lr=None):
    lr = select_lr_marginal_sets(p_cs, y_cs, alpha) if lr is None else lr
    p_stream = np.concatenate([p_fc, p_te])
    s_stream = np.abs(np.concatenate([y_fc, y_te]).astype(float) - p_stream)
    q = eci_lag_sim(s_stream, np.zeros(len(s_stream), int), 1, 0, len(p_fc), lr, alpha=alpha)[len(p_fc):, 0]
    return lac_sets(p_te, q), q, lr


def select_lr_marginal_interval(p_cs, y_cs, alpha):
    y = np.asarray(y_cs, float)
    sc = np.abs(y - p_cs)
    cands = []
    for lr in LR_GRID:
        q = eci_lag_sim(sc, np.zeros(len(sc), int), 1, 0, 0, lr, alpha=alpha)[:, 0]
        lo, hi = marginal_interval(p_cs, q)
        cands.append({"lr": lr, "gap": abs(float(np.mean((y >= lo) & (y <= hi))) - (1 - alpha)), "w": float(np.mean(hi - lo))})
    cands.sort(key=lambda c: (c["gap"], c["w"]))
    return cands[0]["lr"]


def eci_interval(p_fc, y_fc, p_te, y_te, alpha, lr):
    p_stream = np.concatenate([p_fc, p_te])
    sc = np.abs(np.concatenate([y_fc, y_te]).astype(float) - p_stream)
    q = eci_lag_sim(sc, np.zeros(len(sc), int), 1, 0, len(p_fc), lr, alpha=alpha)[len(p_fc):, 0]
    return marginal_interval(p_te, q), q


def recal_interval_w(p_fc, y_fc, p_te, y_te, alpha, W, R=RECAL_R):
    """Marginal thresholds recomputed every R arrivals from the most recent W arrivals strictly before the block (m2_eci_vs_recalibration)."""
    p = np.concatenate([p_fc, p_te])
    y = np.concatenate([y_fc, y_te]).astype(float)
    n_cal, n = len(p_fc), len(p)
    lo = np.zeros(n - n_cal)
    hi = np.zeros(n - n_cal)
    for start in range(n_cal, n, R):
        w = slice(max(0, start - W), start)
        q = _class_threshold(np.abs(y[w] - p[w]), alpha)
        end = min(start + R, n)
        lo[start - n_cal:end - n_cal] = np.clip(p[start:end] - q, 0, 1)
        hi[start - n_cal:end - n_cal] = np.clip(p[start:end] + q, 0, 1)
    return lo, hi


# ------------------------------------------------------------------------------------------ Mondrian constructions
def mondrian_static_sets(p_fc, y_fc, p_te, alpha):
    sets, q = mondrian_lac_sets(p_fc, y_fc, p_te, alpha=alpha)
    return np.asarray(sets, bool), q


def _sets_from_q2(p, q2):
    return np.column_stack([np.abs(0 - p) <= q2[:, 0], np.abs(1 - p) <= q2[:, 1]])


def select_lr_mondrian_eci(p_cs, y_cs, alpha):
    """Guarded selection on calibration_selection (second half evaluated, class-1 coverage closest to 1 - alpha, saturation guard)."""
    y = np.asarray(y_cs, int)
    half = len(p_cs) // 2
    sc = np.abs(y - p_cs)
    trace = []
    for lr in LR_GRID:
        q = eci_lag_sim(sc, y, 2, 0, half, lr, alpha=alpha)
        s = _sets_from_q2(p_cs[half:], q[half:])
        m = set_metrics(s, y[half:], q[half:])
        trace.append({"lr": lr, "c1": m["coverage_class1"], "size": m["mean_set_size"], "rej": m["q1_saturated_frac"] > SAT_GUARD})
    surv = [t for t in trace if not t["rej"]]
    if not surv:
        return min(LR_GRID), True
    return min(surv, key=lambda t: (abs(t["c1"] - (1 - alpha)), t["size"]))["lr"], False


def mondrian_eci_sets_run(p_cs, y_cs, p_fc, y_fc, p_te, y_te, alpha, lr=None):
    fell = False
    if lr is None:
        lr, fell = select_lr_mondrian_eci(p_cs, y_cs, alpha)
    p_stream = np.concatenate([p_fc, p_te])
    y_stream = np.concatenate([y_fc, y_te]).astype(int)
    n_cal = len(p_fc)
    q = eci_lag_sim(np.abs(y_stream - p_stream), y_stream, 2, 0, n_cal, lr, alpha=alpha)[n_cal:]
    return _sets_from_q2(p_te, q), q, lr, fell


def mondrian_recal_sets(p_fc, y_fc, p_te, y_te, alpha, W, R=RECAL_R):
    p = np.concatenate([p_fc, p_te])
    y = np.concatenate([y_fc, y_te]).astype(int)
    s, q = recal_sets(p, y, len(p_fc), W, R, alpha=alpha)
    return s, q


# ------------------------------------------------------------------------------------------ metrics
def operating_characteristics(sets, y):
    """Set type x true class, PPV and sensitivity of {1}, false-positive rate, false alarms per 100 patients, abstention by class."""
    sets = np.asarray(sets, bool)
    y = np.asarray(y, int)
    inc0, inc1 = sets[:, 0], sets[:, 1]
    typ = np.where(inc0 & inc1, "both", np.where(inc1 & ~inc0, "only1", np.where(inc0 & ~inc1, "only0", "empty")))
    out = {"n": int(len(y)), "n_class0": int((y == 0).sum()), "n_class1": int((y == 1).sum()), "cross": {}}
    for t in ("only0", "only1", "both", "empty"):
        out["cross"][t] = {"class0": int(((typ == t) & (y == 0)).sum()), "class1": int(((typ == t) & (y == 1)).sum())}
    s1_total = out["cross"]["only1"]["class0"] + out["cross"]["only1"]["class1"]
    out["ppv_only1"] = float(out["cross"]["only1"]["class1"] / s1_total) if s1_total else None
    out["sensitivity_only1"] = float(out["cross"]["only1"]["class1"] / max(out["n_class1"], 1))
    out["fpr_only1"] = float(out["cross"]["only1"]["class0"] / max(out["n_class0"], 1))
    out["false_alarms_per_100_patients"] = float(100.0 * out["cross"]["only1"]["class0"] / len(y))
    for k, name in ((0, "class0"), (1, "class1")):
        m = y == k
        out[f"abstain_both_{name}"] = float(((typ == "both") & m).sum() / max(m.sum(), 1))
        out[f"abstain_empty_{name}"] = float(((typ == "empty") & m).sum() / max(m.sum(), 1))
        out[f"abstain_any_{name}"] = out[f"abstain_both_{name}"] + out[f"abstain_empty_{name}"]
    return out


def feasibility(n_k, alpha):
    """A class-k conformal quantile at level 1 - alpha exists below the maximum score only if ceil((n_k+1)(1-alpha)) <= n_k."""
    return bool(np.ceil((n_k + 1) * (1 - alpha)) <= n_k)
