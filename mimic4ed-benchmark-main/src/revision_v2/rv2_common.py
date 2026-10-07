"""
Shared helpers for the M2 revision v2 (Phases 1 and 2). Nothing here writes into existing result files.

* data access through the repository's own loaders (train_v2 / test_v2 frames, development-pool split, chronological sort);
* patient-cluster bootstrap utilities (resample patients with all of their stays) with weighted AUROC / AUPRC, paired by
  construction (the same seed yields the same patient draws for any set of stays with the same patient vector);
* small helpers for reporting seed variability and bootstrap uncertainty separately (rule 10).

Never prints rows or identifiers.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "Benchmark_scripts")
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "revision_v2")
SEEDS = [42, 123, 7, 456, 789]
TUNE_SEEDS = [42, 123, 7]
ALPHA = 0.10
BOOT_SEED = 20261007
sys.path.insert(0, SCRIPTS)


def out_dir(phase):
    d = os.path.join(RES, phase)
    os.makedirs(d, exist_ok=True)
    return d


def dump(obj, phase, name):
    p = os.path.join(out_dir(phase), name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return p


# ------------------------------------------------------------------------------------------------ data
def load_frames():
    """Returns dict of frames (with the 4 static features): train period split into the four development subsets, plus test."""
    from m2_hybrid_common import add_hybrid_static_features
    from m2_v2_common import load_split_frame, load_dev_pool_split_v2
    tr = add_hybrid_static_features(load_split_frame("train_v2"))
    te = add_hybrid_static_features(load_split_frame("test_v2"))
    dev = load_dev_pool_split_v2()
    out = {"test": te}
    names = {"training": "fit", "validation": "val", "calibration_selection": "cs", "final_conformal_calibration": "fc"}
    for k, short in names.items():
        ids = set(dev[k]["subject_ids"].tolist())
        out[short] = tr[tr["subject_id"].isin(ids)].copy()
    out["train_period"] = tr
    for k in ("fit", "val", "cs", "fc", "test"):
        for c in ("intime", "outtime"):
            out[k][c] = pd.to_datetime(out[k][c])
        out[k]["duration_h"] = (out[k]["outtime"] - out[k]["intime"]).dt.total_seconds() / 3600.0
    return out


def sorted_stream(df):
    from m2_hybrid_common import sort_chronologically
    return sort_chronologically(df)


def sequences(split_name, stay_ids):
    """Returns values, obs_mask, pad_mask, delta_hours, t_hours aligned to stay_ids (rows looked up by stay_id)."""
    path = os.path.join(DATA, f"m2_irregular_sequences_{split_name}.npz")
    with np.load(path) as z:
        sid = z["stay_id"]
        pos = pd.Series(np.arange(len(sid)), index=sid).loc[np.asarray(stay_ids)].values
        return (z["values"][pos], z["obs_mask"][pos], z["pad_mask"][pos], z["delta_hours"][pos], z["t_hours"][pos])


# ------------------------------------------------------------------------------------------------ bootstrap
class ClusterBoot:
    """Patient-cluster bootstrap over a fixed stay set. The k-th draw is identical for every instance built with the same
    patient vector and seed, so contrasts between models scored on the same stays are paired."""

    def __init__(self, patient_ids, B=1000, seed=BOOT_SEED):
        self.pid = np.asarray(patient_ids)
        self.uniq, self.inv = np.unique(self.pid, return_inverse=True)
        self.P = len(self.uniq)
        self.B = B
        self.seed = seed

    def weights(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.B):
            counts = np.bincount(rng.integers(0, self.P, self.P), minlength=self.P)
            yield counts[self.inv].astype(np.float64)


class Scorer:
    """Weighted AUROC / AUPRC for a fixed (y, score) pair; the sort is done once."""

    def __init__(self, y, score):
        y = np.asarray(y, dtype=np.float64)
        score = np.asarray(score, dtype=np.float64)
        self.order = np.argsort(score, kind="mergesort")
        s = score[self.order]
        self.y = y[self.order]
        new = np.empty(len(s), dtype=bool)
        new[0] = True
        new[1:] = s[1:] != s[:-1]
        self.gid = np.cumsum(new) - 1
        self.G = int(self.gid[-1]) + 1

    def _groups(self, w):
        ws = w[self.order]
        pos = np.bincount(self.gid, weights=ws * self.y, minlength=self.G)
        neg = np.bincount(self.gid, weights=ws * (1.0 - self.y), minlength=self.G)
        return pos, neg

    def auroc(self, w=None):
        w = np.ones(len(self.order)) if w is None else w
        pos, neg = self._groups(w)
        cneg = np.cumsum(neg) - neg
        wp, wn = pos.sum(), neg.sum()
        return float((pos * (cneg + 0.5 * neg)).sum() / (wp * wn))

    def auprc(self, w=None):
        w = np.ones(len(self.order)) if w is None else w
        pos, neg = self._groups(w)
        pos, neg = pos[::-1], neg[::-1]               # descending score
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        prec = tp / np.maximum(tp + fp, 1e-300)
        wp = pos.sum()
        return float((pos / wp * prec).sum())


def pct_ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=float)
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def boot_metric(y, score, boot, metric="auroc"):
    """Point estimate and bootstrap draws for one score vector."""
    sc = Scorer(y, score)
    f = sc.auroc if metric == "auroc" else sc.auprc
    pt = f()
    draws = np.array([f(w) for w in boot.weights()])
    return pt, draws


def seed_summary(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "per_seed": v.tolist()}


def holm(pvals):
    """Holm step-down adjusted p-values (monotone), order preserved."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def boot_p_two_sided(draws):
    """Two-sided bootstrap p for 'difference = 0' from the draws of the difference (proportion of draws on the other side of 0, doubled)."""
    d = np.asarray(draws, dtype=float)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(p, 1.0 / (len(d) + 1)))
