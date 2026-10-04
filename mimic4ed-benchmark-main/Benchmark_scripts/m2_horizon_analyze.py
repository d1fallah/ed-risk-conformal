"""
M2 -- Horizon ablation: patient-cluster bootstrap for the per-horizon deltas
==============================================================================
CLAUDE.md's M2 directive requires a horizons 1h/6h/12h ablation. The models were
trained per horizon by run_horizon_ablation_v2.sh (Temporal-Only GRU-D, Hybrid
GRU-D, XGBoost-Matched; composite-outcome tuned configs, NOT re-tuned per
horizon -- disclosed). This script does the inference.

Declared BEFORE running (confirmatory family, Holm-corrected together):
    D_repr      = Hybrid GRU-D   - XGBoost-Matched        x  {1h, 6h, 12h}
    D_temporal  = Temporal-Only  - XGBoost-Matched        x  {1h, 6h, 12h}
  6 tests. A delta is called significant only if it is Holm-significant AND the
  95% CI excludes zero at all 5 seeds. Hybrid - Temporal-Only is reported as
  EXPLORATORY and unadjusted: it is NOT Delta_static_contribution (that needs
  Frozen-Temporal Head Control, not run at horizons).
  The 1h arm is flagged underpowered a priori (66 test positives, 14 validation
  positives, so early stopping is noise) and is reported with that caveat
  whatever its p-values say.

Inference: the patient-cluster bootstrap is VALID here, unlike for the online
conformal methods, because these are fixed predictions -- each row's score does
not depend on the order or composition of the resample. Patients are resampled
with replacement and all their encounters kept with multiplicity, never
deduplicated: exactly the resampling unit used everywhere else in this module.

Implementation. An AUROC on a resample in which patient j appears c_j times is
EXACTLY the weighted AUROC with row weight c_{patient(row)}. Doing that with
score-tie-aware cumulative sums over pre-computed score groups is ~15x faster
than concatenating index arrays and calling roc_auc_score per resample, and all
three models share each resample (paired). Verified against roc_auc_score before
any bootstrap is run (test_weighted_auc below).
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_horizon_deltas.json")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_horizon_summary.csv")

HORIZONS = ["1h", "6h", "12h"]
SEEDS = [42, 123, 7, 456, 789]
N_BOOT = 10000
MODELS = {"H": "Hybrid GRU-D v2", "T": "Temporal-Only GRU-D v2", "X": "XGBoost-Matched v2"}
DELTAS = {  # name -> (a, b, confirmatory)
    "D_repr": ("H", "X", True),
    "D_temporal": ("T", "X", True),
    "D_hybrid_minus_temporal": ("H", "T", False),
}


class WeightedAUC:
    """Tie-aware weighted AUROC for a FIXED score vector and FIXED labels; only the
    row weights change per call. Score groups are computed once."""

    def __init__(self, y, s):
        self.y = np.asarray(y).astype(float)
        _, self.gid = np.unique(np.asarray(s), return_inverse=True)
        self.G = int(self.gid.max()) + 1

    def __call__(self, w):
        pos = np.bincount(self.gid, weights=w * self.y, minlength=self.G)
        neg = np.bincount(self.gid, weights=w * (1.0 - self.y), minlength=self.G)
        P, N = pos.sum(), neg.sum()
        if P == 0 or N == 0:
            return np.nan
        neg_below = np.cumsum(neg) - neg
        return float((pos * (neg_below + 0.5 * neg)).sum() / (P * N))


def test_weighted_auc():
    rng = np.random.RandomState(0)
    n = 3000
    y = (rng.rand(n) < 0.06).astype(int)
    s = np.round(rng.rand(n) * 0.5 + 0.35 * y, 2)          # heavy ties on purpose
    f = WeightedAUC(y, s)
    assert abs(f(np.ones(n)) - roc_auc_score(y, s)) < 1e-12, "unit-weight AUROC != roc_auc_score"
    for _ in range(25):
        w = rng.poisson(1.0, n).astype(float)
        ref_w = roc_auc_score(y, s, sample_weight=w)
        idx = np.repeat(np.arange(n), w.astype(int))        # literal duplication == 'kept with multiplicity'
        ref_dup = roc_auc_score(y[idx], s[idx])
        assert abs(f(w) - ref_w) < 1e-12 and abs(f(w) - ref_dup) < 1e-12, "weighted AUROC != duplicated-rows AUROC"
    print("test_weighted_auc: matches roc_auc_score (unit weights, sample_weight, and literal row duplication) -- PASS")


def load(h):
    p = pd.read_csv(os.path.join(DATA_DIR, f"m2_v2_hz{h}_predictions.csv"))
    out = {}
    for seed in SEEDS:
        base = None
        for k, name in MODELS.items():
            d = p[(p.model == name) & (p.seed == seed)].sort_values("stay_id").reset_index(drop=True)
            assert len(d) > 0, f"{h}/{name}/seed {seed}: no rows"
            if base is None:
                base = d
            else:
                assert len(d) == len(base) and (d.stay_id.values == base.stay_id.values).all() \
                    and (d.y_true.values == base.y_true.values).all(), f"{h} seed {seed}: models scored different rows"
            out[(seed, k)] = d
        out[(seed, "meta")] = base
    return out


def run_horizon(h):
    data = load(h)
    res = {d: [] for d in DELTAS}
    point = {}
    for seed in SEEDS:
        meta = data[(seed, "meta")]
        y = meta.y_true.values.astype(int)
        subj = pd.factorize(meta.subject_id.values)[0]
        n_pat = subj.max() + 1
        f = {k: WeightedAUC(y, data[(seed, k)].y_prob.values) for k in MODELS}
        rng = np.random.default_rng([42, seed, int(h[:-1])])
        boot = {k: np.empty(N_BOOT) for k in MODELS}
        for b in range(N_BOOT):
            counts = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat).astype(float)
            w = counts[subj]
            for k in MODELS:
                boot[k][b] = f[k](w)
        pt = {k: f[k](np.ones(len(y))) for k in MODELS}
        point[seed] = pt
        for name, (a, b_, _) in DELTAS.items():
            d = boot[a] - boot[b_]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1.0 / (len(d) + 1))
            res[name].append({"seed": seed, "point": pt[a] - pt[b_], "ci_lo": float(lo), "ci_hi": float(hi),
                              "excludes_zero": bool(lo > 0 or hi < 0), "p_proxy": float(p)})
        print(f"  {h} seed {seed}: AUROC H={pt['H']:.4f} T={pt['T']:.4f} X={pt['X']:.4f}", flush=True)
    return res, point, int(data[(SEEDS[0], 'meta')].y_true.sum()), len(data[(SEEDS[0], 'meta')])


def main():
    test_weighted_auc()
    results, summary = {}, []
    for h in HORIZONS:
        print(f"\n=== horizon {h} ===")
        res, point, n_pos, n = run_horizon(h)
        results[h] = {"n": n, "n_pos": n_pos, "deltas": {}}
        for name, rows in res.items():
            pts = np.array([r["point"] for r in rows])
            results[h]["deltas"][name] = {
                "per_seed": rows, "mean": float(pts.mean()), "sd": float(pts.std(ddof=1)),
                "excludes_zero_at_n_of_5": int(sum(r["excludes_zero"] for r in rows)),
                "p_conservative": float(max(r["p_proxy"] for r in rows)),
                "ci_lo_mean": float(np.mean([r["ci_lo"] for r in rows])),
                "ci_hi_mean": float(np.mean([r["ci_hi"] for r in rows])),
                "confirmatory": DELTAS[name][2]}
        for k, name in MODELS.items():
            v = np.array([point[s][k] for s in SEEDS])
            summary.append({"horizon": h, "model": name, "n": n, "n_pos": n_pos,
                            "auroc_mean": v.mean(), "auroc_sd": v.std(ddof=1)})

    keys = [(h, d) for h in HORIZONS for d in DELTAS if DELTAS[d][2]]
    pv = [results[h]["deltas"][d]["p_conservative"] for h, d in keys]
    _, pc, _, _ = multipletests(pv, alpha=0.05, method="holm")
    for (h, d), p_adj in zip(keys, pc):
        r = results[h]["deltas"][d]
        r["holm_p"] = float(p_adj)
        r["holm_significant"] = bool(p_adj < 0.05)
        r["called_significant"] = bool(p_adj < 0.05 and r["excludes_zero_at_n_of_5"] == 5)

    pd.DataFrame(summary).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'horizon':8s}{'delta':26s}{'mean':>9s}{'95% CI (mean of seeds)':>26s}{'CI>0/<0 seeds':>15s}{'Holm p':>9s}  called")
    for h in HORIZONS:
        for d in DELTAS:
            r = results[h]["deltas"][d]
            hp = f"{r['holm_p']:.4f}" if "holm_p" in r else "  n/a"
            call = ("YES" if r.get("called_significant") else "no") if r["confirmatory"] else "(explor.)"
            print(f"{h:8s}{d:26s}{r['mean']:>+9.4f}   [{r['ci_lo_mean']:+.4f}, {r['ci_hi_mean']:+.4f}]{r['excludes_zero_at_n_of_5']:>10d}/5{hp:>9s}  {call}")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
