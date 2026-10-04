"""
M2 revision E3 -- adaptive set-valued baselines on the marginal LAC score
==============================================================================
Executes E3 of M2_Revision_PreSpecification.md (supervisor sign-off 2026-10-03).
Nothing here is tuned against the outcome.

Methods (all on the marginal LAC score s_t = 1 - p_hat_{y_t}(x_t), set {y : 1 - p_hat_y(x) <= q_t}):
  static_lac      static split-conformal LAC sets, threshold from the final-calibration fold
  aci_lac         ACI (vendored core.methods.aci, unchanged)
  eci_lac         ECI (vendored core.methods.ECI, unchanged)
Reference for the class-1 metrics: static Mondrian-LAC (m2_mondrian_conformal.mondrian_lac_sets).

For a binary label with p = P(y=1|x), 1 - p_hat_y = |y - p|, so the LAC score is the score the
established ECI interval arm already uses; a set and the interval with the same radius cover the
same labels. The check T14 below uses this: static LAC coverage must equal the reported static
split-conformal coverage exactly.

Protocol (identical to the established ECI arm, run_eci_4way):
  - learning rate from {0.01, 0.05, 0.1} on the calibration-selection subset, run self-contained:
    coverage closest to 0.90, then smallest mean set size;
  - one online stream: final-calibration subset (warm-up) followed by the test stream in the
    established chronological order; metrics on the test suffix only;
  - label delay 0 and 240 arrivals (middle of the pre-specified realistic band). For ECI the delay
    is applied to the test stream only, with the verified event simulator of m2_label_lag.py
    (eci_lag_sim, checked there against the vendored code). For ACI the vendored `ahead=L+1` mode is
    used; it applies the delay to the warm-up as well, which is disclosed.
ACI window_length = 10,000,000 (whole history), the value used throughout the vendored repository's
configs (tests/configs/*.yaml); T_burnin = 5 as for ECI.

Decision rule (fixed in the pre-specification): an adaptive set method "improves on static" only if
marginal coverage >= 0.89 and mean set size < static LAC on all three models.
"""
import os, sys, json, warnings
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_rev_e3_adaptive_sets.csv")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_rev_e3_adaptive_sets_verdict.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_ablations import eci_method  # vendored core.methods.ECI (puts the vendored repo on sys.path)
from core.methods import aci as aci_method  # vendored, unchanged
from m2_label_lag import get_all_predictions, eci_lag_sim, T_BURNIN
from m2_mondrian_conformal import mondrian_lac_sets
from m2_mondrian_eci_comparators import _class_threshold as class_threshold
from m2_mondrian_eci_pilot import set_metrics
from m2_conformal_v2 import SEED_DIRS_V2, FINAL_SEEDS

ALPHA = 0.1
LR_GRID = (0.01, 0.05, 0.1)
LAGS = (0, 240)
ACI_WINDOW = 10_000_000
VALID_MIN = 0.89
METHODS = ("static_lac", "aci_lac", "eci_lac")


def lac_sets(p, q):
    """{y : 1 - p_hat_y <= q}; column 0 = label 0 (score p), column 1 = label 1 (score 1 - p)."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    return np.column_stack([p <= q, (1.0 - p) <= q])


def online_q(method, scores, lr, L, lag_start):
    """Radius in force at each position. lag_start = index from which the delay applies (ECI)."""
    if method == "eci_lac":
        if L == 0:
            return np.asarray(eci_method(scores=scores, alpha=ALPHA, lr=lr, T_burnin=T_BURNIN, ahead=1,
                                         proportional_lr=True)["q"], float)
        return eci_lag_sim(scores, np.zeros(len(scores), int), 1, L, lag_start, lr)[:, 0]
    if method == "aci_lac":
        return np.asarray(aci_method(scores=scores, alpha=ALPHA, lr=lr, window_length=ACI_WINDOW,
                                     T_burnin=T_BURNIN, ahead=L + 1)["q"], float)
    raise ValueError(method)


def select_lr(method, p_cs, y_cs, L):
    scores = np.abs(y_cs.astype(float) - p_cs)
    cands = []
    for lr in LR_GRID:
        q = online_q(method, scores, lr, L, lag_start=0)
        m = set_metrics(lac_sets(p_cs, q), y_cs)
        cands.append((abs(m["coverage_marginal"] - (1 - ALPHA)), m["mean_set_size"], lr))
    cands.sort()
    return cands[0][2]


def main():
    preds, y, _year = get_all_predictions()
    y_cs, y_fc, y_te = y["cs"], y["fc"], y["te"]
    recal = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_eci_vs_recalibration.csv")).set_index(["model", "seed", "method"])
    mapie = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_conformal_summary.csv")).set_index(["model", "seed"])

    rows = []
    for model in SEED_DIRS_V2:
        for seed in FINAL_SEEDS:
            P = preds[(model, seed)]
            p_cs, p_fc, p_te = P["cs"], P["fc"], P["te"]
            n_cal = len(p_fc)
            p_stream = np.concatenate([p_fc, p_te])
            s_stream = np.abs(np.concatenate([y_fc, y_te]).astype(float) - p_stream)

            # static references
            q_static = class_threshold(np.abs(y_fc.astype(float) - p_fc), ALPHA)
            st = set_metrics(lac_sets(p_te, np.full(len(p_te), q_static)), y_te)
            # T14: static LAC == reported static split-conformal (same score, same fold)
            ref = recal.loc[(model, seed, "M1"), "coverage"]
            assert abs(st["coverage_marginal"] - ref) < 1e-12, f"T14 FAILED {model} {seed}: {st['coverage_marginal']} vs {ref}"
            assert abs(st["coverage_marginal"] - mapie.loc[(model, seed), "mapie_coverage"]) <= 0.003, "T6-equivalent FAILED"
            rows.append({"model": model, "seed": seed, "lag": 0, "method": "static_lac", "lr": np.nan, **st})
            mon = mondrian_lac_sets(p_fc, y_fc, p_te, alpha=ALPHA)
            mon_sets = mon[0] if isinstance(mon, tuple) else mon
            mm = set_metrics(np.asarray(mon_sets, bool), y_te)
            # T15: static Mondrian-LAC reproduces the reported value (same function, same fold)
            assert abs(mm["coverage_class1"] - mapie.loc[(model, seed), "mondrian_lac_coverage_class1"]) < 1e-9, \
                f"T15 FAILED {model} {seed}"
            rows.append({"model": model, "seed": seed, "lag": 0, "method": "static_mondrian_lac", "lr": np.nan, **mm})

            for L in LAGS:
                for method in ("aci_lac", "eci_lac"):
                    lr = select_lr(method, p_cs, y_cs, L)
                    q = online_q(method, s_stream, lr, L, lag_start=n_cal)[n_cal:]
                    m = set_metrics(lac_sets(p_te, q), y_te)
                    m["q_inf_frac"] = float(np.isinf(q).mean())
                    rows.append({"model": model, "seed": seed, "lag": L, "method": method, "lr": lr, **m})
            print(f"{model} seed {seed}: T14 pass; static cov {st['coverage_marginal']:.4f} size {st['mean_set_size']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    verdict = {}
    cols = ["coverage_marginal", "coverage_class0", "coverage_class1", "mean_set_size", "informative_class1"]
    for L in LAGS:
        for method in ("aci_lac", "eci_lac"):
            per_model = {}
            for model in SEED_DIRS_V2:
                g = df[(df.model == model)]
                a = g[(g.method == method) & (g.lag == L)][cols].mean()
                s = g[g.method == "static_lac"][cols].mean()
                per_model[model] = {"method": a.round(4).to_dict(), "static_lac": s.round(4).to_dict(),
                                    "valid": bool(a["coverage_marginal"] >= VALID_MIN),
                                    "smaller_than_static": bool(a["mean_set_size"] < s["mean_set_size"])}
            improves = all(v["valid"] and v["smaller_than_static"] for v in per_model.values())
            verdict[f"{method}_lag{L}"] = {"improves_on_static_all_models": improves, "per_model": per_model}
            print(f"{method} lag {L}: improves on static on all models = {improves}")
    with open(OUT_JSON, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
