"""
Phase 2 diagnostic (descriptive) -- why do marginal LAC sets rarely contain the deterioration label at the landmark?
For the primary model, per seed: the static marginal threshold q_hat (alpha = 0.10, final-calibration fold), the risk level 1 - q_hat that a stay needs to
include label 1, quantiles of the predicted risk among events and non-events on the final-calibration fold and on the test period, and the share of events
and non-events above 1 - q_hat. No p-values; no selection.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import p2_5_conformal as P5  # noqa: E402


def main():
    res = {"alpha": C.ALPHA, "runs": []}
    for seed in C.SEEDS:
        S = P5.load_streams(seed)
        p_fc, y_fc, p_te, y_te = S["fc"]["p"], S["fc"]["y"], S["test"]["p"], S["test"]["y"]
        _s, q = RC.static_lac_sets(p_fc, y_fc, p_te, C.ALPHA)
        thr = 1.0 - q
        rec = {"seed": seed, "q_hat": float(q), "risk_needed_for_label_1": float(thr)}
        for name, p, y in (("final_calibration", p_fc, y_fc), ("test", p_te, y_te)):
            rec[name] = {"median_risk_events": float(np.median(p[y == 1])), "q75_risk_events": float(np.quantile(p[y == 1], 0.75)),
                         "q90_risk_events": float(np.quantile(p[y == 1], 0.90)), "median_risk_non_events": float(np.median(p[y == 0])),
                         "share_events_above_threshold": float((p[y == 1] >= thr).mean()), "share_non_events_above_threshold": float((p[y == 0] >= thr).mean())}
        res["runs"].append(rec)
        print(seed, round(q, 3), round(thr, 3), {k: round(v, 3) for k, v in rec["test"].items()}, flush=True)
    C.dump(res, "phase2", "p2_9_score_distribution.json")


if __name__ == "__main__":
    main()
