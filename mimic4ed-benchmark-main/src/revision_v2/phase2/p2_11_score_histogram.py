"""
Phase 2 figure data (aggregates only) -- histogram of the predicted risk of the primary model on the test period (seed 42, 25 equal-width bins on [0, 1])
for stays with and without the outcome, with the marginal LAC threshold of that seed (risk needed for label 1 = 1 - q_hat) and the class-conditional
(Mondrian) thresholds. Counts only; no patient-level values are stored.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import p2_5_conformal as P5  # noqa: E402


def main():
    S = P5.load_streams(42)
    p_fc, y_fc, p_te, y_te = S["fc"]["p"], S["fc"]["y"], S["test"]["p"], S["test"]["y"]
    _s, q = RC.static_lac_sets(p_fc, y_fc, p_te, C.ALPHA)
    _s2, qm = RC.mondrian_static_sets(p_fc, y_fc, p_te, C.ALPHA)
    edges = np.linspace(0, 1, 26)
    res = {"seed": 42, "bin_edges": edges.tolist(),
           "counts_events": np.histogram(p_te[y_te == 1], bins=edges)[0].tolist(), "counts_non_events": np.histogram(p_te[y_te == 0], bins=edges)[0].tolist(),
           "marginal_threshold_risk_for_label_1": float(1 - q), "mondrian_class1_threshold_risk_for_label_1": float(1 - qm[1]), "mondrian_class0_max_risk_for_label_0": float(qm[0]),
           "n_events": int(y_te.sum()), "n_non_events": int((y_te == 0).sum())}
    C.dump(res, "phase2", "p2_11_score_histogram.json")
    print({k: v for k, v in res.items() if "threshold" in k or "max_risk" in k})


if __name__ == "__main__":
    main()
