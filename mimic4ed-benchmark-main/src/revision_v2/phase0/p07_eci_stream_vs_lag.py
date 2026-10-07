"""
Phase 0, task 0.7 -- stream length versus label lag (counts only; the ECI algorithm is not touched).

For each pre-registered lag L, counts how many labels (all, class 0, class 1) have been resolved by the end of
  (a) the calibration-selection stream when the learning rate is selected under that lag (select_lr_marginal runs the whole
      stream with lag_start = 0; select_lr_mondrian evaluates the second half with lag_start = n//2),
  (b) the test stream (lag applies from the first test arrival).
An item j is resolved before arrival j + L + 1 is predicted, so the number resolved by the end of a stream of n arrivals that
starts lagging at index s is max(0, n - 1 - L - s). L = None means no label ever arrives. Only the cached label vectors are
read (data/_lag_cache/preds.npz, key 'y'); nothing patient-level is printed or written.
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "revision_v2", "phase0")
LAGS = [0, 60, 240, 1000, 4000, 16000, None]


def resolved(y, L, start):
    n = len(y)
    if L is None:
        return {"all": 0, "class0": 0, "class1": 0, "after_start_resolved": 0}
    last = n - 2 - L                                   # last item index resolved before the final arrival is predicted
    idx_end = max(start - 1, last)                     # items before `start` resolve immediately
    res = y[start:idx_end + 1] if idx_end >= start else y[:0]
    return {"all": int(len(res)), "class0": int((res == 0).sum()), "class1": int((res == 1).sum()), "after_start_resolved": int(len(res))}


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "data", "_lag_cache", "preds.npz"), allow_pickle=True)
    y = {k: np.asarray(v).astype(int) for k, v in z["y"].item().items()}
    res = {"stream_lengths": {k: {"n": int(len(v)), "positives": int(v.sum())} for k, v in y.items()}, "per_lag": {}}
    n_cs, half = len(y["cs"]), len(y["cs"]) // 2
    res["calibration_selection_half_used_by_mondrian_selection"] = int(n_cs - half)
    for L in LAGS:
        tag = "inf" if L is None else str(L)
        res["per_lag"][tag] = {
            "marginal_selection_stream_labels_resolved_by_end": resolved(y["cs"], L, 0),
            "mondrian_selection_evaluated_half_labels_resolved_by_end": resolved(y["cs"], L, half),
            "test_stream_labels_resolved_by_end": resolved(y["te"], L, 0),
            "L_exceeds_calibration_selection_stream": bool(L is not None and L >= n_cs),
            "L_exceeds_evaluated_half_of_selection_stream": bool(L is not None and L >= n_cs - half),
            "L_as_share_of_test_stream": None if L is None else float(L / len(y["te"]))}
    json.dump(res, open(os.path.join(OUT, "p07_eci_stream_vs_lag.json"), "w"), indent=2)
    print(res["stream_lengths"], "half:", res["calibration_selection_half_used_by_mondrian_selection"])
    for tag, r in res["per_lag"].items():
        print(tag, "marg-sel resolved", r["marginal_selection_stream_labels_resolved_by_end"]["all"],
              "| mond-sel half resolved (all/c1)", r["mondrian_selection_evaluated_half_labels_resolved_by_end"]["after_start_resolved"],
              r["mondrian_selection_evaluated_half_labels_resolved_by_end"]["class1"],
              "| test resolved", r["test_stream_labels_resolved_by_end"]["all"], r["test_stream_labels_resolved_by_end"]["class1"])


if __name__ == "__main__":
    main()
