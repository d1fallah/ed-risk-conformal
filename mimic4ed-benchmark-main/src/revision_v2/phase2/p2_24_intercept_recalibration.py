"""
Phase 2 diagnostic (reviewer M9): is conformal updating needed once the MODEL is recalibrated? Non-conformal competitor: periodic recalibration of the intercept of the
primary model's score on the most recent labelled arrivals. The logit z of the score is shifted by a block-wise estimated delta (maximum likelihood with the slope fixed
at 1), re-estimated every R = 2,000 arrivals from the most recent W = 10,000 labelled arrivals strictly before the block, as in the periodic quantile recalibration.
Stream: final-calibration subset then test period, each in the stream order (approx_real_year, intime) of p2_5.
  Variant I : rolling intercept; the conformal quantile is the static one, computed on the final-calibration subset after an intercept shift fitted on it.
  Variant II: rolling intercept and a rolling quantile (W = 10,000, R = 2,000) on the recalibrated scores.
Reported on the test period, mean and SD over five seeds: calibration of the recalibrated score (Brier, ECE with ten equal-frequency bins, calibration-in-the-large),
and for the marginal LAC sets (alpha = 0.10) the marginal and class-1 coverage and the mean set size; static Mondrian-LAC class-1 coverage on the recalibrated score.
  Variant III (added after the author asked whether the comparison was fair: in I the calibration scores are shifted by an intercept fitted on the calibration subset itself,
              about 0.14 on average, and the test blocks by about 0.2 to 0.4): for each block the SAME intercept is applied to the final-calibration scores and to the test block, and the
              static quantile is recomputed on the shifted calibration scores. Hypothesis stated before running: with calibration and test shifted alike, static marginal coverage moves
              toward 0.90.
Hypothesis stated before running (variants I and II): the intercept shift restores calibration-in-the-large but not class-1 coverage of the marginal sets (a property of rare outcomes),
and marginal coverage of Variant I stays close to that of the original static method.
Aggregates only.
"""
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import p2_common as P  # noqa: E402
import p2_5_conformal as P5  # noqa: E402
from m2_mondrian_eci_comparators import _class_threshold  # noqa: E402

ALPHA, W, R = 0.10, 10000, 2000


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_delta(z, y, iters=30):
    """Maximum-likelihood intercept shift with the slope fixed at 1 (Newton)."""
    d = 0.0
    for _ in range(iters):
        p = sigmoid(z + d)
        g = float((y - p).sum())
        h = float((p * (1 - p)).sum())
        if h < 1e-9:
            break
        step = g / h
        d += step
        if abs(step) < 1e-8:
            break
    return d


def ece(y, p, bins=10):
    order = np.argsort(p)
    return float(sum(len(c) * abs(y[c].mean() - p[c].mean()) for c in np.array_split(order, bins)) / len(y))


def main():
    res = {"alpha": ALPHA, "W": W, "R": R, "runs": []}
    for seed in C.SEEDS:
        S = P5.load_streams(seed)
        p_fc, y_fc, p_te, y_te = S["fc"]["p"], S["fc"]["y"].astype(float), S["test"]["p"], S["test"]["y"].astype(float)
        n_cal = len(p_fc)
        z = np.concatenate([logit(p_fc), logit(p_te)])
        y = np.concatenate([y_fc, y_te])
        d0 = fit_delta(z[:n_cal], y[:n_cal])
        p_re = np.empty(len(z))
        p_re[:n_cal] = sigmoid(z[:n_cal] + d0)
        deltas = []
        for start in range(n_cal, len(z), R):
            w = slice(max(0, start - W), start)
            d = fit_delta(z[w], y[w])
            end = min(start + R, len(z))
            p_re[start:end] = sigmoid(z[start:end] + d)
            deltas.append(d)
        pt = p_re[n_cal:]
        yt = y_te.astype(int)
        # Variant I: static quantile from the recalibrated final-calibration scores
        q = _class_threshold(np.abs(y_fc - p_re[:n_cal]), ALPHA)
        setsI = RC.lac_sets(pt, np.full(len(pt), q))
        mondI, _ = RC.mondrian_static_sets(p_re[:n_cal], y_fc.astype(int), pt, ALPHA)
        # Variant II: rolling quantile on recalibrated scores
        lo, hi = RC.recal_interval_w(p_re[:n_cal], y_fc.astype(int), pt, y_te, ALPHA, W)
        covII = (y_te >= lo) & (y_te <= hi)
        # Variant III: same intercept on the calibration scores and on the test block; quantile recomputed per block
        covIII = np.zeros(len(pt), bool)
        sets_c1, sets_cov, sets_size = [], [], []
        mond_c1 = np.zeros(len(pt), bool)
        z_fc = z[:n_cal]
        for bi, start in enumerate(range(n_cal, len(z), R)):
            w = slice(max(0, start - W), start)
            d = fit_delta(z[w], y[w])
            end = min(start + R, len(z))
            pf = sigmoid(z_fc + d)
            qb = _class_threshold(np.abs(y_fc - pf), ALPHA)
            sl = slice(start - n_cal, end - n_cal)
            pb = sigmoid(z[start:end] + d)
            sb = RC.lac_sets(pb, np.full(len(pb), qb))
            sets_size.append(sb.sum(1))
            yb = y_te[sl].astype(int)
            covIII[sl] = sb[np.arange(len(yb)), yb]
            mb, _ = RC.mondrian_static_sets(pf, y_fc.astype(int), pb, ALPHA)
            mond_c1[sl] = mb[np.arange(len(yb)), yb]
        # original (not intercept-recalibrated) score, for reference
        qo = _class_threshold(np.abs(y_fc - p_fc), ALPHA)
        setsO = RC.lac_sets(p_te, np.full(len(p_te), qo))

        def cov(sets):
            c = sets[np.arange(len(yt)), yt]
            return {"coverage": float(c.mean()), "class1": float(c[yt == 1].mean()), "size": float(sets.sum(1).mean())}
        run = {"seed": seed, "delta_fc": d0, "delta_test_mean": float(np.mean(deltas)), "delta_test_min": float(np.min(deltas)), "delta_test_max": float(np.max(deltas)),
               "calibration_original": {"brier": float(np.mean((p_te - y_te) ** 2)), "ece": ece(y_te, p_te), "mean_pred": float(p_te.mean()), "observed": float(y_te.mean())},
               "calibration_recalibrated": {"brier": float(np.mean((pt - y_te) ** 2)), "ece": ece(y_te, pt), "mean_pred": float(pt.mean()), "observed": float(y_te.mean())},
               "static_original": cov(setsO), "variant_I_static_quantile": cov(setsI),
               "variant_III_marginal_coverage": float(covIII.mean()), "variant_III_class1_coverage": float(covIII[yt == 1].mean()),
               "variant_III_mondrian_class1": float(mond_c1[yt == 1].mean()), "variant_III_set_size": float(np.concatenate(sets_size).mean()),
               "variant_II_marginal_coverage": float(covII.mean()), "variant_II_class1_coverage": float(covII[yt == 1].mean()),
               "mondrian_static_on_recalibrated": {"coverage": float(mondI[np.arange(len(yt)), yt].mean()), "class1": float(mondI[np.arange(len(yt)), yt][yt == 1].mean())}}
        res["runs"].append(run)
        print(seed, "delta_fc", round(d0, 3), "delta_test mean", round(run["delta_test_mean"], 3),
              "| mean pred", round(run["calibration_original"]["mean_pred"], 4), "->", round(run["calibration_recalibrated"]["mean_pred"], 4),
              "obs", round(float(y_te.mean()), 4), "| brier", round(run["calibration_original"]["brier"], 5), "->", round(run["calibration_recalibrated"]["brier"], 5),
              "| cov orig", round(run["static_original"]["coverage"], 4), "I", round(run["variant_I_static_quantile"]["coverage"], 4), "II", round(run["variant_II_marginal_coverage"], 4),
              "| c1 orig", round(run["static_original"]["class1"], 3), "I", round(run["variant_I_static_quantile"]["class1"], 3), "II", round(run["variant_II_class1_coverage"], 3),
              "| III cov", round(run["variant_III_marginal_coverage"], 4), "c1", round(run["variant_III_class1_coverage"], 3), "mond c1", round(run["variant_III_mondrian_class1"], 3), "| mond c1 I", round(run["mondrian_static_on_recalibrated"]["class1"], 3), flush=True)
    C.dump(res, "phase2", "p2_24_intercept_recalibration.json")


if __name__ == "__main__":
    main()
