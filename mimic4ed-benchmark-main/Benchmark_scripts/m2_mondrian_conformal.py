"""
M2 -- Mondrian / Conditional Conformal Layer (Q1-blocker #2)
==============================================================
External Q1-readiness review: headline conformal coverage (~90%) is
marginal, achieved almost entirely by covering the 94% majority-negative
class. Measured directly in this module (`m2_hybrid_conformal_summary.csv`
/ `results_hybrid.csv`): Hybrid GRU-D class-1 coverage is 0.0759 (MAPIE) /
0.1037 (ECI) at a ~5.9% base rate, and MAPIE returns an EMPTY prediction
set for ~48.6% of the patients who actually deteriorate. A marginal
90%-coverage headline does not survive that.

`M2_Related_Work_Notes.md` records that Yang et al. (JMIR, 2024) already
apply Mondrian (class-conditional split) conformal prediction to MIMIC-IV
sepsis data for exactly this reason. This script closes the equivalent gap
here, and goes one step further for the ECI arm (a class-conditional
ONLINE conformal construction, which split-conformal Mondrian is not).

Three constructions, run against the 3 GRU-D-family models already
established by the v3.8 Hybrid pipeline (Temporal-Only GRU-D,
Frozen-Temporal Head Control, Hybrid GRU-D), reusing their EXISTING
trained checkpoints via `m2_hybrid_conformal.py::get_predictions` --
inference only, no retraining, so this runs independently of the Gate v2
sign-off that blocks the cohort rebuild:

  1. Label-conditional (Mondrian) split-conformal on MAPIE's LAC
     prediction sets. Standard construction (Vovk 2003; the label-
     conditional variant of Sadinle, Lei & Wasserman, "Least Ambiguous
     Set-Valued Classifiers with Bounded Error Levels," JASA 114, 2019 --
     the SAME paper `run_mapie_classification`'s docstring already cites
     for the marginal LAC score used throughout this module). Computes a
     SEPARATE score threshold per true class from the calibration fold,
     so a label k is included in a test point's set iff its own
     nonconformity score clears class k's own threshold -- deployable
     exactly like ordinary MAPIE sets (no test-time knowledge of the true
     label is needed to decide set membership).

  2. Mondrian-ECI: two independent vendored `core.methods.ECI` online
     streams, split by REALIZED label within the calibration+test stream
     in true chronological order (reusing `sort_chronologically`).
     EXPLICITLY NOT a deployable predictor: unlike (1), there is no
     analogous "evaluate both streams, keep what clears its own
     threshold" rule for a single continuous interval, since ECI produces
     one interval, not a per-candidate-label set. This construction
     answers a different, purely retrospective question -- "if the true
     label were known, how tight and how well-calibrated would this
     module's online conformal procedure be, separately, for patients who
     do and do not deteriorate?" -- and is reported as a research
     diagnostic only, honestly labeled as such wherever it appears. Goes
     beyond Yang et al. (2024), which is a SPLIT-conformal Mondrian
     construction.

  3. Observable-taxonomy group-conditional regression interval on the
     Hybrid GRU-D risk score, via MAPIE's own vendored
     `ConditionalSplitConformalRegressor` (Gibbs, Cherian & Candes,
     "Conformal Prediction With Conditional Guarantees," 2023/2025) --
     grouped by TRIAGE ACUITY BAND (already one of the 4 Hybrid static
     features), an observable available at prediction time, unlike the
     true label. This is the deployable analog of (1) for the continuous
     risk-score interval, where the true class cannot be used to pick a
     threshold.

SCOPING NOTE, stated once rather than repeated per section: Track A's own
mandatory-core regression interval (GRU-D + MAPIE on the flat-packed T=2
representation) has NO saved per-seed checkpoint anywhere in this repo
(`m2_multiseed_run.py` persists only aggregate metrics, never model
weights) -- reusing it here would require retraining, which this script
deliberately avoids so it can run ahead of Gate v2 sign-off. Track A's
group-conditional interval is deferred to Phase 4 of the session plan, on
the rebuilt irregular-series cohort, where Track A gets fresh checkpoints
regardless.

Output: data/m2_mondrian_conformal.csv (one row per model x seed x
construction), with coverage, class-conditional coverage, empty-set rate,
and width, so the price of conditional validity (always paid in width) is
visible next to the gain.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FINAL_SEEDS = [42, 123, 7, 456, 789]
ALPHA = 0.1  # 90% target coverage, matching every other M2 conformal result

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    add_hybrid_static_features, prepare_temporal_tensors, build_fixed_dev_split,
    prepare_static_features, conformal_coverage_report, sort_chronologically,
)
from m2_hybrid_conformal import get_predictions, SEED_DIRS
from m2_ablations import eci_method, _sorted_bounds  # vendored ECI wiring, reused as-is


# ── Construction 1: label-conditional (Mondrian) MAPIE sets ────────────────
def mondrian_lac_sets(y_prob_cal, y_cal, y_prob_test, alpha=ALPHA):
    """Label-conditional split-conformal classification sets (Vovk 2003 /
    Sadinle, Lei & Wasserman JASA 2019's class-conditional variant).

    For each class k, the nonconformity score is s(x,k) = 1 - p_hat(k|x)
    (identical convention to MAPIE's own LACConformityScore). The
    threshold q_k is the finite-sample-corrected (1-alpha) quantile of
    {s(x_i,k) : i in calibration, y_i = k} -- i.e. calibrated ONLY on
    examples that truly belong to class k, not on the full calibration
    fold. A test point's set includes label k iff its OWN s(x,k) clears
    q_k. This guarantees P(Y in C(X) | Y=k) >= 1-alpha for each k
    separately (class-conditional coverage), at the cost of set size /
    width relative to the marginal-only construction.
    """
    y_prob_cal = np.asarray(y_prob_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    y_prob_test = np.asarray(y_prob_test, dtype=float)

    q = {}
    for k in (0, 1):
        # p_hat(k|x): y_prob is P(y=1|x), so class 1 uses it directly and
        # class 0 uses its complement. (An earlier version selected
        # y_prob_cal whenever the ROW's label equalled k, which is right for
        # k=1 but returns P(y=1|x) for k=0 -- calibrating class 0 on 1-p while
        # the test-time rule below scores label 0 as p. q_0 then sat near 1,
        # label 0 was always included, class-0 coverage was exactly 1.0000
        # instead of ~0.90, and singleton {1} sets were impossible. Class-1
        # coverage was unaffected, which is why the bug survived review.)
        p_hat_k = y_prob_cal if k == 1 else 1.0 - y_prob_cal
        # scores for the calibration examples truly in class k only
        s_k = 1.0 - p_hat_k[y_cal == k]
        n_k = len(s_k)
        if n_k == 0:
            q[k] = 1.0  # no calibration mass for this class: degrade to "always include" (fail-open, not fail-closed)
            continue
        level = min(1.0, np.ceil((n_k + 1) * (1 - alpha)) / n_k)
        q[k] = float(np.quantile(s_k, level, method="higher"))

    # test-time set membership: label k in the set iff s(x,k) <= q_k
    s_test_0 = 1.0 - (1 - y_prob_test)  # = y_prob_test  (score for label 0)
    s_test_1 = 1.0 - y_prob_test        # score for label 1
    include_0 = s_test_0 <= q[0]
    include_1 = s_test_1 <= q[1]
    sets = np.column_stack([include_0, include_1])
    return sets, q


# ── Construction 2: Mondrian-ECI (retrospective two-stream diagnostic) ─────
def mondrian_eci_streams(y_prob_final_cal, y_true_final_cal, y_prob_test, y_true_test,
                          lr=0.1, alpha=ALPHA):
    """Two independent online ECI streams, one per realized label, run in
    chronological arrival order over final_conformal_calibration (warm-up)
    + test (evaluated). At each step, the observation is routed to
    stream y_i's ECI updater using that stream's own accumulated q.
    Retrospective evaluation only (see module docstring for why this is
    not directly deployable): a test point's covered/not-covered outcome
    uses the interval produced by ITS OWN realized-label stream, which
    requires knowing the true label -- exactly the information a real
    deployment does not have at prediction time.

    `lr=0.1` matches this module's own ECI lr-sweep result for the
    Hybrid pipeline (m2_eci_lr_sweep.csv / M2_Findings_Dossier.md Sec 5:
    lr=0.1 selected at 14 of 15 model x seed runs there), used as a fixed
    default here rather than re-running a 2-stream-aware sweep, which
    would roughly double this diagnostic's cost for a result this script
    only reports as a secondary construction.
    """
    y_prob_final_cal = np.asarray(y_prob_final_cal, dtype=float)
    y_true_final_cal = np.asarray(y_true_final_cal, dtype=float)
    y_prob_test = np.asarray(y_prob_test, dtype=float)
    y_true_test = np.asarray(y_true_test, dtype=float)

    y_prob_stream = np.concatenate([y_prob_final_cal, y_prob_test])
    y_true_stream = np.concatenate([y_true_final_cal, y_true_test])
    n_cal = len(y_prob_final_cal)

    covered = np.zeros(len(y_prob_stream), dtype=bool)
    widths = np.full(len(y_prob_stream), np.nan)

    for k in (0, 1):
        mask_k = y_true_stream == k
        if mask_k.sum() < 2:
            continue
        idx_k = np.where(mask_k)[0]
        p_k = y_prob_stream[idx_k]
        y_k = y_true_stream[idx_k]
        scores_k = np.abs(y_k - p_k)
        res = eci_method(scores=scores_k, alpha=alpha, lr=lr, T_burnin=5, ahead=1, proportional_lr=True)
        qs_k = res["q"]
        lo_k, hi_k = _sorted_bounds(p_k - qs_k, p_k + qs_k)
        lo_k, hi_k = np.clip(lo_k, 0.0, 1.0), np.clip(hi_k, 0.0, 1.0)
        covered[idx_k] = (y_k >= lo_k) & (y_k <= hi_k)
        widths[idx_k] = hi_k - lo_k

    covered_test = covered[n_cal:]
    widths_test = widths[n_cal:]
    y_true_test_int = y_true_test.astype(int)
    return covered_test, widths_test, y_true_test_int


# ── Construction 4: Mondrian-ECI SETS (deployable online label-conditional) ─
def mondrian_eci_sets(y_prob_stream, y_true_stream, n_cal, lr=0.1, alpha=ALPHA,
                       T_burnin=5):
    """Online, label-conditional conformal prediction SETS -- the deployable
    counterpart of construction 2, and the online counterpart of
    construction 1.

    The design hinge: ECI's conformity score is |y - p|, which for binary
    labels is exactly the LAC score mondrian_lac_sets() already uses
    (s(x,1) = 1 - p, s(x,0) = p). So a per-class ONLINE radius q_k can be
    substituted for construction 1's per-class STATIC calibration quantile
    with no other change -- same score, same set-membership rule. The two
    constructions therefore differ in exactly one variable (where the
    threshold comes from), which is what makes them directly comparable.

    Deployability, the property construction 2 lacks. Set membership is
        k in C(x_t)  iff  |k - p_t| <= q_k^(t),
    evaluated for each CANDIDATE label k from p_t alone. The true label is
    never consulted at prediction time. Labels are used only to ROUTE the
    online update afterwards, which is legitimate delayed feedback: in this
    cohort the outcome is known within ~12h, and q_k^(t) is built strictly
    from class-k points that arrived BEFORE t (enforced by the exclusive
    cumulative count below, and asserted in the pilot's causality test).

    Returns (sets_test, q_test, diagnostics) where sets_test is (n_test, 2)
    boolean and q_test is (n_test, 2) of the per-class radii actually in
    force at each test point.
    """
    p = np.asarray(y_prob_stream, dtype=float)
    y = np.asarray(y_true_stream, dtype=int)
    assert len(p) == len(y), "stream arrays misaligned"
    n = len(p)

    q_global = np.zeros((n, 2), dtype=float)
    diagnostics = {}

    for k in (0, 1):
        mask_k = (y == k)
        n_k = int(mask_k.sum())
        if n_k < 2:
            # No usable stream for this class: fail-OPEN (always include),
            # matching mondrian_lac_sets()'s own empty-class convention.
            q_global[:, k] = 1.0
            diagnostics[f"n_stream_{k}"] = n_k
            continue

        idx_k = np.where(mask_k)[0]
        scores_k = np.abs(k - p[idx_k])  # == LAC score s(x,k) on class-k points
        res = eci_method(scores=scores_k, alpha=alpha, lr=lr, T_burnin=T_burnin,
                          ahead=1, proportional_lr=True)
        qs_k = np.asarray(res["q"], dtype=float)

        # Map stream-k's radius trajectory onto the global timeline.
        # j[t] = number of class-k points at positions STRICTLY BEFORE t
        # (exclusive cumulative count). qs_k[j] is by ECI's own convention
        # the radius computed from class-k points 0..j-1, so indexing by
        # j[t] uses only information available before t -- this is what
        # makes the construction causal. Between class-k arrivals j[t] is
        # constant, i.e. the radius is forward-filled (and the minority
        # stream is therefore stale between its own arrivals).
        j = np.cumsum(mask_k) - mask_k.astype(int)
        # After the final class-k point j would reach n_k, one past the last
        # radius ECI computes (its loop stops writing at T-1). Clip: the tail
        # reuses the radius built from n_k-1 points instead of n_k.
        j = np.clip(j, 0, n_k - 1)
        q_global[:, k] = qs_k[j]

        diagnostics[f"n_stream_{k}"] = n_k
        diagnostics[f"q{k}_max"] = float(qs_k.max())
        diagnostics[f"q{k}_saturated_frac"] = float((q_global[n_cal:, k] >= 1.0).mean())

    # Set membership from p alone -- no test-time label anywhere here.
    s_0 = p            # |0 - p|
    s_1 = 1.0 - p      # |1 - p|
    include_0 = s_0 <= q_global[:, 0]
    include_1 = s_1 <= q_global[:, 1]
    sets = np.column_stack([include_0, include_1])

    return sets[n_cal:], q_global[n_cal:], diagnostics


# ── Construction 3: observable-taxonomy group-conditional interval ─────────
class _PrecomputedProbaRegressor:
    """Prefit regressor adapter exposing a precomputed continuous risk
    score as predict(), for MAPIE's ConditionalSplitConformalRegressor --
    same row-index pattern as m2_hybrid_common.py's
    PrecomputedProbaClassifier, applied to the regression API instead."""

    def __init__(self, y_prob):
        self.y_prob = np.asarray(y_prob, dtype=float)

    def fit(self, X, y):
        return self

    def predict(self, X):
        idx = np.asarray(X).ravel().astype(int)
        return self.y_prob[idx]


def triage_band(triage_acuity):
    """3-band grouping of MIMIC-IV-ED's 5-level triage acuity (1=highest
    urgency ... 5=lowest), chosen because acuity 1 and 5 are sparse in
    this cohort (n=4,767 and n=235 respectively out of 83,332 test rows --
    measured directly) and would give an unstable per-group quantile on
    their own."""
    triage_acuity = np.asarray(triage_acuity, dtype=float)
    band = np.full(len(triage_acuity), "unknown", dtype=object)
    band[np.isin(triage_acuity, [1, 2])] = "high_acuity_1_2"
    band[triage_acuity == 3] = "mid_acuity_3"
    band[np.isin(triage_acuity, [4, 5])] = "low_acuity_4_5"
    return band


def group_conditional_interval(y_prob_cal, triage_cal, y_cal, y_prob_test, triage_test, y_test,
                                confidence_level=1 - ALPHA):
    from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor

    band_cal = triage_band(triage_cal)
    band_test = triage_band(triage_test)
    all_bands = ["high_acuity_1_2", "mid_acuity_3", "low_acuity_4_5"]
    band_lookup_cal = {i: b for i, b in enumerate(band_cal)}
    band_lookup_test_offset = len(y_prob_cal)

    combined = np.concatenate([np.asarray(y_prob_cal, dtype=float), np.asarray(y_prob_test, dtype=float)])
    combined_band = np.concatenate([band_cal, band_test])

    def feature_map(X):
        idx = np.asarray(X).ravel().astype(int)
        b = combined_band[idx]
        return np.column_stack([(b == g).astype(float) for g in all_bands])

    reg = _PrecomputedProbaRegressor(combined)
    ccr = ConditionalSplitConformalRegressor(
        feature_map=feature_map, estimator=reg, confidence_level=confidence_level, prefit=True)
    idx_cal = np.arange(0, len(y_prob_cal)).reshape(-1, 1)
    ccr.conformalize(idx_cal, np.asarray(y_cal, dtype=float))
    idx_test = np.arange(len(y_prob_cal), len(y_prob_cal) + len(y_prob_test)).reshape(-1, 1)
    _, y_pis = ccr.predict_interval(idx_test)
    lo, hi = _sorted_bounds(y_pis[:, 0, 0], y_pis[:, 1, 0])
    lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
    y_test = np.asarray(y_test, dtype=float)
    covered = (y_test >= lo) & (y_test <= hi)
    widths = hi - lo
    return covered, widths, band_test


def main():
    print("Loading data and fixed splits...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_train = add_hybrid_static_features(df_train)
    df_test = add_hybrid_static_features(df_test)

    dev_split = build_fixed_dev_split()
    final_cal_ids = set(dev_split["final_conformal_calibration"]["subject_ids"].tolist())
    df_final_cal = sort_chronologically(df_train[df_train["subject_id"].isin(final_cal_ids)])
    df_test_sorted = sort_chronologically(df_test)

    Xf, Mf, Df, yf = prepare_temporal_tensors(df_final_cal)
    Xt, Mt, Dt, yt = prepare_temporal_tensors(df_test_sorted)
    Sf = prepare_static_features(df_final_cal)
    St = prepare_static_features(df_test_sorted)
    triage_f = df_final_cal["triage_acuity"].values
    triage_t = df_test_sorted["triage_acuity"].values

    rows = []
    for model_name in SEED_DIRS:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        for seed in FINAL_SEEDS:
            p_final_cal = get_predictions(model_name, seed, Xf, Mf, Df, S=Sf)
            p_test = get_predictions(model_name, seed, Xt, Mt, Dt, S=St)

            # --- Construction 1: label-conditional MAPIE sets ---
            sets, q = mondrian_lac_sets(p_final_cal, yf, p_test)
            set_sizes = sets.sum(axis=1)
            cov1 = conformal_coverage_report(
                yt, sets[np.arange(len(yt)), yt], set_size=set_sizes, prefix="mondrian_lac_")
            rows.append({
                "model": model_name, "seed": seed, "construction": "1_mondrian_lac_sets",
                "coverage": cov1["mondrian_lac_coverage"],
                "coverage_class0": cov1["mondrian_lac_coverage_class0"],
                "coverage_class1": cov1["mondrian_lac_coverage_class1"],
                "empty_set_rate": cov1["mondrian_lac_empty_set_rate"],
                "empty_set_rate_class0": cov1["mondrian_lac_empty_set_rate_class0"],
                "empty_set_rate_class1": cov1["mondrian_lac_empty_set_rate_class1"],
                "mean_set_size": float(set_sizes.mean()),
                "threshold_class0": q[0], "threshold_class1": q[1],
                "deployable": True,
            })

            # --- Construction 2: Mondrian-ECI (retrospective diagnostic) ---
            covered2, widths2, yt_int = mondrian_eci_streams(p_final_cal, yf, p_test, yt)
            cov2 = conformal_coverage_report(yt_int, covered2, prefix="mondrian_eci_")
            rows.append({
                "model": model_name, "seed": seed, "construction": "2_mondrian_eci_retrospective",
                "coverage": cov2["mondrian_eci_coverage"],
                "coverage_class0": cov2["mondrian_eci_coverage_class0"],
                "coverage_class1": cov2["mondrian_eci_coverage_class1"],
                "empty_set_rate": float("nan"), "empty_set_rate_class0": float("nan"),
                "empty_set_rate_class1": float("nan"),
                "mean_set_size": float("nan"),
                "mean_width": float(np.nanmean(widths2)),
                "mean_width_class0": float(np.nanmean(widths2[yt_int == 0])) if (yt_int == 0).any() else float("nan"),
                "mean_width_class1": float(np.nanmean(widths2[yt_int == 1])) if (yt_int == 1).any() else float("nan"),
                "deployable": False,
            })

            # --- Construction 3: observable-taxonomy group-conditional interval ---
            covered3, widths3, band3 = group_conditional_interval(
                p_final_cal, triage_f, yf, p_test, triage_t, yt)
            cov3 = conformal_coverage_report(yt, covered3, prefix="group_cond_")
            row3 = {
                "model": model_name, "seed": seed, "construction": "3_group_conditional_triage_band",
                "coverage": cov3["group_cond_coverage"],
                "coverage_class0": cov3["group_cond_coverage_class0"],
                "coverage_class1": cov3["group_cond_coverage_class1"],
                "empty_set_rate": float("nan"), "empty_set_rate_class0": float("nan"),
                "empty_set_rate_class1": float("nan"),
                "mean_set_size": float("nan"),
                "mean_width": float(widths3.mean()),
                "deployable": True,
            }
            for band in ("high_acuity_1_2", "mid_acuity_3", "low_acuity_4_5"):
                m = band3 == band
                row3[f"coverage_{band}"] = float(covered3[m].mean()) if m.any() else float("nan")
                row3[f"width_{band}"] = float(widths3[m].mean()) if m.any() else float("nan")
                row3[f"n_{band}"] = int(m.sum())
            rows.append(row3)

            print(f"  seed {seed}: LAC class1 cov {cov1['mondrian_lac_coverage_class1']:.4f} "
                  f"(empty-set class1 {cov1['mondrian_lac_empty_set_rate_class1']:.4f}) | "
                  f"ECI-Mondrian class1 cov {cov2['mondrian_eci_coverage_class1']:.4f} | "
                  f"group-cond class1 cov {cov3['group_cond_coverage_class1']:.4f}")

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "m2_mondrian_conformal.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Headline comparison against the marginal numbers already on record
    # (results_hybrid.csv / m2_hybrid_conformal_summary.csv), for the
    # Hybrid GRU-D model specifically, since that is the model the
    # reviewer's cited numbers (0.0759 / 0.1037 class-1 coverage, ~48.6%
    # empty-set rate) refer to.
    hybrid_lac = out_df[(out_df.model == "Hybrid GRU-D") & (out_df.construction == "1_mondrian_lac_sets")]
    print("\n--- Hybrid GRU-D, label-conditional MAPIE sets vs. the marginal-only numbers on record ---")
    print(f"Marginal-only (results_hybrid.csv, on record): class-1 coverage 0.0759, empty-set-class1 ~0.486")
    print(f"Label-conditional (this script), 5-seed mean: "
          f"class-1 coverage {hybrid_lac['coverage_class1'].mean():.4f} +/- {hybrid_lac['coverage_class1'].std():.4f}, "
          f"empty-set-class1 {hybrid_lac['empty_set_rate_class1'].mean():.4f} +/- {hybrid_lac['empty_set_rate_class1'].std():.4f}, "
          f"mean set size {hybrid_lac['mean_set_size'].mean():.4f} "
          f"(price of conditional validity: set size inflation vs. marginal-only, must be reported alongside)")


if __name__ == "__main__":
    main()
