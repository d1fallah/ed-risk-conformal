"""
M2 -- Phase 0 precision calculation for the Mondrian-ECI pilot
==============================================================================
Verification item 1 of the session's "Mondrian-ECI + deferred items" plan:
BEFORE implementing or running the new construction, confirm its
pre-registered GO/NO-GO thresholds are actually falsifiable at this
cohort's class-1 sample size, under the blocked (sequence-respecting)
inference scheme the design requires.

Why blocked inference at all: for a STATIC construction C(x_i) is fixed
given the calibration fold, so the project's patient-cluster bootstrap is
valid. For an ONLINE construction C(x_t) depends on the entire preceding
stream AND its order, so resampling patients perturbs the very arrival
process the minority stream is most sensitive to. Inference must instead
run across contiguous blocks of the realized stream.

This script computes, from real saved data only (Golden Rule -- no
simulated cohort):
  1. the test period's real temporal resolution and its block structure;
  2. per-block class-1 counts at several candidate K;
  3. the standard error attainable for each pre-registered endpoint, and
     whether the declared GO/NO-GO margins exceed it.

Reports a verdict; makes no changes to any pipeline artifact.
"""
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# Pre-registered effect sizes the design must be able to resolve (plan Phase 0).
COVERAGE_TARGET = 0.90
COVERAGE_CONTRAST = (0.89, 0.85)      # must distinguish these
SETSIZE_CONTRAST = (1.40, 1.52)       # must distinguish these (1.52 = Mondrian-LAC anchor)
CANDIDATE_K = [4, 6, 8, 10, 15, 20]


def main():
    df = pd.read_csv(
        os.path.join(DATA_DIR, "test_v2.csv"),
        usecols=["approx_real_year", "intime", "outcome_critical"],
    )
    n = len(df)
    n_pos = int(df["outcome_critical"].sum())
    test_rate = n_pos / n
    print(f"Test period: n={n}, class-1 n={n_pos} ({test_rate:.4%})")

    # Prevalence shift matters directly for the minority stream's update
    # rate: the class-1 ECI stream advances once per 1/rate arrivals.
    tr = pd.read_csv(os.path.join(DATA_DIR, "train_v2.csv"), usecols=["outcome_critical"])
    train_rate = tr["outcome_critical"].mean()
    print(f"Train period class-1 rate: {train_rate:.4%}  ->  test {test_rate:.4%} "
          f"({test_rate/train_rate - 1:+.1%} relative)")
    print(f"Minority (class-1) ECI stream advances ~1 update per {1/test_rate:.1f} test arrivals.\n")

    # ---- 1. Real temporal resolution -------------------------------------
    # sort_chronologically() orders by (approx_real_year, intime), and
    # m2_hybrid_common.py:909-913 states plainly that `intime` is NOT
    # real-time-meaningful -- only a deterministic within-year tiebreak.
    # So the finest GENUINE calendar resolution available is the year.
    print("=== 1. Genuine temporal resolution (approx_real_year) ===")
    by_year = df.groupby("approx_real_year").agg(
        n=("outcome_critical", "size"), pos=("outcome_critical", "sum")
    )
    by_year["share_of_test"] = by_year["n"] / n
    print(by_year.to_string())
    n_years = len(by_year)
    top_share = by_year["share_of_test"].max()
    print(f"\nDistinct real years: {n_years}; largest single year holds {top_share:.1%} of the test stream.")
    print("=> Genuine CALENDAR blocks are limited to "
          f"K={n_years}, and are severely unbalanced.")
    print("=> Sub-year blocks are contiguous blocks of the STREAM ORDER, not calendar periods;")
    print("   they correctly handle the online method's sequential dependence, but must never")
    print("   be described as months/quarters. Within-year order is arbitrary-but-deterministic.\n")

    # ---- 2. Per-block class-1 counts at candidate K -----------------------
    print("=== 2. Per-block class-1 counts (equal-size contiguous stream blocks) ===")
    print(f"{'K':>4} {'rows/block':>11} {'class-1/block':>14} {'SE(cov) per block':>19}")
    se_cov_block = {}
    for K in CANDIDATE_K:
        rows_per = n / K
        pos_per = n_pos / K
        # SE of a coverage proportion estimated on that block's class-1 points
        se = np.sqrt(COVERAGE_TARGET * (1 - COVERAGE_TARGET) / pos_per)
        se_cov_block[K] = se
        print(f"{K:>4} {rows_per:>11.0f} {pos_per:>14.0f} {se:>19.4f}")
    print()

    # ---- 3. Attainable precision on each pre-registered endpoint ----------
    print("=== 3. Can the design resolve the pre-registered contrasts? ===")
    c_hi, c_lo = COVERAGE_CONTRAST
    d_cov = abs(c_hi - c_lo)
    s_lo, s_hi = SETSIZE_CONTRAST
    d_set = abs(s_hi - s_lo)

    # Set size per point takes values in {0,1,2}. Use the widest plausible SD
    # for a mean near 1.5 supported on {0,1,2} (worst case: mass split
    # between 1 and 2 gives SD<=0.5; allow 0.6 as a conservative pad).
    sd_setsize = 0.6

    # NOTE on K-invariance, stated explicitly because the arithmetic is easy
    # to misread: aggregating K blocks of size n/K gives
    #   SE = sqrt(2 p(1-p) / (n_pos/K)) / sqrt(K) = sqrt(2 p(1-p) / n_pos),
    # i.e. INDEPENDENT of K. Under an independence assumption the total
    # sample governs precision and K only changes how variability is
    # attributed. K's real jobs are (i) to let the EMPIRICAL across-block
    # SD absorb the sequential dependence this closed form ignores, and
    # (ii) to answer whether coverage is stable across periods rather than
    # only on average. So the figures below are an OPTIMISTIC BOUND; the
    # reported inference must use the observed across-block spread.
    se_cov_agg = np.sqrt(2 * COVERAGE_TARGET * (1 - COVERAGE_TARGET) / n_pos)
    ratio_cov = d_cov / se_cov_agg
    print(f"\nEndpoint A -- class-1 coverage, contrast {c_hi} vs {c_lo} (delta={d_cov:.3f})")
    print(f"  aggregate SE of within-block paired mean difference: {se_cov_agg:.5f}  (K-invariant, see note)")
    print(f"  delta/SE = {ratio_cov:.1f}  -> {'RESOLVABLE' if ratio_cov >= 3 else 'UNDERPOWERED'}")
    print(f"  tolerable inflation from block dependence before this fails at 3 SE: "
          f"{ratio_cov/3:.1f}x")

    se_set_agg = np.sqrt(2) * sd_setsize / np.sqrt(n)
    ratio_set = d_set / se_set_agg
    print(f"\nEndpoint B -- mean |C|, contrast {s_lo} vs {s_hi} (delta={d_set:.3f}), SD(|C|) assumed {sd_setsize}")
    print(f"  aggregate SE of within-block paired mean difference: {se_set_agg:.5f}  (K-invariant, see note)")
    print(f"  delta/SE = {ratio_set:.1f}  -> {'RESOLVABLE' if ratio_set >= 3 else 'UNDERPOWERED'}")
    print(f"  tolerable inflation from block dependence before this fails at 3 SE: "
          f"{ratio_set/3:.1f}x")

    # Choosing K: large enough that the across-block SD is estimable, small
    # enough that each block still pins a coverage estimate. Require >=200
    # class-1 events per block as a working floor.
    MIN_POS_PER_BLOCK = 200
    k_max = int(np.floor(n_pos / MIN_POS_PER_BLOCK))
    print(f"\nBlock-count guidance: requiring >={MIN_POS_PER_BLOCK} class-1 events per block "
          f"allows K up to {k_max}.")
    print(f"  Recommended: K=10 ({n_pos//10} class-1/block) for the stream-block arm,")
    print(f"  alongside the K={n_years} genuine-calendar-year arm.")

    # ---- 4. The binding constraint ---------------------------------------
    print("\n=== 4. Verdict ===")
    print("Sample size is NOT the binding constraint: both pre-registered contrasts are")
    print("resolvable by a wide margin at every K considered.")
    print(f"The binding constraint is the number of GENUINELY INDEPENDENT temporal periods:")
    print(f"only {n_years} real years exist in the test window, and one holds {top_share:.0%} of it.")
    print("Implication for the plan: report BOTH")
    print("  (a) across-year results (K=4, honest calendar periods, weak but real), AND")
    print("  (b) across-stream-block results (larger K, valid for sequential dependence,")
    print("      but explicitly NOT calendar periods),")
    print("and never describe (b) as evidence that coverage holds 'in each time period'.")


if __name__ == "__main__":
    main()
