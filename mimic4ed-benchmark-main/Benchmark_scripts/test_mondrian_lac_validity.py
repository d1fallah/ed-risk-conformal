"""
Regression test: mondrian_lac_sets() must be label-conditionally valid.

Why this exists. Until 2026-09-20 mondrian_lac_sets() calibrated class 0 on
1 - p while scoring label 0 as p at test time, so q_0 sat near 1, label 0 was
always included, and class-0 coverage was EXACTLY 1.0000 instead of ~0.90.
Class-1 coverage was unaffected, so every headline number about the
minority class looked right and the defect survived two review passes. The
only signature was on the majority class: coverage 1.0000 (a guarantee
holding with equality to four decimals is never a coincidence) and singleton
{1} sets being impossible.

The test is deliberately the one a correct implementation must pass for ANY
prevalence and class separation, because label-conditional validity is
distribution-free: mean coverage of each class must sit at 0.90 (never
systematically below; at most a finite-sample sliver above). The tolerance
is derived from n and the replicate count rather than fixed, since a fixed
tolerance mis-fires at low prevalence where a class has few points.

Run:  python Benchmark_scripts/test_mondrian_lac_validity.py   (exit 0 = pass)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_mondrian_conformal import mondrian_lac_sets

REPLICATES = 40
N = 40000
SETTINGS = [(0.08, 0.45), (0.054, 0.30), (0.02, 0.60), (0.30, 0.20)]  # (prevalence, class separation)


def main():
    all_ok = True
    for rate, sep in SETTINGS:
        cov = {0: [], 1: []}
        for r in range(REPLICATES):
            rng = np.random.RandomState(1000 * r + int(rate * 1000))

            def draw(m):
                y = (rng.rand(m) < rate).astype(int)
                return np.clip(rng.beta(2, 25, m) + sep * y, 0.001, 0.999), y

            p_cal, y_cal = draw(N)
            p_test, y_test = draw(N)
            sets, _ = mondrian_lac_sets(p_cal, y_cal, p_test)
            covered = sets[np.arange(len(y_test)), y_test]
            for k in (0, 1):
                cov[k].append(covered[y_test == k].mean())
        for k in (0, 1):
            n_k = N * (rate if k == 1 else 1 - rate)
            se = np.sqrt(2 * 0.09 / n_k) / np.sqrt(REPLICATES)
            mean = float(np.mean(cov[k]))
            ok = (0.90 - 3 * se) <= mean <= (0.90 + 3 * se + 2.0 / n_k)
            all_ok &= ok
            print(f"prev={rate:.3f} class{k}: mean coverage {mean:.4f} (SE {se:.4f})  {'ok' if ok else 'OUT OF BAND'}")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
