# ECI vs. periodic recalibration on MARGINAL coverage — pre-registration

**Written 2026-09-20, BEFORE `m2_eci_vs_recalibration.py` exists or any comparator in this document has been run.**

## Why this exists

CLAUDE.md §2/M2 names the module's Track A primary research contribution: *replace standard split-conformal calibration with ECI for adaptive intervals under
temporal non-stationarity.* The measured support is `M2_Findings_Dossier.md` §24.3 — MAPIE under-covers on the chronological primary split (0.877–0.885)
while ECI holds 0.9013–0.9016.

`M2_MondrianECI_PreRegistration.md` Part 4 then found that, for the *label-conditional* problem, a static method with periodically refreshed thresholds restores
validity with smaller sets than an online method (verdict NOT NEEDED). The same cheap remedy has **never been tried against ECI on the marginal problem**. If it
matches ECI there too, the module's stated primary contribution is not needed either, and a manuscript must say so. Finding that out here is better than a
reviewer finding it.

This is a test of ECI, not of anything the module has built on top of it. It is symmetric: an outcome favouring ECI is reported exactly as prominently.

## Comparators

All evaluated on the same stream (final-conformal-calibration fold followed by the chronological test window), the same predictions, the same 3 GRU-D-family
models × 5 seeds. Labels assumed immediate (lag 0) for every method. All are deployable: only labels that arrived before the point being predicted are used.

| id | method | thresholds |
|---|---|---|
| M1 | static split-conformal (MAPIE-equivalent) | one marginal quantile of \|y − p\| from the final-calibration fold, never refreshed |
| M2 | **ECI** — the module's established implementation, unchanged | `run_eci_4way`: two-stage lr sweep on `calibration_selection`, then one online stream over final-cal + test |
| M3, M4 | **M1 + periodic recalibration** | recomputed every R = 2,000 arrivals from the most recent W arrivals, W ∈ {10,000; 40,000} |

M2 is run exactly as the module has always run ECI — no saturation guard, no re-tuning — so the comparison is against ECI as it was reported, not a
handicapped or specially-tuned variant. (`m2_hybrid_common.run_eci_4way`.) M3/M4 reuse the recalibration logic and window grid already used in Part 4 of the
Mondrian pre-registration and are not retuned; only two windows are tried.

All four produce an interval `[p − q, p + q]` clipped to [0, 1], so coverage (`y ∈ interval`, identical to `|y − p| ≤ q`) and mean interval width are directly
comparable and use the module's own ECI convention.

## Definitions (fixed now)

- **Valid**: marginal coverage ≥ 0.89 (5-seed mean, test window) — the same band used in Part 4 and applied identically to every method.
- **Efficiency**: mean clipped interval width, compared **only among valid methods**. Tie tolerance 0.02.
- Class-1 coverage is reported for completeness but is **not** part of the decision: every method here is marginal and all are known to collapse on the minority class.

## Decision rule for ECI's marginal claim (Hybrid GRU-D v2 primary; the other two models are replication)

- **ECI NOT NEEDED** if M3 or M4 is valid with mean width ≤ M2's width + 0.02.
- **SUPPORTED — sole valid method** if M2 is valid and neither M3 nor M4 is.
- **SUPPORTED — efficient** if M2 is valid and each of M3/M4 is either invalid or wider than M2's width + 0.02.
- **ECI INVALID** if M2 itself fails the validity band.
- Anything else: **INCONCLUSIVE**, with the numbers.

## Correctness checks that must pass before any comparison is read

- **T6**: M1's marginal coverage must match the MAPIE classification-set marginal coverage already recorded in `data/m2_v2_conformal_summary.csv`
  (same score, same fold) to within 0.003 on every model. This ties the new code to the module's existing MAPIE numbers.
- **T7**: M3 with a window covering the whole calibration fold and no further recalibration reproduces M1 exactly.

## Inference

No patient-cluster bootstrap (invalid for online/recalibrating methods). Reported: 5-seed SD, and descriptive coverage across 10 contiguous stream blocks and the
4 real calendar years (unbalanced: 2017 holds 82% of the window). Stream blocks are not calendar periods. Block spreads are descriptive, not a significance test.

## Commitments

- The result is reported whatever it is. If ECI is NOT NEEDED, the dossier, report skeleton and framing note say the module's primary contribution is not
  supported against the cheap remedy, and the contribution is restated as the empirical characterization.
- Thresholds, windows and the decision rule above are not adjusted after seeing results.
- Not tested and not to be assumed either way: label lag, other recalibration windows, or any calendar-time window (none exists in this cohort).

---

## RESULT (recorded 2026-09-20). Verdict: **ECI NOT NEEDED**, on all three models.

Pre-registration timestamp 02:24:22; script `m2_eci_vs_recalibration.py` first run 02:25:56. Checks T6 (M1 reproduces the module's recorded MAPIE marginal coverage to within 0.003 —
in fact to ≈0.0002 on every one of the 15 model × seed runs) and T7 (recalibration that never fires reproduces M1) passed as hard asserts before any comparison was printed.
5-seed means, marginal coverage and mean clipped interval width:

| model | static (MAPIE-equiv.) | **ECI** (established impl.) | recal W = 10k | recal W = 40k |
| :--- | :--- | :--- | :--- | :--- |
| Hybrid GRU-D v2 (primary) | 0.8853 / 0.342 (invalid) | 0.9016 / **0.552** | 0.8996 / **0.519** | 0.8984 / **0.499** |
| Temporal-Only GRU-D v2 | 0.8769 / 0.527 (invalid) | 0.9015 / **0.722** | 0.8995 / **0.697** | 0.8976 / **0.681** |
| Frozen-Temporal Head Control v2 | 0.8774 / 0.497 (invalid) | 0.9013 / **0.726** | 0.8997 / **0.713** | 0.8979 / **0.693** |

**Decision rule outcome.** On every model, M3 and M4 are valid (marginal coverage ≥ 0.89) with width **narrower** than ECI's → **ECI NOT NEEDED**.

**What is and is not being said.** ECI does what it claims relative to a *never-refreshed* split-conformal baseline (0.9016 vs 0.8853): that comparison, reported in
`M2_Findings_Dossier.md` §24.3, is true. What this test shows is that the comparison was against a strawman for this purpose — the same static method with its
threshold simply refreshed every 2,000 arrivals achieves the same validity with tighter intervals. Under this cohort's shift, immediate labels, and these two windows, the online
adaptivity is not needed for marginal validity. The module's roadmap-stated primary contribution is therefore **not supported against the cheap remedy** and may not be presented as if it were.

**Descriptive nuance the decision rule does not credit (Hybrid, 10 contiguous stream blocks — not calendar periods).** ECI's coverage is steadier: SD 0.0041 vs 0.0073 (W=10k), 0 of
10 blocks below 0.89 vs 1 of 10 (minimum 0.8827). Recalibration is narrower on average (ECI wider by 0.033), though per block ECI is wider in only 5 of 10. Real calendar years:
ECI 0.901 / 0.904 / 0.910 / 0.880, recal W=10k 0.899 / 0.902 / 0.912 / 0.890 (years 2017–2020; 2020 has 723 rows). A stability argument for ECI exists but is small, rests on a single
under-covered block, and is descriptive — the pre-declared rule was mean validity plus width, and the verdict is not revised on this basis.

**Limits.** Labels assumed immediate for every method — ECI and recalibration both require them, so this is symmetric, but a lag sweep may separate them. *(Update 2026-09-21: the lag sweep was run — `M2_LabelLag_PreRegistration.md`, dossier §24.3d — and did not change the verdict at realistic lags.)* Only two recalibration windows
were tried; the windows are in arrivals, not calendar time (no real calendar resolution finer than a year exists in this cohort). The test window has 4 real years, 82% in 2017.
