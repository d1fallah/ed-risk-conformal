# M2 revision analyses: pre-specification (for supervisor sign-off)

Written 2026-10-03, before any revision analysis is run. Prompted by the Q1 review forwarded by the
supervisor. Every analysis reuses the saved per-stay predictions (`data/m2_v2_predictions.csv`; five
seeds; chronological split) and the existing streaming code. No model is retrained, and no
threshold, quantile or base rate already reported is changed. The original results stay the primary
record; these are labelled as revision analyses.

Review points already answered in text (no analysis needed), see `M2_Revision_Response.md`:
- task timing;
- the between-period scope;
- presentation;
- the stale points that describe the pre-alignment-fix results.

## E3. Adaptive set-valued baselines (review point 1)

- **Question.** Is "adaptive updating added nothing" an artifact of running ECI on score intervals?
  Would an adaptive method that produces *prediction sets* behave differently?
- **Methods.** Run on the same marginal LAC score s_t = 1 − p̂_{y_t}(x_t) used by static split-conformal,
  each returning the set {y : 1 − p̂_y(x) ≤ q_t}:
  - ACI (Gibbs and Candès; `aci` in the vendored ECI repository);
  - ECI (vendored `ECI`, unchanged).

  This complements Mondrian-ECI, which is already reported and is class-conditional.
- **Protocol (identical to the existing ECI arm).**
  - Three GRU-D-family models, five seeds, α = 0.10.
  - Learning rate chosen from {0.01, 0.05, 0.1} on the calibration-selection subset: coverage closest to target, then smallest mean set size.
  - Warm-up on the final-calibration subset, then the test stream in the existing order.
  - Labels immediate, plus one delay at the middle of the pre-specified realistic band (240 arrivals).
- **Metrics.**
  - Marginal coverage.
  - Class-0 and class-1 coverage.
  - Mean set size and the share of deteriorating patients receiving {1}.
  - All compared with static LAC sets (and static Mondrian-LAC for the class-1 metrics).
- **Decision rule (fixed now).** An adaptive set method "improves on static" only if it is valid
  (marginal coverage ≥ 0.89) and has a smaller mean set size than static LAC on all three models.
  Otherwise the manuscript's conclusion stands and now also covers set-valued adaptive methods.
- **Reported whatever the outcome**, in a supplementary table and one Results paragraph.
- **Cost.** Streaming over the saved predictions; estimated well under one hour on CPU.

## E4. Controlled within-period drift: stress test (review point 2)

- **Question.** The chronological stream cannot test drift within a deployment period, because order
  within a year is not recoverable. Does the evaluation detect a benefit from adaptive updating when
  such a drift is known to exist? A comparison that could only ever favour static calibration would be
  weak evidence.
- **Design.** Two drift types are imposed on the test stream of each model and seed, starting at its
  midpoint. The model, the calibration and the labels are untouched.
  1. **Score miscalibration:** logit(p̂) shifted by δ ∈ {+0.5, +1.0} for every stay after the
     change point.
  2. **Case-mix change:** the stream re-ordered so that the stays after the change point are the
     upper half by triage acuity severity (the base rate rises).
- **Methods compared:**
  - static split-conformal;
  - periodic recalibration (W = 10k, R = 2k);
  - ECI (intervals);
  - ACI and ECI on LAC sets (from E3).
- **Metrics.**
  - Coverage in rolling windows of 2,000 arrivals.
  - Coverage and width/set size before and after the change point.
  - Time to return within 0.01 of target after the change.
- **Expected direction, written in advance.** Static calibration should lose coverage after the change
  and the adaptive methods should recover it. If they do not, that is reported.
- **Scope in the paper.** A supplementary stress test, labelled as synthetic drift imposed on real
  predictions. It is not a new finding about the hospital, and it does not change the headline, which
  concerns the observed between-period shift.
- **Cost.** Minutes per configuration; estimated about one hour in total.

## S1. Significance of the width differences (review point 5)

- **Current state.** The Limitations say the width differences "carry no significance test".
- **Test.** For each model, the contrasts ECI − static, recal(10k) − static and recal(40k) − static in
  mean interval width:
  - patient-cluster bootstrap within each seed (10,000 resamples, patients drawn with all their stays);
  - Holm correction across the 9 contrasts.

  This is the paper's existing rule for discrimination: significant only if Holm-significant and all five
  per-seed 95% CIs exclude zero. A per-seed signed-rank test is not used, because with 5 seeds it cannot
  go below p = 0.0625.
- **Requires** saving per-stay widths from the streaming scripts. The runs are deterministic, so the
  reported means must reproduce exactly; that is checked first.

## S2. Uncertainty for the deterministic tree model (review point 5)

- XGBoost-Matched is deterministic, so its "± 0.0000" seed SD will puzzle readers. Its AUROC/AUPRC
  cells will report the patient-cluster bootstrap 95% CI (10,000 resamples, seed 42; all seeds are
  identical), marked as a different uncertainty measure from the neural models' seed SD.
- The deltas already use bootstrap CIs and are unchanged.

## Sign-off

Supervisor sign-off received 2026-10-03, relayed by the author in session:
- [x] E3 approved: run.
- [x] E4 approved: run, reported as supplementary/secondary only.
- [x] S1 approved: run.
- [x] S2 approved: run.
