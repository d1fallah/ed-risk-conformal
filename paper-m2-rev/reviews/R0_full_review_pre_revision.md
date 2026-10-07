# Round 0: full-paper review of the M2 manuscript as it stood before the revision

Reviewer stance: strict SCI-journal reviewer for IEEE JBHI (Q1). The manuscript read is `paper-m2/` as of 2026-10-04 (`main.pdf`, 8 pages;
abstract, introduction, related work, methods, results, discussion, conclusion; supplement not re-read). The evidence checked against it is
Phase 0 and Phase 1 of the revision (`mimic4ed-benchmark-main/results/revision_v2/`), which was produced after this version was written.
This is a self-assessment made by the same assistant that edits the paper; it is a judgement, not an external review, and says so.

## Verdict

| | |
|---|---|
| **Overall verdict** | **Weak as submitted** (the conformal-calibration analysis is Moderate-to-Strong; the clinical task around it is Weak) |
| **SCI readiness** | **4 / 10** |
| **Q1 potential** | **Low** as submitted; **Medium** if the task is redefined at a stated prediction time and the conformal analysis is repeated on it |

## Main strengths

1. A chronological, patient-level split with fixed seeds, five-seed reporting, pre-specified contrasts with Holm correction and a "significant only if all five per-seed intervals exclude zero" rule.
2. Honest negative result on adaptive calibration, with a controlled-drift test that shows the evaluation could have detected a benefit, and with label-delay analysis.
3. A reproducibility discipline (every number read through a provenance accessor, separate verification script) that most clinical-ML papers lack.
4. Class-conditional coverage reported for the deteriorating class, which is the clinically relevant quantity.

## Main weaknesses

1. The prediction task is not defined at a prediction time. Inputs run to ED departure and beyond.
2. The headline discrimination figure is not interpretable as risk prediction, and the comparison with NEWS2 mixes information sets.
3. The conformal conclusions rest on a test stream with no real within-year order and one step change, so the negative result is narrow.
4. No cost of the recommended class-conditional sets (false alarms) is reported.

## Critical problems (must be fixed before submission)

| # | Problem | Evidence (Phase 0 / Phase 1) | Fix |
|---|---|---|---|
| C1 | **Inputs are not available at any stated prediction time.** The sequence holds the last ten ED rows of the whole stay (`tail(10)`), including rows charted after ED departure. | Retained rows charted after departure: 82,570 (32,897 stays) in training and 24,663 (7,810 stays) in test. 85% to 88% of stays have a retained row charted after 2 h from arrival; 93% after 1 h. | Redefine the task at landmarks (triage; 2 h) with inputs restricted to charttime up to the landmark (Phase 2, R2, R5). |
| C2 | **A workflow shortcut explains most of the discrimination.** Measurement counts, sequence length, ED duration and triage missingness are post-decision workflow quantities. | A logistic regression on these alone reaches AUROC 0.843 (95% CI 0.837 to 0.848); with triage missingness 0.868; Hybrid GRU-D 0.937. | Report the shortcut baselines; train landmark models whose inputs exclude post-landmark information; judge them against timing-aware tree and logistic baselines (R7). |
| C3 | **Time-anchored labels (1, 6, 12 h) do not test a horizon.** The inputs extend past the horizon for most stays. | Test stays with a retained measurement charted after 1 / 6 / 12 h: 93.0% / 44.1% / 16.3%. | Replace with a fixed 24 h horizon after a landmark (R5), or drop. |
| C4 | **The truncation statement in the configuration is wrong.** The build counter reports zero truncated stays because the helper truncates first. | 9,779 training stays (3.39%) and 4,837 test stays (5.74%) exceed ten rows and lose their earliest rows, most of them within 2 h of arrival. | Report the measured figures; the landmark inputs are rebuilt anyway. |

## Major problems

| # | Problem | Fix |
|---|---|---|
| M1 | **NEWS2 is not an information-matched comparator** (triage snapshot against the whole stay), and the stated shortfall (AUROC 0.55) is explained largely by treating missing parameters as normal. | At a 2 h landmark compare against NEWS2 recomputed from the last values charted by the landmark; keep the sensitivity analysis. |
| M2 | **Unequal tuning effort.** Temporal-Only GRU-D used one fixed configuration, XGBoost-Matched a two-parameter override, mTAN eight candidates. | Equal budget for every tuned model of the landmark task, fixed before training (`TUNING_BUDGET.md`). |
| M3 | **Calibration shift confounded with a recording-practice shift.** 4.6% of training stays and 0.02% of test stays have no ED vital signs after triage; prevalence rises from 5.4% to 8.1%. | State both in the cohort description, and show the effect on the landmark cohort (2 h cohort: 41.8% against 0.08%). |
| M4 | **Conformal guarantee statements without an alpha sweep, without operating characteristics.** | Alpha sweep (0.20, 0.10, 0.05, 0.01) with the feasibility check of the class-1 quantile; PPV, sensitivity, false alarms per 100 patients, abstention by class. |
| M5 | **ECI settings differ from the reference configurations** (T_burnin 5 against 100 to 1,000; step-size grid including 0.01), and the paper says it ran the authors' implementation "with no change". | State the differences in the Methods and the Limitations. |
| M6 | **Single site, single cohort, proxy label.** No external validation. | State clearly; no overclaiming; a second-site validation is named as the experiment that would be needed. |
| M7 | **Order within a year is not real.** The stream is sorted by shifted timestamps inside each year block. | Permutation analysis (Phase 1.8: 20 permutations, conclusions unchanged) reported as a robustness check; keep the limitation. |

## Minor problems

Abstract states a headline AUROC for a task without a prediction time; "temporal modeling helps" depends on the metric and the task; the introduction
promises three contributions but the second is a negative result and the third is a known phenomenon (marginal against class-conditional coverage);
related work claims absence ("we found no comparison") from a targeted search, which is acceptable only with the search stated; several tables were
moved to the supplement, so the main text cannot be audited without it; second-best values are not underlined; author, affiliation, ORCID,
repository URL and the acknowledgment and conflict-of-interest statements are placeholders; the AI-use disclosure required by IEEE is missing.

## Most important improvements before submission

1. Landmark redesign with an explicit prediction time and the information available then (C1 to C3).
2. Timing-aware and logistic baselines under the same inputs, with equal tuning (M2), and the shortcut diagnostic in the paper (C2).
3. Operating characteristics and alpha sweep for the recommended class-conditional sets (M4).
4. An honest scope statement: one site, retrospective, proxy outcome, no clinical benefit demonstrated.

## Scores by section (as submitted), strict reviewer

| Section | Score /10 | Reason |
|---|---|---|
| Abstract | 5 | Clear structure, but the headline AUROC is for a task without a prediction time, and the key limitation (end-of-stay inputs) is absent from it. |
| Title | 6 | Descriptive and honest, but silent on the prediction time and on the negative result. |
| Introduction | 5 | The gap (calibration under shift) is stated; the clinical use case is not, and "retrospective classification at the end of the stay" undercuts the motivation of early warning. |
| Related work | 6 | Well organised and verifiable; no comparison of what a landmark formulation requires; the novelty claim rests on a targeted search. |
| Methods | 5 | Reproducible detail, but the input definition is the flaw (C1), tuning budgets unequal (M2), ECI deviations not stated (M5). |
| Results | 5 | Careful, but dominated by comparisons that are not interpretable (C1, C2). |
| Discussion | 6 | Good reasoning on the negative result; relies on the flawed task for the clinical reading. |
| Conclusion | 6 | Calibrated, but inherits the task problem. |
