# Round 6: Methods and Results after revision (2026-10-07)

Same rubric and weights as round 5 (`R5_rubric_rescore.md`); nothing re-weighted. Scorer: the drafting assistant (an independent reviewer may score lower).

## What changed

Methods
- New subsection "Comparators and Diagnostic Analyses": shortcut diagnostic with all four model sets, acuity-stratified diagnostic with the 20-event rule,
  duration quartiles with training-period cut points, comorbidity ablation, landmark-time comparison and the fixed set (seed-averaged tree score).
- New subsection "Statistical Analysis": Brier, ECE (ten equal-frequency bins), calibration slope and intercept (cited), reliability curves with bootstrap
  intervals, net benefit with its formula and grid (cited), subgroup and calendar-year discrimination, standardized differences with the formula (cited).
- A paragraph separating the analyses fixed in the analysis plan from those added after the first results, listed by name.
- New subsection "Reporting and Reproducibility": TRIPOD+AI statement with a 27-item checklist (Supplementary Table S3) and the literature search protocol.
- Conformal subsection: within-year permutations, coverage by calendar year, the risk threshold for label 1, and the suite on the two other models.
- Four references added after checking against Crossref: Collins 2024 (TRIPOD+AI), Vickers 2006, Van Calster 2019, Austin 2009.

Results
- Every added analysis is marked as added (two subsubsection titles, inline elsewhere).
- Interpretive clauses removed ("so the signal lies in ...", "the workflow quantities therefore ...", "so the trees were better", "at the price of wider
  intervals", the explanation of the comorbidity shift); each remains in the Discussion.
- Hand-typed "0.90 target" for label delay replaced by the stored values; "10,000" by the bootstrap macro.
- 1,782 to 1,498 words.
- Supplementary tables now cited in order S1 to S12 (label references).

## Methods: 9.5 (was 8.3)
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Data source, ethics, access, version | 1 | 0.75 | dataset version of the hospital module is still an author placeholder |
| Cohort, split, leakage prevention | 1.5 | 1 | |
| Outcome and prediction time | 1.5 | 1 | |
| Inputs, preprocessing, timing audit | 1 | 1 | |
| Models, tuning budget, training reproducible | 1.5 | 1 | |
| Every reported analysis described | 1.5 | 1 | checked against each Results paragraph, including NEWS2 on complete stays, coverage by year, the label-1 threshold |
| Conformal methods with equations and deviations | 1 | 1 | |
| Reporting guideline (TRIPOD+AI) | 0.5 | 1 | statement and checklist |
| Code availability | 0.5 | 0.5 | repository URL is still an author placeholder |

Both remaining deductions need input from the author. Not in the rubric but a reviewer may raise it: Methods is about 2,600 words.

## Results: 9.8 (was 8.3)
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Cohort description with characteristics table | 1.5 | 1 | |
| Each analysis has a Methods description | 1 | 1 | |
| Uncertainty reported | 1.5 | 1 | |
| Pre-specified and added analyses distinguished | 1 | 1 | |
| No interpretation in Results | 1 | 1 | remaining framing sentences state comparability, not meaning |
| Organization, question first | 1 | 1 | |
| Concision, tables not narrated row by row | 1 | 0.75 | 1,498 words; the coverage-level paragraph still restates eight cells of Table X |
| Floats cited in order and consistent with text | 1 | 1 | main and supplementary |
| Calibration, subgroups, usefulness reported | 1 | 1 | |

Checks: verifier PASS (1,429 records), 0 undefined references in main and supplement, main floats and supplementary tables in citation order,
PDF text proofread for the new subsections.
