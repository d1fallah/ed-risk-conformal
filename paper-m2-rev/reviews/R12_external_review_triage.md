# Round 12: triage of an external review of the revised manuscript (2026-10-07)

The review (by another model, pasted by the author) gave 7/10, Medium Q1 potential, "Moderate at the upper end". Every claim was checked against the paper or the data before any
change. The scorer of this file is the drafting assistant; the review's score is another model's estimate.

## Done in this round (no long compute)

| Point | Check | Outcome |
|---|---|---|
| C1 recording shift | `p2_19`, existing data | **Mechanism found**: from 2013 to 2016 the first ED vital-sign row came to repeat the triage measurement (2% of stays in 2011, 44% in 2014, 97% in 2016, 99.6% in the test period); the review's guess (late batch entry) was not what the data show. Reported in Results (Table S5) and Limitations. Models not refitted. |
| M1 audit of own unpublished model | wording | "by the authors" to "by the author"; contribution 1 reframed as a demonstration of a design the pipeline makes easy; no claim of correcting the literature. |
| M2 workflow baseline re-encodes triage | `p2_20` | **Review correct, and larger than it said**: arrival information alone reaches 0.880 (workflow alone 0.843); workflow adds +0.031 (CI 0.028 to 0.034), ED duration alone +0.003. The headline "reproduces most of the discrimination" was withdrawn in the Abstract, Introduction, Discussion and Conclusion. |
| M5 arrival versus two hours, different populations | `p2_21` | On the same 78,492 stays and the same outcome the two-hour trees exceed the arrival trees by +0.027 (CI 0.024 to 0.029), more than the headline comparison (0.887 vs 0.905); the sentence "most of the usable information is in the triage profile" was replaced by the paired figures. |
| M6 validity decisions, static calibration | `p2_22` | Review correct. Trees: 0.888 (0.885 to 0.890), threshold not met (2.4% of resamples reach it). Primary: 0.892 (0.890 to 0.894), indistinguishable from 0.89. Logistic regression: 0.895. "Only narrowly valid" replaced. (A first statement of this was fixed before the review was pasted.) |
| M7 PPV is a property of the model | `p2_23` | **Review correct and verified**: the static {1} is an exact threshold rule; PPV 0.395 at sensitivity 0.687 (plain threshold: 0.395), Mondrian-ECI 0.372 at 0.710 (plain threshold: 0.375). "The new information here is the cost" removed; what conformal adds is the class-1 coverage guarantee. Supplementary Fig. S1. |
| M9 intercept recalibration competitor | `p2_24` | Calibration in the large restored (6.2% to 7.8%, observed 7.8%), Brier 0.0494 to 0.0485. A first variant (calibration scores shifted by their own fitted intercept, test blocks by a different one) gave static coverage 0.885; the author asked whether that was fair and it was not. With the same shift on calibration and test scores and the quantile recomputed, static coverage is 0.893 (0.892 without recalibration); a rolling quantile on the recalibrated score gives 0.900. Only the matched variant is reported. Table S20. |
| Minor: 525 extra positives | `p2_25` | All 525 are ICU admissions later than 12 h after departure but within 24 h of the landmark; sentence added to Methods. |
| Minor: ECI scale c | vendored code | c = 1 in the reference code; stated. |
| Minor: 1,780-day window, 7 rows | text | Window of the benchmark pipeline code, about five years; 7 rows keep every row by the landmark for nearly every stay. |
| Minor: abstract denominators | text | Abstract now gives the 78,492 landmark-eligible test stays. 248 words. |
| Minor: C4 interval asymmetric | evidence file | Not a typo: -0.00105 with interval -0.00145 to -0.00065, rounded to -0.0010 and [-0.0015, -0.0007]. |
| Minor: reference [37] DOI prefix | Crossref, doi.org | Prefix 10.64898 resolves; the Crossref record exists (medRxiv posted-content). |
| Minor: literature search scope | OpenAlex | Two broader queries added (7 in total). Found informative presence (Sisk 2021, Groenwold 2020) and label-leakage framework (Davis 2023) literature; added to Related Work; the novelty claim is limited accordingly. |

## Done after the author asked for the cheap compute first

| Point | Check | Outcome |
|---|---|---|
| C1 and M3 on the classical models | `p2_26` (6.5 min) | Removing the copied triage row changes the AUROC by -0.0005 (trees) and -0.0006 (logistic regression); removing the timing features lowers it by 0.006 and 0.005; trees minus logistic regression is +0.0087 to +0.0099 in every variant. The timing features and the copied row do not explain the advantage of the trees. Supplementary Table S6. The recurrent models are not covered. |
| C1 on the recurrent models (inference only) | `p2_27`, rule fixed first (DEVIATIONS P2-30) | Hybrid GRU-D end to end: seed-mean change -0.00022 (-0.00066 to +0.00019), below the 0.002 threshold, so the refit was not run and C1 stands. Temporal-Only: -0.0019 (just below). |

## Deferred by the author (long compute)

| Point | What it needs |
|---|---|
| C1 sensitivity (recurrent models) | Refit Hybrid GRU-D and Temporal-Only (about 5.5 h of GPU in the earlier run) after removing ED vital-sign rows identical to the triage measurement and charted shortly after arrival, or on training years from 2016 only. Report beside the pre-specified result, which stays primary. |
| M3 timing-feature ablation | Trees without timing features; GRU-D without the mask and elapsed-time inputs. |
| M4 tuning grid | Extend the GRU-D grid until the selected configuration is interior; same number of configurations for the trees (random search). Keep or drop the ranking claim by the result. |
| M8 split and comorbidity ablation | Alternative split with the later stays of returning patients in the test period; comorbidity ablation at the landmark (trees and logistic regression are cheap). |

## Author-only items
Placeholders (MIMIC-IV hospital-module version, funding, acknowledgment, conflict of interest, repository URL, affiliation and ORCID, AI-use wording); check JBHI's current abstract and
page limits; confirm that the analysis plan existed before training as the Methods state.

## Not done, with reasons
- "Deterioration" in the title: the abstract defines it; a longer title naming ICU admission or death was rejected earlier (R9).
- Moving the decision-curve and subgroup paragraphs to the supplement: the author asked not to trim content for length.
- A fresh independent score: the rubric of rounds 5 to 11 does not measure contribution size, which is where the review places the largest deduction.

## Effect on the section scores
Methods, Results, Discussion, Abstract, Introduction and Related Work changed in content (claims narrowed, analyses added). I have not re-scored them in this round; the previous scores
for those sections should be treated as stale until the deferred analyses are in.
