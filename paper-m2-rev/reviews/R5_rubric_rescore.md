# Round 5: rubric re-score of every section (2026-10-07)

Why: the author questioned why every section of round 4 ended at exactly 9.0. Those scores were anchored to the target. Here each section is scored with a weighted rubric
written before judging (criteria from the JBHI exemplars read in round 4 and from the drafting guidance). Credit: met = 1, partly = 0.75 or 0.5, not met = 0.
Score = 10 x (weighted credit / total weight). The scorer is still the drafting assistant; an independent reviewer may score lower.

## Title: 8.5
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Names task and setting | 3 | 1 | "Predicting Deterioration in the Emergency Department" |
| Conveys the key design idea | 2 | 1 | "Two Hours After Arrival" |
| Length within JBHI norm (IQR 10 to 15 words) | 1 | 0.5 | 16 words |
| Plain words, no coined jargon | 2 | 0.5 | "Workflow Shortcuts" is coined |
| Accurate to the content | 2 | 1 | |

## Abstract: 9.0
| Criterion | W | Credit | Reason |
|---|---|---|---|
| At most 250 words, one paragraph, self-contained | 1 | 1 | 241 words |
| Problem and gap | 1.5 | 1 | |
| What was done, data, design ("In this study, we") | 1.5 | 1 | |
| Key results with uncertainty | 1.5 | 1 | CI on the main contrast |
| Number density appropriate | 1 | 0.5 | more numbers than typical JBHI abstracts |
| Conclusion supported, no overclaim | 1.5 | 1 | |
| Covers all main contributions | 1 | 0.5 | online updating under drift and the landmark-time result omitted |
| Abbreviations minimal and defined | 1 | 1 | |

## Introduction: 8.9
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Clinical problem with a cited stake | 1.5 | 1 | |
| Current practice and prior approaches | 1 | 1 | |
| Specific gap tied to evidence | 1.5 | 1 | recorded search |
| One explicit aim | 1 | 1 | |
| Approach summarized | 1 | 1 | |
| Contributions concrete and matched to evidence | 1.5 | 0.75 | second contribution vague ("where the recurrent model does and does not improve") |
| Honest scope | 0.5 | 1 | |
| Roadmap | 0.5 | 1 | |
| Language and paragraphing | 1 | 0.75 | fourth paragraph carries gap, aim, and three steps (about 200 words) |

## Related Work: 8.6 (scored in round 4, Section 7; CAREBench since verified and added, criterion 4 unchanged at 0.75 because the closest-work comparison is still prose only)

## Methods: 8.3
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Data source, ethics, access, version | 1 | 0.75 | dataset version of the hospital module is an author placeholder |
| Cohort, split, leakage prevention | 1.5 | 1 | |
| Outcome and prediction time | 1.5 | 1 | end-of-stay task now defined |
| Inputs, preprocessing, timing audit | 1 | 1 | |
| Models, tuning budget, training reproducible | 1.5 | 1 | |
| Every reported analysis described | 1.5 | 0.5 | characteristics/SMD, reliability, decision curve, year blocks, acuity strata, fixed set, literature search are in Results but not in Methods |
| Conformal methods with equations and deviations | 1 | 1 | |
| Reporting guideline (TRIPOD+AI) | 0.5 | 0 | no checklist or statement |
| Code availability | 0.5 | 0.5 | verification pipeline described; repository URL is a placeholder |

## Results: 8.3
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Cohort description with characteristics table | 1.5 | 1 | |
| Each analysis has a Methods description | 1 | 0.5 | see Methods |
| Uncertainty reported | 1.5 | 1 | |
| Pre-specified and added analyses distinguished | 1 | 0.5 | the added analyses (acuity strata, fixed set, decision curve, year blocks) are not labelled as such |
| No interpretation in Results | 1 | 0.75 | a few interpretive clauses ("so the signal lies in ...") |
| Organization, question first | 1 | 1 | |
| Concision, tables not narrated row by row | 1 | 0.5 | about 2,500 words against 400 to 1,600 in the exemplars |
| Floats cited in order and consistent with text | 1 | 1 | |
| Calibration, subgroups, usefulness reported | 1 | 1 | |

## Discussion (including Limitations): 9.0
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Principal-findings opening | 1 | 1 | |
| Each finding interpreted against prior work | 2 | 1 | |
| Mechanisms tested or labelled as hypotheses | 1.5 | 1 | |
| Implications | 1 | 1 | |
| Limitations specific, enumerated, with remedies | 1.5 | 1 | |
| Specific future work | 0.5 | 1 | |
| Does not repeat Results numbers | 1 | 0.5 | many Results values restated |
| Concision | 1 | 0.5 | about 2,300 words against 620 to 1,200 |
| No overclaim | 0.5 | 1 | |

## Conclusion: 9.5
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Restates what was done | 1.5 | 1 | |
| Key findings with few numbers | 2 | 1 | |
| Supported, no new results | 2 | 1 | |
| Implications | 1.5 | 1 | |
| Specific future work | 1.5 | 1 | |
| Length 90 to 400 words | 1.5 | 1 | 251 words |
| Not a repetition of the Discussion | 1 | 0.5 | the reporting recommendations repeat the Implications subsection |

## Tables: 7.4
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Generated and verified | 2 | 1 | |
| Self-contained captions, proxy label | 2 | 1 | |
| Numbered in citation order | 1 | 0.5 | main tables yes; supplementary tables cited out of order (S9 first, in the Introduction) |
| Number of main tables appropriate | 1.5 | 0.5 | eleven main tables; JBHI papers typically have three to six |
| Consistent formatting, typographic minus signs | 1.5 | 0.75 | hyphen used as minus sign |
| No duplication between tables and figures | 2 | 0.5 | contrasts (Table VI, Fig. 2), landmark sweep (Table VII, Fig. 3), coverage levels (Table X, Fig. 6) |

## Figures: 7.4
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Each figure adds information beyond the tables | 2 | 0.5 | three of six duplicate a table |
| Vector, legible at print width | 2 | 0.75 | small labels in Fig. 6 |
| Self-contained captions, proxy label | 2 | 1 | |
| Colour-blind safe, markers distinguish series | 1.5 | 0.75 | colour plus markers, not checked with a simulator |
| Uncertainty shown where relevant | 1.5 | 0.5 | reliability and decision curves have no intervals although the reliability intervals are stored |
| Cited in order | 1 | 1 | |

## References: 8.4
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Every entry verified | 3 | 1 | |
| Complete metadata | 2 | 0.75 | several entries without DOI |
| Recent and relevant | 2 | 1 | |
| Consistent IEEE style | 1.5 | 0.5 | mixed "doi:" notes and "Preprint: arXiv" suffixes |
| Limited reliance on preprints | 1.5 | 0.75 | five preprints among the cited works |

## Summary

| Section | Round 4 (impressionistic) | Round 5 (rubric) |
|---|---|---|
| Title | 9.0 | 8.5 |
| Abstract | 9.0 | 9.0 |
| Introduction | 9.0 | 8.9 |
| Related Work | 8.6 | 8.6 |
| Methods | 9.0 | 8.3 |
| Results | 9.0 | 8.3 |
| Discussion | 9.0 | 9.0 |
| Conclusion | 9.0 | 9.5 |
| Tables | 8.6 | 7.4 |
| Figures | 8.5 | 7.4 |
| References | 8.7 | 8.4 |

Fixable without new data, in order of gain: Methods (describe the added analyses, TRIPOD+AI checklist), Results (label added analyses, trim narration), Tables and Figures (remove
duplicated floats or move them to the supplement, intervals on the reliability plot, minus signs, supplementary numbering), References (DOIs, style), Title (drop the coined phrase).
