# Round 3: strict-reviewer pass after the author asked why the sections stopped below 9

The reviewer is the assistant that wrote the text. The scores are judgements with reasons; nothing was raised to reach a target.

## What round 2 got wrong

Round 2 put the ceiling of every section except Methods and Limitations on "missing external validity" and stopped. That was too quick: four of the listed deductions did not need new evidence
(an interval in the abstract, a calibration plot, a forest plot of the contrasts, a recorded literature search), and two further reviewer questions could be answered from the stored predictions
(temporal stability of the ranking, clinical usefulness by decision curve). Round 3 did that work. Reading the built PDF line by line also found a defect that rounds 1 and 2 had missed:
an unescaped percent sign in the statistics paragraph of the Methods turned "intervals exclude zero. Conformal coverage is judged" into a LaTeX comment, so the sentence that defines the
significance rule and the sentence that opens the coverage criterion were cut in the PDF. It was fixed, all sections were scanned for the same pattern (none left), and references of the form
"Section\ref" (missing the non-breaking space, printed as "SectionII") were corrected. The Methods score of 9.0 in round 2 was therefore too high for the PDF as it stood.

## Changes in this round

| Item | What was done | Source of the numbers |
|---|---|---|
| Abstract | Interval for the tree-minus-recurrent difference added (247 words). | `ph2_contrasts` via `derived.json` |
| Figure: forest plot of the five contrasts (Fig. 2) | Filled markers meet the significance rule, open marker does not. | `ph2_contrasts.json` |
| Figure: reliability plot and decision curve (Fig. 4) | Ten equal-frequency bins with the three best models; net benefit against threshold. | new `p2_12_reliability.py`, `p2_13_year_blocks_decision.py` |
| Temporal stability | AUROC by approximate calendar-year block of the test period (Supplementary Table S8). Ranking unchanged; the tree ensemble is at or above the primary model in every block. | `p2_13` |
| Clinical usefulness | Decision curve; the primary model and the trees have a net benefit above treating all at every reported threshold; their difference is within a few ten-thousandths at 0.10. The text states that the threshold was not elicited and that the models underestimate risk in the upper deciles. | `p2_13` |
| Related work | Recorded search (OpenAlex, five queries, Supplementary Table S9; counts and screening decisions come from `p2_14_literature_search.py` and `p2_15_screening.py`). Four verified references added (label leakage through diagnostic codes in MIMIC-IV; a conformal ED-triage and a clinical-AI overview; a concordant class-conditional preprint). | `p2_14`, `p2_15` |
| Methods | The comorbidity count is tied to the label-leakage literature (index-admission codes excluded). The percent-sign defect was repaired. | text |
| Limitations | Split into four paragraphs; decision-curve limits added. | text |

Verification at this round: 1,218 recorded values replayed, 0 mismatches; no hand-typed result numbers in the sections; the Overleaf package compiles on its own (main 12 pages, supplement 2 pages, no undefined references).

## Verdict

| | Round 0 | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|
| Overall verdict | Weak | Moderate | Moderate to Strong | **Moderate to Strong** |
| SCI readiness | 4 / 10 | 7 / 10 | 7.5 / 10 | **7.8 / 10** |
| Q1 potential | Low | Medium | Medium | **Medium** |

## Scores by section

| Section | R2 | R3 | Why not 9 |
|---|---|---|---|
| Abstract | 8.5 | **8.7** | Wording is at the level of the evidence. A 9 would need a claim about another site; the abstract cannot make it. |
| Title | 8.0 | **8.0** | Descriptive and accurate; it names three ideas (shortcuts, conformal sets, landmark) and no result. |
| Introduction | 8.5 | **8.6** | Gap now tied to a recorded search. Still a three-question framing around one database. |
| Related work | 8.5 | **8.8** | Recorded search with counts and screening decisions. It uses one index, one screener and title-level screening, not a PRISMA review. |
| Methods | 9.0 | **9.0** | Reproducible and now complete in the PDF (the cut sentence is back). Hyperparameter grids are in the supplement. |
| Results | 8.5 | **8.7** | Forest plot, reliability, decision curve and period stability added; the section is long and dense. Single site. |
| Discussion | 8.5 | **8.6** | Interprets, labels hypotheses, adds usefulness and its limits. Mechanisms for the shortcut and for later landmarks remain untested. |
| Limitations | 9.0 | **9.0** | Specific and quantified; names what would remove each limit. |
| Conclusion | 8.5 | **8.5** | Calibrated, no new result. |
| Tables | 8.5 | **8.6** | Ten main tables, all generated and replayed; Table IX is large. |
| Figures | 8.0 | **8.5** | Six figures that each carry information. Fig. 1 is a schematic. |
| References | 8.5 | **8.7** | Four additions with metadata checked against Crossref. |

## What would still be needed for 9 or more

1. Results, Discussion, Abstract, Introduction: a second site or a second database with the same frozen protocol. No such dataset is available in this project, so a 9 here is not reachable without new data, and the loop stopped there rather than inflate the text.
2. Related work: a protocol with two screeners and several indexes (PubMed, IEEE Xplore, arXiv) to count as a systematic search. PubMed was not reachable from this environment.
3. Discussion: the two untested explanations (monitoring intensity as a record of clinical judgment; why later landmarks score lower) would need a designed analysis, for example matching on acuity or a fixed cohort followed across landmarks.
4. Results: a clinician-elicited threshold range for the decision curve and a cost model for the sets.

## Open items that only the author can close

Dataset version of the hospital module, acknowledgment, conflict of interest, repository URL, affiliation and ORCID, and the wording of the AI-use disclosure (red markers in the text).
The decisions in `DEVIATIONS.md` (P0-5, P2-1, P2-7, and the new P2-10 to P2-12) need confirmation.
