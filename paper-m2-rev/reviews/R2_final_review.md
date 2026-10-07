# Round 2: strict-reviewer pass after the round-1 fixes (paper-m2-rev, 11 pages main + 2 pages supplement)

The reviewer is the assistant that wrote the text. The scores are judgements and are shown with reasons. Nothing was raised to reach a target:
the user asked for 9 out of 10 in every section but also for no faked score, and where a 9 is blocked by missing evidence rather than by wording, that is stated.

Changes since round 1: abstract (singleton set defined, caveats, 247 words); introduction tightened; related work (ED triage models, deep early-warning systems, framing of the prediction-time question);
methods (preprocessing, training, hardware and software, shortcut-diagnostic protocol, p-value definition, static LAC computed in closed form and checked against MAPIE to a coverage difference of 0.0002);
results (landmark-time sensitivity with Table VI and Fig. 2; score-distribution explanation with Fig. 3; mechanism claims moved to the Discussion); discussion (timing features still describe monitoring practice; why late landmarks score lower is flagged as untested);
a wrong hard-typed count in a table caption found and replaced by a recorded value (888 was the end-of-stay figure; the landmark has 774 final-calibration positives).
Verification at this round: 1,070 recorded values replayed, 0 mismatches; no hand-typed result numbers in the sections.

## Verdict

| | Round 0 | Round 1 | Round 2 |
|---|---|---|---|
| Overall verdict | Weak | Moderate | **Moderate to Strong** (Moderate on evidence, Strong on reporting discipline) |
| SCI readiness | 4 / 10 | 7 / 10 | **7.5 / 10** |
| Q1 potential | Low | Medium | **Medium** |

## Scores by section

| Section | R1 | R2 | Why not higher |
|---|---|---|---|
| Abstract | 8.0 | **8.5** | Complete, within 250 words, caveats present. The tree-versus-recurrent difference has no interval in the abstract (space). |
| Title | 8.0 | **8.0** | Descriptive and accurate. |
| Introduction | 8.0 | **8.5** | Clear gap with a hedged novelty statement and a roadmap. |
| Related work | 8.0 | **8.5** | Organized by question; absence claims tied to a targeted search and the closest benchmark cited. A systematic search was not run. |
| Methods | 8.5 | **9.0** | Reproducible: cohort, split, task, inputs and timing, models, equal tuning budget, training, hardware and software, conformal methods, statistics, the verification step. Remaining gap (hyperparameter grids and search spaces) is in Supplementary Table S1. |
| Results | 8.0 | **8.5** | Every number is generated; claim ownership respected; each analysis answers one question. Dense; the ceiling is evidential (one site). |
| Discussion | 8.0 | **8.5** | Interprets, marks hypotheses, compares with the closest prior benchmark. |
| Limitations | 8.5 | **9.0** | Specific, quantified, and names what would remove each limit (a second site, prospective validation, a larger search, a cost model). |
| Conclusion | 8.0 | **8.5** | Calibrated; no new results. |
| Tables | 8.0 | **8.5** | Interval layers kept apart; verified cell by cell against the evidence. |
| Figures | 7.5 | **8.0** | Three figures that carry information (design, landmark time, score distribution) plus the coverage-level figure. |
| References | 8.5 | **8.5** | Only verified entries. |

## What would be needed for 9 or more in the remaining sections

1. Abstract, Introduction, Related work, Conclusion: the wording is at the level of the evidence; a 9 would need a claim the evidence does not yet carry (external validity).
2. Results, Discussion: a second site or a second database with the same frozen protocol; the present evidence supports, at most, a careful single-site description.
3. Figures: a forest plot of the contrasts and a calibration plot would help a reader but add no new evidence.
4. Related work: a systematic literature search with a recorded protocol.

## Open items that only the author can close

Dataset version of the hospital module, acknowledgment, conflict-of-interest statement, repository URL, affiliation and ORCID, and the wording of the AI-use disclosure (all red markers in the text).
The decisions logged in `mimic4ed-benchmark-main/results/revision_v2/DEVIATIONS.md` (P0-5, P2-1, P2-7) were taken while the user was away and need confirmation.
