# Round 4: section-by-section loop modelled on JBHI papers

Rule for this round (author's instruction, 2026-10-07): take one section below 9 at a time and improve it by comparison with sections of papers published in IEEE JBHI.
No second dataset is used. Scores are the drafting assistant's judgement; an independent reviewer may score lower.

## Reference set (IEEE JBHI, full text read through PMC / Europe PMC open-access records)

| PMC | Paper | Why chosen |
|---|---|---|
| PMC12914552 | Discrete time neural networks for mortality over different time horizons (2026) | clinical risk prediction, time horizons |
| PMC13030920 | Benchmarking foundation models with multimodal public EHR (MIMIC-IV) (2025) | MIMIC-IV benchmark, contribution list |
| PMC11970999 | Fusion network for mechanical circulatory support need prediction in the pediatric cardiac ICU (2025) | ICU deterioration prediction |
| PMC12682437 | CalDiff: calibrating uncertainty of diffusion models (2026) | uncertainty and calibration |
| PMC7615082 | Hi-BEHRT (2023) | EHR risk prediction |
| PMC7872470 | Privileged information and label uncertainty for ARDS detection (2021) | label uncertainty, ICU |
| PMC5219944 | Hypotension risk prediction from ICU blood pressure (2016) | early warning, false alarms |

## Section 1: Introduction (8.6 -> 9.0)

Measured against the set (intros of 3 to 10 paragraphs, 363 to 1,363 words): all seven open with the clinical problem and current practice, give one explicit aim
sentence, and connect the gap to the aim; four use a contribution list; four end with a roadmap.

Deficits found in our version and what was done:
1. No clinical stake or current practice -> opened with delayed ICU transfer and mortality (Chalfin et al. 2007, checked against its abstract), triage acuity and NEWS2, and prior triage ML (claims narrowed to what each cited abstract supports).
2. No single aim sentence -> "The aim of this study is to determine how well ... and what its conformal sets guarantee for the patients who deteriorate."
3. Timing and uncertainty problems not linked -> both framed as conditions on a score that is acted on (stated prediction time; trustworthy statement for those who deteriorate).
4. False-alarm cost not motivated -> stated as the quantity that matters to a department.
5. Proofreading at this round: an overclaim ("reflect what clinicians decided") weakened to match the acuity-stratified result; GRU-D expansion corrected; repeated abbreviation definitions removed from Related Work and Methods (AUROC, NEWS2, ECI, ED, GRU-D).

Remaining limits: one clinical-stake citation; single-site scope stated. Length 840 words.

## Section 2: Discussion (8.6 -> 9.0)

Measured against the four papers of the set with a dedicated Discussion (PMC12914552, PMC13030920, PMC7615082, PMC12682437; 620 to 1,196 words): all open with a
principal-findings paragraph, interpret each finding against prior work, three include practical implications, and two close with enumerated limitations.

Deficits found and what was done:
1. No principal-findings opening -> added (three findings).
2. Two explanations stated as untested -> both tested with new analyses on MIMIC only:
   - shortcut within triage-acuity strata (`p2_16_shortcut_by_acuity.py`, Supplementary Table S10): workflow quantities still discriminate within strata, so they are not a restatement of acuity; judgment versus deterioration-driven monitoring remains inseparable and is stated so;
   - fixed set of stays across landmarks (`p2_17_fixed_cohort_landmarks.py`, Supplementary Table S11, dashed lines in Fig. 3): AUROC rises with the landmark on one fixed set, so the decline in the main sweep is a population effect; the gain mixes later values and more measurements and is stated so. Reproduction check against the stored sweep: logistic regression identical, trees within 0.0009.
3. Few comparisons with prior work -> added label leakage through diagnostic codes, triage ML versus acuity, and the published NEWS2 ED range.
4. No implications subsection -> "Implications for Reporting and Practice" (three reporting practices; sets as a trade-off, not an alarm).
5. Limitations as a dense block -> six enumerated limitations, each naming what would resolve it.
6. Proofreading at this round: an over-reading of the fixed-cohort gain corrected; a mismatched remedy in the limitations corrected; a repeated word and a signed number in prose fixed.

Remaining limits: single site; source of the shortcut not identifiable by this design; long section.

## Section 3: Conclusion (8.5 -> 9.0)

Measured against the five papers of the set with a Conclusion (91 to 397 words): each restates what was done, gives the key findings with a few numbers, and names future work; three use promotional
phrasing ("paving the way", "novel", "for the first time"), which was not copied.
Done: opening sentence on what was done; the two new findings added (acuity strata, fixed set of stays); a separate paragraph for the reporting recommendations and specific next steps
(validation of the frozen protocol at another site and prospectively; elicitation of the acceptable false-alarm burden). Two claims softened on proofreading (within-strata wording; causal "because" replaced by
the observed fixed-set result). Length 251 words.

## Section 4: Title (8.0 -> 9.0)

Reference: 100 JBHI titles 2021 onward on clinical prediction (OpenAlex, ISSN 2168-2194): median 12 words (IQR 10 to 15), 30% with a colon, most name the task ("... Prediction for ... Patients").
Old: "Workflow Shortcuts and Conformal Prediction Sets for Emergency-Department Deterioration at a Two-Hour Landmark" (task not named, noun stack, jargon "landmark", two ideas joined by "and").
New: "Predicting Deterioration in the Emergency Department Two Hours After Arrival: Workflow Shortcuts and Conformal Prediction Sets" (16 words). Supplement title changed to match, without a double colon.
Remaining limit: "workflow shortcuts" is a coined phrase, defined in the abstract.

## Section 5: Abstract (8.7 -> 9.0)

Reference: 100 JBHI abstracts (same query): median 220 words (IQR 191 to 244), all unstructured single paragraphs, about half with numbers; read in full: five close to this topic.
Pattern: problem, limitation of practice, "In this study, we ...", data, a few key results, implication.
Done: "In this study, we" sentence; methods named (GRU-D, gradient-boosted trees, logistic regression); acuity-strata finding added; result numbers reduced (the marginal coverage
value replaced by "near-nominal", accurate because 0.892 is just under the 0.90 target and above the 0.89 validity threshold); wording "within each level" corrected to "within the evaluable
triage-acuity levels" because acuity 4 to 5 had one outcome. Length 241 words (limit 250).
Remaining limit: still more numbers than the typical JBHI abstract; the drift result is not in the abstract.

## Section 6: Results (8.7 -> 9.0)

Reference: Results sections of the five papers of the set that have one (405 to 1,588 words). Three open with the cohort ("Cohort Description", "Descriptive Analysis of the Cohorts",
"Characteristics of the benchmark dataset") and a characteristics table; each paragraph is anchored to a table, then states the comparison and the test; about three subsections.

Deficits found and what was done:
1. No cohort subsection or characteristics table -> new "Study Cohort" subsection and Table III (age, sex, acuity, comorbidity, vital-sign rows by L, ED length of stay, outcome rates, by
   period, with standardized differences; `p2_18_cohort_characteristics.py`, aggregates only).
2. The new table exposed an unreported shift: stays with at least one comorbidity fell from 47.4% (training period) to 22.7% (test period), because the split keeps patients with earlier stays in
   the training period and the count is built from earlier admissions. Reported in Results and added to the third limitation.
3. Seven subsections -> four (cohort; timing audit and shortcut; discrimination, calibration and usefulness; conformal prediction), with numbered subsubsections; meta-commentary removed.
4. Table numbering did not follow the order of first citation (the cohort table is cited in Methods first) -> cohort table moved to Methods; a script now checks that all tables and figures are numbered
   in citation order (both pass).
5. Precision: training-period versus training-subset rates made consistent with the table; the two acuity 4 to 5 counts labelled with their cohorts; delayed-label and drift paragraph condensed
   to its finding (values remain in Table XI).

Remaining limit: still longer and more numerical than the JBHI Results sections read (about 2,500 prose words against 400 to 1,600), because the study reports two tasks and a conformal suite.

## Note on the scores above (added after the author questioned why every section ended at exactly 9.0)

The scores for Sections 1 to 6 were impressionistic and anchored to the author's target: each section was revised until the drafting assistant judged it "at 9", then reported as 9.0.
They are not reliable measurements. From Section 7 onward each section is scored with an explicit weighted rubric derived from the JBHI exemplars; the score is the weighted share of
criteria met (met = 1, partly = 0.5 or 0.75, not met = 0), times 10. Sections 1 to 6 will be re-scored the same way.

## Section 7: Related Work (rubric: 7.4 -> 8.6)

Reference: the two papers of the set with a Related Work section (PMC11970999, 306 words; PMC12682437, 1,342 words, two thematic subsections) plus the drafting guidance.

| # | Criterion | Weight | Before | After |
|---|---|---|---|---|
| 1 | Organized by theme | 1.0 | 1 | 1 |
| 2 | Strengths and the specific limit of each theme | 1.5 | 1 | 1 |
| 3 | Explicit positioning paragraph | 1.5 | 0.5 | 1 |
| 4 | Closest prior work covered and described correctly | 1.5 | 0.75 | 0.75 |
| 5 | Absence claims tied to a documented search | 1.0 | 0.75 | 0.75 |
| 6 | Fair to prior work | 0.5 | 1 | 1 |
| 7 | Current | 0.5 | 1 | 1 |
| 8 | Concise, no citation catalogue | 0.5 | 0.5 | 0.75 |
| 9 | No duplication of the Introduction, clean signposting | 0.5 | 0.5 | 1 |
| 10 | Comparison table of the closest studies | 0.5 | 0 | 0 |

Factual corrections found while checking criterion 4 (full text of Xie et al. 2022 read through Europe PMC): the benchmark predicts hospitalization and the critical outcome at triage and
ED reattendance at disposition; the paper had implied that the benchmark's models read the whole stay. Corrected in the Introduction and Related Work; the end-of-stay task (from the authors'
earlier analysis) is now defined in Methods; the benchmark's finding that deep sequence models did not outperform simpler ones is cited in the Discussion. MDS-ED (Lopez Alcaraz et al.,
Comput. Biol. Med. 2025) added after checking Crossref. CAREBench (named in CLAUDE.md as a reference) could not be found in OpenAlex; not cited.
Remaining: comparison table (criterion 10), second index and screener (criterion 5), CAREBench.
