# Length plan: proposed cuts, for approval (nothing edited yet), 2026-10-07

Limits as decided by the author: 14 pages in total including the supplement; main text at most 9 pages including references; supplement at most 5 pages; the TRIPOD+AI
checklist is assumed to be a separate reporting document that does not count (to be confirmed with the editorial office).
Two versions: the submitted paper (`paper-m2-rev/`) and an extended version (`paper-m2-rev/extended/`, to be posted as a preprint) that keeps everything that is in the paper and
the supplement today.

## 1. Page arithmetic (from the current build; accurate to about 0.3 page)
| | Now | Target |
|---|---|---|
| Main text without references | 13.7 pages, 10,894 words in the section files, 14 floats (about 3.3 pages) | about 7.2 pages: 5,620 words, 7 floats (about 1.8 pages) |
| References | 56 cited entries, about 2.1 pages | 31 to 35 entries, about 1.3 pages |
| Main text with references | 15.9 | **about 8.5** (range 8.2 to 8.8) |
| Supplement without the checklist | 9.4 | 2.7 for the five items you listed, up to 5.0 with the optional items in section 5 |
| Total | 27 | 11.2 to 13.5 |

Charges: pages beyond the eighth cost $250 each, so 8.5 pages means one charged page. Reaching exactly 8.0 pages needs about 400 words less than the plan below; I can do it if you prefer
(candidates: the Implications subsection and the second half of the limitations paragraph).

## 2. Extended version (step 1, before any cut)
- `extended/` receives a copy of `main.tex`, `supplement.tex`, `sections/` and the float wrappers exactly as committed today (commit 0f43a53), with the build script building it as
  `main_extended.pdf` and `supplement_extended.pdf`. Generated floats, `rv_numbers.tex` and the evidence stay shared, so there is one source of numbers.
- `verify_assets_v2.py` is extended to scan both directories; both must pass before every commit.
- The submitted version then changes by editing the files in `paper-m2-rev/` only. New merged tables get new names (the old files stay for the extended version).
- Pointers in the submitted text to cut material read "Supplementary Table S?" or "the extended version [ref]"; the extended-version reference stays a placeholder until you post it.

## 3. Main text, section by section
| Section | Words now | Target | Stays | Moves to the extended version (and where, if to the supplement) |
|---|---|---|---|---|
| Abstract | 248 | 240 | Findings on arrival information, copied row, trees versus logistic regression, class-1 coverage | A few secondary numbers |
| Introduction | 890 | 800 | Problem, gap, the three contributions | Two sentences of background |
| Related Work | 981 | 380 | Paragraph 1: leakage, informative presence, landmarking, benchmarks. Paragraph 2: conformal prediction (class-conditional, adaptive, delayed labels) | Subsections "Temporal Models for Vital Signs", "Early Warning Scores" and "Position of This Study" (its content is in Table S2, extended only) |
| Methods | 3,334 | 1,700 | Data and chronological split; landmark and outcomes; inputs and timing; models and tuning (short); conformal methods and the metrics; statistics (seeds, bootstrap, Holm); reporting and the analysis-plan statement | Details of: shortcut diagnostic by acuity and duration; recording and copied-row analyses (one sentence each plus the supplement pointer); landmark sweep; calibration, subgroups, decision curve; NEWS2; static-coverage bootstrap; PR check; intercept recalibration; GRU-D scoring rule (the rule stays in the supplement entry) |
| Results | 2,501 | 1,300 | Cohort and recording finding (Table I, copied row); input audit (Table II); shortcut with arrival increment (Table IV); discrimination (Table V, contrasts in two sentences); conformal sets (Table VII) and coverage levels (Fig. 6) | One sentence each, with the number and a pointer: acuity strata, duration quartiles, landmark sweep, calibration, subgroups, decision curve, label delay and drift, stream order, static-coverage bootstrap, intercept recalibration |
| Discussion | 2,573 | 1,000 | What the end-of-stay task measures; discrimination at the landmark (short); coverage is not alarm precision; static versus adaptive calibration (short); limitations (condensed to the six, one sentence each) | "Implications for Reporting and Practice" reduced to two sentences; the landmark-time paragraph; the net-benefit paragraph |
| Conclusion | 367 | 200 | Three findings and the scope | Restated numbers |
| **Total** | **10,894** | **5,620** | | |

## 4. Floats in the main text
| Now | Decision |
|---|---|
| Fig. 1 landmark design | Keep |
| Table I cohort + Table III characteristics | Merge into one Table I: the cohort rows plus the key rows of the characteristics table (period difference in recording: share without an ED vital sign by 2 h, first row repeats triage). New file `tab_cohort_merged` |
| Table II timing audit | Slim: one line per input group, class and the reason, no per-column detail. New file `tab_timing_slim` |
| Table IV shortcut diagnostic | Keep, and add the arrival-information increment rows (arrival only, workflow, arrival + workflow, arrival + duration) from Supplementary Table S8 |
| Table V discrimination | Keep |
| Fig. 2 forest of contrasts | To the extended version; the contrasts stay in the supplement as a table |
| Fig. 3 landmark sweep, Fig. 4 reliability, Fig. 5 score histogram | Extended version only |
| Table VI score intervals | Extended version only (retired method; a sentence reports it) |
| Table VII conformal sets | Keep |
| Fig. 6 coverage level | Keep |
| Table VIII label delay and drift | Extended version only, **unless you take option A in section 6** |

## 5. References
Of 56 entries now cited, 31 are cited outside Related Work. With Related Work reduced to two paragraphs, 31 to 35 remain; the list will be recomputed from the cut text,
the uncited entries removed from the submitted bibliography, and the extended version keeps the full list. Target 31 to 35 (you asked for 30 to 35).

## 6. Supplement (submitted version): at most 5 pages
Your five items, with the current table that supplies each (sizes are rough, 0.3 page):
| Item | Current table | Pages |
|---|---|---|
| Tuning budget | S3 | 0.7 |
| Pre-specified contrasts | S11 | 0.4 |
| Recording and copied-row robustness | S5 + S6, merged (panels A and B) | 0.7 |
| Alpha sweep | S20 | 0.5 |
| Comorbidity audit and ablation | S7 | 0.4 |
| Subtotal | | 2.7 |

2.3 pages are free. Items that would otherwise exist only in the extended version and that I recommend using it for, in this order:
- A. Label delay and imposed drift (current main Table VIII): two pre-specified conformal analyses that would otherwise shrink to one sentence (0.4).
- B. Literature search record (S1): the novelty statements in the Introduction and Related Work cite it; without it they must be softened further (0.5).
- C. Static-coverage bootstrap (S17), which supports the "static calibration fails the threshold for the trees" statement (0.3).
- D. Intercept recalibration (S21), which answers the model-recalibration question (0.6).
- E. Calibration (S15), the arrival-increment panel if it does not fit in Table IV (0.4).
A to D total 1.8 pages and fit. If the editorial office says the checklist counts (about 1.0 page when compact), drop D and E.

## 7. Order of work (commit after each step; verifier passes in both versions)
1. Extended copy, build, verifier on both. 2. Merged Table I, slim Table II, Table IV with the arrival rows (new generated files). 3. Related Work and references.
4. Methods. 5. Results. 6. Discussion, Introduction, Abstract, Conclusion. 7. Supplement reorganization and renumbering of the cross-references. 8. Page count of both PDFs and a read of
the built PDFs. Each step ends with a commit.

## 8. Decisions needed
1. Options A to E in section 6 (my recommendation: A to D).
2. 8.5 pages (one charged page) or cut to 8.0 (400 words more).
3. Whether the label-delay and drift analyses should appear as a main-text sentence plus supplement table (A) or as a sentence alone.
4. The wording of the pointer to the extended version (a preprint reference to be filled once it is posted).

## 9. Decisions received (2026-10-07)
1. Supplement: include A (label delay and imposed drift) and B (literature-search table). No tables for C, D, E: the static-coverage bootstrap intervals go inline in the Results text, the
   intercept recalibration gets one sentence with its numbers, and the calibration slope and intercept become two columns of the main discrimination table (Table V); the separate calibration table is dropped.
   Supplement size with A and B: 2.7 + 0.4 + 0.5 = 3.6 pages, which still holds if the checklist counts.
2. Target 8.5 pages for the main text including references. Later cuts, if needed, come from the Implications subsection, never from the Limitations paragraph.
3. Pointer to the extended version: "An extended version with all analyses is available [ref]", the reference left as a placeholder (`[DATA_GAP: extended-version reference; depends on whether
   JBHI review is anonymous]`); no arXiv identifier or repository URL. Before posting on arXiv: check the current IEEE preprint policy and update the preprint after acceptance as it requires.
Note on the bibliography: BibTeX prints only cited entries, so the submitted version needs no separate trimmed `.bib`; the reference count follows from the cut text.

## 10. Progress
- Step 1 (extended copy): done. `extended/` builds to text identical to the current paper (16 pages) and supplement (11 pages); the verifier passes on both versions.
