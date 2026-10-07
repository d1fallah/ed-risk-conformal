# Round 7: Tables and Figures after revision (2026-10-07)

Same rubric and weights as round 5. Scorer: the drafting assistant.

## Correction to round 5
Round 5 gave "Self-contained captions, proxy label" full credit for the tables. That was wrong: 18 of the 20 generated table captions did not carry the
proxy label required by CLAUDE.md rule 5. The round-5 table score should have been 6.4, not 7.4. Fixed in this round.

## What changed
- Tables of the contrasts, the landmark sweep and the coverage levels moved to the supplement (Supplementary Tables S7, S9, S15); Figs. 2, 3 and 6 carry the
  same values and the text cites the figure first. Main tables: 11 to 8.
- Minus signs: every negative number in macros and tables is a typographic minus (`\ensuremath{-}`), produced by the formatter that the verifier replays.
- Proxy label appended to every data-table caption (generator pass); the literature-search table holds no patient data and is unlabelled.
- Tables IV and V: shorter row labels (no longer shrunk below about 7 pt); Table IV caption corrected (seed SD applies to all lower-block models).
- Figures: Okabe-Ito colour-blind-safe palette, distinct markers and line styles for every series, fonts of at least 7 pt at print width, double-column figures
  at full text width, legends moved off the data, outcome histogram drawn as an outline so neither class hides the other.
- Reliability panel: patient-cluster bootstrap 95% intervals of the observed rate (stored in the evidence, now plotted); caption updated.
- Results wording corrected while checking the figure: in the upper deciles the points lie above the diagonal (predicted below observed); the text said
  "below it".

## Tables: 9.3 (round 5 reported 7.4; corrected 6.4)
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Generated and verified | 2 | 1 | verifier PASS, 1,436 records |
| Self-contained captions, proxy label | 2 | 1 | label on every data table |
| Numbered in citation order | 1 | 1 | main I to VIII, supplement S1 to S15 |
| Number of main tables appropriate | 1.5 | 0.5 | eight main tables against three to six in the exemplars |
| Consistent formatting, typographic minus signs | 1.5 | 1 | |
| No duplication between tables and figures | 2 | 1 | |

The remaining deduction is a trade-off for the author: reaching six would mean moving the score-interval table (VI) and the label-delay and drift table (VIII)
to the supplement. Both hold pre-specified results of the primary contribution, so I have not moved them.

## Figures: 9.6 (was 7.4)
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Each figure adds information beyond the tables | 2 | 1 | |
| Vector, legible at print width | 2 | 1 | |
| Self-contained captions, proxy label | 2 | 1 | |
| Colour-blind safe, markers distinguish series | 1.5 | 1 | Okabe-Ito palette plus markers and line styles; not checked with a simulator |
| Uncertainty shown where relevant | 1.5 | 0.75 | decision curve has no bands (intervals stored only for differences at six thresholds); the dashed fixed-set lines of Fig. 3 have no intervals although they are stored; Fig. 6 shows seed means without SD |
| Cited in order | 1 | 1 | |
