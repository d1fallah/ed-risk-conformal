# Round 11: Related Work (2026-10-07)

Same rubric and weights as round 4 (weights sum to 9.0; score normalized). Scorer: the drafting assistant.

## Sources read
Full text for nine of the ten closest studies (Xie 2022, PMC9610299; Lopez Alcaraz 2025, arXiv 2407.17856v4; Solaiman 2026, arXiv 2602.20168; Keoliya 2025, arXiv
2510.14286; Ramadan 2025, PMC12743287; Yang 2024, PMC10985608; Dalal 2025, PMC12174992; Kwon 2026, PMC13022086; Millar 2026, PMC13193150). Abstract only for
Abdulai 2025 (no open full text); the table says so.

## What changed
- Supplementary Table S2: the ten closest studies against this one on data and setting, prediction time, handling of input timing, uncertainty of individual
  predictions, class-1 coverage, operating characteristics, and evaluation under shift. Cited from "Position of This Study".
- Corrections found by reading the full texts:
  - Kwon et al. predict sepsis in the ICU (PhysioNet 2019) with deferral to clinicians; "triage decisions" read as ED triage. Reworded.
  - The conformal studies were evaluated on random splits, external cohorts, or later-admitted patients (Kwon et al.), not only externally. Reworded.
  - Solaiman et al. call their loss under reduced inputs a "graceful degradation"; the Discussion said performance "fell only modestly". Aligned.
  - CAREBench matches observed-history length between classes to stop models exploiting it, which is the closest design to the shortcut studied here. Added
    to the leakage subsection, with the distinction that these designs prevent a timing shortcut and do not measure it.
- The gap statements still hold after reading: no study quantifies the workflow shortcut of an end-of-stay model; class-1 coverage is reported by Yang et al.
  and Millar et al., neither in the ED and neither with the false-alarm burden; no study compares online with static and periodic calibration.

## Related Work: 9.6 (was 8.6)
| # | Criterion | Weight | Credit | Reason |
|---|---|---|---|---|
| 1 | Organized by theme | 1.0 | 1 | |
| 2 | Strengths and the specific limit of each theme | 1.5 | 1 | |
| 3 | Explicit positioning paragraph | 1.5 | 1 | |
| 4 | Closest prior work covered and described correctly | 1.5 | 1 | full texts read; three descriptions corrected |
| 5 | Absence claims tied to a documented search | 1.0 | 0.75 | one index (OpenAlex) and one screener; a second screener cannot be supplied by the author alone |
| 6 | Fair to prior work | 0.5 | 1 | |
| 7 | Current | 0.5 | 1 | |
| 8 | Concise, no citation catalogue | 0.5 | 0.75 | 843 words (772 before) |
| 9 | No duplication of the Introduction, clean signposting | 0.5 | 1 | |
| 10 | Comparison table of the closest studies | 0.5 | 1 | Supplementary Table S2 |
