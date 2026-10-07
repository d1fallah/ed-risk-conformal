# Progress log (M2 revision v2)

Updated by the assistant while the user is away (instruction of 2026-10-07: finish everything, manuscript to a genuine 9/10 per section, never inflate).

| Time (local) | Item | State |
|---|---|---|
| 2026-10-07 ~01:30 | Phase 0 tasks 0.1 to 0.7, `phase0/AUDIT_REPORT.md`, `DEVIATIONS.md` | done |
| ~01:40 to 02:00 | Phase 1.2 to 1.8 scripts in `src/revision_v2/phase1/`, outputs `results/revision_v2/phase1/*.json` | done (1.1 is in `phase0/p05_cohort_counts.json`); `PHASE1_REPORT.md` to write |
| ~02:00 | Phase 2.1 cohorts (`p2_1_cohorts.py`), `phase2/TUNING_BUDGET.md` written before any tuning | done |
| 02:02 | `p2_2_classical.py` (LR, XGBoost at triage; LR, timing-aware XGBoost at 2 h), background | running; log `phase2/p2_2_classical.log` |
| 02:02 | `p2_3_neural.py temporal` and `p2_3_neural.py e2e`, background (GPU shared with the M4 E8 NCC queue) | running; logs `phase2/p2_3_*.log`; resumable (runs in `data/revision_v2/runs/`) |
| next | `p2_3_neural.py hybrid_frozen` after `temporal` finishes; contrasts (R7, 10,000 resamples); conformal 2.4; Phase 2 report; manuscript | pending |

Restart recipe if a job died: rerun the same command, finished (stage, config, seed) runs are skipped.
`.venv/Scripts/python.exe src/revision_v2/phase2/p2_3_neural.py <temporal|hybrid_frozen|e2e>` from `module2/mimic4ed-benchmark-main`.

## Update 2026-10-07 09:00

All phases completed. Neural runs: 34 (12 + 12 tuning, 10 final) and the hybrid_frozen stage, finished 07:54. Contrasts (10,000 resamples), NEWS2, calibration and subgroups, conformal suite
(primary model, plus trees and logistic regression), stream-order permutations, score distribution and histogram, landmark-time sweep: all done, outputs in `phase2/`.
Reports: `phase0/AUDIT_REPORT.md`, `phase1/PHASE1_REPORT.md`, `phase2/PHASE2_REPORT.md`. Evidence frozen in `evidence/` (MANIFEST.json). Manuscript: `module2/paper-m2-rev/` (main 11 pages, supplement 2 pages),
review rounds in its `reviews/`, Overleaf package `overleaf_export/M2rev_Overleaf.zip`.
