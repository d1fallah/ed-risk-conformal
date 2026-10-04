# Finding: Track A M2 conformal scripts pair inputs with the wrong rows (found 2026-10-01, Track B export)

**Mechanism.** `m2_hybrid_common.sort_chronologically()` ends with `reset_index(drop=True)`.
`m2_v2_common.prepare_temporal_tensors_v2()` selects the stored sequence rows by `df.index.values`
and only asserts that the index is in range. A chronologically sorted frame therefore gets the
sequence tensors (X, M, D, pad) of rows 0..n-1 of the split file, while labels and static features
come from the sorted frame. The temporal inputs no longer belong to the labelled patient.

**Affected Track A scripts** (sort, then prepare): `m2_conformal_v2.py`, `m2_eci_vs_recalibration.py`,
`m2_label_lag.py`, `m2_mondrian_eci_pilot.py`, `m2_mondrian_eci_comparators.py`,
`m2_mondrian_eci_confirmatory.py`, `m2_mondrian_eci_anchor_correction.py` (and any result built on them).
**Not affected:** `m2_hybrid_grud_v2.py`, `m2_temporal_only_grud_v2.py`, `m2_frozen_temporal_control_v2.py`
and the tuning scripts (unsorted frames). The model-comparison AUROC/AUPRC numbers stand.

**Measured** (`auroc_alignment_check.py`, `conformal_alignment_impact.py`; composite Hybrid GRU-D v2, seed 42):

| | as Track A runs it | aligned |
|---|---|---|
| test AUROC of the predictions used for conformal | 0.7324 | 0.9375 |
| calibration-fold AUROC | 0.7094 | 0.9327 |
| static split-conformal marginal coverage | 0.8841 | 0.8995 |
| recalibration (W=10k) marginal coverage | 0.8999 | 0.9007 |
| ECI marginal coverage | 0.9021 | 0.9013 |
| positive-class coverage static / recal / ECI | 0.025 / 0.042 / 0.046 | 0.204 / 0.208 / 0.211 |
| mean width static / recal / ECI | 0.330 / 0.518 / 0.551 | 0.237 / 0.241 / 0.248 |

The widths in the M2 draft (recalibration 0.519, ECI 0.552) match the misaligned column.

**Track B handling.** `track_b/aligned.py` sorts while keeping each row's original position and
checks `stay_id` against the stored sequences; `export_model.py` and `test_model.py` use it.
No Track A file was modified.
