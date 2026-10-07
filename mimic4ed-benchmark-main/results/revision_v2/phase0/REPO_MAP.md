# Repo map (Phase 0, task 0.1)

Written 2026-10-07 from file listings and code only. No patient-level data were opened or displayed for this map; the only
data-derived facts are file schemas (column names) and aggregate counts, quoted in section 5.

## 0. Location and identity

- Git repository: `module2/` (remote `origin` = `ed-risk-conformal`, branch `main`). Working root for all study code and data:
  `module2/mimic4ed-benchmark-main/` (called the repo root below).
- `REVISION_PLAN.md` is at `module2/paper-m2/REVISION_PLAN.md`, not at `module2/`. It holds the plan, then the addendum that
  replaces R2, adds R5 to R7 and reporting rule 10, and replaces task 0.3.
- Python: the repo's own `.venv` (`.venv/Scripts/python.exe`). Raw MIMIC folders (`mimic-iv-ed-2.2/`, `mimic-iv-clinical/`,
  `mimic4_full/`, plus demo copies) sit inside the repo root and are read only by code.

## 1. Data flow

```
raw MIMIC-IV-ED + MIMIC-IV hosp/icu
  -> Benchmark_scripts/extract_master_dataset.py      -> data/master_dataset.csv   (one row per ED stay; 120 columns)
  -> Benchmark_scripts/m2_build_irregular_cohort.py   -> data/train_v2.csv, data/test_v2.csv,
                                                         data/m2_v2_dev_pool_split.json,
                                                         data/m2_irregular_sequences_{train,test}_v2.npz
  -> Benchmark_scripts/m2_add_t_hours.py              -> adds t_hours to the .npz tensors
  -> model scripts (*_v2.py)                          -> data/m2_v2_predictions.csv (+ checkpoints, tuned configs, seed results)
  -> conformal scripts (m2_conformal_v2.py, ...)      -> data/m2_v2_*.csv and *_verdict.json
  -> src/generate_*.py, src/verify_paper_assets.py    -> paper_assets/ (tables, figures, provenance sidecars; not tracked)
```

## 2. Where each thing lives

| Concern | Script or file | Notes |
|---|---|---|
| Raw extraction, outcome labels, Elixhauser | `Benchmark_scripts/extract_master_dataset.py`; `medcode_utils.py` (`commorbidity()`, `diagnosis_with_time()`); `helpers.py` | `outcome_critical` = in-hospital death or ICU transfer within 12 h of ED departure (`--icu_transfer_timerange 12`). `commorbidity(..., timerange=356*5)` is called at `extract_master_dataset.py:99`. |
| Cohort filters, horizon labels, outlier removal | `m2_build_irregular_cohort.py::load_and_filter_master` | age >= 18 and non-null triage acuity; `build_horizon_labels` makes the 1 h / 6 h / 12 h labels; filters documented in `M2_Outcome_Definition_Gate.md` (v2). |
| Chronological split | `m2_chronological_eci.py::chronological_split`, `CUT_YEAR = 2016` | `approx_real_year <= CUT_YEAR` is the training period; future-period patients who were seen earlier are dropped, not moved. |
| 4-way development split | `m2_build_irregular_cohort.py::build_dev_pool_split_v2` | patient-level and time-ordered inside the training period: training / validation / calibration_selection / final_conformal_calibration, fractions 70/90, 10/90, 5/90, 5/90. IDs are in `data/m2_v2_dev_pool_split.json` (never to be printed). |
| Irregular sequences | `m2_build_irregular_cohort.py::build_irregular_sequences`; `T_MAX_ED = 10`, `T_MAX_TOTAL = 1 + T_MAX_ED` | triage vector in slot t = 0; `m2_add_t_hours.py` adds measurement times (`.tail(t_max)`). |
| Static features | `m2_hybrid_common.py::HYBRID_STATIC_COLS = [age, sex, triage_acuity, comorbidity_elixhauser_count]` (line 805) | the count is the row sum of the 31 `eci_*` flags (lines 815 to 833). |
| Models (six, 5 seeds each) | `m2_temporal_only_grud_v2.py`, `m2_hybrid_grud_v2.py` (primary), `m2_frozen_temporal_control_v2.py`, `m2_mtan_final_v2.py`, `m2_static_only_mlp_v2.py`, `m2_xgboost_matched_v2.py` | tuning: `m2_xgboost_tuning_v2.py`, `m2_hybrid_head_tuning_v2.py`, `m2_frozen_control_tuning_v2.py`, `m2_mtan_tuning_matched_v2.py`; shared code `m2_v2_common.py`, `m2_hybrid_common.py`, `m2_grud_cell.py`. |
| Stored predictions | `data/m2_v2_predictions.csv` | columns `subject_id, stay_id, model, seed, y_true, y_prob`; 6 models x 5 seeds x 84,241 test stays = 2,527,230 rows. Horizon variants: `data/m2_v2_hz{1h,6h,12h}_predictions.csv`. |
| NEWS2 comparator | `run_news2_baseline.py`, `m2_recompute_news2.py`, `m2_news2_sensitivity.py`; `data/score_NEWS2_*.csv` | |
| Static conformal (LAC via MAPIE), Mondrian-LAC | `m2_conformal_v2.py`, `m2_mondrian_conformal.py`; regression guard `test_mondrian_lac_validity.py` | summary in `data/m2_v2_conformal_summary.csv`. |
| ECI (interval and set), recalibration | `m2_ablations.py::eci_method` (wraps `core.methods.ECI` from the vendored clone `module2/Error-quantified-Conformal-Inference/`), `m2_hybrid_common.py::run_eci_4way` (lines 667 to 720), `m2_eci_vs_recalibration.py`, `m2_chronological_eci.py`, `m2_eci_lr_sweep.py` | `eci_lr` grid {0.01, 0.05, 0.1}, `T_burnin = 5`, alpha = 0.10. |
| Mondrian-ECI | `m2_mondrian_eci_{pilot,confirmatory,anchor_correction,comparators,precision}.py` | verdict JSONs in `data/`. |
| Label delay | `m2_label_lag.py` (`eci_lag_sim`, line 67), `m2_generate_table_lag.py` | lags L as in the pre-registration. |
| Controlled drift, adaptive sets | `m2_rev_e4_drift_stress.py`, `m2_rev_e3_adaptive_sets.py`; bootstraps `m2_rev_s1_width_bootstrap.py`, `m2_rev_s2_xgb_bootstrap.py` | |
| Patient-cluster bootstrap, Holm deltas | `m2_v2_analyze.py`, `m2_hybrid_analyze.py` | 10,000 resamples, seed 42 base, five AUROC deltas with Holm correction. |
| Horizon ablation | `run_horizon_ablation_v2.sh`, `m2_horizon_analyze.py` | |
| One-command reproduction | `experiments/run_reproducible_m2.py --config configs/paper_track_m2.yaml` | stage list and order in `CORE_STAGES_V2`; `track: v2` is primary, `t2` is the superseded random-split track. |
| Provenance accessor | `src/paper_data.py`, class `Evidence` (`get`, `reduce`, `reduce_csv`, `series`, `external`, `literal`; `SUPERSEDED_ARTIFACTS`; `archive_*` directories excluded) | every reported number goes through it (global rule 6); used by `src/generate_latex_tables.py` and `src/generate_paper_macros.py`; checked by `src/verify_paper_assets.py`; frozen by `src/freeze_evidence.py` into `results/m2_evidence_manifest.json`. |
| Track B runtime | `track_b/` (`m2_runtime.py`, `recalibration.py`, `news2_fallback.py`, `export_model.py`) | deployment code, out of scope for the revision. |

## 3. Frozen parameters the revision must not change (global rule 3)

Seeds {42, 123, 7, 456, 789}; tuning seeds {42, 123, 7}; alpha = 0.10; ECI `T_burnin` = 5; `eci_lr` grid {0.01, 0.05, 0.1}; bootstrap
10,000 resamples; chronological cut year 2016; dev split fractions 70/90, 10/90, 5/90, 5/90. Source: `configs/paper_track_m2.yaml`.
The validity threshold 0.89, windows W in {10k, 40k} and R = 2000 live in the recalibration and pre-registration scripts.

## 4. Superseded or archived material (do not build on)

`data/archive_*` (pre-irregular, pre-alignment-fix, pre-S1/S2, pre-DEV7 snapshots), `paper_assets_pre_alignment_fix_2026-10-01/`,
files named `*_PRE_*` or `*.PRE_*.bak`, `m2_v2_predictions_PRE_MTAN_MATCHED.csv`, and the whole `t2` track (`run_baseline_models.py`,
`data_general_processing.py`, `m2_baseline_multiseed.py`, and others), kept only for the secondary random-split track.

## 5. Facts checked while mapping (schemas and aggregates only)

- `data/master_dataset.csv`: 120 columns; name families `eci_*` (31), `cci_*` (17), `n_*` (11), `chiefcom_*` (10), `ed_*` (9), `triage_*` (8),
  `outcome_*` (5), `next_*` (3), `time*`, `intime*`, `anchor*`, `subject_id`, `stay_id`, `race`.
- `data/m2_v2_predictions.csv`: 2,527,230 rows = 6 models x 5 seeds x 84,241 test stays (421,205 rows per model).
- The "six existing models" of the plan are: Temporal-Only GRU-D, Hybrid GRU-D, Frozen-Temporal Head Control, mTAN, Static-Only MLP,
  XGBoost-Matched.

## 6. Points to resolve in later Phase 0 tasks (flagged here, not concluded)

1. `configs/paper_track_m2.yaml` says of `t_max_ed: 10` "measured 0 stays truncated (Deviation 13)", while the build script comment says 10 is
   the p90 of ED observations per stay, which would imply truncation of roughly a tenth of stays. Task 0.4 measures it.
2. `commorbidity(..., timerange=356*5)` is 1,780 days, not 365 x 5 = 1,825. Task 0.2 reports it as an observation.
3. The 31 `eci_*` columns in `master_dataset.csv` are the Elixhauser flags, not the conformal ECI; the name collision is only in column names.
4. `data/m2_v2_dev_pool_split.json` and `data/m2_v2_random_split_ids.json` contain patient identifiers; they are read only through counts.
5. `results/` currently holds only the evidence manifests; the Phase 0 outputs are the first files under `results/revision_v2/`.

## 7. Proposed locations for the revision (global rule 2)

- New code: `src/revision_v2/phase0/` (created, empty), later `src/revision_v2/phase1/` and `phase2/`.
- New outputs: `results/revision_v2/phase0/` (created), later `phase1/` and `phase2/`; deviations in `results/revision_v2/DEVIATIONS.md`.
- Nothing under `data/`, `results/m2_*` or `paper_assets/` is written by the revision. The new folders are untracked and are not to be
  committed or pushed to the public repository unless the user asks.
