# Phase 0 audit report (M2 revision, tasks 0.1 to 0.7)

Written 2026-10-07. Phase 0 only: no model was trained, no split, seed, conformal parameter or ECI algorithm was changed, and nothing
existing was overwritten. Every number below comes from a script in `src/revision_v2/phase0/` whose outputs (JSON files next to this
report) contain counts and aggregates only. No identifier, row or free text was printed or written (global rule 1).
Terms: "prediction origin" is the time at which a prediction is made; "not available at prediction time" is used where information
would be missing at that origin, and "incorrect" only where an implementation departs from its specification.

## Summary of verdicts

| Item | Verdict |
|---|---|
| **R1, Elixhauser count** | **Valid by design: R1 is not triggered.** Codes come only from `hosp.diagnoses_icd` of earlier admissions discharged before ED arrival; `ed.diagnosis` is never read; the index admission is excluded by rule. **One exception, quantified:** 42 of 425,087 stays (22 with a changed count inside the analysed cohort: 17 training, 5 test) have their own linked admission accepted as "prior" because its discharge time precedes their ED arrival. Decision on the literal reading of "ANY" is yours (see 0.2). |
| Sequence truncation (0.4) | The triage slot is never dropped. Truncation is not zero as the config states: 9,779 training stays (3.39%) and 4,837 test stays (5.74%) exceed ten ED rows and lose their **earliest** rows; the build script's own counter is always 0 by construction. |
| Timing of model inputs (0.3) | Vital-sign sequences are the **last ten rows charted during the stay, including rows charted after ED departure**. Only the triage vector, age, sex, acuity and the Elixhauser count are available at arrival. The as-built task is an end-of-stay task, not an arrival or 2 h task. |
| Cohort counts (0.5) | All plan counts reproduced exactly except one: the plan says 425,088 stays, the master table has 425,087. |
| ED out-time minus admit-time (0.6) | 97.6% positive, 2.4% negative, 0.09% zero; median 112 min. Admission is registered a median 289 min after arrival, and the median gap from admission registration to ED departure is 112 min. |
| ECI check (0.7) | The update equations, the logistic scale c = 1 and the delay handling match the reference. Differences are parameter choices (T_burnin 5, not the 100 of the reference configs; a learning-rate grid that includes 0.01) and one documented design difference in how the learning-rate scale is computed under delay. The calibration-selection stream is shorter than L = 16,000 and its evaluated half is shorter than L = 16,000 (and 4,000 leaves 2,037 resolved labels): the learning-rate selection under those lags is not informed by feedback. |

---

## 0.1 Repo map

See `REPO_MAP.md` (written and shown before this report). Two points settled since: `REVISION_PLAN.md` lives in `module2/paper-m2/`, not
the repository root; the vendored ECI clone is `module2/Error-quantified-Conformal-Inference/` at upstream commit
`05086e7` (2025-02-02).

---

## 0.2 Elixhauser audit (decides R1)

### Code, quoted (code only)

`Benchmark_scripts/extract_master_dataset.py`
```
:50  df_diagnoses = read_diagnoses_table(os.path.join(mimic_iv_hosp_path, hosp_filename_dict['diagnoses_icd']))
:99  df_master = commorbidity(df_master, df_diagnoses, df_admissions, timerange = 356*5)
:29  ed_filename_dict = {'diagnosis':'diagnosis.csv', 'edstays':'edstays.csv', ...}     # filename dictionary only; never read
```
`Benchmark_scripts/medcode_utils.py`
```
:89   def diagnosis_with_time(df_diagnoses, df_admissions):
:90       df_diagnoses_with_adm = pd.merge(df_diagnoses, df_admissions.loc[:, ['hadm_id', 'subject_id', 'dischtime']],
                                         on=['subject_id', 'hadm_id'], how='left')
:164  def commorbidity(df_master, df_diagnoses, df_admissions, timerange):
:165      timerange = timedelta(days=timerange)
:195          if row['intime'] > df_diagnoses_sorted['dischtime'][j] and row['intime']-df_diagnoses_sorted['dischtime'][j] <= timerange:
```
`Benchmark_scripts/m2_hybrid_common.py`
```
:805  HYBRID_STATIC_COLS = ["age", "sex", "triage_acuity", "comorbidity_elixhauser_count"]
:830      eci_cols = [c for c in df.columns if c.startswith("eci_")]     # 31 flags, asserted
:832      df["comorbidity_elixhauser_count"] = ( df[eci_cols]...sum(axis=1) )
```

### What the code does

1. **Table.** ICD codes come from `hosp/diagnoses_icd.csv` (hospital discharge coding). `ed/diagnosis.csv` appears only in a filename
   dictionary (`extract_master_dataset.py:29`) and is never read; no `m2_*` script refers to it.
2. **Admissions used.** Every admission of the same patient whose **discharge time is before the ED arrival** (`intime >
   dischtime`) and within `356*5 = 1,780` days. Timestamps come from `hosp.admissions.dischtime`. The admission linked to the index ED
   stay (same `hadm_id`) is discharged after arrival and is therefore excluded by the same condition.
3. **Window.** `intime - dischtime <= 1,780 days` (not 365 x 5 = 1,825; an observation, not a defect of consequence). The largest
   observed gap among accepted admissions is 1,779.9993 days; no accepted admission lies beyond 1,825 days.
4. **Can any code postdate arrival?** Not by the rule, apart from the 42-stay exception below.
5. **The loop is pointer-based** (`medcode_utils.py:182-190`) and silently loses diagnoses if the master table is not sorted by
   patient. Checked: the master table is sorted by `subject_id` with contiguous blocks (both True).

### Verification (counts only; `p02_elixhauser_audit.py`, `p02b_index_admission_anomaly.py`)

| Check | Result |
|---|---|
| Stays / patients in the master table | 425,087 / 205,504 |
| Admissions / diagnosis rows | 546,028 / 6,364,488 |
| Stay-admission links accepted by the rule | 1,124,376 (stays with at least one source admission: 201,471; none: 223,616) |
| (i) accepted links with `dischtime >= intime` | **0** |
| (i) accepted links with missing `dischtime` | 0 |
| Same-patient links excluded because `dischtime >= intime` | 1,342,229 (these include later and overlapping admissions) |
| Same-patient links excluded as older than the window | 283,090 |
| **Independent recomputation of the 31 flags** from the accepted admissions | **identical to the stored columns on all 425,087 stays (0 differing cells)** |
| (ii) accepted links whose `hadm_id` equals the stay's own linked admission | **42 links, 42 stays** (0.0099% of stays) |
| Of these, with `admittime >= intime` (discharge stamped before admission) | 41 |
| Own `intime - dischtime` for the 42 | 0.001 to 0.89 days (median 0.45) |
| Of the 42, count would change if the own admission were removed | 34 stays (mean change 2.17 flags, maximum 7) |
| In the analysed cohort | training 24 stays (17 with a changed count), test 5 (5 changed); the other 13 are in the returning-patient stays dropped at the 2016 cut |
| Stays linked to an admission whose recorded discharge precedes ED arrival (all) | 42 (so the 42 are exactly the linked stays with this timestamp inconsistency) |

Gap between ED arrival and the end of the most recent accepted admission (quantiles of days, per stay): minimum 0.0007; 1% 0.48; 5% 2.42;
25% 23.97; 50% 134.9; 75% 506.3; 95% 1,346; 99% 1,669; maximum 1,779.9993. Stays whose most recent accepted admission ended within 1 day:
4,288; within 7 days: 23,824; within 30 days: 56,224 (of 425,087). The stored count has mean 2.32 and is zero for 56.5% of stays.

### Verdict under R1

R1 text: the count is removed from all models if it "uses ANY information not available at ED arrival (e.g., ed.diagnosis, or
diagnoses_icd of the index hadm_id)".

* **Design level: not triggered.** Neither named example applies: `ed.diagnosis` is not used, and the index admission is excluded by
  its discharge time. The recomputation reproduces the stored flags exactly, so no other source feeds the count.
* **Literal reading, quantified exception.** For 42 stays the linked index admission is accepted as a prior admission because its
  recorded discharge precedes the ED arrival (in 41 of them the discharge is stamped before the admission itself, a timestamp
  inconsistency in the source table). For 22 stays in the analysed cohort the count changes. This is 0.0099% of stays and 0.0059% of
  analysed stays (22 of 372,449); it cannot move any reported metric at the reported precision, but under the letter of "ANY" it is
  contamination. **Decision for you:** (a) treat R1 as not triggered and keep the Elixhauser count in Phase 2, reporting the 42 as a
  known data anomaly and Phase 1.3 as the sensitivity analysis; or (b) apply R1 literally. My recommendation is (a) with a fix applied
  in the Phase 2 feature builder (reject an accepted admission whose `hadm_id` equals the index `hadm_id` or whose `dischtime <
  admittime`), which needs no retraining decision today and is listed as a pending item in `DEVIATIONS.md`.
* **Timing class (d), not an R1 trigger.** Discharge codes are finalised after discharge. For 4,288 stays (1.0%) the most recent
  admission ended less than one day before arrival, and for 23,824 (5.6%) less than seven days. In a prospective deployment those
  codes might not yet be available; the retrospective data cannot show this. It belongs under class (d) in 0.3.

The existing full-stay results are **not** flagged as affected by R1.

---

## 0.3 Feature timing audit (addendum version)

Classes: (a) available at or before arrival; (b) charted within the window up to the landmark; (c) only available after the landmark;
(d) ambiguous. "Landmark" is the prediction origin under test: arrival (triage), 2 h after arrival (R2), or, for the horizon labels
as built, arrival with horizons 1, 6, 12 h. Time type: measurement time (when the quantity was measured) or documentation time (when it
was recorded, which can differ).

| # | Model input | Source table and timestamp column | Time type | Class | Models using it | Notes (aggregates from `p04_truncation_timing.json`) |
|---|---|---|---|---|---|---|
| 1 | ED vital-sign values, slots 1 to 10 (temperature, heart rate, respiratory rate, SpO2, SBP, DBP, pain) | `ed.vitalsign.charttime` | documentation (chart) time | (b) if charted by the landmark, **(c) otherwise** | Temporal-Only, Hybrid, Frozen, mTAN (full sequence); XGBoost-Matched (aggregates) | The slots hold the **last ten rows of the whole stay** (`helpers.py:416`, `.tail(t_max)`). Stays with at least one retained row charted after 1 h / 2 h / 6 h / 12 h from arrival, training: 92.3% / 85.4% / 36.3% / 11.3%; test: 93.0% / 87.6% / 44.1% / 16.3%. Retained rows charted after ED departure: 82,570 (training, 32,897 stays) and 24,663 (test, 7,810 stays). Retained rows charted before arrival: 709 and 102. |
| 2 | Observation masks | `ed.vitalsign` missing pattern per row | documentation | same as row 1 | same | derived from row 1 |
| 3 | Measurement times `t_hours` (charttime - intime) | `ed.vitalsign.charttime`, `edstays.intime` | documentation | (b)/(c) as row 1 | mTAN (time embedding) | the last retained time is itself a lower bound on ED length: median 4.9 h (training), 5.4 h (test) among stays with rows |
| 4 | Time since last observation per vital (delta) | derived from charttimes | documentation | (b)/(c) as row 1 | GRU-D family | |
| 5 | Sequence length (pad mask) and per-vital counts (`*_count`) | derived from the number of charted rows | documentation | **(c) for any origin before departure** | pad mask in all sequence models; `*_count` in XGBoost-Matched | encodes how long and how intensively the patient was monitored, a post-decision workflow quantity; this is the shortcut tested in Phase 1.2 (group A) |
| 6 | Triage slot (index 0): triage vitals | `ed.triage` (no timestamp) | measurement at triage; time unstated | **(a)** under the R2 anchor (`edstays.intime` as the operational triage time) | all sequence models; XGBoost-Matched (as slot 0); NEWS2 | timing is assumed, not recorded. Slot 0 is present for all 288,208 training and 84,241 test stays (pad = 1, t = 0); no observed triage value in 3,459 and 3,028 stays |
| 7 | Age | `patients.anchor_age`, `anchor_year`, `edstays.intime` | derived at arrival | (a) | Hybrid, Frozen head, Static-only, XGBoost-Matched | |
| 8 | Sex | `patients.gender` | static | (a) | same | |
| 9 | Triage acuity (ESI) | `ed.triage.acuity` | assigned at triage; time unstated | (a), accepted by the plan | same | |
| 10 | Elixhauser count | `hosp.diagnoses_icd` via `hosp.admissions.dischtime` of prior admissions | documentation (discharge coding) | (a) by design; **(d)** for coding lag and for the 42-stay anomaly | same as row 7 | see 0.2 |
| 11 | NEWS2 inputs | `ed.triage`: respiratory rate, temperature, SBP, heart rate, SpO2 (partial NEWS2) | triage | (a) | NEWS2 comparator | |
| 12 | ED duration (`outtime - intime`, `ed_los_hours`) | `edstays.outtime` | measurement | (c) | **not an input** to any model | verified: absent from every feature list; reachable indirectly through rows 3 and 5 |
| 13 | Fields derived from `outtime` (`time_to_icu_transfer`) | `icustays.intime - edstays.outtime` | measurement | (c) | outcome construction only | not an input |
| 14 | Encounter metadata (`arrival_transport`, `disposition`, race, insurance, admission and discharge times, `hadm_id`) | `edstays`, `admissions` | various | `disposition` and admission fields are (c) | none | present in the master table but in no feature list (`HYBRID_STATIC_COLS`, sequence tensors, XGBoost 39 columns) |
| 15 | Prior-utilisation counts (`n_ed_*`, `n_hosp_*`, `n_icu_*`), medication counts, chief-complaint flags | `edstays`, `admissions`, `icustays`, `pyxis`, `medrecon`, `triage` | various | mixed | none | present in the master table, not model inputs |
| 16 | Outcome-related columns (`outcome_critical`, `y_1h/6h/12h`, `hours_to_event`, `horizon_eligible`) | `admissions.deathtime`, `icustays.intime`, `edstays.outtime` | outcome | labels | labels only | `horizon_eligible` removes 42 training and 12 test stays whose outcome time cannot be recovered or is negative, so the horizon evaluation population is defined partly by the outcome (54 stays in total, 0.014%) |
| 17 | Stream ordering for the conformal methods | `approx_real_year` (from `anchor_year_group`), then `intime` | arrival order within a year block | (a) for order; the **primary analysis assumes immediate labels (L = 0)** while the outcome is observed up to 12 h after ED departure | static, Mondrian, ECI, recalibration | `sort_chronologically` states that order within a year is a deterministic tiebreak "NOT real-time-meaningful" (`m2_hybrid_common.py:908-916`). The label-delay analysis (L up to 16,000) is where realistic feedback is treated |
| 18 | Scaler and split fits | training patients of the development pool | n/a | (a) | all | scalers are fitted on the development `training` subset only (`get_static_scaler_v2`) |

**Reading of the table for the framing.** As built, the sequence models see the last ten charted rows of the whole stay (rows 1 to 5). For
an origin at arrival or at 2 h, 85% to 93% of stays would have retained rows that are not available at prediction time, and the
sequence length and counts are post-decision workflow quantities. The horizon labels (1, 6, 12 h) count from arrival while the inputs
run to departure, so for the 6 h label 36% (training) to 44% (test) of stays carry retained measurements charted after the horizon
has closed. This is what R5 to R7 address.

---

## 0.4 Sequence truncation

Code: `T_MAX_ED = 10`, `T_MAX_TOTAL = 1 + T_MAX_ED` (`m2_build_irregular_cohort.py:74-75`); the triage vector is written to slot 0
before any ED row; ED rows go to slots 1 to 10 (`m2_build_irregular_cohort.py`, `build_irregular_sequences`); truncation keeps the
**latest** ten rows (`helpers.py:416`, `.tail(t_max)`).

| | Training (288,208 stays) | Test (84,241 stays) |
|---|---|---|
| Stays with zero ED vital-sign rows | **13,302 (4.6%)** | **17 (0.02%)** |
| Rows per stay: median / p90 / p99 / max | 2 / 7 / 15 / 109 | 3 / 8 / 18 / 74 |
| **Stays with more than 10 rows (truncated)** | **9,779 (3.39%)** | **4,837 (5.74%)** |
| Rows dropped in total | 44,281 | 23,132 |
| Dropped rows per truncated stay: median / p90 / max | 3 / 10 / 99 | 3 / 10 / 64 |
| Truncated stays losing a row charted within 2 h of arrival | 8,599 | 4,835 |
| Truncated stays losing a row charted within 1 h of arrival | 7,816 | 4,831 |
| Stays whose first retained ED row is later than 2 h after arrival | 108,380 | 3,336 |
| Triage slot present (pad = 1) and at t = 0, all stays | True | True |
| Tensor slots equal the retained raw rows | True | True |

* **The triage measurement is never dropped.** It has its own slot, outside the truncation.
* **The config statement is wrong.** `configs/paper_track_m2.yaml` says "measured 0 stays truncated (Deviation 13)". The build script's
  counter (`n_truncated`) is always 0 because `merge_vitalsign_sequence_on_edstay` has already cut each stay to ten rows before the loop
  compares lengths. The real figures are above. The comment in the build script ("p90 of ED-vitalsign observations per stay") matches
  the data in the training period (p90 = 7) only approximately and is a different quantity from a truncation count.
* **Large era difference.** 4.6% of training stays have no ED vital-sign row against 0.02% of test stays, and the mean rows per stay is
  3.33 against 4.41. Documentation intensity changes across the two periods; this is relevant to any shortcut built on counts or
  sequence length.

---

## 0.5 Cohort counts

`p05_cohort_counts.py`. Plan figures reproduced exactly unless noted.

| Quantity | Found | Plan |
|---|---|---|
| Master table stays | **425,087** | 425,088 (difference of one; reported as found) |
| After age >= 18 and non-null triage acuity | 418,025 | 418,025 |
| Training period: stays / patients | 288,208 / 139,371 | same |
| Test period: stays / patients | 84,241 / 61,843 | same |
| Excluded returning-patient stays / patients | 45,576 / 18,968 | same |
| Train-test overlap (patients, stays) | 0, 0 | |
| Development subsets, patients | 108,400 / 15,486 / 7,743 / 7,742 | same; disjoint and exhaustive of the training-period patients |

Stays and positives (primary composite target `outcome_critical`; horizon targets among horizon-eligible stays):

| Subset | Stays | Patients | Positives | Prevalence | y_1h eligible / pos. | y_6h pos. | y_12h pos. |
|---|---|---|---|---|---|---|---|
| training | 240,353 | 108,400 | 12,598 | 5.24% | 240,319 / 111 | 7,565 | 11,090 |
| validation | 24,026 | 15,486 | 1,438 | 5.99% | 24,023 / 14 | 838 | 1,235 |
| calibration_selection | 12,075 | 7,743 | 643 | 5.33% | 12,072 / 8 | 367 | 558 |
| final_conformal_calibration | 11,754 | 7,742 | 888 | 7.55% | 11,752 / 13 | 462 | 752 |
| *training period total* | 288,208 | 139,371 | 15,567 | 5.40% | 288,166 / 146 | 9,232 | 13,635 |
| test period | 84,241 | 61,843 | 6,815 | 8.09% | 84,229 / 66 | 3,841 | 5,875 |
| excluded returning-patient stays | 45,576 | 18,968 | 2,500 | 5.49% | not computed (not in the split files) | | |

Prevalence rises from 5.4% in the training period to 8.1% in the test period. The 1 h label has only 66 positive test stays, 8 in
the calibration-selection fold and 13 in the final calibration fold.

---

## 0.6 `edouttime - admittime` for stays linked to an admission

`p06_ed_admit_gap.py`. Stays with a `hadm_id`: 203,016; with a matching admission row: 202,990; `edouttime` missing: 0. `edouttime` equals
`edstays.outtime` for 78.4% of them. Values in minutes.

| | n | negative | zero | positive | q05 | q25 | median | q75 | q95 |
|---|---|---|---|---|---|---|---|---|---|
| All linked stays, `edouttime - admittime` | 202,990 | 2.35% | 0.09% | 97.55% | 38 | 77 | 112 | 287 | 1,280 |
| Training stays | 136,069 | 2.22% | 0.11% | 97.67% | 40 | 77 | 111 | 286 | 1,246 |
| Test stays | 39,690 | 2.61% | 0.05% | 97.34% | 34 | 75 | 116 | 285 | 1,342 |
| All linked, `edstays.outtime - admittime` | 202,990 | 0.31% | 0.00% | 99.69% | 1.0 | 44 | 88 | 153 | 1,119 |
| All linked, `admittime - edstays.intime` | 202,990 | 0.30% | 0.00% | 99.70% | 85 | 186 | 289 | 429 | 822 |

Quantile 0 / 1% / 99% / 100% for the first row: -2,818 / -107 / 2,496 / 11,473 minutes. So for admitted patients the admission is
registered a median 4.8 h after arrival, and the median gap from admission registration to ED departure is 1.9 h. For a prediction made at ED
departure the admission decision is already known, and the sequence length and ED duration (rows 3 and 5 above) partly encode it.

---

## 0.7 ECI implementation check (report only)

Reference: `Error-quantified-Conformal-Inference/core/methods.py::ECI` (upstream commit `05086e7`). Repository use: `m2_ablations.py:39`
imports it directly (`from core.methods import ECI as eci_method`), called at `m2_hybrid_common.py:696` and `:709` with `alpha=0.1, T_burnin=5,
ahead=1, proportional_lr=True`; the event-driven delayed-label simulator is `eci_lag_sim` in `m2_label_lag.py:67`.

**Vendored clone vs upstream.** `git diff` against the upstream commit shows 10 changed lines in `core/methods.py` and 1 in `core/quantile.py`, all of them `np.infty` to
`np.inf`. No other edit.

**Update equations.** For `ECI` (`ahead = 1`), the repository calls the reference function itself, so the equations are the reference ones.
For the simulator, term by term:

| Term | Reference `ECI` | `eci_lag_sim` | Same? |
|---|---|---|---|
| Gradient | `alpha` if covered else `-(1-alpha)`, covered = `qs[t_pred] >= scores[t_pred]` | `alpha if qj >= s else -(1-alpha)` | yes |
| Error-quantification term | `(s-q)*c*exp(-c(s-q))/(1+exp(-c(s-q)))**2` with **c = 1** | `d*e/(1+e)**2` with `e = exp(-abs(d))` | yes (the logistic density is even; the form avoids overflow) |
| Scale | `lr_t = lr*(max-min of scores[t-T_burnin:t])` | same, over the last `T_burnin = 5` scores | yes for L = 0 |
| Update | `qts[t+1] = qts[t] - lr_t*grad`; `q = qts + lr_t*integrator` | `qt[k] -= lr_t*grad`; `q[k] = qt[k] + lr_t*eq` | yes |
| Radius used for the update | the radius **at prediction time** of the resolved item (`qs[t_pred]`) | `qarr[j]`, the radius in force when item j arrived | yes |
| Delayed label | `ahead = L + 1`: the label of arrival t - L is applied before arrival t + 1 | item j resolved before arrival j + L + 1 | yes |
| Scale under delay | window over the last `T_burnin` scores before t, **including unresolved ones** | window over **resolved** scores only | **no, by design** (documented in the script header; the reference form would use unresolved labels) |

Existing equivalence tests re-run read-only in this audit (`p07b_run_existing_eci_tests.py`, Hybrid GRU-D seed 42): T8 simulator equals
vendored ECI at L = 0 (max |dq| 5.6e-17); T9 per-class equals the Mondrian implementation (one tail row of 84,241 excluded, a known
end-of-stream clip); T10 unresolved-label flip leaves the radius unchanged; T11 recalibration controls; T12 L = infinity gives a
frozen radius; T13 equals the vendored `ahead = L + 1` when proportional scaling is off. All pass.

**Parameter differences from the reference (not defects).** (1) The 11 reference test configs use `T_burnin` between 100 and 1,000 (100 in eight of them); the repository uses 5 (fixed
by the pre-registration), so its proportional-step scale is the range of the last five scores, not of the last hundred or more. (2) Every reference config runs ECI at lr in {0.05, 0.1, 0.5, 1.0}; the repository's selection grid is {0.01, 0.05,
0.1}, so 0.01 lies outside the reference grid and 0.5 and 1.0 are not tried. (3) Calibration uses the score `|y - p|` on the
probability scale for the interval method; the logit-transform ablation is a separate study. None of these were changed.

**Calibration-selection stream versus lag** (`p07_eci_stream_vs_lag.py`). Streams (chronologically sorted): calibration_selection 12,075
arrivals (643 positives), final calibration 11,754 (888), test 84,241 (6,815). The Mondrian selection evaluates the second half, 6,038
arrivals. Labels resolved by the end of each stream:

| Lag L | Marginal selection (whole stream) | Mondrian selection, evaluated half (all / class 1) | Test stream (all / class 1) |
|---|---|---|---|
| 0 | 12,074 | 6,037 / 310 | 84,240 / 6,814 |
| 60 | 12,014 | 5,977 / 309 | 84,180 / 6,805 |
| 240 | 11,834 | 5,797 / 298 | 84,000 / 6,790 |
| 1,000 | 11,074 | 5,037 / 259 | 83,240 / 6,737 |
| 4,000 | 8,074 | **2,037 / 116** | 80,240 / 6,552 |
| 16,000 | **0** | **0 / 0** | 68,240 / 5,636 |
| infinity | 0 | 0 / 0 | 0 / 0 |

For L = 16,000 the lag exceeds the whole calibration-selection stream (12,075), so no label arrives during the selection run and the
learning rate chosen under that lag is not informed by feedback; for L = 4,000 the Mondrian selection run has 2,037 resolved labels (116
positives) behind its class-1 coverage estimate. The pre-registered realistic band (60, 240, 1,000) is unaffected: at least 5,037
labels (259 positives) are resolved in the evaluated half. The script comment records that for L = infinity the lag-0 learning rate is used.

**Other observations.** (a) The primary analysis (L = 0) applies each label before the next arrival, although the outcome is defined up
to 12 h after ED departure; the lag study is the realistic treatment. (b) Within a year block the stream order is a deterministic tiebreak
on the shifted `intime`, so order inside a year is not real-time-meaningful (`m2_hybrid_common.py:908-916`); Phase 1.8 addresses this.

---

## Items for your decision before Phase 1

1. **R1 reading (0.2):** keep the Elixhauser count (recommended, with the 42-stay fix in the Phase 2 builder and Phase 1.3 as the
   sensitivity analysis) or apply R1 literally.
2. **Config statement on truncation (0.4):** the yaml comment "measured 0 stays truncated" is contradicted by the data. I did not edit it
   (global rule 2); it should be corrected in the manuscript and, when you allow edits to existing files, in the config.
3. **Plan count (0.5):** 425,088 versus 425,087; I report the found value.
4. **Reference ECI parameters (0.7):** keep T_burnin = 5 and the {0.01, 0.05, 0.1} grid as pre-registered, and say so explicitly in
   the Methods, or run a sensitivity to T_burnin = 100 in Phase 1.

Phase 0 ends here. Phase 1 has not been started.
