# Module 2 (M2) — Outcome-Definition Gate

## 1. Deterioration Event Definition
The primary clinical outcome being predicted is the **Critical Outcome** (Task 2 of the MIMIC-IV-ED benchmark), which is a composite outcome defined as:
* In-hospital mortality, OR
* Transfer to an Intensive Care Unit (ICU) directly from the Emergency Department (ED).

## 2. Prediction Origin
The prediction origin (the moment the prediction is requested) is dynamically assessed at two distinct clinical timestamps:
* **T=0**: At ED Triage (using triage vital signs and patient history).
* **T=1**: At the latest recorded vitals during the ED stay (representing the streaming update before disposition).

## 3. Prediction Horizon (h)
* The binary benchmark task (Task 2) evaluates deterioration for the **current ED encounter / hospital admission**.
* The roadmap requires a multi-horizon evaluation at **h ∈ {1h, 6h, 12h}**, defined identically across NEWS2 / GRU-D / MAPIE / ECI.

> **Open discrepancy — not yet resolved, flagged for supervisor decision.**
> The horizon ablation as currently implemented (`m2_ablations.py`) does **not** meet this requirement. It filters the test set on `ed_los_hours ≤ h`, producing three *nested subsets of the same cohort carrying the same encounter-level label* (n = 1,318 / 46,426 / 73,193 at 1h / 6h / 12h). No label is re-derived per horizon and the prediction origin does not move, so "AUROC declines with horizon" is more accurately read as "AUROC is higher on short-stay patients."
>
> A true horizon evaluation requires a timestamped deterioration event, so that the label becomes "event occurs within h of the prediction origin." The composite critical outcome (in-hospital mortality OR direct ICU transfer) has an ICU-transfer timestamp available in `icustays`, but no comparable timestamp for the mortality component at ED-encounter resolution. Resolving this needs an explicit decision: re-scope the multi-horizon analysis to the ICU-transfer component alone, or state the LOS-filtered version as a labelled proxy and drop the horizon claim from the contribution.
>
> Related: `ed_los_hours` and the `ed_*_last` vitals are only knowable at ED **exit**, so the T=1 prediction origin is disposition time, not a mid-stay streaming update. This should be stated explicitly wherever the pipeline is described as "streaming".

## 4. Observation Window
The input observation window ($X_{1:t}$) consists of the sequence of vital signs up to the prediction origin. Specifically:
* **Static Profile**: Demographics (Age, Gender), Triage Acuity, Chief Complaints, and historical utilization (ED/Hosp/ICU visits in past 30/90/365 days).
* **Temporal Vitals**: The trajectory of 7 vital signs (Temperature, Heart Rate, Respiratory Rate, SpO2, Systolic BP, Diastolic BP, Pain Score), represented as a multi-step sequence with corresponding observation masks ($M_t$) and time decay ($\Delta_t$) between readings.

## 5. Cohort Inclusion Criteria
Patients are included if they meet the standard MIMIC-IV-ED benchmark criteria:
* Admitted to the ED.
* Possess at least one recorded triage vital sign.
* Age >= 18 years.

## 6. Exclusion Criteria
* Patients without any valid outcome disposition mapping.
* Encounters with extreme missingness in static history (where median imputation cannot safely reconstruct the clinical profile).

## 7. Patient-level Split Method
* **Split type**: **Random patient-level split.** `train_test_split` is applied to the unique `subject_id` list (`data_general_processing.py`), so every encounter of a given patient lands on one side only — 34% of patients in this cohort have more than one ED visit and 68% of rows belong to a repeat-visit patient, so a row-level split would leak the large majority of patients across the boundary.
* **Ratio**: Train (80%) / Test (20%), `random_state=42`. Verified: 0 patients overlap between train and test.
* **Validation**: a fixed, drawn-once 4-way partition of the 80% development pool (training / validation / calibration-selection / final-conformal-calibration), persisted to `data/m2_hybrid_dev_pool_split.json` and reused identically by every model and every seed.

> **Correction, 2026-09-01.** Earlier versions of this section described the split as "Chronological / Patient-level". That was inaccurate: `train_test_split` on subject IDs is a **random** partition, not a chronological one, and the two are not interchangeable. The split is and always was random; only the description was wrong.
>
> This matters for the module's primary research contribution rather than for leakage. Under a random split, train and test are temporally interleaved, so the deterioration-risk distribution does not shift from calibration to test — and ECI's reason for existing is precisely adaptation under temporal non-stationarity. A "ECI ≈ MAPIE" result on this design is a property of the evaluation, not evidence about ECI. Adding a chronological (rolling-origin) split as a second evaluation condition is the open item that would let the contribution actually be tested; it changes the frozen cohort and so requires supervisor sign-off before it is run.

---
**Status**: ⚠️ **PENDING SUPERVISOR APPROVAL — NOT SIGNED OFF.**

This gate has never been formally approved, yet training, the ECI contribution and the full ablation suite have all been executed against it. Under the project contract that ordering is a hard stop (`CLAUDE.md` §2 M2, "approved before any training begins"; §5 items 2 and 6). Recording this plainly rather than leaving the status line ambiguous: every result produced so far was produced ahead of sign-off, and two substantive items above (the horizon definition in §3 and the split type in §7) are open questions for the supervisor rather than settled protocol.

**Sign-off, when granted, should be recorded here with a date and the approver's name.**
