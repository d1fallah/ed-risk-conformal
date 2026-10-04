# Module 2 (M2) — Outcome-Definition Gate — v2

**Status: ✅ SIGNED OFF — 2026-09-19, approved by Akram.**
**Supersedes:** `M2_Outcome_Definition_Gate.v1.md` (preserved verbatim in this repo for review-trail purposes; do not delete). v1 was itself never signed off — see its own status line — and all M2 training/results to date were produced ahead of any sign-off, in violation of `CLAUDE.md` §5 item 2. This v2 does not repeat that: nothing in Phase 2 onward of the session's M2 Q1-blockers plan (cohort rebuild and everything downstream) starts before a dated sign-off is recorded below.

**Why v2 exists.** An external Q1-readiness review of M2 surfaced four defects. Two of v1's own open items — the LOS-filtered horizon (§3) and the random-vs-chronological split question (§7) — are root causes of two of them. v1 asserted, in §3, that a true horizon label was infeasible because "no comparable timestamp for the mortality component" exists. **That assertion was checked against the raw MIMIC-IV data in this session and is false**: `hosp/admissions.csv.deathtime` is populated for 11,790 of 11,801 in-hospital deaths (99.91%), and `icu/icustays.csv.intime` is 100% populated. A genuine time-to-event horizon label is constructible. This version replaces the LOS filter with it.

---

## 1. Deterioration Event Definition — unchanged from v1

The primary clinical outcome is the **Critical Outcome** (Task 2 of the MIMIC-IV-ED benchmark):
* In-hospital mortality, OR
* Transfer to an ICU directly from the Emergency Department.

`outcome_critical = outcome_inhospital_mortality | outcome_icu_transfer_12h`, as implemented in `helpers.py::add_inhospital_mortality` / `add_outcome_icu_transfer` (`extract_master_dataset.py:63`). This definition, and the ≤12h ICU-transfer window it is built on, are unchanged — reopening them would require re-deriving `outcome_critical` itself, which is out of scope for this gate.

## 2. Prediction Origin — revised

* **T=0**: ED triage (`intime`), using triage vitals + static profile. Unchanged from v1.
* **v1's T=1 ("latest recorded ED vitals … representing the streaming update before disposition") is retracted.** `ed_los_hours` and every `ed_*_last` vital are only computable at `outtime` — ED exit. That is disposition time, not a mid-stay update, and describing it as "streaming" was a mislabel. This is not a new finding; it was already flagged as an open item in v1 §3's closing paragraph and is now resolved by removal rather than by caveat.
* **v2 origin, with the irregular-series rebuild (Phase 2 of the session plan):** every genuine vitals observation `charttime` between `intime` and `outtime` is a valid prediction origin. The mandatory-core streaming claim is evaluated at **T=0 (triage)** only, since that is the one origin available before disposition; later origins are used for the model's internal recurrence (its input sequence), not claimed as separate streaming evaluation points, unless a future gate revision proposes rolling-origin evaluation explicitly.

## 3. Prediction Horizon (h) — revised: real time-to-event label replaces the LOS filter

**v1's implementation is retracted.** `m2_ablations.py`'s `ed_los_hours <= h` filter produced nested subsets of the same cohort carrying the same encounter-level label (n = 1,318 / 46,426 / 73,193 at h = 1h/6h/12h) with a fixed origin — "AUROC declines with horizon" was actually "AUROC is higher on short-stay patients." This is now replaced.

**v2 definition:**
```
event_time = min( icustays.intime  for the first ICU stay after ED outtime within 12h,
                   admissions.deathtime )     # whichever is earlier and non-null
y_h(origin) = 1{ 0 <= (event_time - origin) <= h },  h in {1h, 6h, 12h}
```
with `origin = intime` (triage, T=0) for the mandatory-core evaluation. The full cohort (n = 425,088 pre-filter) is retained at every horizon — only the positive label changes — so the three horizons are genuinely nested-in-*time*, not nested-in-*cohort*.

**Measured by the actual implementation** (`Benchmark_scripts/m2_build_irregular_cohort.py::build_horizon_labels`, run against the real, gate-filtered cohort — supersedes this gate's earlier preliminary manual estimate, which used a looser event-time definition; corrected here so the gate reflects the code that actually runs, not a hand-computed approximation, per `CLAUDE.md` §1.1's Golden Rule):

| h | Positive rate (eligible cohort, n=417,964) | Positive n |
|---|---|---|
| 1h | 0.00055 | 228 |
| 6h | 0.03391 | 14,175 |
| 12h | 0.05152 | 21,534 |

The 1h horizon has very few positives (228) — flagged now, not discovered mid-training: report it with a wide bootstrap CI and do not over-interpret AUROC at that horizon. This is a real small-effective-sample-size finding, not a pipeline defect. Note the ICU component of `event_time` is gated on `outcome_icu_transfer_12h == True` (a DIRECT ED→ICU transfer within 12h, matching `outcome_critical`'s own definition exactly) rather than any ICU admission ever — this is why these final numbers differ from the gate's original preliminary estimate, which did not apply that gate and so counted some later, indirect ICU admissions as if they were the outcome event.

**Exclusions, both small and both to be reported by count in every table footnote using this label:**
* **54 of 418,025 gate-eligible encounters (0.013%) have no recoverable `event_time`** — `outcome_critical=1` but neither a valid direct-transfer `intime_icu` nor a valid `deathtime` exists on that encounter (an inherent data-quality gap in the source tables, not a pipeline defect). Excluded from horizon-labeled analyses only; retained in the non-horizon composite-outcome analyses that don't need `event_time`.
* **7 encounters have `event_time < intime`** (event recorded before the ED stay that supposedly preceded it) — a MIMIC-IV date-consistency artifact. Excluded from horizon-labeled analyses.

Total horizon-eligible cohort: 418,025 − 54 − 7 = 417,964 (0.015% dropped).

## 4. Observation Window — revised for the irregular-series rebuild

* **Static Profile**: unchanged from v1 — demographics, triage acuity, chief complaints, 30/90/365-day utilization history.
* **Temporal Vitals — replaces v1's fixed T=2.** The full `ed/vitalsign.csv` per-stay sequence, not just triage + last. Measured this session: 408,146 of ~425k stays have ≥1 ED vitalsign row (mean 3.83 obs/stay, median 3, p75=5, p90=8, p95=10, p99=16, max=109).
  * **`T_max = 10`** (the measured p90) most-recent observations per stay, right-truncated if longer, kept in chronological order; triage (T=0) is always included as the first element regardless of truncation. Truncation rate (fraction of stays exceeding `T_max`) must be reported in the manuscript — measured this session at approximately 10% of stays with ≥1 ED observation.
  * **Stays with exactly 1 ED-vitals observation (~25% of stays with any ED vitals)**: retained, sequence length 2 (triage + the one observation) or 1 if triage itself is missing that vital — handled by the existing mask `M_t`, not a special case.
  * **Stays with zero ED-vitals rows** (~425k − 408k ≈ 17k, ~4% of cohort): retained with triage-only input (sequence length 1), consistent with GRU-D's mask-and-decay design; report this subgroup's size and outcome rate separately, since it is a real missing-data population, not an artifact to hide.
  * Observation mask `M_t` and time-decay `Δt` (real inter-observation hours from consecutive `charttime`, not the current `ed_los_hours / median_LOS` proxy) computed per `Benchmark_scripts/m2_hybrid_common.py::prepare_temporal_tensors`, to be rewritten under Phase 2 of the session plan.

## 5. Cohort Inclusion Criteria — unchanged from v1

* Admitted to the ED.
* At least one recorded triage vital sign.
* Age ≥ 18.

## 6. Exclusion Criteria — extended

* Patients without any valid outcome disposition mapping (v1, unchanged).
* Encounters with extreme missingness in static history where median imputation cannot safely reconstruct the profile (v1, unchanged).
* **New (§3): the 41 + 13 = 54 encounters lacking a recoverable or consistent `event_time`, for horizon-labeled analyses only** (not for the non-horizon composite-outcome analyses, where they are retained exactly as in v1).

## 7. Patient-level Split Method — revised: chronological promoted to primary

v1 retained a **random** patient-level split (`train_test_split` on `subject_id`, `random_state=42`, 80/20) as primary, with a chronological arm as an auxiliary evaluation only. v1 itself recorded why that is the wrong way around for this module's primary contribution: under a random split, train and test are temporally interleaved, so the deterioration-risk distribution does not shift from calibration to test — and ECI's reason for existing is precisely adaptation under temporal non-stationarity. An "ECI ≈ MAPIE" result on this design is a property of the evaluation, not evidence about ECI. Consistent with that, and now backed by the already-executed auxiliary arm's own result (MAPIE undercovers 0.8837 vs. nominal 0.90 under the real chronological shift while ECI holds 0.9008, non-overlapping across all 5 seeds) —

**v2 primary split: chronological (rolling-origin).** Patient-level, cut by `anchor_year_group`-recovered calendar time exactly as `m2_chronological_eci.py::chronological_split` already implements, train = earlier period, test = later period, encounters straddling the cut dropped and counted. **v2 secondary split: the existing random 80/20 split, retained as a robustness arm**, not deleted — a result that holds under both splits is a stronger claim than one that holds under only one.

**Both splits reuse the existing machinery** (`m2_hybrid_common.py::build_fixed_dev_split`, `m2_chronological_eci.py`), inverted in priority rather than rewritten. Subject-ID SHA-256 persistence and the 0-patient-overlap assertion apply identically to whichever split is primary.

## 8. Mandated Comparator Set — new in v2

Settled here, at gate time, rather than left as an open item for review to catch:

| Comparator | Status | Role |
|---|---|---|
| NEWS2 (partial, 5/7 RCP params) | existing | clinical baseline; forensic sensitivity analysis (composite vs. mortality-alone vs. ICU-alone vs. complete-case) required in every table this score appears in, given the disclosed gap to the published 0.71-0.86 ED-validation range |
| LR, RF, XGBoost (untuned + matched-budget tuned) | existing | classical-ML baselines |
| GRU-D (+MAPIE, +ECI) | existing | Track A mandatory core / primary contribution vehicle |
| **mTAN** (Shukla & Marlin, *Multi-Time Attention Networks for Irregularly Sampled Time Series*, ICLR 2021) | **new, recommended** | modern irregular-time-series comparator — vendor per `CLAUDE.md` §1.7, run under the same matched-tuning protocol as XGBoost-Matched once the irregular-series cohort exists (Phase 2 of the session plan). The T=2 architecture made this comparator arguably unnecessary; the rebuild removes that argument, so this is close to mandatory for the target venues rather than optional. A measured compute cost will be brought to the supervisor before this run starts (Phase 5 of the session plan). If declined, record the decision here with a date and reclassify flaw #1 as a stated manuscript limitation rather than an open item. |
| SeFT (Horn et al., ICML 2020) | **not included** | mTAN is judged the stronger single comparator for the available budget; adding both is not judged worth the marginal cost. Revisit only if a reviewer specifically requests it. |
| Hi-BEHRT (named once in `SihhaDZ_Plan_Consolide`, weeks 12-14) | **not included** | never run to date; no plan to add it under the current scope |

---

**Status**: ✅ **SIGNED OFF.**

This v2 resolves both substantive open items carried from v1 (§3 horizon, §7 split) using facts verified against the raw MIMIC-IV tables in this session, and adds a comparator decision (§8) that v1 left implicit. Per `CLAUDE.md` §5 item 2, no training against this v2 gate started before sign-off.

**Sign-off record:**
- **Date:** 2026-09-19
- **Approved by:** Akram
- **Scope of approval:** the gate as written, including §8's mTAN recommendation — the supervisor's follow-up instruction was explicit: "You are now fully unblocked to begin Phase 2 ... and the mTAN execution," so §8's conditional ("a measured compute cost will be brought to the supervisor before this run starts") is superseded by this direct authorization. mTAN proceeds without a separate compute-cost check-in.
- Phase 2 (cohort rebuild) and downstream phases of the session's M2 Q1-blockers plan are unblocked as of this sign-off.
