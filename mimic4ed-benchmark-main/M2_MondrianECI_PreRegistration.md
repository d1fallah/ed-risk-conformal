# Mondrian-ECI sets — pilot record and confirmatory pre-registration

**Status: Part 1 (pilot) CLOSED with a NO-GO verdict. Part 2 (confirmatory) PRE-REGISTERED, written 2026-09-20 BEFORE the confirmatory run was executed.**

This document exists so that the distinction between the exploratory pilot and the confirmatory run cannot be blurred after the fact. Part 1's verdict is permanent and is not revised by anything in Part 2.

---

## Part 1 — Pilot (exploratory). CLOSED, verdict NO-GO.

### What was declared in advance

The construction, endpoints, and GO/NO-GO thresholds were fixed in the session plan before any result was seen (`m2_mondrian_eci_pilot.py` docstring, same values):

- **Primary endpoint:** `P(C = {1} | y = 1)` — *informative* minority coverage (a singleton `{1}`), deliberately **not** `P(1 ∈ C | y = 1)`, because the latter is satisfiable by an always-`{0,1}` predictor.
- Empty sets count as a **miss** for both classes.
- **NO-GO** if mean |C| ≥ 1.90, **or** q₁ ≥ 1 on > 25% of the stream, **or** `P(C={1}|y=1)` < 0.10.
- **GO** if class-1 coverage ∈ [0.85, 0.95] **and** mean |C| < the Mondrian-LAC anchor **and** `P(C={1}|y=1)` ≥ anchor + 0.05.
- **MIDDLE** otherwise → contribution reframes to shift-tracking.
- Learning rate selected on the `calibration_selection` fold only (never the test window), from the pre-declared grid {0.01, 0.05, 0.1}. The **full grid was pre-declared as a reported sensitivity check**, on the stated grounds that saturation is lr-driven and one unlucky lr must not silently kill the method.

### Correctness tests (both passed, run before any metric)

- **T1 causality** — flipping every label at positions ≥ t leaves the radii in force at positions ≤ t bit-identical (and does change them afterwards). No future label can influence a current prediction set.
- **T2 static equivalence** — substituting the static per-class calibration quantile for the online radius reproduces `mondrian_lac_sets()` exactly, confirming the two constructions differ in exactly one variable (threshold source).

Independent corroboration that the pilot's data path is sound: for **Temporal-Only GRU-D v2**, whose checkpoint was unchanged since the original conformal run, the pilot reproduced `data/m2_v2_conformal_summary.csv`'s Mondrian-LAC class-1 coverage to **0.000000 on all 5 seeds**.

### Verdict

**NO-GO**, triggered solely by the saturation criterion. Primary model (Hybrid GRU-D v2), 5 seeds, at the lr the declared selection rule chose:

| quantity | value | threshold |
| :--- | ---: | :--- |
| class-1 coverage | 0.9238 | in GO band [0.85, 0.95] ✓ |
| informative `P(C={1}|y=1)` | 0.1820 | NO-GO if < 0.10 ✓ (anchor 0.0000) |
| mean \|C\| | 1.6565 | NO-GO if ≥ 1.90 ✓, but **above** anchor 1.5159 ✗ |
| **q₁ saturation fraction** | **0.4297** | **NO-GO if > 0.25 → TRIGGERED** |

Artifacts: `data/m2_v2_mondrian_eci_pilot.csv`, `data/m2_v2_mondrian_eci_pilot_verdict.json`.

### Pre-declared sensitivity grid (Hybrid GRU-D v2, 5-seed means)

| lr | class-1 cov | informative | mean \|C\| | q₁ saturation |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.9066 | 0.1740 | 1.4872 | 0.012 |
| 0.05 *(selected)* | 0.9216 | 0.1817 | 1.6411 | 0.386 |
| 0.10 | 0.9304 | 0.1821 | 1.7067 | 0.574 |
| *Mondrian-LAC anchor* | *0.8805* | *0.0000* | *1.5159* | — |

### Diagnosed failure mechanism

The declared rule selects lr on `calibration_selection`, which lies in the **train period at 5.4% class-1 prevalence**, where the minority stream is starved and a larger step size is favoured to keep up. The evaluated window runs at **8.09% prevalence** (+49.8% relative), so the minority stream updates ~50% more often and the selected step overshoots into the ratchet/saturation mode — q₁ pins at its bound and label 1 is then included unconditionally.

**The selection rule fails to transfer across exactly the prevalence shift the method exists to handle.** This is a finding about the method's hyperparameter-selection procedure, not about the set construction, and it is reportable in its own right.

### What is explicitly NOT concluded from Part 1

That lr = 0.01 "works". The learning-rate selection rule is **part of the method**: at deployment lr must be chosen without observing test results. Reporting the lr = 0.01 column as the headline result would be post-hoc selection on the evaluation window. Part 1's verdict stands as NO-GO.

---

## Part 2 — Confirmatory run. PRE-REGISTERED 2026-09-20, before execution.

### Exactly one thing changes

A **degeneracy guard is added to learning-rate selection**. Nothing else — not the construction, not the endpoints, not the GO/NO-GO thresholds, not the lr grid, not the seeds, not the evaluation window.

The guard has direct precedent in this project: `m2_xgboost_tuning.py` already rejects any candidate whose ECE exceeds 5× the baseline's, on the principle that a candidate winning the primary metric while exhibiting a known degenerate mode must not be selected. Saturation is this construction's known degenerate mode (declared as such in Part 1, before any result). The guard applies that same principle.

### Guard specification (fixed now)

For each candidate lr in the unchanged grid {0.01, 0.05, 0.1}:

1. Run the construction on the `calibration_selection` fold only — first half as warm-up, second half evaluated. **No test-window data is read at any point during selection.**
2. Compute the q₁ saturation fraction on the evaluated half.
3. **Reject** the candidate if that fraction exceeds **0.25** — numerically the same threshold as the declared NO-GO criterion, so the guard and the verdict cannot disagree about what "degenerate" means.
4. Among surviving candidates, select by |class-1 coverage − 0.90|, tie-broken on mean |C| — the unchanged Part 1 rule.
5. **If every candidate is rejected**, select the smallest lr in the grid and record that the guard eliminated all candidates.

### Evaluation criteria — unchanged from Part 1

Identical endpoints, identical NO-GO/GO/MIDDLE thresholds, identical primary model (Hybrid GRU-D v2), identical 5 seeds, identical anchor (Mondrian-LAC on the same calibration fold). Restated so they cannot drift:

- **NO-GO** if mean |C| ≥ 1.90, or q₁ saturation > 0.25, or `P(C={1}|y=1)` < 0.10.
- **GO** if class-1 coverage ∈ [0.85, 0.95] and mean |C| < anchor and `P(C={1}|y=1)` ≥ anchor + 0.05.
- **MIDDLE** otherwise.

### Commitments

- The confirmatory outcome will be reported **whatever it is**, including a second NO-GO. A NO-GO here is a real result: it would mean the guard does not rescue the selection rule, and the method's dependence on step size is not resolvable from non-test data at this prevalence.
- The guard threshold will **not** be adjusted after seeing the confirmatory outcome. If 0.25 turns out to admit a saturating candidate, that is reported, not re-tuned.
- Any manuscript must state that the pilot was exploratory and hypothesis-generating, that it produced a NO-GO, and that it motivated the guard — the confirmatory result may never be presented as if it were the first and only experiment.
- The pilot's artifacts are retained unmodified alongside the confirmatory ones.

### Anticipated outcome, recorded in advance

The guard is expected to reject lr ∈ {0.05, 0.1} and select 0.01, on the mechanism diagnosed in Part 1. **This is a prediction, not a result**, and is recorded here so that agreement or disagreement with it is itself informative — in particular, if the selection fold's starved minority stream saturates at *every* lr, the guard will eliminate all candidates and fall through to rule 5, which would be evidence that non-test data cannot identify a safe step size at this prevalence.

---

## Part 3 — Comparator defect found after the confirmatory run. Recorded 2026-09-20.

**This part changes what Parts 1 and 2 may be cited for. Read it before citing either.**

### What happened

After the confirmatory run returned **GO**, a consistency check on its secondary numbers exposed a defect in
`mondrian_lac_sets()` — the static Mondrian-LAC construction used as the comparator ("anchor") in Parts 1 and 2.
Class 0 was calibrated on `1 - p` while label 0 was scored as `p` at test time, so the class-0 threshold sat
near 1, label 0 was always included, and the anchor could never emit a singleton `{1}`. Signature: class-0 coverage
of **exactly 1.0000** on every model and seed in both cohorts. Class-1 calibration was correct, so class-1 coverage
— the number the module's headline claims rest on — was unaffected, which is why the defect survived earlier review.

Fixed in `m2_mondrian_conformal.py`; `test_mondrian_lac_validity.py` now guards it (label-conditional validity is
distribution-free, so each class must sit at 0.90 for any prevalence and separation).

### Consequence for this experiment

Parts 1 and 2 compared Mondrian-ECI against a comparator that was handicapped by construction. The pre-registered
criteria were **not** changed; only the anchor was recomputed with the fixed function
(`m2_mondrian_eci_anchor_correction.py`). Mondrian-ECI's own sets are unchanged — the defect was never in them.

| Hybrid GRU-D v2, 5-seed means | defective anchor | **corrected anchor** | Mondrian-ECI |
| :--- | ---: | ---: | ---: |
| class-0 coverage | 1.0000 | **0.8894** | 0.9091 |
| class-1 coverage | 0.8805 | **0.8805** | 0.9066 |
| informative P(C={1} \| y=1) | 0.0000 | **0.2156** | 0.1740 |
| mean \|C\| | 1.5159 | **1.3967** | 1.4872 |
| **verdict, criteria unchanged** | GO | **MIDDLE** | |

**The Part 2 GO is retracted as a statement about efficiency.** Against a correct static comparator Mondrian-ECI is
*less* informative and produces *larger* sets. The confirmatory GO rested on a comparator that scored 0.0000 on the
primary endpoint by construction; it must not be cited as evidence the method beats static Mondrian-LAC.

### What survives

The pre-registered **MIDDLE** branch — "coverage fine, efficiency tied or worse → the contribution reframes to
shift-tracking" — was declared in advance for exactly this outcome, so the reframing is not post-hoc. The supported
claim is narrow: the corrected static anchor **under**-covers on this cohort (Hybrid class-1 0.8805, class-0 0.8894;
Frozen-Control and Temporal-Only class-0 ≈ 0.871), consistent with residual within-class shift, while Mondrian-ECI holds
both classes at ≈0.91. It buys that at the price of larger sets. Whether that trade beats the obvious cheap remedy —
periodic recalibration of the *static* method — is untested, and Phase 4's static-plus-recalibration strawman is now
the decisive experiment, not a formality.

### Standing commitments (unchanged, restated)

- Part 1's NO-GO, Part 2's GO, and this Part 3 correction are all retained; none is deleted.
- A manuscript must state that the pilot was exploratory, that a comparator defect was found afterwards, and that
  the efficiency claim was retracted.
- The confirmatory artifacts (`m2_v2_mondrian_eci_confirmatory*.{csv,json}`) are unmodified; corrected-anchor results
  live in `m2_v2_mondrian_eci_anchor_corrected*.{csv,json}`.

---

## Part 4 — Decisive-comparator pre-registration. Written 2026-09-20 BEFORE any comparator was run.

**Question.** Part 3 left Mondrian-ECI with one supported claim: it restores coverage on both classes under shift where
the corrected static method under-covers. That claim is only worth making if nothing *cheaper* achieves the same
validity. This part fixes, in advance, the comparators and the rule that decides whether the online construction is needed.

### Comparators (all deployable — every one uses only labels that arrived before the point being predicted)

All evaluated on the same stream, same predictions, same 5 seeds × 3 GRU-D-family models, same test window; labels assumed
available immediately (lag 0), which is generous to every method equally — label-lag sensitivity is a separate, later experiment.

| id | method | conditioning | adaptivity |
|---|---|---|---|
| A | corrected static Mondrian-LAC (anchor) | label | none |
| B | **Mondrian-ECI** (the construction under test) | label | online (ECI), guarded lr |
| C1, C2 | **static Mondrian + periodic recalibration** | label | thresholds recomputed every R = 2,000 arrivals from the most recent W arrivals, W ∈ {10,000; 40,000} |
| D | static score-bin Mondrian-LAC | feature-space: 10 quantile bins of the model's own risk score p, edges from the final-calibration fold | none |
| E | **online score-bin ECI** | feature-space: same 10 bins | online (one ECI stream per bin), guarded lr |

C is the obvious cheap remedy for temporal shift; D and E are the feature-space proxy for label conditioning that
POGO/MVP-style group conditioning permits (their groups are arbitrary functions of X — see `M2_Related_Work_Notes.md`
§4.3.6) and which a reviewer will ask for. Window sizes W are in stream arrivals, **not** calendar time: within-year order
is arbitrary-but-deterministic in this cohort. Learning rates for B and E are chosen on the `calibration_selection` fold
only, from the unchanged grid {0.01, 0.05, 0.1}, with the same saturation guard (reject if the saturated fraction exceeds
0.25), and by the same selection rule (|class-1 coverage − 0.90|, tie-broken on mean |C|).

### Definitions (fixed now)

- **Valid**: coverage of **both** classes ≥ 0.89 (5-seed mean, test window). This is the pre-declared meaning of "restores
  coverage"; it is looser than 0.90 by 0.01 to allow for finite-sample and seed noise, and it is applied identically to every method.
- **Efficiency**: mean set size |C|, and informative P(C={1} | y=1). Compared **only among valid methods** — comparing a
  method that covers less against one that covers more is not a comparison of efficiency (the confound Part 3 identified).
- Tie tolerance on |C|: 0.02.

### Decision rule for Mondrian-ECI (Hybrid GRU-D v2 primary; the two other models reported as replication)

- **NOT NEEDED** if any of A, C1, C2, D is valid with mean |C| ≤ B's |C| + 0.02. The online machinery then buys nothing a
  cheaper method does not.
- **DOMINATED** if E is valid with mean |C| < B's |C| − 0.02. B is then superseded by a group-conditional online method
  (which is closer to existing prior art and is not this module's construction).
- **SUPPORTED — sole valid method** if B is valid and none of A, C1, C2, D, E is.
- **SUPPORTED — efficient** if B is valid, and every other valid method has |C| > B's |C| + 0.02.
- Anything else is reported as **INCONCLUSIVE** with the numbers.

### Inference (fixed now)

The patient-cluster bootstrap is **not** used for any online or recalibrating method (see the plan: resampling patients
changes the arrival process the methods depend on). Uncertainty is reported as (i) across-seed SD, and (ii) across
**contiguous stream blocks**: K = 10 equal-size blocks and the 4 real calendar years, with within-block paired differences
against B. Stream blocks handle sequential dependence but are **not calendar periods**; only the 4-year split is calendar,
and it is unbalanced (2017 holds 82% of the window). Block SDs are descriptive, not a significance test.

### Commitments

The outcome is reported whatever it is. A NOT NEEDED or DOMINATED verdict is a real result and will be stated as the
paper's finding, not buried. The thresholds above are not adjusted after seeing results. The Part 1–3 record is unchanged.

### Part 4 — RESULT (recorded 2026-09-20). Verdict: **NOT NEEDED**, on all three models.

Executed by `m2_mondrian_eci_comparators.py` exactly as pre-registered; correctness tests T3 (recalibration that never fires reproduces the static
anchor), T4 (online score-bin ECI is causal) and T5 (score-bin coverage on exchangeable data) passed before any metric. Hybrid GRU-D v2, 5-seed means:

| method | class-0 cov | class-1 cov | mean \|C\| | informative P({1}\|y=1) | valid (both ≥ 0.89) |
| :--- | ---: | ---: | ---: | ---: | :---: |
| A  static Mondrian-LAC (corrected) | 0.8894 | 0.8805 | 1.3967 | 0.2156 | no |
| B  **Mondrian-ECI** (online, label-conditional) | 0.9091 | 0.9066 | **1.4872** | 0.1740 | yes |
| C1 static Mondrian + recalibration, W = 10,000 | 0.8990 | 0.8987 | **1.4291** | 0.1963 | yes |
| C2 static Mondrian + recalibration, W = 40,000 | 0.8969 | 0.8926 | **1.4191** | 0.2002 | yes |
| D  static score-bin Mondrian | 0.9554 | **0.2303** | 1.0870 | 0.0147 | no |
| E  online score-bin ECI | 0.9697 | **0.2897** | 1.1292 | 0.0131 | no |

**Decision rule outcome.** C1 and C2 are valid with mean \|C\| ≤ B's + 0.02 (in fact *smaller* by 0.058 and 0.068) → **NOT NEEDED**. Identical on
Temporal-Only GRU-D v2 and Frozen-Temporal Head Control v2 (replication): recalibration is valid with \|C\| 1.80 / 1.80 against B's 1.82 / 1.83.

**What this means.** A static label-conditional method whose thresholds are simply refreshed every 2,000 arrivals from a recent window restores both
coverage levels to ≈0.90 *more efficiently* than the online construction. The online machinery buys nothing here that a trivial remedy does not. Mondrian-ECI
is therefore **not a supported contribution of this module**, and no manuscript may present it as one.

**What the comparators established, independent of that verdict.**
- **Feature-space conditioning fails on the minority class.** Score-binned group-conditional conformal — the proxy for label conditioning that POGO/MVP-style groups
  permit — reaches class-1 coverage of only 0.09–0.29 (static and online alike) across all three models, against ≥ 0.89 for every label-conditional method.
  Conditioning on the true label is not replaceable by conditioning on covariates of the model's own risk score. This answers the reviewer question flagged in
  `M2_Related_Work_Notes.md` §4.3.6 before it is asked.
- **Static label-conditional calibration under-covers under this shift and is repaired cheaply.** A: class-0 0.871–0.889; C1/C2: ≈0.895–0.899.

**Honest descriptive nuance the decision rule does not credit.** Across 10 contiguous stream blocks (not calendar periods) B's class-1 coverage ranges 0.891–0.917
against 0.871–0.916 for C1; in calendar year 2018 (764 class-1 points) B holds 0.915 where C1/C2 fall to 0.865/0.851. B tracks more tightly. But B used larger sets
than C1 in **10 of 10 blocks**, the year-level evidence rests on very few positives, block SDs are descriptive and not a significance test, and the pre-declared rule
was mean validity plus set size. The verdict is not revised on this basis.

**Limits.** Labels assumed immediate (lag 0) for every method; a lag sweep may favour or disfavour either. *(Update 2026-09-21: the lag sweep was run — `M2_LabelLag_PreRegistration.md`, dossier §24.3d — and did not change the verdict at realistic lags.)* Recalibration windows are in arrivals, not calendar time.
The recalibration grid (R, W) was pre-declared and not tuned, but only two windows were tried. The test window has 4 real calendar years, 82% in 2017.
