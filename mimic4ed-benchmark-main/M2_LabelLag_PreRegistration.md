# Label lag: does ECI (or Mondrian-ECI) beat periodic recalibration once outcomes arrive late? — pre-registration

**Written 2026-09-21, BEFORE `m2_label_lag.py` exists and before any lagged result has been computed.**

## Why this exists

Every pre-registered comparison so far (`M2_ECI_vs_Recalibration_PreRegistration.md`, `M2_MondrianECI_PreRegistration.md` Part 4) assumed **labels arrive immediately (lag 0)**
for every online and recalibrating method, and both concluded the online method is NOT NEEDED. Both documents named label lag as the untested axis most likely to
separate the methods, and the module's own deployment story (outcome known ~12 h after the prediction) makes lag the realistic condition. This experiment tests it.
It is symmetric: a result in which ECI or Mondrian-ECI is needed at realistic lags is reported exactly as prominently as one in which it is not, and would change the
manuscript's framing.

## Lag definition

- **Unit: arrivals, not hours.** Dates are shifted per patient in this cohort and within-year order is not real-time meaningful, so an arrival rate cannot be measured
  from the data. As a coarse conversion only, MIMIC-IV-ED holds ~418k gate-eligible stays over roughly 9 years, ≈ 5 arrivals/hour (uncertain by ±50%).
- A lag of **L arrivals** means: the label of arrival *j* becomes available immediately before arrival *j + L + 1* is predicted. L = 0 is the lag-0 setting of the
  earlier pre-registrations. The final-conformal-calibration fold is fully labelled historical data (already resolved when deployment starts); lag applies to the test stream.
- **Grid (fixed now):** L ∈ {0, 60, 240, 1000, 4000, 16000, ∞}. At ~5 arrivals/h these are ≈ {0, 12 h, 2 d, 8 d, 33 d, 4.4 months, never}.
  L = ∞ means no label ever arrives during deployment: recalibration reduces to the static method and ECI is frozen at its end-of-calibration state.
- **Realistic band (fixed now):** L ∈ {60, 240, 1000} (≈ 12 h to ~1 week), the range in which outcomes of an ED composite-critical-event label plausibly resolve.
  L ∈ {4000, 16000, ∞} are stress tests, reported but not part of the decision.
- Uniform lag for every patient. Real resolution times are heterogeneous (ICU transfer within 12 h resolves fast; in-hospital death can take weeks) and are **not** modelled;
  this is a limitation, stated up front.

## Methods (all deployable: only labels that have arrived are used, verified by tests below)

| id | family | method | under lag L |
|---|---|---|---|
| M1 | marginal | static split-conformal (MAPIE-equivalent) | unaffected by lag |
| M2 | marginal | **ECI** | feedback for arrival *j* applied L arrivals late; lr re-selected per L on `calibration_selection` (same two-stage rule as `run_eci_4way`, lag applied there too) |
| M3, M4 | marginal | M1 + periodic recalibration, W = 10,000 / 40,000, R = 2,000 | window ends at the latest *resolved* arrival, i.e. `[s − L − W, s − L)` at a refresh at arrival *s* |
| A | class-conditional | corrected static Mondrian-LAC | unaffected by lag |
| B | class-conditional | **Mondrian-ECI**, per-class streams, guarded lr (saturation guard 0.25, unchanged) | per-class feedback applied L arrivals late |
| C1, C2 | class-conditional | static Mondrian + periodic recalibration, W = 10,000 / 40,000, R = 2,000 | window ends at latest resolved arrival, as M3/M4 |

The score-bin comparators (D, E) are not repeated: they already failed the minority class at lag 0 (class-1 coverage 0.23–0.29) and delay cannot help them.
ECI's update rule is **not** altered. The vendored implementation already exposes delayed feedback (`ahead` argument, the ECI paper's multi-step-ahead mechanism);
the lag simulator used here reproduces its update recursion event by event so that lag can be applied to the test stream only and per class, and is verified
against the vendored code (tests T8, T9, T13). One deliberate difference from the vendored `ahead` mode: the proportional learning-rate scale `lr_t` in the vendored
code is computed from the last 5 scores *including unresolved ones*; the simulator uses only resolved scores (strictly causal). The vendored `ahead = L + 1`
run is reported as a cross-check.

## Definitions (unchanged from the lag-0 pre-registrations)

- **Marginal family — valid:** marginal coverage ≥ 0.89; efficiency = mean clipped interval width, compared only among valid methods; tie tolerance 0.02.
- **Class-conditional family — valid:** coverage of both classes ≥ 0.89; efficiency = mean |C| (and P(C={1} | y=1)), compared only among valid methods; tie tolerance 0.02.
- 5-seed means on the test window; Hybrid GRU-D v2 is primary, Temporal-Only and Frozen-Control are replication.

## Decision rule (per lag L, per family, primary model; replication reported alongside)

Marginal (M2 vs M3/M4; M1 as reference):
- **ECI NOT NEEDED** if M3 or M4 is valid with width ≤ M2's width + 0.02.
- **ECI SUPPORTED — sole valid method** if M2 is valid and neither M3 nor M4 is.
- **ECI SUPPORTED — efficient** if M2 is valid and each valid M3/M4 is wider than M2 + 0.02.
- **ECI INVALID** if M2 fails validity (with the note whether recalibration is valid). **BOTH INVALID** if neither M2 nor any recalibration is valid.

Class-conditional (B vs A, C1, C2): the same four outcomes with |C| in place of width and both-class validity in place of marginal validity.

## Overall reading (fixed now)

- **The negative result is ROBUST TO REALISTIC LAG** if, for the primary model and both families, no lag in the realistic band {60, 240, 1000} yields an
  ECI/Mondrian-ECI SUPPORTED outcome.
- **The negative result DOES NOT SURVIVE label lag** if any realistic-band lag yields an ECI/Mondrian-ECI SUPPORTED outcome on the primary model **and** on at least one
  replication model. The dossier, report skeleton and framing note are then revised accordingly.
- Otherwise: reported as **MIXED** with the numbers.
- Descriptive, not decision-bearing: the **collapse point** of each method (smallest L at which it fails validity), the stress-test lags, per-block coverage spread.

## Sensitivity arms (declared now, reported alongside, not decision-bearing)

- ECI with lr fixed at the lag-0-selected value instead of re-selected per L.
- Vendored `ahead = L + 1` ECI as a cross-check of the simulator (expected to differ slightly because of the `lr_t` window and warm-up handling).

## Correctness tests that must pass before any lagged result is printed

- **T8**: simulator at L = 0, marginal stream, reproduces the vendored ECI radius trajectory exactly.
- **T9**: simulator at L = 0, per-class streams, reproduces `mondrian_eci_sets`' radii exactly.
- **T10 (causality under lag)**: flipping every label not yet resolved at time *t* leaves the radius at *t* unchanged, for L > 0.
- **T11**: recalibration at L = 0 reproduces the existing `recal_interval` / `recal_sets` output exactly; recalibration with window covering the whole calibration fold and no refresh reproduces static.
- **T12**: L = ∞ leaves the ECI radius constant across the test window.
- **T13**: with `proportional_lr=False` and lag applied over the whole stream, the simulator reproduces the vendored `ECI(..., ahead=L+1)` exactly (validates the delayed-feedback timing against the library's own).

## Inference

No patient-cluster bootstrap (invalid for online/recalibrating methods). Reported: 5-seed SD; descriptive coverage across K = 10 contiguous stream blocks and the 4 calendar
years. Stream blocks are not calendar periods; block spreads are descriptive, not a significance test.

## Commitments

- The result is reported whatever it is. Grid, realistic band, windows, definitions and the decision rule above are not adjusted after seeing results.
- Limitations that bound every conclusion: uniform lag; arrival-count lag whose hour conversion is approximate; one cohort; two recalibration windows; a single recalibration cadence (R = 2,000).
- Any deviation from this document during implementation is logged in `M2_Hybrid_Deviation_Log.md` with its reason.

---

## RESULT (recorded 2026-09-21). Overall reading: **THE NEGATIVE RESULT IS ROBUST TO REALISTIC LAG.**

Pre-registration file written 05:37:22; `m2_label_lag.py` first run after 05:41 (its prediction cache is stamped 05:41:08). Correctness tests T8–T13 passed as hard asserts before any lagged number was printed
(the first two launches stopped at T9 and T11 and the third on a numerical overflow in the simulator, all before any result table existed; see Deviation 21). Full numbers: dossier §24.3d, `data/m2_v2_label_lag.csv`,
`data/m2_v2_label_lag_verdict.json`, `paper_assets/tables/label_lag.tex`.

Per-lag verdicts (same on all three models in the decision band): **marginal — ECI NOT NEEDED at L = 60, 240, 1,000** (and at 0, 4,000, 16,000; both invalid at ∞); **class-conditional — Mondrian-ECI NOT NEEDED at L = 60, 240, 1,000**
(and at 0), **Mondrian-ECI INVALID with recalibration valid at L = 4,000 and 16,000**, both invalid at ∞. No realistic-band lag yields an ECI-supported outcome on the primary model or on any replication model.

Sensitivity arms (not decision-bearing): ECI with lr fixed at the lag-0 value winds up already at L = 1,000 (0.9689 / 0.864); the vendored `ahead = L + 1` cross-check agrees with the simulator for L ≤ 1,000 (0.8997 / 0.521 at L = 1,000) and winds up at L = 4,000.

Commitments honoured: grid, realistic band, windows, definitions and decision rule unchanged after seeing results. Implementation clarifications made during the run (none affecting a decision) are logged as Deviation 21.
