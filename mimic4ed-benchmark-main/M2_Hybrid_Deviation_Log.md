# M2 Hybrid GRU-D — Deviation Log

Per `M2_Hybrid_GRU_D_Student_Execution_Guide_v3_8_EXECUTION_READY.docx` §18: "any deviation from this protocol must be recorded with timestamp, reason, supervisor approval status, and impact assessment." Entries are append-only, in execution order.

---

## Deviation 1 — Development-pool split proportions rescaled to sum to 100%

- **Timestamp:** 2026-08-29, Phase G (fixed 4-way dev-pool split infrastructure).
- **What the guide specifies:** §5.1's table gives Training≈70%, Validation≈10%, Calibration-selection≈5%, Final-conformal-calibration≈5% "of dev [pool]." §2.4 describes this as the development pool being "partitioned into four disjoint subsets."
- **The deviation:** these four percentages sum to 90%, not 100%. A "partition," by definition, must be exhaustive as well as disjoint — as literally stated, 10 percentage points of the development pool are not assigned to any subset.
- **Reason / resolution:** proportionally rescaled all four percentages by dividing by 0.90, preserving the guide's stated 70:10:5:5 relative ratio while covering the development pool exactly: Training≈77.78%, Validation≈11.11%, Calibration-selection≈5.56%, Final-conformal-calibration≈5.56%. Implemented in `Benchmark_scripts/m2_hybrid_common.py`'s `build_fixed_dev_split()` (`DEV_SPLIT_FRACTIONS = [70/90, 10/90, 5/90, 5/90]`). Verified disjoint and exhaustive by direct set comparison against `data/train.csv`'s full subject_id set (125,199 + 17,886 + 8,943 + 8,943 = 160,971, zero overlap) before being persisted.
- **Supervisor approval status:** not yet explicitly confirmed — this is a mechanical rounding resolution rather than a scientific-validity question, made without pausing execution (unlike the four static/architecture questions in `M2_Hybrid_Guide_Clarification_Request.md`, which were paused pending explicit sign-off). Flagged here for visibility and available for the supervisor to override if a different resolution is preferred.
- **Impact assessment:** none of the four proportions was intended to carry independent scientific meaning beyond "a big training pool and small held-out subsets for selection/calibration" — the exact 70 vs. 77.78 split does not change which patients could ever land in a subset, only the relative subset sizes by a few percentage points. No result computed elsewhere depends on knowing the precise rescaled percentages; the persisted subject-ID sets and their SHA-256 hashes (`data/m2_hybrid_dev_pool_split.json`) are the actual source of truth every script reads from.

## Deviation 2 — DeLong test omitted

- **Timestamp:** to be finalized once Phase M's statistical comparison is implemented; documented here in advance since the guide's own text anticipates this exact fallback.
- **What the guide specifies:** §"Statistical Comparison — DeLong (Secondary/Optional)": "report DeLong only if the implementation appropriately accounts for patient-level clustering. Otherwise, omit DeLong and rely on the patient-cluster bootstrap as the primary inferential procedure."
- **The deviation:** no validated, peer-reviewed Python implementation of a patient-cluster-adjusted DeLong test was located. Standard DeLong implementations (including any quick from-scratch reimplementation) assume i.i.d. observations, which repeat-encounter patients (34% of this cohort) violate.
- **Reason / resolution:** DeLong is omitted entirely, per the guide's own explicitly sanctioned fallback. Hand-rolling a nonstandard modification to a well-established statistical test — with no external validation of the modification — would itself be a rigor risk this project's standards are meant to avoid. The patient-cluster bootstrap (Phase M) is the sole inferential procedure for all reported deltas.
- **Supervisor approval status:** pre-approved by the guide's own text (this is the guide's own named fallback path, not a unilateral deviation from it).
- **Impact assessment:** none — the guide already treats DeLong as secondary/optional and names the bootstrap as primary regardless of whether DeLong is reported.

## Deviation 3 — Temporal-Only GRU-D trained at full-batch, not batch_size=128

- **Timestamp:** 2026-08-29, Phase I (Temporal-Only GRU-D, Stage 1).
- **What the guide specifies:** §6.1's fixed configuration table (explicitly "no hyperparameter grid... fixed a priori"): hidden=32, dropout=0.2, weight_decay=1e-4, **lr=0.01, batch_size=128**, stating "The learning rate 0.01 and batch size 128 are documented codebase defaults... these are the reproducible configurations from the existing codebase."
- **The deviation:** this specific claim does not hold up under direct testing. Isolated via a 3-way controlled comparison (all at seed 42, identical data/architecture):
  - `lr=0.01, batch_size=128`: 200 epochs, validation loss flat at ≈0.221 (≈ the entropy of predicting the constant ~6% base rate) and validation AUROC pinned at ≈0.50 (chance) for the entire run — no learning occurs.
  - `lr=0.01, batch_size=None` (full-batch, matching how lr=0.01 is actually validated elsewhere in this codebase, e.g. `m2_ablations.py`'s `SKLearnGRUD`): clean, monotonic learning — validation loss 0.628→0.188, AUROC 0.60→0.76 over 100 epochs, still improving at epoch 100 (not yet plateaued).
  - `lr∈{0.001, 0.0001}, batch_size=128`: also learns cleanly, reaching AUROC ≈0.78–0.80 within just 8 epochs.

  This isolates the problem precisely: lr=0.01 is a real, validated default, but only for full-batch training (1 gradient step/epoch); batch_size=128 is a value with no precedent anywhere in this codebase, and pairing it with the full-batch-tuned lr produces a model that never learns. The guide's framing of "0.01 and 128 together" as a single "documented, reproducible" pair does not match what's actually documented or reproducible.
- **Reason / resolution:** presented to the user with the full 3-way evidence above (not decided unilaterally, since this deviates from an explicitly "fixed a priori, no grid" table). Resolved to keep **lr=0.01 exactly as specified** and revert to **full-batch training** (`batch_size=None`) — the smaller deviation, since lr=0.01 is the part of the guide's claim that is genuinely validated; batch_size=128 is the part that isn't. Implemented via native `batch_size=None` support added to `m2_hybrid_common.py`'s shared training loop (`_minibatch_early_stop_loop`), matching `SKLearnGRUD`'s own existing full-batch convention rather than a special-cased workaround.
- **Supervisor approval status:** not yet sought — resolved with the user directly given the narrow, empirical (not scientific-framing) nature of the question, following the same escalate-rather-than-guess posture established for the four v3.2 items, but handled as a direct decision this time rather than a written round-trip to the supervisor. Available to revisit if the supervisor prefers a different resolution.
- **Impact assessment:** Temporal-Only GRU-D (and, downstream, Frozen-Temporal Head Control and Hybrid GRU-D, which reuse its frozen encoder) trains at full-batch instead of mini-batches. The epoch budget needed for full-batch convergence is being separately, empirically verified (see M2_Findings_Dossier.md's Temporal-Only GRU-D section) before any 5-seed result is trusted, per this project's standing practice after the Static-Only MLP epoch-budget bug.

### RETRACTED, 2026-09-01 — this deviation was an artifact of corrupted inputs, not of the guide's configuration

Deviation 7 (below) found that the vitals reaching the model at the time of the 3-way test above were in Fahrenheit, had bypassed outlier removal entirely (temperature max 986, SpO2 max 9322, SBP max 151103), and were fed **unscaled** into tanh/sigmoid gates. At those magnitudes the gates saturate, which is a complete and sufficient explanation for "lr=0.01 + batch_size=128 never learns": the evidence recorded above was real, but it measured the input corruption, not the learning rate/batch size pairing.

Re-tested at seed 42 against corrected inputs (`data/m2_temporal_only_grud_convergence_diagnostic_dev7_seed42.json`): `lr=0.01, batch_size=128, patience=10` trains cleanly, early stopping fires normally at epoch 31, best val_loss 0.1510 at epoch 20, val AUROC 0.8736. The guide's Sec 6.1 configuration is followed **exactly** again, with no deviation. `m2_temporal_only_grud.py` was reverted accordingly.

A full-batch arm was measured in the same diagnostic (val AUROC 0.8840, best epoch 582 of 600) and deliberately **not** adopted: preferring it would mean retaining two protocol deviations in order to gain validation AUROC, which is exactly the post-hoc configuration selection this project's anti-p-hacking design exists to prevent.

## Deviation 4 — Observation-density thresholds use natural breaks, not literal 25th/75th percentiles; and a pipeline-wide finding about GRU-D's masking mechanism

- **Timestamp:** 2026-08-29, Phase M (subgroup analysis, `m2_hybrid_subgroups.py`).
- **What the guide specifies:** §11's "Observation Density Definition (Prespecified)": Sparse = <=25th percentile of pre-cutoff vital measurement count per patient; Moderate = >25th-75th; Dense = >75th percentile — estimated from the development population and frozen before test evaluation.
- **The deviation, and a significant underlying finding:** computing this from `train.csv`/`test.csv` directly gives a degenerate result — the 25th and 75th percentiles are BOTH exactly 14 (the maximum possible count), because **every single row in `train.csv` (334,693/334,693) has zero missing vitals**. Root cause traced precisely: `Benchmark_scripts/data_general_processing.py` lines 71-76 applies `SimpleImputer(strategy="median")` to all 14 vitals columns (fit on train, applied to test) *before* `train.csv`/`test.csv` are ever written — upstream of every GRU-D model this entire M2 module has ever trained, not something introduced by this Hybrid experiment.

  **This is a genuine, pipeline-wide, previously-undocumented limitation worth surfacing prominently**: GRU-D's mask `M` (used in the decay formula `xc = m*x + (1-m)*(gamma*x)`) has therefore always been all-ones in every M2 GRU-D experiment to date — the decay-based imputation branch (`(1-m)*(gamma*x)`) has never actually been exercised, since `m=1` everywhere. GRU-D's stated missingness-handling capability, repeatedly cited as its core architectural rationale (including this guide's own Primary Subgroup Hypothesis: *"mechanistically motivated because GRU-D's core design feature is explicitly modeling missingness through masks and time intervals"*), is architecturally present in the code but has never been given real missing data to act on anywhere in this project. Flagged prominently in `M2_Findings_Dossier.md` beyond just this subgroup analysis, since it bears on how GRU-D's contribution should be framed generally.

  Genuine missingness *does* exist earlier in the pipeline: `data/master_dataset.csv` (pre-imputation) shows real per-vital missingness of ~4-12%, and combined across all 14 columns, ~80% of encounters are fully observed while ~20% have at least one missing value — a real but heavily right-skewed distribution, not a smooth spread. This means the literal 25th/75th-percentile scheme still degenerates even using the recovered, genuine missingness signal (25th percentile still lands on the ceiling value), though the underlying data-generating variation is real.
- **Reason / resolution:** recovered true pre-imputation missingness from `master_dataset.csv` (joined on `stay_id`, verified as a complete, unique key for every train/test row) rather than declaring the subgroup analysis a dead end when a clean, non-arbitrary source of the intended signal exists. Replaced the degenerate percentile scheme with natural breaks matching the data's own visible structure: Dense = fully observed (count=14); Moderate = 8-13 observed; Sparse = <=7 observed (half or more missing). The formal interaction test compares the two poles (Sparse vs. Dense), excluding Moderate — directly analogous to how this same guide's own "Outcome risk group" subgroup (NEWS2<3 vs. NEWS2>=7, excluding the 3-6 middle band) already compares extremes with an excluded middle, not a novel pattern introduced here.
- **Supervisor approval status:** not yet sought. This combines a mechanical necessity (the literal scheme cannot be computed at all from `train.csv`/`test.csv` alone) with a genuine, non-arbitrary fix (recovering the real signal from an earlier pipeline stage) — resolved directly rather than escalated, since there was no comparably defensible alternative once the literal scheme was confirmed inapplicable. The broader finding (GRU-D's mask has been functionally inert project-wide) is surfaced prominently for the user's and supervisor's awareness regardless, since it may affect how GRU-D's contribution is framed in the eventual manuscript.
- **Impact assessment:** the Sparse subgroup (~20% of encounters) is smaller than a literal-25th-percentile Sparse group would have been, but is well-powered for the bootstrap (thousands of encounters). The broader masking-mechanism finding does not invalidate any already-reported AUROC/calibration/conformal number (those reflect what GRU-D actually does with 2 fully-observed timesteps + elapsed time, which is a real and valid, if narrower than advertised, capability) — it affects only how that capability should be *described* in any future write-up.

### Follow-up, 2026-08-29: escalated from "documented limitation" to "root-caused and fixed, models rebuilt"

Per explicit user decision: *"we should rebuild and retrain again since that finding was a huge bug and we have to make a Q1 paper work and this is unacceptable to make it as a limitation."* Agreed assessment: the guide's own Primary hypothesis (§0) is directly about whether temporal/missingness-aware modeling beats tabular aggregation under matched information — running that comparison on data with no real missingness available to either model doesn't test a limited version of the hypothesis, it fails to test it at all, and a Q1 reviewer would be expected to raise exactly this question.

**Scope of the fix** (`Benchmark_scripts/m2_hybrid_common.py`'s `prepare_temporal_tensors()` and `m2_xgboost_matched.py`'s `build_matched_features()`): both now source vitals from a cached, `stay_id`-indexed lookup over `master_dataset.csv`'s genuine pre-imputation values, instead of from `train.csv`/`test.csv`'s already-imputed columns. Verified immediately after the fix: `prepare_temporal_tensors` now produces real mask variance (79.71% fully observed on the test set, exactly matching the subgroup analysis's earlier finding); `build_matched_features`'s `_count` columns now range 0–2 with real variance (previously exactly 2 for every row, confirmed before fixing it — the completeness-count feature had been completely vacuous, not just GRU-D's mask).

**Scope boundary preserved**: `train.csv`/`test.csv` themselves are not modified or re-derived (same frozen cohort/split); the legacy Track A numbers (0.9009, NEWS2, LR, RF, XGBoost-Rich) are not reopened; Static-Only MLP is not retrained (it never touches vitals). Only XGBoost-Matched, Temporal-Only GRU-D, Frozen-Temporal Head Control, Hybrid GRU-D, and everything computed downstream of them (conformal analysis, the 5 bootstrap deltas, subgroup analysis) are rebuilt.

**Supervisor approval status:** not yet sought for the fix itself (an explicit user decision, made with the reasoning above); the underlying finding and this rebuild are worth surfacing to the supervisor directly at the user's discretion, given they may materially change how GRU-D's contribution is framed in the eventual manuscript.

**Impact assessment:** superseded pre-fix numbers are retained in `M2_Findings_Dossier.md`, explicitly labeled, never silently overwritten — see the Dossier's updated §13–20 for the full old-vs-new comparison.

## Deviation 5 — Rebuild convergence: patience=10 stops too early on the real-missingness data; effectively disabled instead

- **Timestamp:** 2026-08-29, Phase O rebuild (`m2_temporal_only_grud.py`).
- **What the guide specifies:** §6.1's fixed configuration table (same "fixed a priori, no grid" table as the lr/batch_size deviation, Deviation 3): "Early stopping | Validation loss, patience = 10 epochs."
- **The deviation:** with real missingness now reaching the model (Deviation 4's fix), training dynamics changed materially — a fresh convergence diagnostic (required regardless, per this project's standing practice of not reusing a previous epoch-budget decision across a data change) found `patience=10` stops training at epoch ~126 (AUROC ~0.82), while genuine, if noisy, improvement continues to a true optimum around epoch 1611 (val_loss 0.1618, AUROC 0.8457) — a real ~0.02 AUROC gap. Tested `patience=100` directly too: still insufficient, since a legitimate ~200-epoch-wide noisy dip occurs along the way (val_loss temporarily worsening from ~0.169 at epoch 100 to ~0.195 at epoch 200 before recovering to ~0.166 by epoch 300). Beyond ~epoch 2100, training becomes genuinely unstable (AUROC crashing to 0.54–0.61 at times in an extended 2500-epoch diagnostic run) — so uncapped patience is not a safe answer either.
- **Reason / resolution:** presented to the user with the full evidence above (not decided unilaterally, same posture as Deviation 3). Resolved to effectively disable patience-based early stopping (`PATIENCE = MAX_EPOCHS = 2000`, so it never fires) and instead run a fixed, generous epoch budget — capped comfortably below the observed instability zone — always keeping whichever epoch's checkpoint had the single best validation loss (the training loop already tracks this continuously, independent of the patience mechanism). Verified directly: this configuration lands exactly on the epoch-1611 optimum found by the diagnostic.
- **Supervisor approval status:** not yet sought — resolved with the user directly, same posture as Deviation 3.
- **Impact assessment:** Temporal-Only GRU-D (and, downstream, Frozen-Temporal Head Control and Hybrid GRU-D, which reuse its frozen encoder) trains for up to 2000 full-batch epochs per seed (~150s each, ~13 minutes for all 5 seeds) rather than stopping at ~126 — a real compute increase, still modest in absolute terms, in exchange for a materially better and more honest optimum.

### RETRACTED, 2026-09-01 — same root cause as Deviation 3

This deviation followed directly from Deviation 3's full-batch resolution: once training was full-batch on corrupted, unscaled inputs, convergence genuinely did take ~1600 epochs and genuinely did become unstable past ~2100. Both observations were correct about that regime, and both cease to apply once the regime is corrected.

With Deviation 7's fix in place, `patience=10` — Sec 6.1's literal value — fires normally and at the right time (epoch 31, with the best checkpoint at epoch 20, inside a 100-epoch budget). `MAX_EPOCHS = 2000 / PATIENCE = 2000` is reverted to the guide's `MAX_EPOCHS = 100 / PATIENCE = 10`.

## Deviation 6 — Implementation Fingerprint's git commit hash does not version this code

- **Timestamp:** ongoing, first recorded 2026-08-29 (Phase G).
- **What the guide specifies:** §18 "Implementation Fingerprint (Mandatory)" requires a git commit hash per saved manifest.
- **The deviation:** `module 2`'s `.gitignore` excludes `Benchmark_scripts/`, `data/`, and every document this experiment produces or reads — the repository's git history (3 old commits, unrelated to this work) does not track any of the code or data this experiment depends on.
- **Reason / resolution:** `save_manifest()` records whatever `git rev-parse HEAD` currently returns for `module 2`, honestly annotated with a note (`git_note` field) that this hash does not meaningfully version the run. No commit is created as part of this experiment — per standing project practice, commits happen only when explicitly requested, and the user was previously made aware (this session) that `module 2`'s git repository exists but is local-only and not actively used to track this work.
- **Supervisor approval status:** not sought — this is a disclosed limitation of the current repository setup, not a scientific or methodological choice.
- **Impact assessment:** the manifest's other fields (package versions, seed, full config, subject-ID-set SHA-256 hashes) remain fully meaningful and sufficient to verify a given run's inputs were reproduced exactly; only the git-hash field itself is not informative under the current (untracked) repository state.

## Deviation 7 — Vitals cleaning restored, train-only scaling added, and GRU-D's decay mechanism implemented as specified

- **Timestamp:** 2026-09-01, post-hoc pipeline audit (all three GRU-D-family models + XGBoost-Matched).
- **What the guide specifies:** Sec 2.1 requires "the existing GRU-D implementation"; the module's own architecture description (and `M2_Findings_Dossier.md` Sec 1.3) states the decay mechanism as GRU-D's defining feature and the mechanistic basis of the Sec 11 Primary Subgroup Hypothesis. Sec 5.5's leakage checklist requires "scalers fit on train only."
- **The deviation — three coupled defects, all confirmed by direct execution, not by inspection:**

  **(a) GRU-D's decay mechanism had never been active, in any M2 run.** The implementation was `gamma = exp(-relu(d @ W_gamma + b_gamma))`, `xc = m*x + (1-m)*(gamma*x)`. Since `prepare_temporal_tensors()` encodes missing entries as `x = 0`, the second term is `(1-m)*(gamma*0) = 0` for every missing entry, and the first is zeroed wherever `m = 0`. `gamma` was therefore multiplied by zero everywhere it appeared. Verified on all 5 saved seed checkpoints of the previous run: `W_gamma` absmax = 0.000e+00 and `b_gamma` absmax = 0.000e+00 — after 2000 epochs per seed, the decay parameters had never received a single non-zero gradient. Verified independently in eval mode: `|out(D) - out(1000*D)| = 0.0` exactly. Two consequences: the decay/imputation branch that motivates choosing GRU-D over a plain GRU had never run, and **Delta-t never reached the model at all**, since `D` enters only through `gamma`. What was actually being trained was a GRU with mask concatenation and zero-imputation.

  **(b) Deviation 4's fix silently reverted the vitals cleaning.** Moving the vitals source from `train.csv`/`test.csv` to `master_dataset.csv` correctly recovered pre-imputation missingness, but `master_dataset.csv` is the stage *before* `convert_temp_to_celcius()` and *before* `remove_outliers()` (`data_general_processing.py` lines 45-46). Measured on the frozen test cohort: `triage_temperature` arrived in Fahrenheit (mean 98.0, max **986.0**), `triage_o2sat` reached **9322**, `triage_sbp` reached **151103**, against Celsius/valid-range values of 36.7/100/312 in `test.csv`. Confirmed at the model input: `X` min/max/mean = 0.00 / **9102.00** / 69.59.

  **(c) No scaling anywhere in the temporal path, and inconsistent scaling of the static features.** Raw vitals went straight into tanh/sigmoid gates. Separately, `m2_static_only_mlp.py` standardises the 4 Guide static features while `m2_hybrid_experiment.py` passed the same 4 columns raw into the Hybrid's static MLP branch — so `Delta_static_contribution` and `Delta_static_only`, two of the five headline deltas, compared two models that differed in preprocessing as well as in architecture.

  **Why (b)+(c) matter for the headline conclusion specifically:** XGBoost-Matched reads the same corrupted source, but trees are invariant to monotone rescaling and robust to extreme values, whereas the GRU-D family is neither. The primary comparison `Delta_representation = Hybrid - XGBoost-Matched`, and therefore Guide Sec 14's Scenario C classification, was confounded by a preprocessing defect that penalised only the neural arm.

- **Reason / resolution:**
  - `_clean_vitals_frame()` applies exactly `data_general_processing.py`'s own cleaning (Celsius conversion, then two-tier outlier removal) to the master lookup, **keeping NaNs** rather than median-imputing them — restoring parity with `train.csv`/`test.csv` without giving up Deviation 4's recovered missingness. `_assert_valid_range_matches_pipeline()` parses the threshold table out of `data_general_processing.py` at load time and fails loudly if the two copies ever drift.
  - `get_vitals_scaler()` / `get_static_scaler()` fit per-vital and per-static-feature statistics on the fixed dev-pool `training` subject subset **only**. This is leak-free and seed-independent by construction, because that partition is drawn once at `random_state=42` and persisted. Missing entries are encoded as 0.0 *after* scaling, which in scaled space is exactly the training empirical mean — i.e. GRU-D's own `x_tilde` term, not an arbitrary sentinel. Persisted to `data/m2_vitals_scaler.json` and `data/m2_static_scaler.json`.
  - Delta-t is expressed in units of the training-set median ED length of stay, and the 6 negative `ed_los_hours` values in the cohort are clipped to 0 (a negative elapsed time is not a meaningful Delta-t).
  - `GRUDCell` implements Che, Purushotham, Cho, Sontag & Liu, "Recurrent Neural Networks for Multivariate Time Series with Missing Values," *Scientific Reports* 8:6085, 2018, Eqs. 10-13: a **diagonal** input-decay `gamma_x` (the previous code used a full matrix, a second departure from the paper), a full hidden-state decay `gamma_h`, last-observation carry-forward, and interpolation toward the empirical mean. `GRUDWithEncoder` and `TemporalEncoder` now share one `GRUDCell` instance rather than two hand-copied recurrence blocks that could drift.
  - **Initialisation is part of the fix, not incidental:** PyTorch defines `relu'(0) = 0`, so zero-initialising the decay parameters would make `z = w*delta + b` identically zero, pass zero gradient, and freeze them at zero forever — reproducing the original bug by a different route. `gamma_x_w` is drawn `N(0, 0.1)` and `gamma_h` uses `nn.Linear`'s default init; biases start at 0 so that at triage (`delta = 0`) `gamma = exp(-relu(0)) = 1`, i.e. no decay when no time has elapsed, as the paper specifies.
  - `load_frozen_encoder()` now asserts that no expected `cell.*` key was missing. `strict=False` is still required (the checkpoint carries the discarded `clf.*` head) but on its own it would silently tolerate a checkpoint whose encoder keys do not match at all, leaving a randomly-initialised "frozen encoder" in place and invalidating every downstream result without any error.
  - Verified post-fix: decay parameters receive non-zero gradients (`gamma_x_w` 1.56e-04, `gamma_h.weight` 6.71e-04) and Delta-t now measurably changes the output (`|out(D) - out(5D)| = 0.0248` in eval mode, versus exactly 0.0 before). Observed vitals now have mean -0.003 / SD 1.015; the mask is unchanged at 95.47% observed.
- **Supervisor approval status:** not yet sought. This is a defect repair — restoring behaviour the guide, the dossier and the module's own architecture description already claimed — rather than a change of scientific design, so it was executed directly. It nevertheless materially changes what the module is and should be surfaced to the supervisor: the previously-reported GRU-D results did not involve a working decay mechanism, and Deviations 3 and 5 are retracted as consequences of the input corruption rather than genuine properties of the guide's locked configuration.
- **Impact assessment:** all pre-fix numbers for Temporal-Only GRU-D, Frozen-Temporal Head Control, Hybrid GRU-D and XGBoost-Matched are superseded, along with everything computed downstream of them (conformal analysis, the 5 bootstrap deltas, the subgroup analysis). Static-Only MLP is unaffected and is not retrained (it never touches vitals, and its static inputs were already standardised on the same subset). The pre-fix checkpoints, per-seed results, predictions and summary tables are preserved in full under `data/archive_pre_s1s2_fix/`, never overwritten. `Delta_representation` may change magnitude or sign; whichever way it moves will be reported as it comes out, per Guide Sec 19's Final Rule.


## Deviation 8 — The tuning search's "baseline" arm had silently become the previous search's winner

- **Timestamp:** 2026-09-01, Track A re-tuning under Deviation 7's corrected architecture (`m2_grud_tuning.py`).
- **What the guide/project convention specifies:** `m2_grud_tuning.py` is a staged, validation-only search in which each stage compares candidate configurations against a *baseline* arm, adopting a candidate only if it beats the baseline mean by more than one within-baseline standard deviation (an explicit noise guard against GRU-D's seed-to-seed variance). The recommended config is then rolled out to `SKLearnGRUD`'s constructor defaults in `m2_ablations.py`, which is the documented mechanism by which a tuning decision reaches the primary pipeline.

- **The deviation:** those two mechanisms silently compose into a trap. `evaluate_candidate()` built each model as `SKLearnGRUD(seed=seed, **kwargs)`, passing **only that stage's overrides**; every unspecified hyper-parameter fell through to the constructor defaults. Because the *first* tuning pass's winner (`hid=32, dropout=0.2, weight_decay=1e-4`) had already been rolled into those defaults, the unspecified-kwargs "baseline" arm of any subsequent run **was** that winner. A search designed to measure candidates against a neutral reference was instead measuring them against its own previous conclusion.

  This is not an inference from reading the code — the re-run's trace shows it directly, as bit-identical numbers:

  - **Stage 3 measured nothing.** "no reg (baseline)" already carried `dropout=0.2` **and** `weight_decay=1e-4`, so it, the `dropout=0.2` candidate, and the `weight_decay=1e-4` candidate were the same configuration evaluated three times — all three returned exactly 0.9238 ± 0.0014 (dropout=0.2: 0.9241 ± 0.0017; weight_decay=1e-4: 0.9240 ± 0.0014), to four decimals, across three seeds each.
  - **Stage 5 measured nothing.** Its pre-specified follow-up, "combine hid=32 + dropout=0.2 + wd=1e-4", was compared against a `cfg={}` reference that already *was* hid=32 + dropout=0.2 + wd=1e-4. Both arms returned identical numbers.
  - Stages 1 and 2 remained valid, since `pos_weight` and `hid` were passed explicitly in every arm of those stages.

- **Reason / resolution:** `m2_grud_tuning.py` now defines an explicit `UNTUNED_BASE` (`hid=16, clf_hidden=32, dropout=0.0, weight_decay=0.0, pos_weight=None, batch_size=None, epochs=150, lr=0.01, early_stopping_patience=None`) and builds every candidate as `{**UNTUNED_BASE, **best_cfg, **overrides}`, so each stage measures the factor it names regardless of what the module defaults are later set to. The saved trace now records the **full effective config** per row rather than only that stage's overrides — without which the same ambiguity simply reappears in the artefact and cannot be diagnosed after the fact. The pre-fix script is preserved as `data/archive_pre_dev7_trackA/m2_grud_tuning.py.prefix`.

- **Results of the corrected re-search** (validation-only; `test.csv` never opened, asserted at startup; full trace `data/m2_grud_tuning_trace.csv`):

  | Stage | Baseline | Best candidate | Outcome |
  |---|---|---|---|
  | 1 — class imbalance | unweighted 0.9133 ± 0.0053 | pos_weight=15.75 0.9230 ± 0.0009 | **adopted** |
  | 2 — capacity | hid=16 0.9230 ± 0.0009 | hid=64 0.9238 ± 0.0014 (hid=32 0.9234) | **adopted, marginally** |
  | 3 — regularisation | 0.9238 ± 0.0014 | dropout=0.2 0.9241 | baseline retained |
  | 4 — training procedure | 0.9238 ± 0.0014 | batch_size=2048 0.8831 | baseline retained |

  Two findings are worth flagging as *reversals of what the contaminated search reported*. First, `pos_weight=15.75` now clears the noise guard comfortably (0.9230 vs. 0.9133), where the earlier trace had concluded it "did not clear the noise-guard bar" — that conclusion was reached against a baseline that was not the baseline. Second, `batch_size=2048` is now clearly harmful (0.8831 vs. 0.9238, a gap of 0.0408), where the contaminated run had it at a near-competitive 0.9182. The `hid=64` adoption clears the guard only marginally (0.9238 against hid=16's 0.9230, with a baseline SD of 0.0009) and should be read as a weak preference, not a finding.

- **Supervisor approval status:** not yet sought. This is a defect repair to a search harness, not a change of scientific design, so it was executed directly — but the *config rollout that follows from it* is a different matter and is flagged for sign-off: the previous config was adopted "per explicit sign-off," and its replacement (`pos_weight=15.75, hid=64`) has not been. See the calibration caveat below before that sign-off is given.

- **Impact assessment:** every Track A GRU-D number is affected, but they were already superseded by Deviation 7 independently of this, so no otherwise-valid result is invalidated by this deviation alone. The pre-fix Track A artefacts are preserved under `data/archive_pre_dev7_trackA/`. The generalisable lesson, worth stating because the same pattern can recur anywhere in this repository: **a tuning harness must never rely on mutable module defaults for its own control arm.** Rolling a winner into the defaults is a legitimate convention; combining it with implicit-baseline construction is what makes the search self-confirming, and the failure is silent — it produces plausible numbers, not an error.

- **The AUROC-only selection criterion, and why `pos_weight` was NOT adopted:** the search selects on validation **AUROC**, which is invariant to any monotone rescaling of the scores and therefore structurally blind to decalibration. That is not a hypothetical concern here. The first 5-seed confirmatory run at the recommended config (`pos_weight=15.75, hid=64`) showed the signature immediately: AUROC 0.9009 → 0.9261 and AUPRC 0.4506 → 0.5640, alongside a conformal interval width of 0.2824 → 0.7692 at unchanged nominal coverage.

  Rather than assume which of the two simultaneously-changed factors caused it, `m2_posweight_calibration_check.py` isolates them — 4 arms × 5 seeds on the fixed Gate split, identical conformal code, every configuration passed **explicitly** rather than inherited from constructor defaults (the lesson of this very deviation applied to its own follow-up):

  | Arm | AUROC | AUPRC | Brier | ECE | Mean pred. | ECI cov. | ECI width |
  |---|---|---|---|---|---|---|---|
| hid=16, no pos_weight | 0.9151 | 0.5257 | 0.0381 | 0.0017 | 0.0597 | 0.9017 | 0.2619 |
| **hid=64, no pos_weight** (adopted) | 0.9221 | 0.5606 | 0.0365 | 0.0020 | 0.0597 | 0.9002 | 0.2432 |
| hid=16, pos_weight=15.75 | 0.9251 | 0.5568 | 0.1103 | 0.1863 | 0.2451 | 0.8990 | 0.7754 |
| hid=64, pos_weight=15.75 | 0.9259 | 0.5635 | 0.1051 | 0.1781 | 0.2368 | 0.8988 | 0.7584 |

  Test-set positive rate, for reference: **0.0588**.

  The attribution is unambiguous. **`pos_weight` is the sole cause**: it inflates ECE by roughly 109× (0.0017 → 0.1863 at hid=16), widens the ECI interval 3.0× (0.2619 → 0.7754), and drives the mean predicted probability to 0.2368 against a true base rate of 0.0588 — a risk score asserting roughly 4× the real event rate. It buys only +0.0038 AUROC over the adopted arm. **`hid=64` is a pure improvement**: against hid=16 it raises AUROC (0.9151 → 0.9221) and AUPRC (0.5257 → 0.5606), lowers Brier (0.0381 → 0.0365), and *narrows* the interval (0.2619 → 0.2432) at equal calibration.

  **Adopted Track A config: `hid=64`, no `pos_weight`.** M2's Track A primary contribution is the conformal interval; trading a 3.0× width increase for +0.004 AUROC would sacrifice the deliverable to improve a secondary metric.

  **Coverage cannot be used to detect this, and that is the generalisable point.** Empirical coverage sits at nominal in *every* arm above, including the two that are severely decalibrated — because split-conformal and ECI both hold marginal coverage essentially by construction, and they hold it by paying in **width**. Any future report in this repository that offers "coverage is at 90%" as evidence that a model's uncertainty quantification is healthy is offering no evidence at all; width and calibration must be reported alongside it.

- **Recommended follow-up (not yet actioned):** `m2_grud_tuning.py`'s `pick_winner()` should carry a calibration or interval-width guard alongside its AUROC noise guard, so a future search cannot again recommend a config that improves ranking while degrading the module's actual deliverable. Left as an open item rather than changed unilaterally, since it alters the search's selection semantics.


## Deviation 9 — The Track B ONNX export path never received Deviation 7's fix

- **Timestamp:** 2026-09-01, discovered while updating `M2_Findings_Dossier.md` §6 after the Track A
  re-run.
- **What the module claims:** §6 of the dossier and the repository-root `deployment_config.md`
  describe `Benchmark_scripts/grud_risk.onnx` as the Track B deployment artifact — "exported from the
  seed-42, tuned-config GRU-D model." CLAUDE.md §2 (M2, Track B) requires a real deployment engine,
  and §1.7 forbids from-scratch reimplementations where a shared/vendored implementation exists.

- **The deviation:** `Benchmark_scripts/train_grud_conformal.py`, the script that trains and exports
  that artifact, carries its **own third hand-copied GRU-D recurrence** and its **own `prepare_data`**,
  neither of which Deviation 7's repair reached. Deviation 7 fixed `m2_hybrid_common.py` (the v3.8
  pipeline) and, subsequently, `m2_ablations.py` (Track A); this file was missed because it
  duplicates rather than imports both.

  Verified by direct execution on 3,000 rows of the frozen test cohort, not by inspection — all four
  symptoms of Deviation 7 are present:

  | Check | Result | Meaning |
  |---|---|---|
  | Mask fraction observed | **1.0000** | reads the already-median-imputed `test.csv`, so `M` is all-ones and carries no information |
  | `X` min / max / mean | 0.00 / **269.00** / 63.39 | raw, unscaled vitals fed into tanh/sigmoid gates |
  | grad `W_gamma` absmax | **0.0** | decay parameters receive exactly zero gradient |
  | grad `b_gamma` absmax | **0.0** | ″ |
  | \|out(D) − out(1000·D)\| | **0.0** | Delta-t has no effect on the output at all |

  The mechanism is identical to Deviation 7(a): `x_c = m*x + (1-m)*(gamma*x)` with missing entries
  encoded as `x = 0` multiplies `gamma` by zero everywhere it appears. Note this path *does* get
  cleaned vitals (it reads `test.csv`, which is post-`remove_outliers`, hence max 269 rather than the
  master file's 151,103) but gets **no scaling and no real missingness**, and has **no working decay
  mechanism**.

  **What this means for the artifact:** `grud_risk.onnx` was exported from a masked GRU with
  zero-imputation on unscaled inputs, in which the mask is constant and elapsed time is ignored. It
  is not a GRU-D model, and the dossier's §6 description of it was inaccurate.

- **Reason / resolution:** **fixed, 2026-09-01, same session.** `train_grud_conformal.py` was
  rewritten to import the shared `GRUDCell` (`m2_grud_cell.py`) and the corrected `prepare_data`
  from `m2_ablations.py` instead of its own private copies, then re-exported. The export now asserts
  ONNX-vs-PyTorch agreement and non-zero Delta-t sensitivity before writing the artifact, so a
  regression of this exact kind fails loudly rather than shipping silently. Verified post-fix:
  `data/m2_trackB_artifact.json` records AUROC 0.9216, Delta-t sensitivity **2.187784** (nonzero,
  vs. the pre-fix 0.0), ONNX-vs-PyTorch max abs diff **1.9073486328125e-06**, ONNX size 73.7 KB,
  latency 0.0757 ms/patient over 5,000 samples. `M2_Findings_Dossier.md` §7/§10 both record this as
  closed; this entry is updated to match rather than left to contradict them.

- **Supervisor approval status:** not sought for the mechanical repair itself (importing the already
  Track-A-approved shared `GRUDCell` into a third file that was supposed to use it all along is not a
  new scientific decision); the underlying Track A fix this repairs (Deviation 7) was itself flagged
  for the supervisor and remains open on that basis, tracked there, not duplicated here.

- **Impact assessment:** confined to Track B. No Track A or v3.8-pipeline number depends on
  `train_grud_conformal.py` — §§3–5 come from `m2_multiseed_run.py`/`m2_ablations.py` and §§13–20 from
  the `m2_hybrid_*` pipeline, both of which now share the single corrected `GRUDCell` in
  `m2_grud_cell.py`. The measured ONNX latency figure (0.0380 ms/patient) remains a valid *latency*
  measurement of a model of that shape, but the model's weights are not the reported Track A model's.
  **The general lesson matches Deviation 8's:** three hand-copied implementations of one recurrence
  existed in this repository, and fixing two of them left the third silently broken for weeks. Any
  remaining duplicate of a core mechanism should be collapsed onto the shared implementation rather
  than repaired in place.


## Deviation 10 — The vendored NEWS2 implementation scored only 4 of its 7 parameters

- **Timestamp:** 2026-09-01, while updating `M2_Findings_Dossier.md` §3 with the Research
  Synthesist agent's literature-verification findings (`M2_Related_Work_Notes.md` §5, item G4).
- **What the guide/literature specifies:** NEWS2 (Royal College of Physicians) scores 7 physiological
  parameters: respiratory rate, SpO2 (on one of two scales), temperature, systolic blood pressure,
  heart rate, level of consciousness (ACVPU), and supplemental-oxygen use. CLAUDE.md's Golden Rule
  requires every metric to come from an executed script on real data, which implicitly requires the
  score being executed to be the score being claimed.

- **The deviation:** `Benchmark_scripts/helpers.py`'s `add_score_NEWS2()` implemented only 4 of the 7
  parameters (respiratory rate, temperature, SBP, heart rate), leaving SpO2, supplemental-oxygen use
  and level of consciousness entirely absent. This capped the achievable score at 12 rather than
  NEWS2's true range (20, or higher with the scale-2 hypercapnic table), and was not previously
  disclosed as a limitation anywhere in the dossier.

  Found via a cheap, literature-motivated sensitivity check rather than by inspecting the code first:
  the Research Synthesist agent's literature review flagged that M2's NEWS2 AUROC (0.6227) sat well
  below the 0.71–0.86 range reported in published ED-validation studies, and recommended checking
  whether the module's composite `outcome_critical` target (rather than a single outcome like
  in-hospital mortality) explained the gap. It did not: NEWS2 against `outcome_inhospital_mortality`
  alone reproduced essentially the same AUROC (0.6293 vs. 0.6227), ruling out the composite outcome
  and pointing at the score's own computation instead. Inspecting `add_score_NEWS2()` at that point
  found the missing components directly.

- **Reason / resolution:** `triage_o2sat` is present in this extract (0% null, already cleaned by the
  same pipeline) and was simply unused by the scoring function. Restored the missing SpO2 sub-score
  using RCP NEWS2's Scale 1 (the standard scale, used absent a documented target range for
  hypercapnic respiratory failure — the convention published ED NEWS2 replications themselves use
  when supplemental-oxygen and consciousness fields are unavailable, which they are in this extract).
  Supplemental-oxygen use and level of consciousness are **not** added: both are genuinely absent from
  this extract, and adding a placeholder or assumed value for either would be inventing data. NEWS2
  computed by this pipeline remains an explicitly **partial** implementation — now missing 2 of 7
  components instead of 4 — and this is disclosed in the dossier (§3) rather than left implicit.

  Recomputed via `Benchmark_scripts/m2_recompute_news2.py`, which re-applies the corrected function to
  the already-frozen `train.csv`/`test.csv` and overwrites only the `score_NEWS2` column — it does
  **not** re-run the full extraction/imputation pipeline, so no other column and no split is touched.
  The pre-fix column is archived at `data/score_NEWS2_pre_deviation10_{train,test}.csv` before being
  overwritten, per this project's standing practice of never silently discarding a superseded value.

- **Result:** 0.6359 → recomputed at 0.6359 AUROC (+0.0132), AUPRC
  0.1445 → 0.1566, on 8.4% of test rows whose score changed (7,022 of 83,332). GRU-D's margin over
  NEWS2 narrows correspondingly from +0.2992 to +0.2860 AUROC — still a very large, clearly real gap,
  now measured against a materially more complete (if still partial) clinical comparator.

  **The residual gap to the published literature is not fully explained, and this is stated rather
  than papered over.** Restoring SpO2 closed part of the distance to the 0.71–0.86 range cited in ED
  NEWS2 validations, not all of it. The remainder plausibly reflects the still-missing
  consciousness/oxygen components, this cohort's specific composite outcome definition, and case-mix
  differences from the cited studies — but none of this has been independently verified, and no claim
  stronger than "the score is now more complete than before, and still partial" is supported by the
  evidence in hand.

- **Supervisor approval status:** not yet sought. This is a defect repair to a baseline computation
  using data already present in the extract — restoring a documented capability the score's own name
  already claims — rather than a change of scientific design, so it was executed directly, matching
  the posture taken for Deviations 7–9. It nevertheless changes a headline comparator's value and
  should be surfaced to the supervisor: NEWS2's margin against GRU-D narrowed, and the corrected score
  is still disclosed as partial.

- **Impact assessment:** confined to the NEWS2 baseline number in §3 of the dossier and this repository's
  PDF report; no GRU-D, XGBoost, or Hybrid-experiment number is affected (`score_NEWS2` is not an input
  to any of those models). The Hybrid pipeline's exploratory, descriptive-only "NEWS2 risk group"
  subgroup breakdown (§19.3) was computed before this fix and is flagged stale in the dossier rather
  than silently left inconsistent; it has not been re-run, since it is explicitly not the module's
  formal test (§19.2 is) and a full bootstrap re-run for a descriptive table was judged disproportionate
  to what it would change.


## Deviation 11 — The Track A config decision was made by reading `test.csv`

- **Timestamp:** 2026-09-01, found by an external Statistician-agent review of this module's
  finished dossier and scripts.
- **What the guide/project convention specifies:** `m2_grud_tuning.py` asserts at startup that it
  never opens `data/test.csv`, and §2's stated integrity property is "no config may be chosen by
  looking at the test set." That property is meant to hold for every config-adoption decision in
  Track A, not only for the staged search itself.

- **The deviation:** `m2_posweight_calibration_check.py` — the 4-arm study `M2_Findings_Dossier.md`
  §3.1 cites as the evidence for rejecting `pos_weight` and adopting `hid=64` — computes every
  metric (AUROC, ECE, Brier, mean predicted probability, MAPIE/ECI width) directly on `data/test.csv`.
  That is a config-selection decision informed by the held-out test set: exactly the leak the rest of
  Track A is architected to prevent. The bias this introduces is not uniformly conservative — on
  AUROC the test-set peek happened to *lower* the reported number relative to what a validation-only
  read would have shown, but on conformal interval width (the module's stated primary deliverable)
  the adoption decision was made by directly reading the test-set width.

- **Reason / resolution:** rather than assume the decision would survive a validation-only replication,
  it was run: `m2_posweight_calibration_check_valonly.py` repeats the identical 4-arm comparison using
  only the `X_fit_train`/`X_fit_val` fold `m2_grud_tuning.py` already builds (patient-level, disjoint
  from the MAPIE calibration fold, asserted every seed, 3 tuning seeds) — `test.csv` is never opened
  (asserted at startup, matching `m2_grud_tuning.py`'s own convention).

  **Validation-only result** (mean ± SD across 3 seeds; full detail
  `data/m2_posweight_calibration_check_valonly.csv`):

  | Arm | Val. AUROC | Val. ECE | Val. mean pred. |
  |---|---|---|---|
  | A: hid=16, no pos_weight | 0.9133 ± 0.0053 | 0.0038 | 0.0591 |
  | B: hid=64, no pos_weight (adopted) | 0.9204 ± 0.0006 | 0.0032 | 0.0600 |
  | C: hid=16, pos_weight | 0.9229 ± 0.0009 | 0.1917 | 0.2525 |
  | D: hid=64, pos_weight | 0.9239 ± 0.0014 | 0.1804 | 0.2412 |

  **The decision holds on data the model never touched.** `hid=64` beats `hid=16` well outside noise
  (0.9133 → 0.9204, against a baseline SD of 0.0053); `pos_weight` buys only
  +0.0035 AUROC while inflating ECE ~56× and driving the mean predicted
  probability to 0.2412 against a true base rate of 0.0588. Both conclusions match the
  test-set study's qualitatively — the test-set peek did not change which config was adopted — but the
  decision now has the validation-data provenance §2/§3.1 claim for it.

  `m2_posweight_calibration_check.py`'s test-set numbers are retained and still reported in §3.1: they
  remain a valid **confirmatory** measurement of the adopted config's effect on the actual held-out
  set, reported as a downstream check on a decision already made on validation data — not, as before,
  as the basis for the decision itself. `M2_Findings_Dossier.md` §2/§3.1 are updated to cite the
  validation-only study as the decision's evidence and the test-set study as its confirmation.

- **Supervisor approval status:** not yet sought. This is a defect repair restoring an integrity
  property the module already claims (§2's "no config may be chosen by looking at the test set"),
  executed directly per the posture taken for Deviations 7–10. It should be surfaced to the
  supervisor: the config itself is unchanged, but its stated evidentiary basis was wrong until this
  fix.

- **Impact assessment:** the adopted Track A config (`hid=64`, no `pos_weight`) is **unchanged** —
  this deviation corrects the evidence cited for it, not the decision itself, and every downstream
  number computed at that config (§§3–5, §21) stands. The general lesson, stated because it is the
  fourth time this pattern has appeared in this module (Deviations 7, 8, 9, 11 are all "a component
  quietly diverged from the standard the rest of the pipeline enforces"): **every config-adoption
  decision needs the same test-set firewall the primary search has, not just the primary search
  itself** — a follow-up study built to explain or attribute an adopted decision is exactly the kind
  of script that gets written after the discipline has already relaxed, and is worth checking for
  test-set access on principle, not only when a reviewer asks.

## Deviation 12 — Gate v2 replaced Gate v1's outcome/horizon definition and split method

- **Timestamp:** 2026-09-19, `M2_Outcome_Definition_Gate.md` v2.
- **What v1 specified:** the horizon ablation as an `ed_los_hours <= h` filter on a fixed, random
  patient-level split (§3/§7 of v1).
- **The deviation:** v2 replaces the LOS filter with a real time-to-event label
  (`event_time = min(direct-ED→ICU intime, in-hospital deathtime)`, gated on
  `outcome_icu_transfer_12h` to match `outcome_critical`'s own definition), and promotes the
  chronological (rolling-origin) split from an auxiliary arm to the **primary** evaluation split,
  demoting the random split to a secondary robustness arm. Both changes reverse guide v3.8's own
  §5.1/§7 specification (random patient-level split as primary).
- **Reason / resolution:** v1's own §3 asserted a true horizon label was infeasible ("no comparable
  timestamp for the mortality component"); checked directly against `hosp/admissions.deathtime`
  (99.91% populated for in-hospital deaths) and found false. v1's own §7 already recorded that a
  random primary split makes ECI's non-stationarity premise untestable on the headline benchmark.
  Both defects were therefore fixable, not merely disclosable.
- **Supervisor approval status:** **signed off** — Gate v2, approved by Akram, 2026-09-19, recorded
  in `M2_Outcome_Definition_Gate.md`'s own sign-off block. No training against v2 started before this
  sign-off (verified: the cohort rebuild is timestamped after the gate's sign-off line).
- **Impact assessment:** invalidates the T=2/random-split result set as the primary benchmark (it
  remains valid and archived as a secondary result, per §24). Every §24 number is computed under this
  revised definition. The horizon ablation is now genuinely non-nested-in-cohort (see gate v2 §3
  for the measured per-horizon positive rates), and the chronological-primary split is what makes
  §24.3's MAPIE-undercovers/ECI-holds-nominal result the module's headline rather than an auxiliary
  finding.

## Deviation 13 — Irregular vitals sequence truncated to T_max=10 observations

- **Timestamp:** 2026-09-19, `m2_build_irregular_cohort.py`.
- **What changed:** the T=2 pipeline's fixed (triage, latest-ED) representation is replaced by the
  genuine `ed/vitalsign.csv` per-stay sequence, truncated to the 10 most recent observations (plus
  triage), right-truncated if longer.
- **The deviation:** `T_max=10` is not specified by any governing document — it is a value chosen
  this session from the measured p90 of ED-vitalsign observations per stay (mean 3.83, median 3,
  p90=8, rounded up to 10 for headroom). No alternative T_max was benchmarked against it.
- **Supervisor approval status:** not sought; this is an implementation parameter of the Gate-v2-
  mandated rebuild, not a new scientific decision, and is disclosed here rather than left implicit
  in the extraction script alone.
- **Impact assessment:** measured directly on the final cohort — **0 stays were truncated in either
  split period** (train or test), so T_max=10 imposed no actual information loss on this cohort as
  built. This could change if the cohort or the MIMIC-IV extract version changes; the truncation rate
  should be re-measured, not assumed, if either does.

## Deviation 14 — Frozen-Temporal Head Control v2 and Hybrid GRU-D v2 reused T=2 head hyperparameters

- **Timestamp:** 2026-09-19, `m2_frozen_temporal_control_v2.py` / `m2_hybrid_grud_v2.py`.
- **What the guide specifies:** Sec 6.3's Stage-2B grid (`fusion_dropout` × `lr`) and Sec 6.2's
  Stage-2A grid (`static_hidden` × `static_dropout`) — a validation-selected search per model, run
  once for the T=2 cohort and never re-run for v2.
- **The deviation:** both v2 scripts reuse the T=2 pipeline's own validation-selected winners
  (Frozen Control: `fusion_dropout=0.1, lr=5e-4`; Hybrid: `static_hidden=64, static_dropout=0.1,
  fusion_dropout=0.1, lr=1e-3`) rather than re-running Stage-2A/2B against the v2 feature
  distribution.
- **Reason / resolution:** a disclosed scope decision, made to keep the overnight v2 grid (already
  ~3.5h) tractable. Judged low-risk because these are head-only architecture choices (dropout/
  learning-rate/hidden-width on top of a fixed frozen encoder), not choices expected to interact
  strongly with the input representation change the way the encoder's own hyperparameters would.
- **Supervisor approval status:** not sought; disclosed in both scripts' docstrings and in
  `M2_Findings_Dossier.md` §24.5 item 2 as an open item, not hidden.
- **Impact assessment / resolution: RESOLVED, 2026-09-19 (Phase 2b, same session)** —
  `m2_frozen_control_tuning_v2.py` and `m2_hybrid_head_tuning_v2.py` ran the same Stage-2A/2B grids
  against the v2 feature distribution. Selected configs were close to the reused T=2 winners (Frozen
  Control: `lr=1e-3` at the first tuning pass, `lr=5e-4` when the full pipeline was re-run end-to-end on 2026-09-21 (near-tie, GPU nondeterminism) vs. reused `5e-4`; Hybrid: `static_dropout=0.3` vs. reused `0.1`, everything else
  unchanged), and confirmatory AUROC moved negligibly (Frozen Control 0.9200→0.9201, Hybrid
  0.9371→0.9372) — confirming the low-risk judgment above was correct: these head-only choices did not
  interact strongly with the v2 representation change.

## Deviation 15 — mTAN v2 tuned under a lighter protocol than GRU-D/XGBoost

- **Timestamp:** 2026-09-19, `m2_mtan_tuning_v2.py`.
- **What the established protocol is:** `m2_grud_tuning.py` / `m2_xgboost_tuning.py`'s own precedent
  — 3 tuning seeds, ~14–15 candidates, staged with an anti-p-hacking `UNTUNED_BASE` and a
  baseline-vs-SD noise guard.
- **The deviation:** mTAN's tuning pass (`m2_mtan_tuning_v2.py`) used **1 tuning seed (42)**, a
  2-stage, 8-candidate grid (`nhidden`×`embed_time` then `num_ref_points`), and no baseline-vs-SD
  noise guard — selection was by raw validation AUROC only.
- **Reason / resolution:** an explicit, disclosed time-budgeting choice for an overnight run that
  also had to complete the 3-GRU-D-family full grid (~5h). Stated in the script's own docstring
  before it ran, not discovered after the fact.
- **Supervisor approval status:** not sought; disclosed in the script, in
  `M2_Findings_Dossier.md` §24.2, and in §24.5 item 3.
- **Impact assessment:** mTAN v2's reported AUROC (0.8446 ± 0.0053, 5-seed confirmatory) is a
  genuine, fairly-run result under a lighter search than its competitors received. Since mTAN
  underperforms every other v2 model including the no-vitals Static-Only MLP v2, additional tuning
  budget is very unlikely to reverse the qualitative finding (mTAN loses this comparison) — but the
  exact margin should be treated as a lower bound on mTAN's true tuned performance, not a precise
  estimate, until a matched-rigor re-tune is run.
- **RESOLVED, 2026-09-20 — and part of the impact assessment above was wrong.** The matched-rigor
  re-tune (`m2_mtan_tuning_matched_v2.py`: 3 seeds [42,123,7], 4 stages, `UNTUNED_BASE` re-merge,
  1-SD noise guard, 5× ECE guard; stage 5 not applicable to the shared BCE loop) selected
  (nhidden, embed_time) = (16,16) and `weight_decay = 0` (validation AUROC 0.8709 ± 0.0036 vs.
  0.8480 ± 0.0019 for the defaults). Five-seed confirmatory mTAN AUROC moved **0.8446 ± 0.0053 →
  0.8622 ± 0.0031**. The claim above that mTAN "underperforms every other v2 model including the
  no-vitals Static-Only MLP v2" is **false** at matched rigor (Static-Only is 0.8556); what survives is
  that mTAN is the weakest vitals-using model and loses to Hybrid GRU-D by +0.0750 AUROC (was +0.0926),
  still Holm-significant at 5/5 seeds. The lighter search had never touched weight decay, which supplied
  most of the gain. Pre-re-tune artifacts are archived (`*_PRE_MTAN_MATCHED`, `mtan_PRE_MATCHED`,
  `checkpoints/PRE_MTAN_MATCHED`, `m2_v2_deltas_pre_mtan_matched.json`). Only Δ_modern_comparator was
  recomputed; the other four deltas involve no mTAN and were reused, Holm re-applied over all five.

## Deviation 16 — XGBoost-Matched v2 left untuned in the confirmatory run

- **Timestamp:** 2026-09-19 (confirmatory run); tuning search added 2026-09-19 (Phase 2a, this
  session) via `m2_xgboost_tuning_v2.py`.
- **What the deviation was:** the v2 confirmatory grid's `m2_xgboost_matched_v2.py` ran XGBoost-
  Matched v2 at the same untuned pilot config (`n_estimators=100`, library defaults otherwise) the
  T=2 pipeline's own untuned XGBoost-Matched baseline uses, while every neural competitor (GRU-D
  family, mTAN) received a real search budget.
- **Why this matters, stated plainly because §18.1 already demonstrated the consequence once**: an
  apparent Hybrid-GRU-D-beats-XGBoost gap can reverse entirely once search effort is equalized — on
  the T=2 cohort, Δ_representation went from a nominal +0.0005 tie under the Guide-locked comparison
  to −0.0045 (CI excludes zero at 5/5 seeds) under symmetric tuning. `M2_Findings_Dossier.md` §24.2
  disclosed this immediately upon the v2 grid completing, rather than reporting the 1.5-point gap as
  settled: "this 1.5-point gap should be reported as suggestive, not conclusive."
- **Supervisor approval status:** not applicable — this was a disclosed sequencing decision (run the
  grid first, tune the one asymmetric baseline second), not a request for a scope reduction.
- **Impact assessment / resolution: RESOLVED, 2026-09-19 (Phase 2a, same session)** —
  `m2_xgboost_tuning_v2.py`'s 5-stage search selected `n_estimators=1000, learning_rate=0.03`
  (val AUROC 0.9189→0.9245); the tuned 5-seed confirmatory run reached test AUROC 0.9271 (up from
  0.9217). Unlike the T=2 precedent this deviation warned about, **the gap did not reverse**: Hybrid
  GRU-D v2 vs. XGBoost-Matched v2 narrowed from +0.0154 to +0.0101 but remains positive and
  Holm-significant (patient-cluster bootstrap, `M2_Findings_Dossier.md` §24.2b, all 5 seeds exclude
  zero, p<0.005). A second, sharper effect emerged as a side benefit of closing this deviation:
  Δ_temporal_alone (Temporal-Only GRU-D v2 vs. XGBoost-Matched v2) reversed from a non-significant tie
  to a significant XGBoost win (−0.0063, p=0.0022) — see §24.2b for the full analysis.

## Deviation 17 — `mondrian_lac_sets` class-0 calibration defect (static Mondrian-LAC comparator)

- **Timestamp:** found and fixed 2026-09-20, `Benchmark_scripts/m2_mondrian_conformal.py`. The defect had been present
  since the function was first written for Q1-blocker #2 (2026-09-19, T=2 cohort) and was carried into
  `m2_conformal_v2.py` and the Mondrian-ECI pilot/confirmatory runs.
- **What the specification requires:** a label-conditional split-conformal method must give per-class coverage ≥ 1−α
  for *each* class, distribution-free — i.e. ≈0.90 for both classes at α = 0.1 regardless of prevalence.
- **The defect:** class k was calibrated on `1 − np.where(y == k, y_prob, 1 − y_prob)[y == k]`. Since `y_prob` is
  P(y=1|x), that is right for k=1 but returns P(y=1|x) for k=0, so class 0 was calibrated on `1−p` while the test-time
  rule scored label 0 as `p`. q₀ sat near 1, label 0 was always included, and class-0 coverage was **exactly 1.0000
  on every model, seed, and cohort**. Singleton `{1}` sets were impossible.
- **Why it survived review:** class-1 coverage — the number every headline claim was about — was unaffected.
  The only signatures were on the majority class (a guarantee holding with equality to four decimals is not a
  coincidence) and in derived quantities (mean set size, empty-set rate, informative rate). It was found only
  because the Mondrian-ECI pilot reported `informative P(C={1}|y=1) = 0.0000` for the anchor on every seed.
- **Why the first regression test missed it:** the pilot's "T2 static-equivalence" test checked that the *set rule*
  reproduces given a threshold, never that the *threshold* was calibrated correctly.
- **Fix and guard:** corrected in place; `Benchmark_scripts/test_mondrian_lac_validity.py` now asserts per-class coverage
  of 0.90 across four prevalence/separation settings with n-derived tolerances (40 replicates each). A first,
  fixed-tolerance version failed at 2% prevalence on sampling noise alone and was replaced with a
  tolerance derived from n — recorded here because the correct response to a failing test is to find out whether the
  test or the code is wrong, and here it was the test.
- **Impact on reported results (all dispositioned):**
  - *Unaffected:* every class-1 coverage figure, all MAPIE and ECI figures (verified bit-identical), all AUROC/calibration/bootstrap results.
  - *Superseded:* every Mondrian-LAC mean set size (T=2 and v2), empty-set rates, class-0 and marginal coverage, and
    the framing that class-conditional validity "costs only set size". Corrected v2 values: class-0 coverage 0.871–0.889,
    marginal 0.874–0.889 (the static method **under**-covers under this cohort's shift), set size 1.78/1.77/1.40.
  - *Retracted:* the Mondrian-ECI confirmatory **GO** as a statement about efficiency (comparator scored 0.0000 on the primary
    endpoint by construction); corrected-anchor verdict is **MIDDLE** — `M2_MondrianECI_PreRegistration.md` Part 3.
  - *Also corrected:* the G7 table caption falsely said ECI restores class-1 coverage (it does not: 0.05–0.08).
- **Supervisor approval status:** not applicable — a correctness fix, disclosed on discovery.

## Deviation 18 — Stale conformal results after Phase 2b head re-tuning

- **Timestamp:** found and fixed 2026-09-20.
- **What happened:** Phase 2b (2026-09-19) replaced the Frozen-Temporal Head Control and Hybrid GRU-D head checkpoints
  (16:15/16:19) but `m2_conformal_v2.py` was not re-run; `data/m2_v2_conformal_summary.csv` (05:31) and everything derived
  from it (§24.3, the G7 table, the conformal figure, the report) described the pre-tuning checkpoints. Temporal-Only was
  unaffected (checkpoint unchanged) and reproduced bit-identically, which is what exposed the discrepancy.
- **Also fixed:** `get_predictions_v2` hardcoded `static_dropout=0.1` while the checkpoint was trained with 0.3. Numerically
  inert (dropout has no parameters and is disabled at eval), but a silent config/artifact divergence; it now reads the tuned
  config JSONs so the divergence cannot recur.
- **Impact:** small (largest shift: Hybrid Mondrian-LAC class-1 0.8872 → 0.8805) but real; qualitative conclusions unchanged.
  The dependency chain (`analyze` → tables → figures → report) has no automatic staleness check; the reproducibility
  entrypoint runs stages in order, which prevents this only if the whole chain is re-run.

## Deviation 19 — Mondrian-ECI pilot returned NO-GO; a guard was added to lr selection after seeing it, on the same test window

- **Timestamp:** 2026-09-20. Full record: `M2_MondrianECI_PreRegistration.md`.
- **What happened:** the pre-registered pilot returned NO-GO (q₁ saturation 0.43 vs a 0.25 threshold). The diagnosed cause
  is that the lr-selection rule, tuned on the 5.4%-prevalence train period, does not transfer to the 8.09%-prevalence
  test window. A degeneracy guard was added to lr selection (non-test data only) and re-run under a pre-registration written
  before execution.
- **Limitation, stated plainly:** the guard was designed after observing the failure **on this same test window**. The
  guarded run therefore shows the decision rule is fixed a priori and not test-driven; it does not show the rule
  generalizes to fresh data. Any manuscript must describe the pilot as exploratory.
- **Supervisor approval status:** direction chosen by the supervisor (2026-09-20, "pre-register a corrected confirmatory run").

## Deviation 20 — Horizon ablation: composite-tuned hyperparameters, 3 of 6 models, and an underpowered 1h arm

- **Timestamp:** 2026-09-20. Code: `run_horizon_ablation_v2.sh`, `m2_horizon_analyze.py`, `M2_V2_HORIZON` switch in `m2_v2_common.py`.
- **What CLAUDE.md requires:** a horizons 1h/6h/12h ablation for M2. The only prior horizon result (T=2, LOS-filtered) was retracted by Gate v2 §3.
- **What was done:** Temporal-Only GRU-D retrained per horizon (a per-horizon Hybrid needs a per-horizon frozen encoder), Hybrid GRU-D head on it, XGBoost-Matched
  native to `y_h`; 5 seeds; composite-outcome tuned configs; no class reweighting. Selected by the supervisor as "full per-horizon encoders, primary + tabular comparator".
- **Departures, each disclosed:**
  1. *Not re-tuned per horizon.* Every model uses its composite-outcome tuned config. Symmetric across the three models, but not the matched-rigor protocol the composite
     comparison received; a per-horizon re-tune could shift the deltas.
  2. *Only 3 of 6 models.* Frozen-Temporal Head Control, Static-Only MLP and mTAN were not run, so the static-contribution decomposition (Δ_static_contribution) is not repeated per horizon.
  3. *The 1h arm is underpowered a priori and reported as such.* 66 test positives; 14 validation positives, so early stopping on validation loss is noise (Temporal-Only seed 42: 11
     epochs, AUROC 0.7736). No 1h delta is significant; none may be read as evidence about short-horizon performance.
  4. *Different tasks.* `y_h` and `outcome_critical` are different labels with different positive rates; cross-row AUROC comparisons are descriptive.
- **Isolation guarantees, verified rather than asserted:** the horizon switch defaults to the composite target with byte-identical paths; `data/m2_v2_results.csv` and
  `m2_v2_predictions.csv` md5 unchanged after all horizon runs; the XGBoost composite run reproduces bit-for-bit at full float precision; GRU-D checkpoint-loading reproduces saved AUROC to 0.
- **Inference note:** the patient-cluster bootstrap is valid here (fixed predictions). It uses an exact weighted-AUROC implementation, verified against `roc_auc_score`
  (unit weights, `sample_weight`, and literal row duplication, to 1e-12), because the original per-resample implementation cost ~7 min per delta-seed.
- **Consequence for a previously stated finding:** §24.2b's "GRU-D's temporal representation alone is a net negative relative to tabular aggregation" is **specific to the composite
  outcome**; at 6h Temporal-Only beats tuned XGBoost (+0.016, 5/5 seeds, Holm-significant) and at 12h it ties. §24.2b, §24.4, §24.5 and the README were qualified accordingly.
- **Supervisor approval status:** design chosen by the supervisor (2026-09-20).

## Deviation 21 — Label-lag experiment: implementation clarifications made while running the pre-registered protocol

- **Timestamp:** 2026-09-21, `m2_label_lag.py`, executing `M2_LabelLag_PreRegistration.md`.
- **What the pre-registration specifies:** an event-driven reproduction of the vendored ECI recursion, verified against the vendored code (tests T8, T9, T11, T13) before any lagged result is printed; lr re-selected per lag; L = ∞ meaning no label ever arrives.
- **The clarifications (none changes a decision; each is disclosed here):**
  1. **T9 tail row.** The per-class simulator at L = 0 equals `mondrian_eci_sets` on every row up to each class's last arrival and differs on **1 of 84,241 rows** (the very last arrival, class-0 column): `mondrian_eci_sets` clips its class-radius index to n_k − 1 (the vendored ECI loop never writes the update from the final class-k point), so that one row reuses a radius built from one fewer class-0 point. The simulator applies the update. The earlier Mondrian-ECI numbers are unaffected beyond that single row; T9 was written to assert identity up to each class's last arrival and to count the excluded rows (≤ 5 allowed).
  2. **L = ∞ learning rate.** The pre-registration did not say which lr to use when no label ever arrives (selection has no feedback to select on). The lag-0 selection is used; the frozen radius is that of the end of the calibration fold.
  3. **Numerical stability.** A first version of the simulator computed exp(−d) directly and overflowed when the radius wound up at long lag (radius above ~350 in at least one configuration). It was replaced by exp(−|d|), which is mathematically identical because the derivative of the sigmoid is even, and T8/T9/T13 were re-run and pass.
  4. **dtype.** The lagged recalibration first tripped test T11 because it used float32 predictions where `recal_sets` casts to float64 (radii differed by 3×10⁻⁸; sets were identical). It now casts identically.
  5. **Aborted launches.** Three launches ended early (T9 tolerance, T11 dtype, the overflow). No result table is written until a run completes, so no partial output was read or used.
- **Impact assessment:** none on any verdict. The realistic-band verdicts are NOT NEEDED with large margins (recalibration narrower or equal on every model and lag), so a single-row edge case, a dtype and an overflow guard cannot move them.
- **Supervisor approval status:** not sought; disclosed here and in the dossier §24.3d.

## Deviation 22 — Streaming-latency benchmark on v2 measured the recalibration refresh on an 11,754-arrival window for the "W = 40k" row

- **Timestamp:** 2026-09-21, `m2_v2_latency.py`.
- **The deviation:** the benchmark times one periodic-recalibration refresh from the labelled calibration arrivals available before the test stream; the final-calibration fold has 11,754, so the "W = 40k" refresh was timed on 11,754 arrivals (recorded in the JSON as `window_used`). It therefore understates a real 40k-arrival refresh (a quantile over ~3.4× more values).
- **Impact assessment:** negligible for the conclusion: both refreshes cost well under 0.1 ms and are amortized over 2,000 arrivals; the budget is 2,000 ms.

## Deviation 23 — Conformal scripts paired sequence tensors with the wrong patients (row-index lookup after `sort_chronologically`)

- **Timestamp:** found 2026-10-01 while exporting the Track B model; fixed and all affected analyses rerun the same day.
- **The deviation:** `m2_hybrid_common.sort_chronologically()` ends in `reset_index(drop=True)`, and `m2_v2_common.prepare_temporal_tensors_v2()` selected the stored sequence rows with `df.index.values`, asserting only that the index was in range. Any script that sorted a frame before calling it received the sequence tensors (X, M, D, pad) of rows 0..n-1 of the split file, while the labels and static features came from the sorted frame. The temporal inputs no longer belonged to the labelled patient. Nothing failed, because the indices stayed in range.
- **Measured effect** (Hybrid GRU-D v2, seed 42, composite target): test AUROC of the predictions used for conformal 0.9375 -> 0.7324; calibration-fold 0.9327 -> 0.7094.
- **Affected analyses (all rerun):** `m2_conformal_v2`, `m2_mondrian_eci_pilot`, `m2_mondrian_eci_confirmatory`, `m2_mondrian_eci_anchor_correction`, `m2_mondrian_eci_comparators`, `m2_eci_vs_recalibration`, `m2_label_lag` (and `m2_v2_logit_vs_clipping`, which reuses the label-lag prediction cache), plus `m2_v2_latency`. The label-lag prediction cache was built from misaligned predictions and was discarded.
- **Not affected:** every model-training and model-comparison script (they pass unsorted frames; the new lookup is bit-identical to the old one there), the T=2 pipeline (selects rows by `stay_id`), and `m2_v2_analyze` / `m2_horizon_analyze` (read saved predictions that carry their own labels).
- **Fix:** `prepare_temporal_tensors_v2()` now looks rows up by `stay_id` (asserting `stay_id` is unique and every row has a stored sequence), so the result no longer depends on the frame's order or index. Verified: unsorted frames give identical tensors to the old lookup; sorted frames give test AUROC 0.9375 and calibration AUROC 0.9327.
- **Pre-fix outputs** are preserved in `data/archive_pre_alignment_fix_2026-10-01/` (do not cite).
- **Impact assessment:** every pre-fix conformal number (coverage, width, class-conditional coverage, ECI/recalibration comparison, label-lag tolerance, Mondrian-ECI verdicts) was computed from predictions with AUROC about 0.73 and is superseded. Conclusions that depended on them are re-derived from the rerun, not patched.
