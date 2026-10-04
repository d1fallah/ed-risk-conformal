# M2 — Clarification Request: Hybrid GRU-D Execution Guide v3.2

**To:** Supervisor
**From:** Student 2 — M2
**Date:** 2026-08-28
**Re:** `M2_Hybrid_GRU_D_Student_Execution_Guide_v3_2.docx` — four points requiring your decision before Hybrid GRU-D construction begins

## Summary

Before starting on the Hybrid GRU-D model itself, every "frozen"/"already validated" number and architectural claim in the guide was cross-checked directly against the actual M2 codebase and saved result files (not just re-read from the prior reports). Six of eight baseline numbers checked (GRU-D Tuned, XGBoost-Rich, NEWS2, Logistic Regression, Random Forest, and the XGBoost-Rich AUPRC) matched exactly. Four points did not resolve cleanly, and rather than pick an interpretation unilaterally, I'm bringing all four to you before writing any Hybrid GRU-D model code. Work that doesn't depend on these four points — the data/leakage pipeline, the Static-Only MLP, and XGBoost-Matched — is proceeding in parallel (see the closing section).

| # | Item | What's blocked |
|---|---|---|
| 1 | Static-feature confound in the "Frozen"/"Tuned" GRU-D baseline, plus overlapping terminology | Δ_static's validity; which checkpoint is compared against Hybrid GRU-D |
| 2 | Fusion width: 96-d (diagram) vs. 64-d (arithmetic from the frozen hidden dim) | Hybrid GRU-D's fusion-layer/prediction-head shape |
| 3 | GRU-D learning rate: 5e-4 (guide) vs. 0.01 (actual frozen checkpoint) | Which lr the "frozen" temporal branch trains at |
| 4 | Stale "Historical" baseline number, not reproducible from any current file | What the Historical row in the final table reports |

---

## Item 1 — Static-feature confound + overlapping "Frozen" terminology

The guide's §1 table lists **"GRU-D Tuned (Primary Comparator)"** — AUROC 0.9009 ± 0.0042 — as **"Temporal vitals only (2-timestep, irregular)"**. §4.1's four-way model-definition table separately lists **"Frozen GRU-D"** (no "(Historical)" suffix) with the same "✓ temporal / ✗ static" characterization, described in §2.1 as "reuse the existing GRU-D implementation... HYPERPARAMETERS FROZEN: hidden dim = 32, dropout = 0.2, weight decay = 1e-4" — i.e., the identical configuration as "GRU-D Tuned."

The actual code (`Benchmark_scripts/m2_ablations.py:120-145`, `GRUD.forward`) shows this exact model is not temporal-only:

```python
self.clf = nn.Sequential(
    nn.Linear(hid+stat, clf_hidden), nn.ReLU(),
    nn.Dropout(dropout), nn.Linear(clf_hidden, out)
)
...
return self.clf(torch.cat([h, S], 1))   # h = temporal hidden state, S = 24 static features
```

`stat=24` (`static_cols`, `m2_ablations.py:58-68`): age, gender, triage acuity, 9 utilization counts (ED/hosp/ICU visits at 30/90/365 days), 10 chief-complaint flags, med/med-reconciliation counts. This is baked into every saved AUROC for this model, including the 0.9009 figure.

This means Δ_static = AUROC_Hybrid − AUROC_Frozen would not cleanly answer "did adding static information help," since the comparator already has (a different, broader) static information fused in. Three questions:

1. Should a genuinely temporal-only control (`static_size=0`, same architecture otherwise) be added as an additional ablation arm, so Δ_static has a clean denominator? (My recommendation, if you'd like one: yes — it's a cheap addition given the architecture refactor is already needed for the two-branch design, and it makes the causal claim defensible under review.)
2. In the guide's own usage, does "Frozen GRU-D" (§4.1) refer to the same checkpoint as "GRU-D Tuned" (§1), or something else?
3. §10's final comparison table shows blank (`—`) Coverage/Width cells for "GRU-D Tuned," even though this model's MAPIE/ECI coverage and width are already computed and saved (`data/m2_multiseed_overall.csv`: coverage_mapie 90.13%±0.16pp, width_mapie 0.2824±0.0586). Is that omission intentional, or should those cells be filled from the existing data?

## Item 2 — Fusion width: 96-d vs. 64-d

§2's architecture diagram states the fusion layer is 64-d (temporal) + 32-d (static) = 96-d → `Linear(96, 1)`. §6.1's hyperparameter table freezes the GRU-D hidden dim at 32 (matching the actual code's `hid=32`), which would make the real temporal embedding 32-d, not 64-d — so fusion would be 32+32=64-d → `Linear(64, 1)`, not 96.

Which is authoritative: retrain the temporal branch at `hidden_dim=64` to match the diagram (a deviation from "frozen"/"already validated" hidden=32), or treat the diagram's 64-d/96-d labels as a documentation slip and build 32+32=64-d?

## Item 3 — GRU-D learning rate: 5e-4 vs. 0.01

§6.1's table states: "Learning rate | 5e-4 | Already validated as best config in M2." Checked against every artifact that documents the tuning search:

- `M2_Findings_Dossier.md` §2's tuning stages vary `pos_weight`, `hid`, `dropout`, `weight_decay`, `batch_size`, `early_stopping` — no learning-rate sweep appears anywhere.
- `data/m2_grud_tuning_trace.csv`'s header has no learning-rate column.
- The actual tuned/frozen model's code (`m2_ablations.py:158-165`, `SKLearnGRUD.__init__`) hardcodes `lr=0.01` as the default, and its own inline comment names only `hid=32, dropout=0.2, weight_decay=1e-4` as "the winning config from `m2_grud_tuning.py`'s validation-only search" — lr is not part of that list.

Should the frozen temporal branch train at lr=0.01 (what's actually in every saved artifact), or at 5e-4 (the guide's stated value, which would need its own validation since it was never tested in any saved trace)?

## Item 4 — Stale "Historical" baseline number

§1/§10 state: **"GRU-D Frozen (Historical) | ... | 0.8937 ± 0.0128 | 0.4196 ± 0.0424 | ... pre-tuning baseline"**, with §1's instruction "Do not rerun, recompute, or modify."

This number predates a fit/calibration-split leakage fix documented in `M2_Findings_Dossier.md` §1.2 (10,156 overlapping patients found at seed 42 under the old row-level split). The pre-tuning baseline was re-run after that fix, and the currently-archived, reproducible result (`data/m2_seed_results_pretuning/seed_{42,123,7,456,789}.json`, aggregated in `data/m2_multiseed_overall_pretuning.csv`) is **0.8925 ± 0.0136** (AUPRC **0.4176 ± 0.0469**) — not 0.8937/0.0128 or 0.4196/0.0424. No file in the repo currently reproduces the guide's exact stated figures.

Should the Historical row report the reproducible 0.8925 ± 0.0136 (with a note that the guide's figure predates the leakage fix), or the guide's stated 0.8937 ± 0.0128 (footnoted as supervisor-provided, not independently reproducible from current repo artifacts)? Note this doesn't affect anything else — §0.1's Fix #1 already relegates this specific number to a historical footnote regardless of which value is used.

---

## FYI — not blocking, flagged for completeness

- **Outcome-Definition Gate status**: `M2_Outcome_Definition_Gate.md`'s last line still literally reads "Pending Supervisor Approval," even though every M2 script complies with it and both prior reports treat it as binding. Worth a formal status update at your convenience.
- **Elixhauser category count**: §3.1 states "29 AHRQ categories." The vendored mapper (`Benchmark_scripts/medcodes/diagnoses/comorbidities.py:109-111`, citing Quan H, Sundararajan V, Halfon P, et al., "Coding algorithms for defining Comorbidities in ICD-9-CM and ICD-10 administrative data," *Med Care*, 2005 Nov;43(11):1130-9) implements **31** categories (`medcode_utils.py`'s `eci_var_map`). The comorbidity-burden feature will be documented as a sum over the real 31, not 29, per the project's citation-verification standard.
- **Roadmap scope**: `SihhaDZ_Student_Roadmap_v8-1.pdf`'s M2 section lists exactly three closed-menu optional enhancements (frequency-decoupled vitals, test-time adaptation, sensitivity-aware DeepSHAP caching) and doesn't include a hybrid static+temporal architecture among them. This guide's own "Approved for execution after peer review" is being treated as your direct authorization superseding that closed menu — flagging for the record, not as an objection.

## What's proceeding vs. paused

**Paused pending your answers above:** all Hybrid GRU-D model code, its multi-seed training, and everything downstream of it (uncertainty analysis, Δ_temporal/Δ_static, subgroup analysis, final synthesis table).

**Proceeding now**, since none of it depends on the four items above: environment setup (done), the four-static-feature data pipeline and Feature Availability Table, the Pre-Experiment Leakage Checklist, the Static-Only MLP model, and XGBoost-Matched.
