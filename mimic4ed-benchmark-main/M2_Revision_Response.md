# M2: response to the Q1 review (forwarded by the supervisor), status 2026-10-03 (analyses run after sign-off)

Manuscript: `../paper_draft/` (preview `M2_draft_preview.pdf`, third pass after the alignment-defect
rerun). Every point was checked against the current draft and the rendered PDF.

**Note for the supervisor.** Part of the review describes the *pre-fix* results:
- "collapse to 3–8% coverage";
- "periodic refitting matches ECI with narrower intervals";
- "ECI restored marginal coverage";
- width differences of "0.017–0.053".

After the row-alignment defect was fixed (Deviation 23), the findings are different:
- static split-conformal kept nominal coverage;
- ECI and recalibration did not improve on it, and ECI was wider;
- class-1 coverage under marginal calibration is about 20%.

The reviewer was probably given an older PDF. The current preview should be the one circulated.

| # | Review point | Status | Action |
|---|---|---|---|
| headline | "Marginal 90% hides collapse to 3–8%; recalibration matches ECI with narrower intervals" | **Superseded** | Pre-fix results. The current draft reports the corrected findings above. |
| 1 | ECI forced onto classification; add an adaptive set-valued baseline | **Done (E3)** | ACI and ECI on the marginal LAC score with set-valued output (vendored implementations, unchanged), immediate labels and a 240-arrival delay. All were valid on all three models but none gave smaller sets than static LAC (primary model 1.003 and 0.930 against 0.929), so under the pre-specified rule neither improves on static; the conclusion now covers set-valued adaptive methods. Under delay, ACI kept 0.90 coverage only by giving 10% of patients an empty set and most others both labels (0.2% of deteriorating patients received {1}). Supplementary Table S7; Methods and Results paragraphs. Note in the paper: for a binary label the LAC score equals \|y − p̂\|, so the sets cover the same patients as intervals of the same radius. |
| 2 | Streaming partly synthetic; scope to between-period shift or stress-test | **Done (text + E4, supplementary)** | Scope text as before. E4 imposed synthetic drift on the real predictions at the midpoint of the test stream. Static coverage fell after the change (primary model 0.888 and 0.867 under logit shifts +0.5/+1.0, 0.815 under a case-mix change), while ECI and ACI held about 0.90; recalibration held under the shifts but not the case-mix change (0.856). The evaluation can therefore register an adaptive benefit; the negative result reflects the cohort's shift. The pre-specified time-to-recovery measure proved uninformative (2,000-arrival coverage has sampling SD ≈ 0.007, so all methods crossed the ±0.01 band by chance) and is reported as such. Supplementary Table S8; Methods, Results and Discussion sentences. |
| 3 | Task timing: full-stay input vs triage framing | **Done** | The abstract and Introduction now state that the task is a retrospective end-of-stay classification, not triage-time warning. Methods already said so. |
| 4 | NEWS2 reads as a second paper | **No change proposed** | NEWS2 is a comparator subsection in Results and Discussion, not a contribution. None of the three contribution bullets mentions it. It can be shortened further if the supervisor prefers. |
| 5a | Width differences: add paired tests with multiplicity correction | **Done (S1)** | Per-seed patient-cluster bootstrap (10,000) with Holm over the 9 contrasts, the paper's existing rule; recomputed widths reproduced the reported means exactly first. ECI was significantly wider than static on all three models (+0.023 to +0.039); recalibration was wider by +0.001 to +0.011, significant in 5 of 6 model–window pairs (detectable with 84,241 stays, small on the risk scale). Supplementary Table S6; the Limitations sentence "carry no significance test" was replaced. |
| 5b | XGBoost SD of exactly 0 | **Done (S2)** | Predictions verified identical across the five seeds; Table II now gives AUROC 0.9271 [0.9241, 0.9301] and AUPRC 0.6337 [0.6221, 0.6454] (patient-cluster bootstrap, 10,000), footnoted as population uncertainty, not seed variability. Methods and Limitations updated. |
| 6 | "MIMIC-IVED" typo | **Not in the manuscript** | PDF-to-text artifact. The source and the rendered pages read "MIMIC-IV-ED". |
| 6 | Table IV 12h row "+0.0151+0.0134" | **Not in the manuscript** | The cell reads "+0.0151 [+0.0134, +0.0168]". This is an extraction artifact. The caption was rewritten without the emphatic internal wording ("NOT truncated", "must not be read"). |
| 6 | Watermark, affiliation/ORCID, repository URL | **Needs author input** | Preview markers; the journal version needs the real values. |

Also changed:
- To keep the abstract under 250 words (233), the clause on the 6-hour-horizon ranking was moved out
  of the abstract; it remains in Results.
- `src/verify_paper_assets.py` passes (25 assets, 695 values), and the draft builds with no undefined
  references.

Revision analyses (run 2026-10-03 after supervisor sign-off):
- Scripts: `Benchmark_scripts/m2_rev_e3_adaptive_sets.py`, `m2_rev_e4_drift_stress.py`, `m2_rev_s1_width_bootstrap.py`,
  `m2_rev_s2_xgb_bootstrap.py`; outputs `data/m2_v2_rev_*`; logs `data/m2_rev_*.log`.
- Built-in checks passed: T14 (static LAC = reported static coverage, exact), T15 (static Mondrian = reported), T16
  (recomputed widths = reported means, exact), S2 determinism across seeds.
- Implementation notes: ACI window length 10^7 (the vendored repository's configured value); for ACI under delay the
  vendored `ahead` mode also delays the warm-up, while ECI's delay applies to the test stream only (verified simulator).
- Evidence manifest amended (new artifacts only; no existing hash changed); `src/verify_paper_assets.py` PASS
  (28 assets, 1,112 values); the draft builds with no undefined references.
- Also fixed: "below XGBoost-Matched by -0.0063" double negative in Results.

## Supervisor style remarks (2026-10-03), applied to the drafting preview

| Remark | Change |
|---|---|
| Abstract: clear contribution, short sentences, context → "However" → contribution, no numbers or "best" | Abstract rewritten in that order. Sentences are at most about 25 words. There are no numerical results, and models are named generically. |
| No IMRAD titles; descriptive section titles | Every section now has a descriptive title. All labels are unchanged, so cross-references still resolve. |
| Bold the best result in tables | Bolding is computed in the table generators, from the values. Rules: the best per column by metric direction; for coverage, the valid method with the smallest width or set size; significant differences; no bold in descriptive tables or for training-independent columns. Each caption states its rule. |
| References from 2017 on | Kept only the original sources of methods the papers use (Mondrian conformal, XGBoost, Elixhauser coding, DP-SGD). Removed Löfström 2015 and Konečný 2016, and replaced Vovk 2012 with Vovk et al. 2022. |
| Remove sections such as "Relation to prior work" | Removed. Their substantive comparisons were moved into the Discussion paragraphs they bear on. The roadmap paragraphs were deleted. |
| UML-style pipeline figure | New Fig. 1: a component diagram in which each stage shows inputs → outputs and its parameters. Counts come from traced values; the rest is the declared study configuration. |
| Related Work: shorter, with the gap at the end of each paragraph, and no Gap section | Rewritten as four short themed subsections, each ending in its own gap sentence. The "Gap addressed" section and padding citations were removed. |
| No lists | Every itemize/enumerate (contributions, research questions, model and regime lists) is now prose. |

The verifier passes, the draft builds with no undefined references, and the changed pages were render-checked.
