# M2 Related-Work Notes — Literature Synthesis (internal reference, not manuscript prose)

**Status:** Reference notes. Per the project's standing rule, no paper text is written until M1–M5 are all
complete. This document closes the dossier's §7 item 9 / §10 item "Real related-work literature search —
not started." It is organized by the six areas requested, and every entry is graded for evidentiary
weight and tied to a specific M2 claim.

**Verification protocol used.** Every reference below was checked by live web search against a primary
or publisher-controlled source (publisher page, conference proceedings page, PMC/PubMed record, or
arXiv abstract) for title, authors, venue and year. Nothing here is quoted from memory. Where a detail
could not be confirmed from a primary source, it is marked **[UNVERIFIED — do not cite until checked]**
rather than dropped or guessed.

**Search boundaries (state these if a reviewer asks how this review was scoped).**
- Sources searched: general web search reaching NeurIPS/ICML/ICLR/PMLR proceedings, JMLR, Nature
  Scientific Data / Scientific Reports, Taylor & Francis (JASA), Project Euclid (Annals of Statistics),
  PLOS One, JMIR, PMC/PubMed, PhysioNet/MIMIC documentation, and arXiv.
- Date range: no lower bound for foundational statistics (1999–2019 conformal/EWS foundations);
  2020–2026 for "current landscape" claims.
- Language: English only.
- Excluded: blog posts, vendor documentation (except MAPIE's own docs, used only to establish what the
  *library* implements, never as evidence for a scientific claim), and news coverage.
- Not searched: Scopus, Web of Science, Embase, IEEE Xplore full-text, ACM DL full-text. **This is a
  targeted review, not a PRISMA systematic review, and should never be described as one.** Coverage of
  clinical-informatics venues (JBHI, JAMIA, AIM, ICHI) in particular is thin here and is the main
  known gap in this search.

**Evidence tier key used throughout:**
`A` = peer-reviewed, top-tier venue, primary methodological or empirical contribution ·
`B` = peer-reviewed journal/conference, sound but narrower or single-cohort ·
`C` = peer-reviewed but secondary (review/scoping/commentary) ·
`D` = preprint / non-peer-reviewed grey literature — usable as a signal, never as authority ·
`E` = official documentation / standards body (authoritative for what it documents, not a research finding).

---

## 1. GRU-D and irregular clinical time series

### 1.1 The canonical citation — VERIFIED, with one correction to the dossier

> Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). **Recurrent Neural Networks for
> Multivariate Time Series with Missing Values.** *Scientific Reports*, 8(1), 6085.
> DOI: 10.1038/s41598-018-24271-9. (arXiv preprint 1606.01865, 2016.)
> **Tier A.**

Establishes GRU-D: a GRU variant that takes masking and time-interval representations of missingness
and folds them into the architecture via input and hidden-state decay, on the premise that missingness
patterns in clinical records are themselves informative.

*Relation to M2:* This is the correct and complete citation for the dossier's §1.3 architecture. **Two
things to fix in the dossier's citation handling:** (a) the dossier says "Che et al. 2018" without
recording the venue — *Scientific Reports* 8:6085, not a conference; (b) the arXiv preprint predates the
journal version by two years, so any statement like "GRU-D (2018)" understates its age — reviewers who
know the field will read it as a 2016 method. The dossier's decay form
`gamma = exp(-relu(Delta @ W_gamma + b_gamma))` matches the paper's specification.

*Complication M2 should acknowledge:* Che et al. evaluate on long, densely-sampled ICU sequences
(PhysioNet 2012, MIMIC-III). M2 runs GRU-D at **T = 2 timesteps** (triage, latest ED). That is a
defensible engineering choice given ED data availability, but it is far outside the regime the original
paper's mechanism was designed for and validated in, and a reviewer will notice. §19.2's Sparse-vs-Dense
interaction result (DiD positive at 5/5 seeds) is the module's best defense here and should be
foregrounded, not buried.

### 1.2 Where GRU-D sits now — successors a Q1 reviewer will expect to see named

> Horn, M., Moor, M., Bock, C., Rieck, B., & Borgwardt, K. (2020). **Set Functions for Time Series.**
> *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119.
> (arXiv 1909.12064.) **Tier A.**

Reframes classification of irregularly-sampled, unaligned time series as learning a differentiable set
function over (time, variable, value) observations, avoiding imputation entirely. Highly parallelizable
with a favorable memory footprint; positioned explicitly for healthcare and online monitoring.

*Relation to M2:* SeFT is the most direct "you did not need to impute or discretize" alternative to
GRU-D. M2's 2-timestep tensorization is exactly the kind of preprocessing SeFT argues against. This is
the first name a reviewer is likely to raise.

> Shukla, S. N., & Marlin, B. M. (2021). **Multi-Time Attention Networks for Irregularly Sampled Time
> Series.** *International Conference on Learning Representations (ICLR 2021).* **Tier A.**

mTAND: learned continuous-time embeddings plus a time-attention mechanism replacing fixed similarity
kernels, for interpolation and classification of sparse, irregular, multivariate physiological series.
Reports performance at or above prior baselines (GRU-D among them) with substantially faster training.

*Relation to M2:* mTAND is the standard modern attention-based comparator on precisely M2's data type,
and it directly claims to beat GRU-D. Its absence from M2's comparison table is a live reviewer risk.

> Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020). **Neural Controlled Differential Equations
> for Irregular Time Series.** *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*,
> spotlight. **Tier A.**

Neural CDEs: the continuous-time family that, unlike Neural ODEs, can incorporate incoming observations
into the trajectory, making them directly applicable to partially-observed, irregularly-sampled
multivariate series.

*Relation to M2:* The continuous-time branch of the landscape. Less immediately threatening to M2 than
mTAND (Neural CDEs need more than two observations per series to be meaningful), and that is itself the
*argument M2 can make*: at T = 2 the continuous-time family has almost nothing to integrate over. Worth
citing to show the landscape was surveyed, with a stated reason for non-comparison.

> Zhang, X., Li, S., Chen, Z., Yan, X., & Petzold, L. R. (2023). **Improving Medical Predictions by
> Irregular Multimodal Electronic Health Records Modeling.** *Proceedings of the 40th International
> Conference on Machine Learning (ICML 2023)*, PMLR 202. (arXiv 2210.12156.) **Tier A.**

Handles irregularity in both numerical time series and clinical-note sequences, combining hand-crafted
imputation embeddings with learned interpolation embeddings via a gating mechanism, plus time attention
for notes. Reports relative F1 gains of 6.5% (time series), 3.6% (notes), 4.3% (fusion) over
state-of-the-art baselines.

*Relation to M2:* Establishes the 2023 state of the art and that the field has moved to multimodal
irregular EHR modeling. Also useful as a scope defense: M2 is deliberately vitals-only, which is a
narrower but clinically deployable setting (no notes needed at triage).

### 1.3 The counter-literature — this is M2's strongest ally on §3.2/§18.1

> Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). **Why do tree-based models still outperform deep
> learning on typical tabular data?** *Advances in Neural Information Processing Systems 35 (NeurIPS
> 2022), Datasets and Benchmarks Track.* **Tier A.**

Benchmarks tree ensembles against modern deep tabular architectures across 45 datasets **with matched
hyperparameter search budgets**, finding tree-based models remain state of the art on medium-sized data
(~10K samples). Explicitly argues that benchmark conclusions in this area are decided by search budget
and by differing inductive biases, not by architecture class alone.

*Relation to M2 — this is the single most useful citation in this document for M2's most awkward
finding.* §3.2 and §18.1 report that tuned XGBoost beats tuned GRU-D (0.9239 vs 0.9219, p = 0.0172) and
that `Δ_representation` flips from a tie to −0.0045 once the baseline is symmetrically tuned. Grinsztajn
et al. establishes that (a) this result is *expected*, not anomalous, and (b) **tuning-budget symmetry
is a recognized methodological requirement in this comparison, not a self-imposed scruple.** M2's
§18.1 sensitivity analysis is doing methodologically what this paper argues everyone should do. Frame
the finding as a confirmation in a new domain, not as a disappointment.

> Liao, K., Wang, W., Elibol, A., Meng, L., Zhao, X., & Chong, N. Y. (2022). **Does Deep Learning REALLY
> Outperform Non-deep Machine Learning for Clinical Prediction on Physiological Time Series?**
> arXiv:2211.06034. **Tier D (preprint — no peer-reviewed venue confirmed).**

Ten models (three deep, seven classical) on PhysioNet 2019 physiological time series; concludes deep
learning outperforms classical methods only under specific conditions (particular evaluation metrics,
sufficiently large training sets).

*Relation to M2:* Directionally supports §3.2, but it is a preprint and I could not confirm publication.
**Use only as supporting color alongside Grinsztajn et al., never as a load-bearing citation.** If a
peer-reviewed equivalent on physiological time series is needed, that is worth one more targeted search
before writing.

> Bui, H., Warrier, H., & Gupta, Y. (2024). **Benchmarking with MIMIC-IV, an irregular, sparse clinical
> time series dataset.** arXiv:2401.15290. **Tier D (preprint, no venue).**

*Relation to M2:* Noted for completeness only. Thin, unrefereed, and I could not extract a concrete
GRU-D-vs-GBDT comparison from it. **Recommend excluding from the manuscript.** Recorded here so a later
pass does not spend time rediscovering it.

### 1.4 Section 1 verdict

The GRU-D citation is correct and now fully specified. **The real finding in this section is a gap, not
a confirmation:** GRU-D is a 2016/2018 method and at least three well-cited successors (SeFT, mTAND,
Neural CDE) plus a 2023 ICML multimodal successor exist and are the standard comparators in this
literature. M2 currently compares GRU-D to LR / RF / XGBoost and NEWS2 — all *non-temporal* baselines —
so the module has no evidence that GRU-D is a good choice *among temporal architectures*, only that it
beats tabular and clinical baselines (and §18.1 says it does not reliably beat tabular either). See
Gaps & Risks G1.

---

## 2. Conformal prediction for clinical / tabular risk scores

### 2.1 MAPIE's methodological basis — VERIFIED as the dossier states

> Sadinle, M., Lei, J., & Wasserman, L. (2019). **Least Ambiguous Set-Valued Classifiers With Bounded
> Error Levels.** *Journal of the American Statistical Association*, 114(525), 223–234.
> DOI: 10.1080/01621459.2017.1395341. (arXiv 1609.00451.) **Tier A.**

Set-valued classification that guarantees a user-specified coverage level while minimizing expected set
size (ambiguity). This is the LAC conformity score.

*Relation to M2:* Confirms §17's verification chain exactly. MAPIE 1.5.0's
`SplitConformalClassifier(conformity_score="lac")` implements the Sadinle et al. score, and MAPIE's own
source documents `LACConformityScore` as "formerly called `score`". The dossier's claim that the guide's
`MapieClassifier(method="score")` → `conformity_score="lac"` mapping is *a rename across API
generations, not a methodological substitution*, is correct and now externally supported. **No change
needed.** MAPIE's documentation is Tier E and is cited only for what the library implements.

*One thing to add:* LAC minimizes *average* set size, which is precisely why it produces **empty
prediction sets** for low-confidence points. The dossier's §17 finding — MAPIE's empty-set rate on the
positive class reaches 0.4860 at seed 42 — is a **known and intended property of the LAC score**, not a
bug or an artifact. Saying so explicitly (with this citation) turns an alarming-looking number into a
correctly-diagnosed methodological consequence. This is a meaningful improvement to how §17 reads.

### 2.2 Split-conformal foundations

> Vovk, V., Gammerman, A., & Shafer, G. (2022). **Algorithmic Learning in a Random World**, 2nd ed.
> Springer International Publishing. DOI: 10.1007/978-3-031-06649-8. **Tier A (monograph).**
> **VERIFIED, 2026-09-19** — the 2nd edition exists and is the citable current edition (the 1st
> edition, Springer, 2005, ISBN 0387001522, remains valid as a historical citation but the 2nd
> edition should be preferred in new work). Confirmed via Springer's own DOI record and independently
> corroborated by Rutgers' and Royal Holloway's institutional repositories.

> Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). **Distribution-Free
> Predictive Inference for Regression.** *Journal of the American Statistical Association*, 113(523),
> 1094–1111. DOI: 10.1080/01621459.2017.1307116. (arXiv 1604.04173.) **Tier A.**

The reference treatment of split conformal: constructs finite-sample marginally-valid prediction bands
around any regression estimator, and analyzes full vs. split conformal and the jackknife variant,
including their statistical-accuracy/computation trade-offs.

*Relation to M2:* This is the correct citation for "split-conformal prediction" wherever M2 uses the
term for its interval construction, and it is the source of the **marginal** validity guarantee — which
is exactly the guarantee M2's §17/§21.5 shows is insufficient at the class-conditional level. Cite it
for what it *does* promise, so the limitation reads as a scope statement rather than a failure.

> Tibshirani, R. J., Foygel Barber, R., Candès, E. J., & Ramdas, A. (2019). **Conformal Prediction Under
> Covariate Shift.** *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2526–2536.
> **Tier A.**

Weighted conformal prediction restores distribution-free validity when test and training *covariate*
distributions differ, via likelihood ratios between the two distributions.

*Relation to M2:* Important framing for §21. M2's chronological arm shows a shift in **outcome
prevalence** (0.0565 → 0.0826 across the 2016 cut), i.e. a shift in P(Y|X) and/or P(Y) — **not** pure
covariate shift. Weighted conformal would not fix M2's problem, and saying why (with this citation)
demonstrates the alternative was considered rather than overlooked. Reviewers of a conformal paper
routinely ask "why not weighted conformal?"

### 2.3 Conformal prediction applied to clinical risk

> Yang, M., Chen, H., Hu, W., Mischi, M., Shan, C., Li, J., Long, X., & Liu, C. (2024). **Development and
> Validation of an Interpretable Conformal Predictor to Predict Sepsis Mortality Risk: Retrospective
> Cohort Study.** *Journal of Medical Internet Research*, 26, e50369. **Tier B (peer-reviewed journal,
> retrospective, two cohorts with external validation).**

Sepsis-3 cohort from **MIMIC-IV** (20,933 patients), externally validated on eICU-CRD (10,362).
Internal AUC 0.858. Critically: the authors use **Mondrian (class-conditional) conformal prediction
specifically because the data are imbalanced**, so the error rate is controlled *within each class*. At
90% confidence: 60.2% single correct predictions, 29.4% flagged uncertain for clinician review, error
rate 10.5% vs 34.6% without CP; external validation 11.8% vs 43.8%.

*Relation to M2 — the most consequential clinical citation in this document.* This is (a) the strongest
available demonstration that conformal prediction is deployable on MIMIC-IV clinical risk, (b) **prior
art that already made M2's §17/§21.5 "future work" choice**, on the same database, for the same reason,
and (c) a demonstration that the abstain/uncertain framing is clinically legible. It both supports M2's
premise and **partially anticipates M2's proposed fix**. M2 must cite this and position the Mondrian
extension as "known necessary, applied here to the streaming/adaptive setting," not as a novel idea.

> Vazquez, J., & Facelli, J. C. (2022). **Conformal Prediction in Clinical Medical Sciences.** *Journal
> of Healthcare Informatics Research*, 6(3), 241–252. **Tier C (scoping review).**

SCOPUS-based scoping review; identified only **14** papers using conformal prediction in clinical
applications at the time. Notes that most were done in isolation, without practicing-clinician input,
and without comparisons across approaches.

*Relation to M2:* Establishes that clinical conformal prediction was a thin literature as of 2022 —
useful for motivating M2's contribution. **Handle with care on two counts:** it is a 2022 snapshot and
the field has grown since (Yang et al. 2024, the sepsis diagnosis work below), so citing "only 14
papers" in 2026 without qualification would be a stale-review error. And its critique — no clinician
input, no cross-approach comparison — applies to M2 as well: M2 has no clinician-facing evaluation.

> Dalal, S., Khaleghi Ardabili, A., & Bonavia, A. S. (2025). **Time-series deep learning and conformal
> prediction for improved sepsis diagnosis in primarily non-ICU hospitalized patients.** *Computers in
> Biology and Medicine*, 193, 110497. DOI: 10.1016/j.compbiomed.2025.110497 (ScienceDirect PII
> S0010482525008480; medRxiv preprint 2024.11.21.24317716; PubMed 40450820). **Tier B, VERIFIED
> 2026-09-19** — authors confirmed via Penn State's institutional repository (pure.psu.edu) and
> cross-checked against the ScienceDirect/PubMed records.

MIMIC-IV-trained sepsis prediction (83,813 patients) with an external eICU-CRD validation; conformal
framework used to reduce false positives; reported ~57% specificity improvement in external
validation.

*Relation to M2:* A good demonstration that conformal machinery on MIMIC-IV vitals yields an
operationally meaningful alarm-burden reduction — the clinical "so what" M2 currently lacks.

> Wang, X., Albahra, S., Liang, F., Duggal, A., & Dweik, R. A. (2025). **Conformal Prediction in
> Clinical Artificial Intelligence: Enhancing Model Reliability and Interpretability.** *CHEST*.
> PubMed 41672654 (DOI segment S0012-3692(25)05184-0). **Tier B, VERIFIED 2026-09-19** — authors
> confirmed via PubMed and the CHEST journal page: Xiaofeng Wang and Faming Liang (Dept. of
> Quantitative Health Sciences, Cleveland Clinic, and Dept. of Statistics, Purdue University), Samer
> Albahra (Center for Diagnostics and AI, Cleveland Clinic), Abhijit Duggal and Raed A. Dweik (Dept.
> of Pulmonary and Critical Care Medicine, Cleveland Clinic).

*Relation to M2:* A clinician-audience framing citation for why conformal prediction specifically
(not just calibration generally) belongs in a clinical-AI paper — useful for the introduction at a
clinical venue like JBHI/AIM.

---

## 3. Online / adaptive conformal prediction under distribution shift *(most thorough section — this is M2's primary claimed contribution)*

### 3.1 The ECI citation — VERIFIED, with a title correction the dossier needs

> Wu, J., Hu, D., Bao, Y., Xia, S.-T., & Zou, C. (2025). **Error-quantified Conformal Inference for Time
> Series.** *International Conference on Learning Representations (ICLR 2025).* arXiv:2502.00818.
> Code: github.com/creator-xi/Error-quantified-Conformal-Inference. **Tier A.**

ECI smooths the quantile loss so the online update receives a *continuous, magnitude-aware* feedback
signal — the distance between the non-conformity score and the current threshold — instead of the binary
miscoverage indicator used by ACI-family methods. Establishes a long-term coverage guarantee under
**arbitrary dependence and distribution shift**.

*Corrections the dossier requires:*
1. **The exact title is "Error-quantified Conformal Inference for Time Series."** The dossier (§1, §4,
   §21.1) drops "for Time Series" throughout. Fix everywhere before any citation is generated.
2. Authors are Wu, Hu, Bao, Xia, Zou — the dossier records none. Add them.
3. The venue/year (ICLR 2025) as cited is correct.
4. CLAUDE.md §2 lists the ECI repo as a mandatory vendored baseline and the dossier confirms
   `lr = 0.05` is on "the ECI authors' own grid" (§5) — consistent with the released code. Good.

*Relation to M2:* The method is correctly identified and correctly attributed. M2's use of ECI on
binary clinical risk scores is a genuine domain transfer — the paper's own experiments are time-series
forecasting, not clinical classification.

### 3.2 The lineage ECI sits in — M2 must situate it, not present it as free-standing

> Gibbs, I., & Candès, E. J. (2021). **Adaptive Conformal Inference Under Distribution Shift.**
> *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*, 1660–1672. arXiv:2106.00170.
> **Tier A.**

ACI: models distribution shift as an online learning problem in a single parameter (the effective
miscoverage level), continuously re-estimated from observed miscoverage. Achieves the target coverage
frequency over long horizons **with no assumptions on the data-generating distribution**, wrapping any
black-box point or quantile predictor.

*Relation to M2 — load-bearing for the novelty assessment.* ACI is the direct ancestor of ECI and its
binary-feedback update is exactly what ECI replaces. **ACI already establishes, theoretically and
empirically, that split conformal loses coverage under shift while an adaptive online method recovers
it.** M2's §21 result (MAPIE 0.8837, ECI 0.9008) is therefore *not* a novel demonstration of the
mechanism — it is a demonstration in a new setting. See §3.5 below and Gaps & Risks G2.

> Zaffran, M., Féron, O., Goude, Y., Josse, J., & Dieuleveut, A. (2022). **Adaptive Conformal Predictions
> for Time Series.** *Proceedings of the 39th International Conference on Machine Learning (ICML 2022)*,
> PMLR 162, 25834–25866. arXiv:2202.07282. **Tier A.**
> *(Note: the author list is Zaffran, Féron, Goude, Josse, Dieuleveut — verified. Any draft listing
> "Romano" among these authors is wrong.)*

Argues ACI is a sound procedure for time series with general dependency, and — critically for M2 —
studies the **sensitivity of ACI's behavior to its step size γ**, proposing AgACI, a parameter-free
variant built on online expert aggregation to remove that choice.

*Relation to M2 — directly relevant to §5.* M2's ECI learning-rate sweep (lr ∈ {0.05, 0.1, 0.5, 1.0},
width penalty rising monotonically to +75.8% at lr = 1.0) rediscovers, for ECI, the step-size
sensitivity that Zaffran et al. characterized for ACI. That makes M2's sweep a *corroboration in a new
method*, which is worth reporting — but M2 must cite this rather than present step-size sensitivity as
its own discovery. Zaffran et al. also supplies the obvious reviewer question: **why not use a
parameter-free aggregation (AgACI-style) instead of hand-selecting lr = 0.05?** M2 should have an answer.

> Bhatnagar, A., Wang, H., Xiong, C., & Bai, Y. (2023). **Improved Online Conformal Prediction via
> Strongly Adaptive Online Learning.** *Proceedings of the 40th International Conference on Machine
> Learning (ICML 2023)*, PMLR 202. arXiv:2302.07869. **Tier A.**

SAOCP/SF-OGD: minimizes *strongly adaptive regret* — worst-case regret over every interval of a fixed
length — arguing that standard regret minimization is insufficient when the distribution changes
arbitrarily over time.

*Relation to M2:* A second serious ACI successor, contemporaneous with and competitive against ECI.
Establishes that "adaptive online conformal" is a crowded methods space by 2023. M2 chose ECI per the
roadmap; the manuscript should say **why ECI rather than SAOCP or conformal PID**, or at minimum
acknowledge that the choice was not benchmarked against them.

> Angelopoulos, A. N., Candès, E. J., & Tibshirani, R. J. (2023). **Conformal PID Control for Time Series
> Prediction.** *Advances in Neural Information Processing Systems 36 (NeurIPS 2023).* **Tier A.**

Recasts online conformal as control: proportional/integral/derivative terms let the procedure
prospectively model conformal scores and adapt to seasonality, trend and general distribution shift.
Validated on COVID-19 death forecasting (improving on the CDC ensemble's coverage), electricity demand,
market returns, temperature.

*Relation to M2:* The strongest general-purpose competitor to ECI. Its "scorecaster" component
explicitly targets *predictable* shift, which is what M2's slow year-over-year prevalence drift looks
like. A reviewer familiar with this literature will ask why it was not a comparator.

> Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). **Conformal prediction beyond
> exchangeability.** *The Annals of Statistics*, 51(2), 816–845. DOI: 10.1214/23-AOS2276.
> arXiv:2202.13415. **Tier A.**

Non-exchangeable conformal prediction: reweights the calibration set to handle violated exchangeability,
with coverage-gap bounds expressed in terms of the total-variation distance between the shifted and
exchangeable settings.

*Relation to M2:* Gives the **theoretical reason** M2's §21 result had to happen. Split conformal's
guarantee is conditional on exchangeability; a past→future split breaks it; the coverage gap is bounded
by the degree of non-exchangeability. M2 measured a 1.63pp gap. This citation converts that from "we
observed undercoverage" to "we observed the predicted-in-theory coverage gap, and quantified it on
clinical data" — a stronger and more honest framing.

### 3.3 Clinical/EHR temporal shift — prior art that M2's §21 design rests on

> Nestor, B., McDermott, M. B. A., Boag, W., Berner, G., Naumann, T., Hughes, M. C., Goldenberg, A., &
> Ghassemi, M. (2019). **Feature Robustness in Non-stationary Health Records: Caveats to Deployable
> Model Performance in Common Clinical Machine Learning Tasks.** *Proceedings of the 4th Machine
> Learning for Healthcare Conference (MLHC 2019)*, PMLR 106, 381–405. arXiv:1908.00690. **Tier A/B
> (peer-reviewed conference, PMLR).**

Shows that public EHR benchmarks (MIMIC-III) are **"time agnostic"** — records are assigned to train or
test without regard to date of care — so they cannot assess year-to-year generalization. All
state-of-the-art models tested degraded when trained on historical data and evaluated on future data,
notably around a 2008 system-wide record-keeping change.

*Relation to M2 — the best possible external justification for §21 existing at all.* M2's §7 item 4 and
§21.1 say the frozen random split "cannot exhibit the temporal non-stationarity ECI exists to handle."
Nestor et al. made exactly that critique of exactly this data family six years earlier. Cite it as the
motivation for the auxiliary chronological arm; it turns a self-imposed extra experiment into a response
to a documented field-level methodological problem. It also **anticipates the discrimination half** of
M2's §21.3 result (AUROC 0.9219 random → 0.9184 chronological).

> Guo, L. L., Pfohl, S. R., Fries, J., Johnson, A. E. W., Posada, J., Aftandilian, C., Shah, N., & Sung, L.
> (2022). **Evaluation of domain generalization and adaptation on improving model robustness to temporal
> dataset shift in clinical medicine.** *Scientific Reports*, 12, 2726. **Tier B.**

MIMIC-IV ICU patients spanning 2008–2019, binned into year groups; measures temporal shift across
mortality, long-LOS, sepsis and invasive-ventilation tasks (largest shift: sepsis, AUROC −0.090). Finds
domain-generalization and unsupervised-domain-adaptation algorithms **failed** to beat plain ERM,
concluding other approaches are needed.

*Relation to M2 — two distinct uses, both valuable.* (1) It independently establishes that temporal
shift in MIMIC-IV is real and material, corroborating M2's §21.3 prevalence shift (0.0565 → 0.0826).
(2) It is **negative evidence about the alternative family of fixes**: representation-learning
approaches to temporal shift did not work on this data. That is a genuine argument for M2's approach —
adaptive *calibration* rather than shift-robust *representation* — and it is far stronger than asserting
the choice. It also confirms that year-group binning of MIMIC-IV is standard practice (see §6).

### 3.4 What the literature already establishes about split conformal failing under shift

Consolidated, so the novelty claim is calibrated honestly:

| Claim | Already established by | Strength |
|---|---|---|
| Split conformal's validity requires exchangeability | Lei et al. 2018 (JASA); Vovk et al. 2005 | Definitional — this is the theorem's own hypothesis |
| Violating exchangeability produces a bounded coverage gap | Barber et al. 2023 (Ann. Statist.) | Theoretical, tight |
| An adaptive online update recovers long-run coverage under arbitrary shift | Gibbs & Candès 2021 (NeurIPS) | Theoretical + empirical |
| Adaptive online conformal pays for coverage in interval width | Gibbs & Candès 2021; Zaffran et al. 2022 | Well replicated across methods |
| Step size / learning rate governs the width–coverage trade-off | Zaffran et al. 2022 (ICML) | Empirical, characterized in detail |
| ECI's magnitude-aware update improves on binary feedback | Wu et al. 2025 (ICLR) | Single paper, authors' own benchmarks |
| Clinical EHR data is temporally non-stationary and models decay past→future | Nestor et al. 2019; Guo et al. 2022 | Two independent groups, two datasets |

Every qualitative element of M2's §21 result is separately established in the literature above. What is
**not** in the table — and is therefore M2's actual defensible novelty — is the *conjunction*: an
adaptive online conformal method evaluated on **ED vital-sign risk scores under a real MIMIC-IV
past→future split, with class-conditional coverage reported alongside marginal coverage.** I found no
paper doing that.

### 3.5 Novelty verdict on §21 — stated plainly

**M2's §21 finding (MAPIE 0.8837 vs ECI 0.9008 under a chronological MIMIC-IV split) is a replication of
a known mechanism in a new clinical setting, not a novel demonstration that split conformal fails under
shift.** That mechanism is the explicit motivating premise of Gibbs & Candès (2021) and is bounded
theoretically by Barber et al. (2023).

This is not fatal, and the module should not soften it — it should reframe around what actually *is*
novel and hold to that:

1. **Domain-transfer evidence.** ECI was published with time-series forecasting experiments. Applying it
   to binary clinical risk scores on ED vitals, with a real past→future split on a public credentialed
   dataset, is a genuine and citable contribution. It is a *validation* contribution, and clinical
   informatics venues (JBHI, AIM, ICHI) value those more than ML venues do — which matters for the
   module's stated target venues.
2. **Effect size on real clinical shift.** The literature says "split conformal degrades." M2 says *by
   1.63 percentage points, on a ~50% relative prevalence increase across a 2016 cut, with non-overlapping
   5-seed coverage distributions (MAPIE [0.8829, 0.8849], ECI [0.8999, 0.9013]).* That quantification on
   real clinical data is new and is the most defensible sentence in the module.
3. **The dual-split design is itself a methodological result.** M2 runs the *identical conformal code* on
   a random split (no separation) and a chronological split (clean separation). That is a controlled
   demonstration that **the evaluation split determines whether an adaptive conformal method appears to
   have any value at all.** The dossier already recognizes this in §4 ("that is a property of the
   evaluation design, not a finding about ECI"). Given Nestor et al.'s finding that EHR benchmarks are
   routinely time-agnostic, this is a warning the clinical-conformal literature genuinely needs — and it
   is the framing I would build the paper's novelty claim on.
4. **§3.1's decalibration result is an independent, underrated finding.** Across four configs including
   two severely decalibrated ones (ECE up to ~0.18, mean predicted probability ~4× the base rate),
   empirical coverage stayed at ~90%. **Coverage is not evidence of sound uncertainty quantification.**
   I did not find this stated as a headline result anywhere in the clinical conformal literature, and
   given Vazquez & Facelli's observation that clinical CP papers rarely compare approaches, it is likely
   to be a real contribution. Do not let it stay buried in a config-selection subsection.

---

## 4. Class-conditional / Mondrian conformal prediction for imbalanced outcomes

*Context: the dossier's most serious self-identified weakness. §17 reports class-1 coverage 0.0759
(MAPIE) / 0.1037 (ECI) at a 90% marginal target; §21.5 reports 0.0929 / 0.1503 in the chronological arm;
§7 item 2 and §10 both name a Mondrian construction as the fix.*

### 4.1 Foundations

> Vovk, V., Lindsay, D., Nouretdinov, I., & Gammerman, A. (2003). **Mondrian Confidence Machine.**
> Technical report, Royal Holloway, University of London. (Available: alrw.net/old/04.pdf.)
> **Tier D/E (technical report — not peer-reviewed, but it is the primary origin and is universally
> cited as such).**

Introduces the Mondrian taxonomy: an online predictor emitting prediction sets at significance levels
specified *per category of example*, subsuming TCM and ICM as special cases. Motivated explicitly by
asymmetric false-positive/false-negative risk and conditional inference.

*Relation to M2:* This is the origin citation for the class-conditional construction M2 proposes as
future work. **Grading note worth carrying:** it is a technical report, not a refereed paper. Standard
practice is to cite it alongside Vovk et al. (2005) *Algorithmic Learning in a Random World*, which is
the refereed-monograph treatment. Do not cite the tech report alone in a Q1 submission.

> Vovk, V. (2012). **Conditional validity of inductive conformal predictors.** *Proceedings of the
> Asian Conference on Machine Learning (ACML 2012)*, PMLR 25:475–490. arXiv:1209.2673. **Tier A/B.**
> **VERIFIED, 2026-09-19** — confirmed directly against the PMLR proceedings page
> (proceedings.mlr.press/v25/vovk12.html), pages 475–490. An extended journal version also exists:
> *Machine Learning* 92, 349–376 (2013), DOI 10.1007/s10994-013-5355-6 — cite the PMLR conference
> version for the original result or the *Machine Learning* version if a journal citation is
> preferred; both are the same paper's two publication venues, not different works.

Formalizes the varieties of conditional validity for inductive conformal predictors and the
modifications needed to attain them.

*Relation to M2:* Supplies the vocabulary M2's §7 item 2 needs — *marginal* vs *label-conditional*
validity — and establishes that marginal-only coverage is a recognized, named limitation, not an
oversight peculiar to this module.

### 4.2 Class-conditional CP under imbalance — directly on M2's weakness

> Löfström, T., Boström, H., Linusson, H., & Johansson, U. (2015). **Bias reduction through conditional
> conformal prediction.** *Intelligent Data Analysis*, 19(6), 1355–1375. DOI: 10.3233/IDA-150786.
> **Tier B.**

Introduces label-conditional conformal prediction (LCCP), giving a per-class error bound. Frames it as
a *bias* problem: on class-imbalanced data, learners predict the majority class more often than its
relative frequency warrants, and marginal conformal prediction inherits that bias.

*Relation to M2 — the single closest methodological precedent for M2's proposed fix.* It establishes
that (a) the failure mode M2 measured is a **known, named, decade-old** phenomenon, and (b) the fix is
known and characterized. M2 should present §17/§21.5 as *"we measured, in a clinical ED cohort at a ~6%
base rate, the label-conditional bias that Löfström et al. characterized"* — which is a legitimate
empirical contribution — rather than as a discovery.

> Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I., & Tibshirani, R. J. (2023). **Class-Conditional
> Conformal Prediction with Many Classes.** *Advances in Neural Information Processing Systems 36
> (NeurIPS 2023).* arXiv:2306.09335. **Tier A.**

Clustered conformal prediction: groups classes with similar conformal-score distributions and calibrates
at cluster level, because exact class-conditional calibration **fails when there is little labeled data
per class**. Evaluated on four image datasets with up to 1000 classes.

*Relation to M2 — supports the fix but flags the real obstacle.* M2 has only 2 classes, so clustering is
unnecessary; but the paper's core diagnosis is the binding constraint for M2 too. Class-conditional
calibration needs enough *positive* calibration examples to estimate a class-1 quantile at the 90% level.
M2's calibration fold is a small slice of a 5.9%-positive cohort, so the positive-class calibration set
may be too small for a stable quantile. **Before implementing Mondrian, compute the positive count in
the calibration fold and check it against the finite-sample requirement (roughly n ≥ 1/α − 1, i.e. ≥ 9
positives for 90%, and realistically far more for a stable estimate).** This is an actionable next step,
not just a citation.

> Shi, Y., Ghosh, S., Belkhouja, T., Doppa, J. R., & Yan, Y. (2024). **Conformal Prediction for Class-wise
> Coverage via Augmented Label Rank Calibration.** *Advances in Neural Information Processing Systems 37
> (NeurIPS 2024).* arXiv:2406.06818. **Tier A.**

RC3P: guarantees class-conditional coverage while avoiding the prediction-set inflation that naive
class-wise thresholding causes, by applying the class-wise threshold selectively to a subset of classes
and augmenting with label rank. Reports ~26% average reduction in set size versus existing
class-conditional methods.

*Relation to M2:* The current state of the art on the **cost** side of the Mondrian fix. Relevant because
M2's §3.1 established that conformal procedures buy coverage with width — a Mondrian construction that
restores class-1 coverage by emitting uselessly wide intervals would not be a clinical improvement. If
M2 implements the fix, RC3P is the efficiency comparator, and "we restored class-1 coverage at a
tolerable set-size cost" is a much better result than "we restored class-1 coverage."

> Yang, M., et al. (2024). *JMIR* 26:e50369 — **see §2.3.** Repeated here because it is the **only
> verified example found of Mondrian CP applied to imbalanced clinical risk on MIMIC-IV.** Directly
> anticipates M2's proposed fix in M2's own dataset family. This is the citation M2 cannot omit.

### 4.3 Recent grey literature — flagged, weighted low, not to be leaned on

Two 2026 arXiv preprints surfaced making exactly M2's argument (marginal CP under-covers the minority
class; class-conditional calibration restores it) in other domains:

- **[Tier D]** Tursunbadalov, M., & Tursunbadalov, M. (2026). *A Quiet Failure in Calibrated Virtual
  Screening: Marginal Conformal Prediction Under-Covers the Minority Class, and a Class-Conditional Fix
  Recovers It.* arXiv:2607.06605. Reports ~4.2% minority coverage under a 90% global guarantee, with a
  conservation relation between minority undercoverage and majority overcoverage.
  **Author affiliation is a secondary school ("Champions College Prep"); not peer-reviewed. Verified to
  exist; NOT verified for correctness. Do not cite as authority.**
- **[Tier D]** Singh, M., Srikantha, A., & Lakhanpal, S. (2026). **Cost-Sensitive Conformal Prediction
  and Human-in-the-Loop Abstention for Imbalanced High-Stakes Decision Support: A Multi-Domain
  Benchmark.** arXiv:2607.27143 (posted 2026-07-29). **VERIFIED TO EXIST, 2026-09-19** — authors and
  abstract confirmed via arXiv's own listing. Reports Mondrian CP restoring minority coverage by an
  average of 61.7 percentage points over marginal CP across 15 imbalanced tabular datasets, 7 models,
  3 calibration techniques, 10 seeds (3,150 runs). **Still Tier D: arXiv preprint, not peer-reviewed,
  authors' institutional affiliation not established in this pass. Existence and content now verified;
  peer-review status is not — do not upgrade its tier on that basis alone.**
- **[Tier D/B?]** **Class-Conditional Conformal Prediction for Reliable Anomaly Detection Under
  Extreme Class Imbalance.** *Machine Learning and Knowledge Extraction (MAKE)*, MDPI, 8(2), article
  70, 2026. DOI: 10.3390/make8070190. ISSN 2504-4990. **VERIFIED, 2026-09-19** — journal identity
  confirmed (MAKE = "Machine Learning and Knowledge Extraction," ISSN 2504-4990, ed. Andreas
  Holzinger, current 2026 Impact Factor 8.4 — a real, indexed, peer-reviewed MDPI journal, not a
  predatory or throwaway venue). Reports 52.94% → 90.59% anomaly coverage at a 1:345 imbalance ratio
  via class-conditional (Mondrian) calibration. Author names were not resolved in this pass — do not
  cite without confirming them from the DOI record directly.

*Why these are in the notes at all:* they are **independent corroboration that the phenomenon M2
measured is general across domains** — a useful thing to know internally. *Why they are graded low:*
three preprints agreeing does not outweigh one refereed source, and at least one has no discernible
methodological review. **Do not cite any of them in a Q1 submission.** Löfström et al. 2015, Ding et al.
2023, Shi et al. 2024 and Yang et al. 2024 cover the same ground with real evidentiary weight.

### 4.3.5 Online group-conditional CP without a tuned learning rate — found 2026-09-19, directly relevant to M2's own methodology gap

> Bharti, B., Pal, A., Teneggi, J., & Sulam, J. (2026). **Parameter-Free and Group Conditional Online
> Conformal Prediction.** arXiv:2606.00419 (posted 2026-05-29; Johns Hopkins University / Amazon
> Responsible AI). **Tier B/C (preprint, methods paper). VERIFIED, 2026-09-19** — authors and abstract
> confirmed via arXiv's own listing.

Introduces POGO, described as the first *parameter-free* algorithm for group-conditional online
conformal prediction — combining online conformal prediction, parameter-free online learning, and
multivariate betting to provide rigorous per-group coverage guarantees **without tuning a learning
rate**, robust to adversarial/unknown shift.

*Relation to M2 — a direct methodological gap, not just adjacent work.* M2's own ECI-lr selection
(`ECI_LR=0.05` for Track A; the two-stage `{0.01, 0.05, 0.1}` sweep for the Hybrid pipeline and, this
session, the v2 pipeline) is exactly the kind of tuned-learning-rate step POGO exists to eliminate.
This is not a desk-reject risk the way G2's shift-robustness overlap is — M2 does not claim learning-
rate-free online CP as a contribution — but it is the single most relevant "you should at least discuss
why you didn't use this" citation found in this pass, and a legitimate "future work" pointer: a
parameter-free construction would remove the one remaining tuned hyperparameter in M2's primary
contribution's own calibration procedure. Recommended: cite in the manuscript's limitations/future-work
discussion of the ECI-lr sweep, not as a competing result.

### 4.3.6 Online × label-conditional: prior-art check for Mondrian-ECI — searched 2026-09-20

Purpose: decide whether an *online* label-conditional construction (`M2_Findings_Dossier.md` §24.3b) occupies an empty
cell. **Method and its limits:** one web-search pass (six queries) plus direct reading of two papers. This is an
absence-of-evidence result from web search, **not** a database search; IEEE Xplore, PubMed and ACM DL were not
queried systematically. Each row below states how far it was actually verified.

| Work | Online? | Conditions on | Verification |
|---|---|---|---|
| Gibbs & Candès (2021); Wu et al. (2025) ECI; Angelopoulos et al. (2023); Bhatnagar et al. (2023) | yes | marginal only | already catalogued in §3 |
| Bastani, Gupta, Jung, Noarov, Ramalingam & Roth (2022), *Practical Adversarial Multivalid Conformal Prediction*, NeurIPS 2022 (arXiv:2206.01067) | yes | **arbitrary, possibly intersecting subsets of the feature space** | authors and venue confirmed from search results quoting the abstract; body not read |
| Bharti, Pal, Teneggi & Sulam (2026), POGO (arXiv:2606.00419, v4) | yes | grouping functions **c_j(X) ∈ [0,1] of the covariates** — its Eq. (1) | **read directly, 2026-09-20** (PDF pp. 1–3): groups are functions of X only, not of Y |
| Jung, Noarov, Ramalingam & Roth (arXiv:2209.15145), *Batch Multivalid Conformal Prediction* | no (batch) | feature-space subsets | title/authors from search results; **venue not verified** |
| Podkopaev & Ramdas (2021), *Distribution-free uncertainty quantification for classification under label shift*, UAI 2021 (PMLR 161) | no (batch) | label *shift*, not label-conditional coverage | title/venue from search results; body not read |
| Bairaktari, Wu & Wu (2025), *Kandinsky Conformal Prediction: Beyond Class- and Covariate-Conditional Coverage*, arXiv:2502.17264 v2 | no (batch) | class- and covariate-conditional | characterized from a fetch summary only; **venue not verified, paper not read in full** |
| Millar, Román, Gouripeddi & Facelli (2026), *Reliable Uncertainty Under Class Imbalance and Distribution Shift: Class-Conditional Conformal Prediction of Multiple Sclerosis*, medRxiv | no (static split-CP) | class-conditional; shift is scanner/field-strength, not temporal | abstract read via PubMed; **preprint — Tier D, peer-review status unknown, not citable as peer-reviewed** |

**Finding.** Within this search, no work combines *online/adaptive* updating with *label-conditional* (true-label) guarantees.
The online group-conditional line (MVP, POGO) is defined on covariate functions and so cannot express label
conditioning by construction; every label-conditional work found is static. The cell appeared empty. Treat as
provisional until a database search confirms it.

**A comparator this leaves undone, and a reviewer will ask for.** POGO's `c_j(X)` may be *any* function of X — including
**bins of the model's own predicted risk `p(x)`**, a covariate function that correlates strongly with the label. Online
group-conditional conformal with score-bin groups is therefore a feature-space *proxy* for label conditioning, and it
could plausibly recover much of the minority-class coverage. The only feature-space grouping M2 has tested is triage
acuity (§23 blocker #2 (iii): class-1 coverage 0.333), which is a weak proxy. **Score-binned group-conditional conformal,
static and online, is an untested baseline and must be run before any novelty claim for Mondrian-ECI.**

*Relation to the ECI-lr finding.* POGO's stated purpose — removing the tuned learning rate — bears directly on the Mondrian-ECI
pilot, whose first NO-GO was caused by lr selection failing to transfer across the train→test prevalence shift (§24.3b).
A parameter-free variant is the natural remedy and the natural future-work pointer.

### 4.4 Section 4 verdict

The dossier's §7 item 2 diagnosis is **correct, well-supported, and — importantly — not novel.** The
class-conditional coverage failure of marginal conformal prediction under imbalance has been named and
solved since at least 2015, formalized since 2003/2012, refined at NeurIPS in 2023 and 2024, and already
applied to imbalanced clinical risk on MIMIC-IV in 2024. **The dossier is right that this is the module's
most serious weakness; the literature makes it worse, because a reviewer can point to an off-the-shelf
fix that a 2024 JMIR paper already applied to the same database.** See Gaps & Risks G3.

---

## 5. NEWS2 and clinical early-warning-score validation

### 5.1 The score itself

> Royal College of Physicians (2017). **National Early Warning Score (NEWS) 2: Standardising the
> assessment of acute-illness severity in the NHS.** Updated report of a working party. London: RCP.
> **Tier E (standards body — authoritative for the score's definition, not evidence of performance).**

*Relation to M2:* The definitional citation for NEWS2 wherever M2 refers to the score (§3, §19.3's NEWS2
risk-group subgroups). M2 reads `score_NEWS2` precomputed from `test.csv` via the vendored
mimic4ed-benchmark pipeline, so **the manuscript should cite both the RCP definition and the benchmark
repo's implementation of it** — and should state that the score was not independently re-derived.

### 5.2 Validation and critique — why NEWS2 is the right comparator, and what it is worth

> Mhd Yunin, N. S., & Tan, T. L. (2025). **National Early Warning Score 2 (NEWS2) as a prognostic tool
> for adult patients in emergency department: A retrospective observational study.** *PLOS One*, 20(6),
> e0326058. DOI: 10.1371/journal.pone.0326058. **Tier B. VERIFIED, 2026-09-19** — exactly two authors
> confirmed (Nor Safiahani Mhd Yunin and Toh Leong Tan, Dept. of Emergency Medicine, Faculty of
> Medicine, Universiti Kebangsaan Malaysia), cross-checked against the PLOS One publisher page, PMC
> (PMC12169516), and UKM's own institutional author profile. n=1906.

ED cohort. NEWS2 AUROC 0.71 (95% CI 0.68–0.73) for hospitalization, 0.75 (0.67–0.82) for critical care
admission, 0.86 (0.79–0.92) for in-hospital mortality. Subgroup analysis: reduced utility for mortality
prediction in sepsis (AUROC 0.66) and COVID-19 patients.

*Relation to M2 — and a discrepancy M2 must address head-on.* M2 reports NEWS2 at **AUROC 0.6227** on
its composite outcome (in-hospital mortality **or** direct ICU transfer within 12h). This literature
reports 0.75–0.86 for closely related ED outcomes. **A reviewer will read M2's 0.6227 as a suspiciously
weak baseline and suspect the comparison is rigged.** M2 has good defenses, but they must be *stated
explicitly in the results text, not left implicit*:
- M2's outcome is a **composite** (mortality OR ICU transfer), horizon-bounded at 12h — not the
  all-cause in-hospital mortality on which NEWS2 scores 0.86.
- M2's cohort is unfiltered adult ED arrivals (n = 83,332 test), far larger and less selected than
  typical single-center validation cohorts.
- M2's `score_NEWS2` comes from the vendored benchmark's computation on MIMIC-IV-ED triage/vitals
  fields, with whatever coverage and imputation that entails.
This section is **the strongest reason in this whole document to run a sensitivity check**: report NEWS2
AUROC against in-hospital mortality alone, alongside the composite. If it lands near 0.8, the composite
explains the gap and the baseline is vindicated; if it stays near 0.62, there is an implementation
question to answer *before* submission, not after review.

> Gerry, S., Bonnici, T., Birks, J., Kirtley, S., Virdee, P. S., Watkinson, P. J., & Collins, G. S.
> (2020). **Early warning scores for detecting deterioration in adult hospital patients: systematic
> review and critical appraisal of methodology.** *BMJ*, 369, m1501. DOI: 10.1136/bmj.m1501. **Tier A
> (systematic review in a top general medical journal). VERIFIED, 2026-09-19** — complete author list
> is these 7 names (Gary S. Collins was the previously-unconfirmed 7th author); no "et al." is needed
> or correct. Confirmed via ResearchGate's indexed record of the full author byline.

Screened 13,171 references, included 95 articles (11 development-only, 23 development + external
validation, 61 external-validation-only). Critically appraises the *methodology* of the early-warning-score
literature.

*Relation to M2 — the most valuable framing citation in this section.* It licenses two things M2 needs:
(1) treating NEWS2 as the standard-of-care comparator is well founded, because the EWS literature is
large and NEWS2 is its reference point; and (2) **the methodological quality of that literature is
uneven**, which is exactly why M2's insistence on patient-level splits, pre-declared outcome definitions
(the Outcome-Definition Gate), 5-seed reporting and matched tuning budgets is a contribution to how this
comparison is *done*, not only to what it finds. For a PhD-portfolio paper, that methodological framing
is worth more than the AUROC delta.

> Guan, G., Lee, C. M. Y., Begg, S., Crombie, A., & Mnatzaganian, G. (2022). **The use of early warning
> system scores in prehospital and emergency department settings to predict clinical deterioration: A
> systematic review and meta-analysis.** *PLOS One*, 17(3), e0265559. DOI: 10.1371/journal.pone.0265559.
> PMID 35298560. **Tier A (systematic review). VERIFIED, 2026-09-19** — full author list and venue
> confirmed via PubMed/PLOS. 11,565 articles screened, 20 included; pooled DOR for 30-day mortality in
> the ED setting 4.05–6.48 across NEWS/NEWS2/MEWS/RAPS/CART cut-offs — a useful third source
> corroborating §5.2's Mhd Yunin & Tan point estimates with pooled, multi-cohort evidence.

### 5.3 Section 5 verdict

NEWS2 is a defensible and expected comparator; the literature supports using it. The exposure is not the
choice of baseline, it is **M2's unusually low NEWS2 number relative to published ED validations**, which
needs an explicit explanation or a sensitivity analysis. Note also that M2's +0.2992 AUROC over NEWS2 is
the module's least contested headline — every other headline has reversed at least once — so it is worth
protecting.

---

## 6. MIMIC-IV-ED as a research resource, and the date-shifting caveat

### 6.1 Dataset and benchmark citations

> Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. (2023). **MIMIC-IV, a freely accessible electronic
> health record dataset.** *Scientific Data*, 10, 1. DOI: 10.1038/s41597-022-01899-x.
> *(An author correction exists: Sci Data, 2023, DOI 10.1038/s41597-023-02136-9 — cite the correction
> alongside if the journal requires it.)* **Tier A/E.**

> Johnson, A., Bulgarelli, L., Pollard, T., et al. **MIMIC-IV-ED** (PhysioNet, v1.0 released 2021;
> M2 uses **v2.2**). DOI: 10.13026/77z6-9w59 (v1.0). **Tier E.**
> *Note: MIMIC-IV-ED is a PhysioNet data release, **not** a separate Scientific Data journal article —
> I searched for one and it does not exist. Cite the PhysioNet DOI for the version actually used (2.2),
> plus the MIMIC-IV Scientific Data paper for the parent database.*

> Xie, F., Zhou, J., Lee, J. W., Tan, M., Li, S., Rajnthern, L. S. O., & Liu, N. (2022). **Benchmarking
> emergency department prediction models with machine learning and public electronic health records.**
> *Scientific Data*, 9, 658. DOI: 10.1038/s41597-022-01782-9. (arXiv 2111.11017.) **Tier A/B.**

The mimic4ed-benchmark paper — the vendored repository M2 builds on per CLAUDE.md §2. Provides the
open-source ED benchmark platform, cohort construction and baseline task definitions on MIMIC-IV-ED.

*Relation to M2:* **This must be cited prominently.** It is the vendored baseline, the source of
`master_dataset.csv`, the `score_NEWS2` computation, and the comorbidity extraction. It is also the
source of the two defects M2 found and fixed (§1.1's row-level `df_master.sample(frac=0.8)` leaking
patients across train/test; the pre-`train.csv` median imputation of all 14 vitals that made §19.1's
prespecified density percentiles degenerate). Reporting those repairs as **corrections to a published,
widely-used public benchmark** is a real, citable contribution — arguably more durable than the AUROC
results — but it obliges M2 to describe them precisely and neutrally, versioned against the exact repo
state, without overclaiming their scope.

Also cited within the vendored pipeline and already correctly recorded in dossier §11.1:
> Quan, H., Sundararajan, V., Halfon, P., et al. (2005). **Coding algorithms for defining comorbidities
> in ICD-9-CM and ICD-10 administrative data.** *Medical Care*, 43(11), 1130–1139. **Tier A.**
> *(Cited directly in the vendored `comorbidities.py`; carried forward as-is, not independently
> re-verified in this pass.)* **[INHERITED — verify if it appears in the manuscript]**

### 6.2 Date shifting — this is documented, not a module discovery

**Direct finding: the ±1-year resolution limit M2 documents in §21.2 is a widely-known, officially
documented property of MIMIC-IV, not something the module worked out independently.**

The MIMIC-IV documentation (PhysioNet v2.2/v3.x; mimic.mit.edu/docs/IV/) states explicitly:
- All dates are shifted into 2100–2200, with **a single random date shift per `subject_id`**.
- `anchor_year` is the deidentified year; `anchor_year_group` is one of five real 3-year windows
  (2008–2010, 2011–2013, 2014–2016, 2017–2019, 2020–2022).
- The documented worked example is the same reconstruction M2 implements: a patient with
  `anchor_year` 2158 and `anchor_year_group` "2011-2013" had that year's hospitalizations sometime in
  2011–2013.
- The documentation states the limitation in the same terms M2 arrived at: **because the shift is
  per-patient, two patients admitted in deidentified year 2120 cannot be assumed to have been admitted
  in the same real year.**
> **Tier E (official dataset documentation — authoritative).** Cite as: MIMIC-IV documentation /
> PhysioNet project page for the version used (v2.2), plus the Johnson et al. 2023 *Scientific Data*
> paper.

*Relation to M2 — this is a correction to how §21.2 currently reads.* The dossier presents the
de-shifting recipe and its ±1-year limit as a discovery of this pass ("**That was wrong.**" ... "What
*is* recoverable is `anchor_year_group`"). That framing is accurate as a description of what happened
internally, but **in a manuscript it would misattribute a documented dataset property as a novel
methodological insight**, which a MIMIC-fluent reviewer would catch immediately and hold against the
whole paper. Rewrite as: *"following the MIMIC-IV documented reconstruction, we recover approximate
real calendar year via `anchor_year_group`; the resulting ±1-year resolution bounds our temporal
analysis."* Same content, correctly attributed, and it signals dataset fluency rather than the reverse.

*What genuinely is M2's own contribution here, and should be kept:*
- The **empirical observation** that sorting a multi-patient conformal stream by raw `intime` produces
  an arbitrary order "wearing a chronological disguise" — and that this silently corrupts an *online*
  conformal method, for which stream order is part of the method. The documentation states the shift
  property; it does not state this consequence for online conformal inference. **That is a real, novel,
  and directly transferable caution**, and it is worth a paragraph of its own.
- The observed **bucket-start pileup** (spikes at 2011/2014/2017 from encounters on a patient's own
  anchor year) — a concrete artifact of the reconstruction, quantified.
- The explicit accounting of **45,576 encounters (18,968 patients) dropped** for straddling the cut, and
  the conservative rationale for dropping them.

*Precedent for the approach:* Guo et al. (2022, *Sci Rep* 12:2726, §3.3 above) bin MIMIC-IV ICU patients
into year groups for temporal-shift analysis, and Nestor et al. (2019, MLHC) do the date-aware split on
MIMIC-III. **M2's chronological arm is methodologically conventional, and it should say so** — citing
precedent makes the design look standard rather than improvised.

---

## Gaps and Risks — what a Q1 reviewer will raise that the dossier does not yet address

Ordered by how much damage each could do at review.

**G1 — No comparison to any modern architecture for irregular time series.** *(High risk; not in the
dossier's limitations list at all.)* M2's comparison set is GRU-D vs LR / RF / XGBoost / NEWS2 — all
non-temporal. GRU-D is a 2016/2018 method with at least three well-cited successors on exactly this data
type (SeFT, ICML 2020; mTAND, ICLR 2021; Neural CDE, NeurIPS 2020) plus a 2023 ICML multimodal successor,
and mTAND explicitly claims to beat GRU-D. At JBHI/AIM this may pass with a stated justification; at
ICHI or an ML-leaning venue it likely will not. **Options, in order of cost:** (a) add mTAND as one more
arm under the same matched-tuning protocol — the most convincing fix and consistent with the project's
stated PhD-portfolio standard; (b) argue explicitly from T = 2 that attention/continuous-time methods
have almost nothing to attend over or integrate across at two timesteps, making GRU-D's decay the
appropriate minimal mechanism — defensible, cheap, and it converts a weakness into a design rationale,
but it is an argument rather than a measurement; (c) say nothing and hope — not viable at these venues.
**This gap should be added to the dossier's §7 limitations regardless of which option is chosen.**

**G2 — Prior art substantially anticipates the shift-robustness finding.** *(High risk — SHARPENED,
2026-09-19: this risk is now higher, not lower, than when first written.)* Gibbs & Candès (2021)
already establish that split conformal loses coverage under shift and that an adaptive online update
recovers it; Barber et al. (2023) bound the gap theoretically; Zaffran et al. (2022) characterize the
step-size/width trade-off M2's §5 sweep rediscovers; Nestor et al. (2019) and Guo et al. (2022)
establish clinical temporal shift. This item was written against §21, an *auxiliary* chronological
arm on the T=2 cohort. **§24.3 has since promoted the identical result to the module's headline
finding** — the v2 pipeline's chronological split is now primary, so "MAPIE undercovers while ECI
holds nominal coverage" is the paper's lead claim, not a secondary robustness check. That raises the
stakes on the framing risk considerably: a headline built on "we show split conformal fails under
shift" invites a desk-level "this is Gibbs & Candès 2021" at the point where a reviewer's attention is
highest. **The defensible novelty for the v2-era manuscript is specifically:**
1. the **quantified clinical effect size** on a real chronological split of a real ED cohort (not a
   synthetic or theoretical shift) — 87.7–88.5% vs. 90.1–90.2% coverage, a measured 1.5–2.3 point gap;
2. **class-conditional validity under that same shift** (§24.3's Mondrian-LAC result, §23 blocker #2)
   — Gibbs & Candès and the shift-robustness literature address marginal coverage under shift; they do
   not simultaneously address class-conditional coverage under shift on an imbalanced clinical outcome,
   which is the combination M2 actually measures.
Lead with (1)+(2) together, explicitly cast as a domain-transfer-with-quantified-effect-size
contribution, not a claim that split conformal's shift-sensitivity is a new phenomenon.

**REVISED 2026-09-20 — claim (2) as written above no longer holds and must not be used.** It said class-conditional validity *under that same shift* was
demonstrated by Mondrian-LAC. With a defect in the static Mondrian-LAC comparator fixed (Deviation 17), the corrected static method is **not** valid under this
cohort's shift (class-0 coverage 0.871–0.889; valid, 0.902, on the exchangeable T=2 split). Pre-registered comparisons then found that (a) periodic recalibration of
the static method restores both classes to ≈0.90 with smaller sets than the online construction (Mondrian-ECI: NOT NEEDED), and (b) periodic recalibration of
ordinary split-conformal matches ECI's *marginal* validity with tighter intervals (ECI: NOT NEEDED against the cheap remedy). What survives of (1)+(2) is the
**measured characterization** — minority-class collapse of marginal methods including ECI (class-1 0.03–0.08), label conditioning fixing it where score-bin
conditioning does not (0.09–0.29), and static calibration's loss of validity under real shift being cheaply repairable — as an empirical result with effect sizes,
not as a method contribution. See `M2_Findings_Dossier.md` §24.3b, §24.3c and the revision block in §24.4. See §3.5 for
the four novelty claims that survive contact with this literature; claims 3 (the split-design
demonstration) and 4 (coverage-is-not-calibration) remain the strongest and should stay the most
prominent framing in any manuscript draft, now doing double duty as the headline's own defense.

**G3 — The class-conditional fix already exists, and has already been applied to MIMIC-IV.** *(High
risk.)* The dossier's §7 item 2 correctly names the weakness and correctly names Mondrian as the fix.
The literature makes this worse, not better: Löfström et al. (2015) named and solved it; Ding et al.
(2023) and Shi et al. (2024) refined it at NeurIPS; and **Yang et al. (2024, JMIR) applied Mondrian CP
to imbalanced sepsis mortality on MIMIC-IV specifically because of class imbalance.** A reviewer can
therefore say: *the fix is off-the-shelf, and a 2024 paper already used it on your database — why is it
future work?* Two honest responses, both requiring action: **implement it** (highest-value single item
in the module — it converts the largest weakness into a result, and can be reported against RC3P-style
efficiency), or **state explicitly why it was out of scope** and pre-check feasibility. On feasibility:
per Ding et al., class-conditional calibration needs enough positive calibration examples for a stable
class-1 quantile; **compute the positive count in M2's calibration fold now** — if it is marginal, that
is itself a reportable finding ("class-conditional calibration is infeasible at this base rate and fold
size without a larger calibration allocation"), and a far better sentence than silence.

**G4 — The NEWS2 baseline number is well below published ED validations.** *(Medium-high risk; not in
the dossier's limitations.)* M2 reports 0.6227; the ED literature reports 0.71–0.86 for related
outcomes. The composite outcome and cohort breadth plausibly explain it, but a reviewer sees a weak
baseline propping up a +0.2992 headline. **Run and report NEWS2 against in-hospital mortality alone as
a sensitivity check.** Cheap (NEWS2 is deterministic and read from `test.csv`), and it either vindicates
the baseline or surfaces an implementation issue before review rather than after.

**G5 — The date-shifting caveat is presented as a discovery.** *(Medium risk, cheap to fix.)* §21.2's
framing would misattribute documented MIMIC-IV behavior. Rewrite per §6.2 and cite the documentation.
Keep the genuinely novel part — that raw-`intime` ordering silently corrupts an *online* conformal
stream — as the contribution, because it is one.

**G6 — Citation hygiene items to fix before any prose is written.** *(Low risk individually, cumulative
credibility damage if left.)*
- ECI's title is "Error-quantified Conformal Inference **for Time Series**" — the dossier truncates it
  in §1, §4 and §21.1. Authors (Wu, Hu, Bao, Xia, Zou) are recorded nowhere.
- Che et al. needs its venue recorded (*Scientific Reports* 8:6085), and its 2016 arXiv date
  acknowledged where the method's age matters.
- MIMIC-IV-ED has **no** Scientific Data paper — cite the PhysioNet DOI for **v2.2** (the version
  actually used) plus Johnson et al. 2023 for the parent database.
- **RESOLVED, 2026-09-19 — all 8 items previously marked [UNVERIFIED]/[PARTIALLY UNVERIFIED] are now
  verified** (Phase 7 of the session's "Clearing the Runway to the Manuscript" plan): Vovk et al.
  edition (2nd ed., 2022, Springer, DOI 10.1007/978-3-031-06649-8); Vovk 2012 venue (PMLR 25:475–490,
  confirmed against the proceedings page); the *Computers in Biology and Medicine* sepsis paper's
  authors (Dalal, Khaleghi Ardabili & Bonavia); the CHEST 2025 piece's authors (Wang, Albahra, Liang,
  Duggal & Dweik) and full title; the PLOS One NEWS2 author list (exactly 2: Mhd Yunin & Tan); the
  Gerry et al. full author list (7 names, including the previously-missing Gary S. Collins); arXiv
  2607.27143's authors (Singh, Srikantha & Lakhanpal — existence and content verified, peer-review
  status still Tier D); DOI 10.3390/make8070190's venue (MAKE, MDPI, ISSN 2504-4990, a real indexed
  journal — its authors were NOT resolved and still require confirmation before citing). The PLOS One
  EWS meta-analysis (PMID 35298560) mentioned as a possible third source was also resolved (Guan, Lee,
  Begg, Crombie & Mnatzaganian, 2022) and is now recommended as a real citation, not just a maybe.
- **Do not cite the Tier-D 2026 preprints (arXiv 2607.06605, arXiv 2607.27143) as authority in §4.3.**
  Existence-verified now, but still preprints, not peer-reviewed. Three preprints agreeing is not
  evidence; the refereed sources (Löfström 2015, Ding 2023, Shi 2024, Yang 2024) cover the same claim
  with real evidentiary weight.

**G7 — No clinician-facing or decision-relevant evaluation.** *(Medium risk at clinical venues
specifically.)* Vazquez & Facelli's (2022) central critique of the clinical CP literature is that
studies are done in isolation, without practicing-clinician input, and without cross-approach
comparison. M2 is currently exposed on all three. The cheapest partial answer is a **decision-relevant
framing** of the conformal output — Yang et al. (2024) report the fraction of patients flagged
"uncertain, requiring clinician review" and the error rate among confidently-predicted patients. M2 has
the raw material for the same table (MAPIE set sizes, empty-set rates, per-class coverage) and does not
currently present it that way. At JBHI/AIM, that reframing is likely worth more than another decimal
place of AUROC.

**G8 — RESOLVED, 2026-09-19 (Phase 7 of the session's "Clearing the Runway to the Manuscript" plan).**
*(Was: "not searched: IEEE Xplore, PubMed/Embase systematically, ACM DL... run one targeted search
restricted to JBHI/AIM/ICHI before the manuscript is written.")* Six targeted web searches were run this
session, restricted to and naming each target venue plus the exact problem combination (GRU-D +
conformal prediction + ED deterioration + MIMIC-IV): "IEEE JBHI conformal prediction emergency
department deterioration," "Artificial Intelligence in Medicine ... conformal prediction ICU emergency
deterioration GRU-D," "ICHI ... conformal prediction clinical time series," "IEEE Journal of Biomedical
and Health Informatics conformal prediction vital signs deterioration MIMIC," and "GRU-D emergency
department mortality ICU transfer prediction conformal MIMIC-IV."

**No directly competing paper was found** — no JBHI/AIM/ICHI (or any other venue) publication combining
GRU-D (or an equivalent decay-based recurrent architecture), conformal/adaptive-conformal prediction,
and MIMIC-IV-ED deterioration prediction under a chronological evaluation was located. This is a
disclosed **negative result of a real, executed search** (this pass genuinely searched and did not
find), not the prior "not searched" gap — a meaningfully different and stronger claim for a manuscript's
related-work section to make. Two adjacent-but-non-competing papers surfaced and were evaluated
directly (not just noted): **THCM-CAL** (Zhang et al., EMNLP 2025) applies conformal calibration to
multi-label ICD coding from clinical text/causal graphs — a different modality and problem, not a
competitor. **POGO** (Bharti et al., arXiv 2026) is a general parameter-free group-conditional online CP
method, not a clinical application — see §4.3.5, added as a genuinely useful methodological citation for
M2's own tuned-lr limitation, not as competing prior art.

**Caveat, stated because it is real:** this was still a web-search pass, not a formal IEEE Xplore/
PubMed/ACM DL database query with systematic-review-grade recall — the risk of a missed paper is lower
than before this pass but not zero. If venue selection narrows further (e.g., to JBHI specifically), one
additional IEEE-Xplore-native search restricted to that venue's own archive is still worth 15 minutes
before final submission.

---

## Consolidated map: what this literature does to each M2 claim

| M2 claim / finding | Literature verdict | Strength of M2's position |
|---|---|---|
| GRU-D is an appropriate architecture for irregular ED vitals | Supported by Che et al. 2018, but **dated**; three successors exist and one claims to beat it | **Weak — needs a comparator or an explicit T=2 argument (G1)** |
| Tuned XGBoost ≥ tuned GRU-D (§3.2, §18.1) | **Strongly supported** — Grinsztajn et al. 2022 predicts exactly this and validates the matched-budget method | **Strong — reframe as confirmation, not disappointment** |
| MAPIE = split conformal with the LAC score (§17) | Verified against Sadinle et al. 2019 + MAPIE source; empty sets are an intended LAC property | **Strong — and the empty-set finding is now correctly diagnosable** |
| Split conformal undercovers under temporal shift (§21) | **Already established** by Gibbs & Candès 2021, bounded by Barber et al. 2023 | **Replication, not discovery — reframe per §3.5 (G2)** |
| ECI holds nominal coverage under shift, paying in width | Consistent with the whole ACI lineage; ECI's own guarantee covers it | Supported; **novelty is the clinical setting + quantified effect size** |
| ECI lr governs the width–coverage trade-off (§5) | Anticipated by Zaffran et al. 2022 for ACI | Corroboration; **cite, don't claim; and answer "why not a parameter-free variant?"** |
| Marginal coverage ≠ class-conditional coverage (§17, §21.5) | **Well established** since Vovk 2003/2012, Löfström et al. 2015; already applied on MIMIC-IV by Yang et al. 2024 | **Correct diagnosis, non-novel; the fix is off-the-shelf (G3)** |
| Coverage alone cannot show UQ is sound (§3.1) | **No prior statement of this as a headline found** in the clinical CP literature | **Strongest under-promoted claim in the module — promote it** |
| The evaluation split determines whether adaptive CP shows any value (§4 vs §21) | Motivated by Nestor et al. 2019's "time agnostic benchmark" critique; not previously demonstrated for conformal methods | **Genuinely novel and the best available framing for the contribution** |
| NEWS2 comparator at AUROC 0.6227 | Below published ED validations (0.71–0.86) | **Explain or sensitivity-test (G4)** |
| ±1-year limit from `anchor_year_group` | **Documented MIMIC-IV property**, not a discovery | **Reattribute (G5); keep the online-stream-ordering consequence, which is novel** |
| Repairs to the vendored mimic4ed-benchmark (patient leakage, pre-split imputation) | No prior report found | **Real, durable contribution — report precisely and version it** |


---

## 7. Delayed-feedback conformal prediction, and clinical conformal risk under temporal shift — venue-restricted and combination search (2026-09-21)

*Internal reference, not manuscript prose. Requested question: does prior work combine **clinical risk prediction + temporal shift + conformal prediction + delayed feedback**? Every entry below was opened at its source
(arXiv abstract page, PMC or publisher page) on 2026-09-21 and its title, authors, year and venue/status were read from that page, per CLAUDE.md §1.10. Nothing here is recalled from memory.*

### 7.1 Answer to the combination question

**No prior work combining all four was found.** Each pair exists; the four together do not, in this search. The closest matches, and how each falls short:

| Work (verified) | clinical | temporal shift | conformal | delayed feedback | class-conditional | real data |
|---|---|---|---|---|---|---|
| El Halabi & Brandt, arXiv:2609.07251 (7 Sep 2026) | no | yes (simulated) | yes (ACI) | **yes** | not addressed | **no (simulation only)** |
| Kwon & Kim, *Sci Rep* 2026, doi:10.1038/s41598-026-40637-w | yes (sepsis, PhysioNet 2019) | yes (later-admitted OOD cohort) | yes (static split CP; Mondrian by gender) | no (no online update) | no (gender strata) | yes |
| Dalal, Khaleghi Ardabili & Bonavia, *Comput Biol Med* 2025, doi:10.1016/j.compbiomed.2025.110497 | yes (sepsis, MIMIC-IV → eICU) | external validation only | yes (trust-set p-values, static) | no | no | yes |
| Wang, Zecchin & Simeone, arXiv:2503.10345 (13 Mar 2025) | no | yes | yes (IM-OCP) | intermittent (missing) feedback, not delay | not addressed | not stated |
| Skalse, Pona, Simeone & Paoletti, arXiv:2608.07139 (7 Aug 2026) | no | yes | yes (OCPQ, label queries) | no (labels queried, not delayed) | not addressed | not stated |

Position of this module's experiment (dossier §24.3d): an **empirical** delayed-feedback study on **real clinical data**, **label-conditional** as well as marginal, against a **periodic-recalibration baseline** — none of which the delayed-feedback ACI paper covers. It should be cited and positioned against, not ignored: it is two weeks old at the time of writing.

### 7.2 Entries

- **El Halabi, L., & Brandt, A. (2026). *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic.* arXiv:2609.07251, submitted 7 September 2026.** Shows that the τ-delayed ACI recursion decomposes into τ interleaved ACI-like sequences, gives a finite-sample long-run coverage bound with explicit dependence on τ, and introduces the delay-to-memory ratio r = τ/L (L: the time scale over which the signal driving non-exchangeability decays). Simulation only (AR(1), GARCH(1,1), Markov switching, abrupt mean/variance shifts); no clinical application; class-conditional coverage not addressed. Status: arXiv preprint (no venue stated on the page). **Relevance:** this is the theoretical companion to the integrator-windup behaviour observed at long lag (§24.3d): its interleaving decomposition is a natural place to look for why marginal ECI's radius overshoots on stale feedback. Note the object differs: ACI adjusts the miscoverage *level*; ECI adjusts a quantile *radius*; the mapping is an analogy until checked.
- **Wang, B., Zecchin, M., & Simeone, O. (2025). *Mirror Online Conformal Prediction with Intermittent Feedback.* arXiv:2503.10345, submitted 13 March 2025.** IM-OCP: runtime calibration under potentially intermittent feedback, with deterministic long-term coverage and sub-linear regret. Intermittent (missing) feedback, not delayed; no clinical application on the page. Status: arXiv preprint.
- **Skalse, J., Pona, E., Simeone, O., & Paoletti, N. (2026). *Online Conformal Prediction Beyond Feedback.* arXiv:2608.07139, submitted 7 August 2026.** OCPQ: the learner chooses per round between a prediction set and querying the true label; O(T^{2/3}) expected regret with expected coverage ≥ β − O(T^{−1/3}) while querying a small fraction of rounds. Delay, clinical use and class-conditional coverage not addressed on the abstract page. Status: arXiv preprint. **Relevance:** the label-query framing is the deployment-side counterpart of "labels arrive late or not at all".
- **Vovk, V., Gammerman, A., & Shafer, G. (2022). *Algorithmic Learning in a Random World* (2nd ed.). Springer.** Cited by the `online-cp` package documentation (Hallberg Szabadváry, Löfström & Matela; PyPI) for the result that asymptotic validity is retained for invariant conformal predictors with a **fixed-lag** delay (equally spaced feedback, the strongest guarantee) and with a "lazy teacher" whose feedback arrives on more than a logarithmic fraction of trials. I read this through the package's documentation, not the book: **the theorem numbers quoted there (ALRW2 §3.3, Thm 3.7–3.11) are unverified against the book and must be checked before citation.** This is the exchangeable-online-CP lineage; ECI/ACI are the non-exchangeable adaptive lineage.
- **Kwon, H., & Kim, D.-J. (2026). *Conformal selective prediction with cost aware deferral for safe clinical triage under distribution shift.* Scientific Reports, doi:10.1038/s41598-026-40637-w.** PhysioNet 2019 Sepsis Challenge (2,155 patients, 7.0% prevalence); split, Mondrian (gender-stratified) and importance-weighted conformal; deferral threshold chosen once on calibration data; later-admitted patients form the OOD test cohort. No online or adaptive update. **Relevance:** a clinical, temporal-shift, static conformal comparator in a peer-reviewed venue (quartile not checked); it stratifies by gender, not by the outcome, so the minority-class collapse studied here is not examined.
- **Dalal, S., Khaleghi Ardabili, A., & Bonavia, A. S. (2025). *Time-series deep learning and conformal prediction for improved sepsis diagnosis in primarily non-ICU hospitalized patients.* Computers in Biology and Medicine, doi:10.1016/j.compbiomed.2025.110497.** Deep time-series sepsis model with a nonconformity (trust-set distance) framework applied post hoc; static; external validation on eICU-CRD; no class-specific or temporal-shift coverage analysis. **Relevance:** an example of conformal calibration on a clinical time-series model (peer-reviewed venue; quartile not checked) that reports indeterminacy (~17% in eICU-CRD) rather than coverage under shift.
- **Gibbs, I., & Candès, E. J. (2022; revised 2023). *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts.* arXiv:2208.08401.** Adaptive step-size selection for online conformal inference without knowing the shift rate (applications: stock volatility, COVID-19 cases). The arXiv page does not state a journal; publication venue unverified here. Complements the 2021 ACI paper already in §3.
- **Kladny, K.-R., Schölkopf, B., Koch, L., Baumgartner, C. F., & Muehlebach, M. (2025). *A Critical Perspective on Finite Sample Conformal Prediction Theory in Medical Applications.* arXiv:2512.14727, submitted 9 December 2025.** Argues that finite-sample guarantees are of practical use only with sufficiently large calibration sets, a concern for data-scarce medical settings. The abstract does not treat temporal shift or delayed labels. Status: arXiv preprint. Relevant to the calibration-set sizes used here (calibration folds of ~12k).
- **Kim, G. Y. E., Corbin, C. K., Grolleau, F., Baiocchi, M., & Chen, J. H. (2025). *Monitoring Strategies for Continuous Evaluation of Deployed Clinical Prediction Models.* Journal of Biomedical Informatics.** Addresses **label modification by intervention** (alerts change outcomes, so post-deployment labels are biased) and proposes adherence- and sampling-weighted monitoring; inverse probability weighting, no conformal prediction. **Relevance — a limitation this module does not model:** once a risk score triggers care, the observed outcome that ECI or recalibration would learn from is itself altered. That feedback loop is a further reason label availability in deployment is not the retrospective label used here.

### 7.3 Venue-restricted search (JBHI, Artificial Intelligence in Medicine, ICHI) — what was and was not done

Searches were run on 2026-09-21 with the queries: online conformal prediction + delayed feedback; conformal + clinical risk + temporal shift + delayed labels; conformal + emergency-department deterioration + MIMIC-IV-ED; and venue-named queries for IEEE JBHI, Artificial Intelligence in Medicine and IEEE ICHI (also with IEEE Xplore, PubMed and ScienceDirect domain filters).
**No paper in any of the three target venues was found that combines clinical risk prediction, temporal shift, conformal prediction and delayed feedback.** This is a *negative search result with limited power*: the search tool does not index the three venues' full tables of contents, returned mostly arXiv and PMC items for the venue-named queries, and cannot substitute for a database search.
**The paper author should run the venue search directly** (IEEE Xplore for JBHI and ICHI; ScienceDirect/Elsevier for AIM) with: `("conformal prediction" OR "conformal inference") AND (clinical OR patient OR ICU OR "emergency department") AND ("distribution shift" OR "temporal shift" OR "dataset shift" OR drift OR "delayed feedback" OR "label delay")`, restricted to 2022–2026, before the Introduction and contribution claims are finalised.

### 7.4 Effect on M2's claims

- The **delayed-feedback dimension is not novel as a theory topic** (Vovk-lineage fixed-lag results; the 2026 delayed-feedback ACI paper). The module's contribution on it is empirical and clinical: a real-data lag sweep, label-conditional as well as marginal, against a periodic-recalibration baseline, with a pre-registered negative reading.
- **Do not claim** that no delayed-feedback conformal work exists; claim that none was found that is clinical, real-data, label-conditional and compared to recalibration.
- Two additions to the limitations that come from this search and are not yet in the manuscript skeleton: label modification by intervention (Kim et al. 2025), and calibration-set-size sensitivity of finite-sample guarantees in medical data (Kladny et al. 2025).
