# Discussion cut: current text beside the proposed text (for approval; nothing applied)

Numbers in both columns are `\RV...` macros, so every value stays tied to the frozen evidence. The Limitations subsection is kept word for word, as you decided (any cut falls on the other subsections). The extended version keeps the current Discussion unchanged. Word counts are those of the LaTeX source.

## Opening
**Current (144 words)**

```latex
This study separated two questions that the end-of-stay benchmark task conflates: how well a deterioration model discriminates, and when its score could be used.
Three findings follow. First, the end-of-stay task cannot support a claim of early warning: it uses rows charted after departure, and quantities that describe how the stay was monitored add to what is
known at arrival. Second, at a stated prediction time two hours after arrival, the recurrent model did not improve on gradient-boosted trees with timing features, and its
gain over a model of the vital-sign sequence alone came mainly from the static profile. Third, conformal sets that meet their marginal coverage rarely contain the
deterioration label for the patients who deteriorate, and the false-alarm burden of class-conditional sets that do contain it is the precision of the score at that sensitivity, which the coverage guarantee does not reveal.
```

**Proposed (144 words)**

```latex
This study separated two questions that the end-of-stay benchmark task conflates: how well a deterioration model discriminates, and when its score could be used.
Three findings follow. First, the end-of-stay task cannot support a claim of early warning: it uses rows charted after departure, and quantities that describe how the stay was monitored add to what is
known at arrival. Second, at a stated prediction time two hours after arrival, the recurrent model did not improve on gradient-boosted trees with timing features, and its
gain over a model of the vital-sign sequence alone came mainly from the static profile. Third, conformal sets that meet their marginal coverage rarely contain the
deterioration label for the patients who deteriorate, and the false-alarm burden of class-conditional sets that do contain it is the precision of the score at that sensitivity, which the coverage guarantee does not reveal.
```

## What the End-of-Stay Task Measures
**Current (390 words)**

```latex
A model that reads the last ten vital-sign rows of a stay is scoring the stay after most of it has happened. The shortcut diagnostic puts a number on what that allows. ED
duration, the number of measurements, and the per-vital counts, none of which is a physiological value, reached an AUROC of \RVShortAAuroc{} against
\RVEosHybridAuroc{} for the full model, but that figure should not be read as the leaked share of the full model, because much of it restates what is known at arrival: arrival information alone reached \RVIncArr{}, and acuity alone \RVAcuOverallAcu{}. The increment that monitoring adds is smaller, \RVIncWorkOverArr{} (\RVIncWorkOverArrLo{} to \RVIncWorkOverArrHi{}) when the workflow quantities are added to arrival information, of which ED duration alone contributes \RVIncDurOverArr{}; the remainder of the full model rests on later vital-sign values. A plausible reading of the increment is that clinicians monitor sicker patients more often and keep admitted patients in the department for
longer, so that monitoring intensity records the clinical judgment that the model is meant to anticipate. The workflow quantities kept discriminating within acuity
strata (\RVAcuHighWork{} for acuity 1 or 2 and \RVAcuMidWork{} for acuity 3), so they do not merely restate the triage acuity. That is consistent with judgment exercised
after triage, but equally with monitoring that becomes denser because the patient is deteriorating, and the design cannot separate the two. What the diagnostic does
establish is that the end-of-stay AUROC combines arrival information, monitoring, and later physiology, and cannot be used to argue for early warning. The audited task comes from an earlier analysis by the author, so this is a demonstration of a design that the benchmark pipeline makes easy and not a correction of published studies. A restricted-input benchmark on the same database, limited to the first hour
of presentation, described the loss of discrimination when inputs were reduced to vital signs as a graceful degradation~\cite{solaiman2026earlydeterioration}. The two designs differ in
the window, the outcome, and the cohort, so the figures are not comparable, but both point away from the whole-stay record as the basis for an early prediction. The
same concern has been raised for inpatient models that use diagnostic codes finalized after discharge~\cite{ramadan2025icdleakage}; the shortcut here comes from the
monitoring record rather than from coding, but the remedy is the same: tie every input to the time at which it exists.
```

**Proposed (239 words)**

```latex
A model that reads the last ten vital-sign rows of a stay is scoring the stay after most of it has happened. ED
duration, the number of measurements, and the per-vital counts, none of which is a physiological value, reached an AUROC of \RVShortAAuroc{} against
\RVEosHybridAuroc{} for the full model, but that figure is not the leaked share of the full model, because much of it restates what is known at arrival: arrival information alone reached \RVIncArr{}. The increment that monitoring adds is smaller, \RVIncWorkOverArr{} (\RVIncWorkOverArrLo{} to \RVIncWorkOverArrHi{}), of which ED duration alone contributes \RVIncDurOverArr{}. A plausible reading is that clinicians monitor sicker patients more often and keep admitted patients in the department for
longer, so that monitoring intensity records the clinical judgment that the model is meant to anticipate; it is equally consistent with monitoring that becomes denser because the patient is deteriorating, and the design cannot separate the two. What the diagnostic does
establish is that the end-of-stay AUROC combines arrival information, monitoring, and later physiology, and cannot be used to argue for early warning. The audited task comes from an earlier analysis by the author, so this is a demonstration of a design that the benchmark pipeline makes easy and not a correction of published studies. The same concern has been raised for inpatient models that use diagnostic codes finalized after discharge~\cite{ramadan2025icdleakage}; the remedy is to tie every input to the time at which it exists.
```

## Discrimination at the Landmark
**Current (762 words)**

```latex
Restricting the inputs to what had been recorded two hours after arrival lowered the AUROC of the best model from \RVEosHybridAuroc{} to about \RVAurocXgbT{}, which is
the figure that bears on an early prediction. Among the landmark models the differences were small, and the sequence-free models were competitive: timing-aware trees
were better than the primary recurrent model by \RVTreesOverHybrid{} (95\% CI \RVTreesOverHybridLo{} to \RVTreesOverHybridHi{}), which is detectable with
\RVHtwoTestEligible{} test stays and small in absolute terms, and the recurrent model exceeded logistic regression (difference \RVCTwoDelta{}). The largest single gain came from
static features (\RVCThreeDelta{}). This agrees with reports that gradient-boosted trees remain strong on tabular data~\cite{grinsztajn2022treebased}, with the MIMIC-IV-ED benchmark, in which deep sequence models did not outperform simpler ones~\cite{xie2022mimic4ed}, and with the
finding that models of triage data outperform a model of the triage acuity alone~\cite{raita2019triage}; most of the discrimination is
already available at arrival. On the same stays and the same outcome, the two-hour trees exceeded the arrival trees by \RVCommonTreesGain{} (\RVCommonTreesGainLo{} to \RVCommonTreesGainHi{}), so the two hours add an amount that is detectable and small (Supplementary Table~\ref{S-tab:arrival}). Training the encoder jointly with the static branch did not help. These statements apply to
this cohort, this outcome, and a tuning budget of four configurations per model, in which the selected recurrent configuration lay at a corner of the grid (the larger
hidden size and the smaller step size); a larger search could narrow or reverse the differences.

Discrimination fell at landmarks later than two hours, but this was not information being lost. The eligible population at a later landmark consists of patients who
stayed longer without an event, and on one fixed set of such patients the AUROC rose with every landmark, from \RVFixTreesOne{} at one hour to \RVFixTreesSix{} at
six hours for the trees. The decline in the main sweep therefore reflects a harder population, not a loss of signal. For the patients who remain, the record
accumulated after two hours adds information, but it contains both later vital-sign values and more measurements, and the design does not
separate the two. For these patients the AUROC stayed below that of the two-hour cohort as a whole (\RVAurocXgbT{}), so discrimination is lowest for the
patients who stay longest, who are also the patients for whom a later prediction would be made. The timing features of the tree ensemble, which count and space the measurements made by $L$, are legitimate inputs at $L$,
but they still describe monitoring practice, so their value may not transfer to a hospital that charts differently. The concern is concrete in these data: in the later years of the training period the first ED vital-sign row came to repeat the triage measurement (\RVRecEqTriC{} of stays in 2011, \RVRecEqTriTest{} in the test period), so a count of rows by $L$ means something different in the two periods. For logistic regression and the trees this concern did not materialize: refitting without the copied rows changed the AUROC by \RVDdTreesChange{} for the trees and \RVDdLrChange{} for logistic regression (Supplementary Table~\ref{S-tab:dedup}), removing the timing features lowered it by about the same amount in both models, and the advantage of the trees over logistic regression stayed near \RVOrigTreesOverLr{} in every variant. The timing features therefore contribute to both models and do not explain the advantage of the trees. The recurrent models were not refitted, but the stored Hybrid GRU-D models, which take the mask and elapsed-time inputs that encode the same recording practice, changed by \RVGrudHybridChange{} in AUROC when the copied rows were removed from the test stays, well below the \RVCOneDelta{} of the pre-specified contrast with the trees. The Temporal-Only models, which use vital signs and their timing alone, changed by \RVGrudTempChange{}. Scoring without a refit shows that the test-time rows do not distort the stored models; it does not show what models trained without the copied rows would have learned.

The three best models also separated little in usefulness: their net benefit differed by a few thousandths at a threshold of 0.10, and all exceeded treating every stay
as positive. The decision curve assumes that a score is a probability and that the threshold is the same for every clinician, and the models underestimated risk in the
upper deciles (extended version). NEWS2 performed well below every learned model and below the AUROCs of \RVRefNewsLo{} to \RVRefNewsHi{} reported for it in
another ED cohort~\cite{mhdyunin2025news2}. The difference is consistent with three features of this study: five of its seven parameters were available, a missing
parameter scores as normal, and the score was not designed for a 24-hour intensive-care or death outcome.
```

**Proposed (390 words)**

```latex
Restricting the inputs to what had been recorded two hours after arrival lowered the AUROC of the best model from \RVEosHybridAuroc{} to about \RVAurocXgbT{}, which is
the figure that bears on an early prediction. Among the landmark models the differences were small: timing-aware trees
were better than the primary recurrent model by \RVTreesOverHybrid{} (95\% CI \RVTreesOverHybridLo{} to \RVTreesOverHybridHi{}), the recurrent model exceeded logistic regression (\RVCTwoDelta{}), and the largest single gain came from
static features (\RVCThreeDelta{}). This agrees with reports that gradient-boosted trees remain strong on tabular data~\cite{grinsztajn2022treebased} and with the MIMIC-IV-ED benchmark, in which deep sequence models did not outperform simpler ones~\cite{xie2022mimic4ed}. On the same stays and the same outcome, the two-hour trees exceeded the arrival trees by \RVCommonTreesGain{} (\RVCommonTreesGainLo{} to \RVCommonTreesGainHi{}), so most of the discrimination is
already available at arrival. These statements apply to
this cohort, this outcome, and a tuning budget of four configurations per model, in which the selected recurrent configuration lay at a corner of the grid; a larger search could narrow or reverse the differences. The decline of the AUROC at landmarks later than two hours reflects a harder, shrinking population and not a loss of signal: on one fixed set of stays the AUROC of the trees rose from \RVFixTreesOne{} at one hour to \RVFixTreesSix{} at six hours (extended version).

The timing features of the tree ensemble are legitimate inputs at $L$, but they describe monitoring practice, and their value may not transfer to a hospital that charts differently. The concern is concrete in these data: in the later years of the training period the first ED vital-sign row came to repeat the triage measurement (\RVRecEqTriC{} of stays in 2011, \RVRecEqTriTest{} in the test period). For logistic regression and the trees it did not materialize: refitting without the copied rows changed the AUROC by \RVDdTreesChange{} for the trees and \RVDdLrChange{} for logistic regression (Supplementary Table~\ref{S-tab:dedup}), and the advantage of the trees over logistic regression stayed near \RVOrigTreesOverLr{} in every variant, so the timing features do not explain it. The recurrent models were not refitted, but the stored Hybrid GRU-D models changed by \RVGrudHybridChange{} in AUROC when the copied rows were removed from the test stays, well below the \RVCOneDelta{} of the pre-specified contrast with the trees; scoring without a refit does not show what models trained without the copied rows would have learned.
```

## Coverage Is Not Alarm Precision
**Current (337 words)**

```latex
Conformal prediction sets guarantee, under exchangeability, that the true label lies in the set at the stated rate. For a rare outcome, the marginal guarantee is met by
sets that almost never contain the deterioration label: class-1 coverage was \RVSetCovOneStaticLac{} for static LAC at a marginal coverage of \RVSetCovStaticLac{}. The
marginal threshold is set by the many non-events, and the model rarely gives an event a risk as high as the threshold requires (Section~\ref{sec:results-coverage}).
Calibrating within each class closes the gap, as the theory of class-conditional validity predicts~\cite{vovk2003mondrian,ding2023classconditional}.
Yang et al.\ applied class-conditional calibration to sepsis mortality for the same reason~\cite{yang2024sepsisconformal}, and a preprint on multiple sclerosis imaging
reported the same under-coverage of the minority class~\cite{millar2026classconditionalms}. What a class-conditional set adds is a guarantee on class-1 coverage, not information about precision. At a nominal level of 0.90, the singleton
set $\{1\}$ identified \RVSetSensMondEci{} of the patients with the outcome and had a PPV of \RVSetPpvMondEci{}, so about \RVSetFaMondEci{} in 100 patients received it
without the outcome, and about \RVSetAbstOneMondEci{} of the patients with the outcome received an abstention. The singleton of a static set is issued exactly for the stays whose score exceeds one threshold, so its PPV is a point on the precision--recall curve of the score: a plain threshold with the sensitivity of static Mondrian-LAC reached a PPV of \RVPrMondSame{} (the set: \RVPrMondPpv{}), and one with the sensitivity of Mondrian-ECI reached \RVPrEciSame{} (the set: \RVPrEciPpv{}; Supplementary Fig.~\ref{S-fig:pr}). The false-alarm burden is therefore the cost of that sensitivity with this model, which the coverage level selects and does not change. A hospital that reads a conformal singleton as an alarm would be reading a statement about coverage as one about precision. Fig.~\ref{fig:alpha} shows that precision rose to \RVAFivePpvMondEci{} at a nominal level of 0.95 and \RVAOnePpvMondEci{} at 0.99 only as sets became larger. Whether
such sets would be useful depends on what is done with a set that contains both labels, a decision that needs clinician input and a cost model, neither of which we
have.
```

**Proposed (251 words)**

```latex
For a rare outcome, the marginal guarantee is met by
sets that almost never contain the deterioration label: class-1 coverage was \RVSetCovOneStaticLac{} for static LAC at a marginal coverage of \RVSetCovStaticLac{}, because the
marginal threshold is set by the many non-events and the model rarely gives an event a risk as high as the threshold requires.
Calibrating within each class closes the gap, as the theory of class-conditional validity predicts~\cite{vovk2003mondrian,ding2023classconditional}, and has been applied to sepsis mortality~\cite{yang2024sepsisconformal}; a preprint on multiple sclerosis imaging
reported the same under-coverage of the minority class~\cite{millar2026classconditionalms}. What a class-conditional set adds is a guarantee on class-1 coverage, not information about precision. At a nominal level of 0.90, the singleton
set $\{1\}$ identified \RVSetSensMondEci{} of the patients with the outcome and had a PPV of \RVSetPpvMondEci{}, so about \RVSetFaMondEci{} in 100 patients received it
without the outcome. The singleton of a static set is issued exactly for the stays whose score exceeds one threshold, so a plain threshold with the sensitivity of static Mondrian-LAC reached a PPV of \RVPrMondSame{} (the set: \RVPrMondPpv{}). The false-alarm burden is therefore the cost of that sensitivity with this model, which the coverage level selects and does not change; precision rose to \RVAFivePpvMondEci{} at a nominal level of 0.95 and \RVAOnePpvMondEci{} at 0.99 only as sets became larger (Fig.~\ref{fig:alpha}). Whether
such sets would be useful depends on what is done with a set that contains both labels, a decision that needs clinician input and a cost model, neither of which we
have.
```

## Static, Adaptive, and Periodic Calibration
**Current (290 words)**

```latex
The static quantile kept marginal coverage close to the target on the later period for the primary model (\RVIntCovStatic{}; bootstrap interval \RVStaticBootHybLo{} to \RVStaticBootHybHi{}, which includes the validity threshold of 0.89), but it did not meet the threshold for every model: coverage was \RVOtherXgbStaticCov{} for the trees (\RVStaticBootXgbLo{} to \RVStaticBootXgbHi{}), below the threshold, and \RVOtherLrStaticCov{} for logistic regression. The mean predicted risk was lower than
the observed rate in the test period (\RVCalMeanPredEndToEnd{} against \RVCalObserved{}), so the later period differs from the calibration subsets in prevalence. Shifting the intercept of the score on recent labels restored calibration in the large and lowered the Brier score slightly (Supplementary Table~\ref{S-tab:intercept}), but, with the same shift applied to the calibration scores, it left the coverage of the static sets unchanged (\RVIrCovOne{} against \RVIrCovOrig{}). Correcting the shift in prevalence therefore does not change static coverage, which was already near the target for this model. A quantile recomputed on the recalibrated scores gave \RVIrCovTwo{}, as ECI (\RVIntCovEci{}) and periodic recalibration of the original score (\RVIntCovRecalTen{}) did, at wider intervals; in these data the gain from updating the quantile is small, and its value shows mainly in the drift test. Permuting the order of arrivals within each year block did not change this, so the result does not depend on the unknown within-year order of this
database. The controlled-drift test shows what static calibration loses when a shift is larger, and that ECI recovered coverage after the case-mix change where
recalibration did not. Since the drift is imposed on real predictions, it demonstrates the behavior of the methods and not a property of the hospital. Label delays of up
to 1{,}000 arrivals left both online methods valid; longer delays were not examined on the landmark task.
```

**Proposed (192 words)**

```latex
The static quantile kept marginal coverage close to the target for the primary model (\RVIntCovStatic{}; bootstrap interval \RVStaticBootHybLo{} to \RVStaticBootHybHi{}, which includes the validity threshold of 0.89), but it did not meet the threshold for every model: coverage was \RVOtherXgbStaticCov{} for the trees (\RVStaticBootXgbLo{} to \RVStaticBootXgbHi{}), below the threshold, and \RVOtherLrStaticCov{} for logistic regression. The mean predicted risk was lower than
the observed rate in the test period (\RVCalMeanPredEndToEnd{} against \RVCalObserved{}). Shifting the intercept of the score on recent labels restored calibration in the large but, with the same shift applied to the calibration scores, left the coverage of the static sets unchanged (\RVIrCovOne{} against \RVIrCovOrig{}). A quantile recomputed on the recalibrated scores gave \RVIrCovTwo{}, as ECI (\RVIntCovEci{}) and periodic recalibration (\RVIntCovRecalTen{}) did, at wider intervals; in these data the gain from updating the quantile is small, and its value shows mainly in the drift test, where ECI recovered coverage after the case-mix change and recalibration did not. Since the drift is imposed on real predictions, it demonstrates the behavior of the methods and not a property of the hospital. Label delays of up
to 1{,}000 arrivals left both online methods valid.
```

## Implications for Reporting and Practice
**Current (191 words)**

```latex
Three reporting practices follow for studies of ED deterioration models. First, a study should state the prediction time and restrict both the inputs and the
population at risk to it; an end-of-stay AUROC should be reported, if at all, next to a workflow-only baseline such as the shortcut diagnostic, which costs one logistic
regression. Second, a conformal analysis of a rare outcome should report class-specific coverage together with the PPV and sensitivity of the singleton set and the
false alarms per 100 patients, at more than one coverage level, and relate them to the precision--recall curve of the score, because the marginal guarantee constrains none of them. Third, an online calibration method should be
compared with periodic recalibration on a stream with a stated order and label delay; in our data both restored coverage, and the static quantile met the threshold for the primary model but not for the trees. For practice, class-conditional sets offer a trade-off to be chosen, not an alarm to be trusted: a department would set the coverage level from the number of
false alarms it accepts per patient correctly flagged, a choice that needs clinical input.
```

**Proposed (98 words)**

```latex
A study of an ED deterioration model should state the prediction time, restrict both the inputs and the population at risk to it, report an end-of-stay AUROC only next to a workflow-only baseline, and report class-specific coverage with the PPV and sensitivity of the singleton set and the false alarms per 100 patients at more than one coverage level. Class-conditional sets offer a trade-off to be chosen, not an alarm to be trusted: a department would set the coverage level from the number of
false alarms it accepts per patient correctly flagged, a choice that needs clinical input.
```

## Limitations

Unchanged (433 words).

Totals without Limitations: current 2114 words, proposed 1314 words; with Limitations: current 2547, proposed 1747.

## What the proposal removes from the submitted version (kept in the extended version)
- The subsection-length detail on the benchmark comparison with a restricted-input benchmark (Solaiman et al.).
- The fixed-set landmark paragraph (reduced to one sentence, so that the fall of the AUROC at later landmarks is not read as a loss of signal), the net-benefit and NEWS2 paragraph, and the sentence that training the encoder jointly did not help.
- The within-year permutation sentence and the statement that longer label delays were not examined.
- The three numbered reporting practices (reduced to two sentences).
No claim of substance is changed, so there is nothing to sync; the extended Discussion stays as it is.

## Page forecast (measured, not estimated, for the Discussion step)
A throwaway build with the proposed Discussion (and the current Introduction, Conclusion, Methods and Results) gives **10.5 pages with references** (11 pages, the last one 55% full), against 12 pages now.
The plan target of 8.5 pages is therefore **not reached by the Discussion cut alone**. Methods (2,389 words) and Results (1,532 words) are above the word targets of the plan (1,700 and 1,300), the floats take about 2.6 pages, and the 38 references about 1.3 pages.

Further levers, each with its estimated saving (words of the section files over about 1,065 words per text page; floats and references from the build):
| Lever | Saving (pages) | Running total |
|---|---|---|
| After the Discussion cut as proposed | | 10.5 |
| Introduction and Conclusion trimmed as planned (800 and 200 words) | 0.25 | 10.2 |
| Methods to 1,700 words (cut the equations' text, the NEWS2 paragraph, the reporting paragraph) | 0.65 | 9.6 |
| Results to 1,300 words | 0.2 | 9.4 |
| Table II (timing audit, full width) to the supplement, three sentences stay in Methods | 0.3 | 9.1 |
| References cut to about 33 with the shorter text | 0.15 | 8.9 |
| Discussion body (without Limitations) from 880 to 580 words | 0.3 | 8.6 |
| Fig. 6 (full width) to one column | 0.15 | 8.5 |
A forecast is uncertain by about 0.3 page, so 8.5 needs all of these and 9.0 needs roughly the first five. At 9.0 pages the cost is one charged page; at 10.5, two.

## Decisions needed
1. Approve the proposed Discussion text (or mark the paragraphs to change).
2. Which of the further levers to use, and whether 9.0 pages is acceptable for now.
