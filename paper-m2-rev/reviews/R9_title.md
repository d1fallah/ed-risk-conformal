# Round 9: Title (2026-10-07)

Same rubric and weights as round 5. Scorer: the drafting assistant.

Before: "Predicting Deterioration in the Emergency Department Two Hours After Arrival: Workflow Shortcuts and Conformal Prediction Sets" (16 words; 8.5).
After: "Predicting Deterioration Two Hours After Emergency Department Arrival: Avoiding Leakage and Evaluating Conformal Prediction Sets" (15 words).

Reference: recent JBHI titles from OpenAlex (source S2495854775, 2023 onward, 14 titles on related topics): median 13 words, interquartile range about 11 to 16,
57% with a colon; capitalization follows IEEE title case (After, With, From).

Candidates considered and rejected:
- "... Shortcut Learning and Conformal Prediction Sets": overclaims. The paper shows that workflow quantities alone reproduce the discrimination of the
  end-of-stay task, not that the models learned them.
- "Predicting ICU Admission or Death Two Hours After Emergency Department Arrival ...": names the outcome exactly (TRIPOD+AI item 1) but needs 17 words or more.
- "... Data Leakage and Conformal Prediction Sets": could be read as saying the two-hour models leak.
- "... Input Timing and Conformal Prediction Sets" (14 words): accurate and plain, but does not convey the leakage finding. Kept as the fallback.

## Title: 10.0 on this rubric (was 8.5)
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Names task and setting | 3 | 1 | predicting deterioration, emergency department |
| Conveys the key design idea | 2 | 1 | two hours after arrival; avoiding leakage |
| Length within JBHI norm | 1 | 1 | 15 words |
| Plain words, no coined jargon | 2 | 1 | "leakage" is the established term (Kapoor et al.; the keywords) |
| Accurate to the content | 2 | 1 | the landmark design restricts inputs to what is recorded by two hours; the conformal sets are evaluated, not proposed |

Not weighed by the rubric: "deterioration" is less specific than the outcome (ICU admission or in-hospital death within 24 h), which the abstract defines; the
word "leakage" does not appear in the abstract, only in the keywords and the main text.
