# Round 8: Introduction (2026-10-07)

Same rubric and weights as round 5. Scorer: the drafting assistant.

## Arithmetic correction to round 5
All rubric tables of rounds 5 to 7 were recomputed by script (score = 10 x sum(W x credit) / sum(W)). The Introduction weights sum to 9.5, not 10, and round 5
reported the raw sum (8.9) instead of the normalized score (9.3). No other section was affected; Related Work (round 4, weights 9.0) was normalized correctly (8.6).

## What changed
- The fourth paragraph (gap, aim and three steps, about 200 words) is split in two: the gap from the recorded search, then the aim, the intended use, and the approach.
- Intended use stated (TRIPOD+AI item 3b): a score for ED clinicians two hours after arrival, between triage and disposition; checklist updated.
- Contributions rewritten as concrete findings that match the Results: (1) workflow quantities reproduce most of the end-of-stay discrimination above chance
  ((0.843 - 0.5) / (0.937 - 0.5), about 78%), also within the evaluable acuity levels; (2) timing-aware trees slightly better than the recurrent model (C1), and the
  recurrent model's gain over the sequence-only model comes from the static profile (C3); (3) marginal sets rarely contain the deterioration label, with the
  operating characteristics of class-conditional sets and online versus periodic recalibration.
- One term for the shortcut inputs: "workflow quantities", defined in the bullet; the gap sentence no longer uses "monitoring intensity".
- 846 words.

## Introduction: 10.0 on this rubric (corrected round 5: 9.3)
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Clinical problem with a cited stake | 1.5 | 1 | |
| Current practice and prior approaches | 1 | 1 | |
| Specific gap tied to evidence | 1.5 | 1 | recorded search |
| One explicit aim | 1 | 1 | |
| Approach summarized | 1 | 1 | |
| Contributions concrete and matched to evidence | 1.5 | 1 | each bullet states a finding that the Results support |
| Honest scope | 0.5 | 1 | |
| Roadmap | 0.5 | 1 | |
| Language and paragraphing | 1 | 1 | paragraphs of 3 to 6 sentences; one term per concept |

The rubric is now saturated, so it no longer separates a good Introduction from an excellent one. Points a strict reviewer could still raise, which the rubric
does not weigh: the Introduction defines ECI although the paper does not propose a method; at about 850 words it is at the upper end of the exemplars; and the
scorer is the drafting assistant.
