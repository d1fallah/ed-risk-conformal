# Phase 2 tuning budget (fixed before any tuning run; plan item 2.3)

Written 2026-10-07 02:05 local time, before `p2_*` tuning scripts were run. Cohort files (`p2_1_cohorts.py`) had been built; no model had been
trained on them.

**Rule.** Every tuned model receives the same number of configurations (**4**) and the same three tuning seeds {42, 123, 7}. Selection is on
the `validation` development subset (22,070 eligible 2 h stays; 24,026 triage stays), by **validation AUROC on the primary outcome Y_L**,
averaged over the three tuning seeds (ties: first configuration in the order listed). This is the metric the original tuning scripts used
(`val_auroc` in `m2_mtan_tuning_matched.py`, `m2_hybrid_head_tuning_v2.py`). After selection each model is refit on the `training` subset with
its selected configuration for the five final seeds {42, 123, 7, 456, 789}. Neural models stop on validation BCE loss, as in the original
shared loop (`_minibatch_early_stop_loop`, patience 10, at most 100 epochs, batch 128). The test period is never used for tuning or selection.
Deterministic learners (logistic regression) ignore the tuning seeds; this is stated, not hidden.

| Model | Landmark | Configurations (order = tie-break order) | Fixed settings |
|---|---|---|---|
| Logistic regression | triage, 2 h | L2 penalty C in {0.01, 0.1, 1, 10} | median imputation fitted on `training`, missing indicators, standardisation |
| XGBoost (triage), timing-aware XGBoost (2 h) | triage, 2 h | (max_depth, learning_rate) in {(4, 0.05), (6, 0.05), (4, 0.1), (6, 0.1)} | up to 1,000 trees, early stopping 50 rounds on validation AUC, subsample 0.8, colsample_bytree 0.8, min_child_weight 1, reg_lambda 1, random_state = seed |
| Temporal-Only GRU-D encoder | 2 h | (hidden, lr) in {(32, 1e-2), (64, 1e-2), (32, 3e-3), (64, 3e-3)} | dropout 0.2, weight decay 1e-4, classifier width 32 (the original architecture) |
| Hybrid GRU-D, frozen encoder (ablation) | 2 h | (static_hidden, static_dropout) in {(32, 0.1), (32, 0.3), (64, 0.1), (64, 0.3)} on the encoder of the same seed | fusion dropout 0.1, lr 1e-3, the original head routine |
| Hybrid GRU-D end-to-end (primary, R6) | 2 h | (hidden, lr) in {(32, 1e-2), (64, 1e-2), (32, 3e-3), (64, 3e-3)} | static_hidden 64, static_dropout 0.3, fusion dropout 0.1 (the original Hybrid head), weight decay 1e-4, joint training of encoder, static branch and head |

Where the original pipeline tuned with fewer or different budgets (the Temporal-Only GRU-D v2 used one fixed configuration; XGBoost-Matched
used a 2-parameter override search), the end-of-stay results are unchanged and are not re-tuned here; the Phase 2 comparisons use this
equal budget for all models of the landmark task.

Sequence input length at the 2 h landmark: 7 ED slots + the triage slot (the smallest slot count covering 99.5% of eligible stays; 1,321 of
343,153 eligible stays are truncated to their latest 7 rows). Vital-sign and static scalers are the existing ones fitted on the
development `training` subjects (never refitted on validation or test).

**Erratum (2026-10-08).** The line above that gives the writing time as 02:05 is inaccurate. The file system records this file as created at 01:59:24 on 2026-10-07 and not modified since; the first tuning logs were created at 02:00:56 (`p2_2_classical.log`) and 02:02:35 (`p2_3_e2e.log`). The budget therefore preceded every tuning run by about one and a half minutes. The text above is left unchanged.
