#!/usr/bin/env bash
# M2 horizon ablation (CLAUDE.md M2 required ablation: horizons 1h/6h/12h).
#
# For each horizon h, in order of decreasing signal so an uninformative arm
# cannot block the informative ones:
#   1. Temporal-Only GRU-D   -- retrained on y_<h> (5 seeds; ~15 min/seed). A
#                               per-horizon Hybrid needs a per-horizon frozen
#                               encoder: reusing the composite-outcome encoder
#                               would not be an ablation of the temporal
#                               representation.
#   2. Hybrid GRU-D          -- head on top of that horizon's frozen encoder
#   3. XGBoost-Matched       -- trained natively on y_<h>
#
# Hyperparameters are the composite-outcome tuned configs, NOT re-tuned per
# horizon (disclosed in the dossier): the head config, the XGBoost config, and
# the encoder's HIDDEN/DROPOUT/WEIGHT_DECAY/LR. No class reweighting at any
# horizon, matching the composite run's protocol (plain BCE).
#
# M2_V2_HORIZON routes every path, checkpoint and result file to a parallel
# `_hz<h>` location (see m2_v2_common.py); the primary results are never touched.
# A failed stage is logged and the script continues to the next horizon.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
LOGDIR="data/horizon_logs"
FAILS=0
mkdir -p "$LOGDIR"

for h in 12h 6h 1h; do
  echo "=== horizon $h: $(date +%H:%M:%S) ==="
  export M2_V2_HORIZON="$h"

  $PY -u Benchmark_scripts/m2_temporal_only_grud_v2.py > "$LOGDIR/temporal_only_$h.log" 2>&1
  ec=$?; echo "TEMPORAL_ONLY_EXIT_$h=$ec" | tee -a "$LOGDIR/status.log"
  if [ $ec -ne 0 ]; then
    FAILS=$((FAILS+1))
    echo "horizon $h: encoder training failed; skipping Hybrid (it needs this horizon's encoder)" | tee -a "$LOGDIR/status.log"
  else
    $PY -u Benchmark_scripts/m2_hybrid_grud_v2.py > "$LOGDIR/hybrid_$h.log" 2>&1
    ec=$?; echo "HYBRID_EXIT_$h=$ec" | tee -a "$LOGDIR/status.log"; [ $ec -ne 0 ] && FAILS=$((FAILS+1))
  fi

  $PY -u Benchmark_scripts/m2_xgboost_matched_v2.py > "$LOGDIR/xgboost_$h.log" 2>&1
  ec=$?; echo "XGBOOST_EXIT_$h=$ec" | tee -a "$LOGDIR/status.log"; [ $ec -ne 0 ] && FAILS=$((FAILS+1))
done
unset M2_V2_HORIZON
echo "HORIZON_ABLATION_DONE failures=$FAILS $(date +%H:%M:%S)" | tee -a "$LOGDIR/status.log"
[ "$FAILS" -eq 0 ]
