#!/bin/bash
# Reruns every M2 analysis affected by Deviation 23, in the orchestrator's dependency order.
# Run from module2/mimic4ed-benchmark-main. Stops at the first failure. Logs in track_b/results/rerun_logs/.
PY=./.venv/Scripts/python.exe
LOG=track_b/results/rerun_logs
for s in m2_conformal_v2 m2_mondrian_eci_pilot m2_mondrian_eci_confirmatory m2_mondrian_eci_anchor_correction \
         m2_mondrian_eci_comparators m2_eci_vs_recalibration m2_label_lag m2_v2_logit_vs_clipping m2_v2_latency; do
  start=$(date +%s)
  echo "=== $s  start $(date +%H:%M:%S)" | tee -a $LOG/_progress.txt
  if $PY Benchmark_scripts/$s.py > $LOG/$s.log 2>&1; then
    echo "=== $s  OK   $(( $(date +%s) - start ))s" | tee -a $LOG/_progress.txt
  else
    echo "=== $s  FAILED after $(( $(date +%s) - start ))s (see $LOG/$s.log)" | tee -a $LOG/_progress.txt
    exit 1
  fi
done
echo "ALL DONE $(date +%H:%M:%S)" | tee -a $LOG/_progress.txt
