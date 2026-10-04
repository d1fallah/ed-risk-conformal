#!/usr/bin/env bash
# Runs AFTER m2_mtan_tuning_matched_v2.py finishes (closes Deviation 15).
#   1. archive the lighter-protocol mTAN result so it is never silently overwritten
#   2. 5-seed confirmatory mTAN at the matched-rigor config (~20 min GPU)
#   3. recompute ONLY Delta_modern_comparator (Hybrid - mTAN) with the module's established
#      patient-cluster bootstrap; the other four deltas involve no mTAN and are reused verbatim,
#      and Holm is re-applied across all five.
# Refuses to run if the tuning config is missing or came from a --smoke run.
set -eu
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
CFG="data/m2_v2_mtan_tuned_config_matched.json"
[ -f "$CFG" ] || { echo "missing $CFG -- tuning has not finished"; exit 2; }
grep -q '"smoke": true' "$CFG" && { echo "$CFG is from a --smoke run"; exit 2; }

echo "== archiving pre-matched mTAN artifacts =="
cp -n data/m2_v2_results.csv        data/m2_v2_results_PRE_MTAN_MATCHED.csv
cp -n data/m2_v2_predictions.csv    data/m2_v2_predictions_PRE_MTAN_MATCHED.csv
cp -n data/m2_v2_deltas.json        data/m2_v2_deltas_pre_mtan_matched.json
[ -d data/m2_v2_seed_results/mtan_PRE_MATCHED ] || cp -r data/m2_v2_seed_results/mtan data/m2_v2_seed_results/mtan_PRE_MATCHED
mkdir -p data/checkpoints/PRE_MTAN_MATCHED
cp -n data/checkpoints/*mtan*_v2_seed*.pt data/checkpoints/PRE_MTAN_MATCHED/ 2>/dev/null || true

echo "== mTAN 5-seed confirmatory at matched config =="
$PY -u Benchmark_scripts/m2_mtan_final_v2.py

echo "== Delta_modern_comparator only =="
$PY -u Benchmark_scripts/m2_v2_analyze.py --reuse-from data/m2_v2_deltas_pre_mtan_matched.json \
    --skip Delta_representation Delta_static_contribution Delta_temporal_alone Delta_static_only
echo POST_MTAN_DONE
