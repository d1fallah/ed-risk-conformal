#!/bin/bash
# M2 -- unattended overnight run: mTAN tuning -> full 5-seed grid (all 5
# v2 models) -> v2 conformal re-run (MAPIE + ECI + Mondrian-LAC).
# Sequential (not parallel) to respect the 6GB GPU and the dependency
# order: Frozen-Temporal Head Control v2 and Hybrid GRU-D v2 each require
# ALL 5 of Temporal-Only GRU-D v2's seed checkpoints to exist first.
# `set -e`: stop at the first failure rather than continuing on broken
# state -- the log will show exactly which stage failed and why.
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "STAGE 0: cheap CPU/small-GPU models (5 seeds each)"
echo "=========================================="
python m2_static_only_mlp_v2.py
python m2_xgboost_matched_v2.py

echo "=========================================="
echo "STAGE 1: mTAN tuning pass (1 seed, 8 candidates)"
echo "=========================================="
python m2_mtan_tuning_v2.py

echo "=========================================="
echo "STAGE 2: mTAN, full 5-seed run (tuned config)"
echo "=========================================="
python m2_mtan_final_v2.py

echo "=========================================="
echo "STAGE 3: Temporal-Only GRU-D v2, full 5-seed run (+ checkpoints)"
echo "=========================================="
python m2_temporal_only_grud_v2.py

echo "=========================================="
echo "STAGE 4: Frozen-Temporal Head Control v2, full 5-seed run"
echo "=========================================="
python m2_frozen_temporal_control_v2.py

echo "=========================================="
echo "STAGE 5: Hybrid GRU-D v2 (primary model), full 5-seed run"
echo "=========================================="
python m2_hybrid_grud_v2.py

echo "=========================================="
echo "STAGE 6: v2 conformal re-run (MAPIE + ECI + Mondrian-LAC)"
echo "=========================================="
python m2_conformal_v2.py

echo "=========================================="
echo "ALL STAGES COMPLETE"
echo "=========================================="
