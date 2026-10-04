"""
M2 -- Frozen-Temporal Head Control v2, its own Stage-2B-style tuning pass
==============================================================================
Phase 2b of the session's "Clearing the Runway to the Manuscript" plan.
Mirrors m2_frozen_temporal_control.py's OWN independent tuning grid
(fusion_dropout in {0.1,0.3} x lr in {1e-3,5e-4}, Sec 6.3, 3 tune seeds)
exactly, on the v2 (irregular-series) cohort, replacing the disclosed
reuse of the T=2 pipeline's own winner
(m2_frozen_temporal_control_v2.py's docstring, Deviation 14).

Validation-only (train_v2/validation dev-pool subsets); test_v2.csv is
never opened here. Cheap: heads train on precomputed frozen embeddings
(train_on_frozen_encoder_v2 re-encodes once per call, no gradient), not
GRU-D re-runs.

Output: data/m2_v2_frozen_control_tuning_trace.csv,
        data/m2_v2_frozen_control_tuned_config.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRACE_CSV = os.path.join(DATA_DIR, "m2_v2_frozen_control_tuning_trace.csv")
CONFIG_JSON = os.path.join(DATA_DIR, "m2_v2_frozen_control_tuned_config.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2,
    load_frozen_encoder_v2, train_on_frozen_encoder_v2, predict_proba_frozen_v2,
)

TUNE_SEEDS = [42, 123, 7]
GRID = [(fd, lr) for fd in (0.1, 0.3) for lr in (1e-3, 5e-4)]  # Sec 6.3, matching m2_frozen_temporal_control.py
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 100, 10


def main():
    print(f"Device: {DEVICE}")
    df_train = load_split_frame("train_v2")
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())

    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]
    print(f"Fixed partitions: fit={len(df_fit)} rows, val={len(df_val)} rows")

    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_fit, "train_v2", device=DEVICE)
    Xv, Mv, Dv, Pv, yv = prepare_temporal_tensors_v2(df_val, "train_v2", device=DEVICE)
    yf_np, yv_np = yf.cpu().numpy(), yv.cpu().numpy()

    print(f"\n{'='*60}\nStage 2B-style tuning: {len(GRID)} configs x {len(TUNE_SEEDS)} seeds\n{'='*60}")
    trace_rows = []
    best_cfg, best_mean_auroc = None, -1.0
    for fusion_dropout, lr in GRID:
        val_aurocs = []
        for seed in TUNE_SEEDS:
            encoder = load_frozen_encoder_v2(seed)
            head, hist = train_on_frozen_encoder_v2(
                encoder, Xf, Mf, Df, Pf, yf_np, Xv, Mv, Dv, Pv, yv_np, seed=seed,
                static_hidden=None, fusion_dropout=fusion_dropout, lr=lr,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            )
            p_val = predict_proba_frozen_v2(encoder, head, Xv, Mv, Dv, Pv)
            auroc_val = roc_auc_score(yv_np, p_val)
            val_aurocs.append(auroc_val)
            trace_rows.append({"fusion_dropout": fusion_dropout, "lr": lr, "seed": seed,
                                "epochs_trained": len(hist), "val_auroc": auroc_val})
        mean_auroc = float(np.mean(val_aurocs))
        print(f"  fusion_dropout={fusion_dropout} lr={lr}: val AUROC = {mean_auroc:.4f} +/- {np.std(val_aurocs):.4f}")
        if mean_auroc > best_mean_auroc:
            best_mean_auroc, best_cfg = mean_auroc, (fusion_dropout, lr)

    pd.DataFrame(trace_rows).to_csv(TRACE_CSV, index=False)
    fusion_dropout, lr = best_cfg
    print(f"\nSelected config: fusion_dropout={fusion_dropout}, lr={lr} (val AUROC {best_mean_auroc:.4f})")

    config = {"fusion_dropout": fusion_dropout, "lr": lr, "batch_size": BATCH_SIZE,
              "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "static_hidden": None}
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote {CONFIG_JSON}")


if __name__ == "__main__":
    main()
