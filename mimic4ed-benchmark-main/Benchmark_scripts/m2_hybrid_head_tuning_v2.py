"""
M2 -- Hybrid GRU-D v2, its own Stage-2A/2B tuning pass
==============================================================================
Phase 2b of the session's "Clearing the Runway to the Manuscript" plan.
Mirrors m2_hybrid_experiment.py's own two-stage grid exactly, on the v2
(irregular-series) cohort, replacing the disclosed reuse of the T=2
pipeline's own winner (m2_hybrid_grud_v2.py's docstring, Deviation 14):

  Stage 2A: static MLP architecture -- static_hidden in {16,32,64} x
    static_dropout in {0.1,0.3} (6 configs), evaluated with a HELD-FIXED
    fusion_dropout=0.1, lr=1e-3 (Sec 6.2's holding value, same convention
    as the T=2 script). Winning (hidden,dropout) is frozen as an
    ARCHITECTURE choice.
  Stage 2B: fusion_dropout in {0.1,0.3} x lr in {1e-3,5e-4} (4 configs),
    using the frozen Stage-2A architecture (fresh weights each run).

3 tune seeds, validation-only; test_v2.csv is never opened here. Cheap:
heads train on precomputed frozen embeddings, not GRU-D re-runs.

Output: data/m2_v2_hybrid_stage2a_tuning_trace.csv,
        data/m2_v2_hybrid_stage2b_tuning_trace.csv,
        data/m2_v2_hybrid_head_tuned_config.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRACE_2A_CSV = os.path.join(DATA_DIR, "m2_v2_hybrid_stage2a_tuning_trace.csv")
TRACE_2B_CSV = os.path.join(DATA_DIR, "m2_v2_hybrid_stage2b_tuning_trace.csv")
CONFIG_JSON = os.path.join(DATA_DIR, "m2_v2_hybrid_head_tuned_config.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import HYBRID_STATIC_COLS, add_hybrid_static_features, DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
    load_frozen_encoder_v2, train_on_frozen_encoder_v2, predict_proba_frozen_v2,
)

TUNE_SEEDS = [42, 123, 7]
GRID_2A = [(h, d) for h in (16, 32, 64) for d in (0.1, 0.3)]  # Sec 6.2
GRID_2B = [(fd, lr) for fd in (0.1, 0.3) for lr in (1e-3, 5e-4)]  # Sec 6.3
HOLDING_FUSION_DROPOUT, HOLDING_LR = 0.1, 1e-3  # for Stage 2A's search only
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 100, 10


def main():
    print(f"Device: {DEVICE}")
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())

    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]
    print(f"Fixed partitions: fit={len(df_fit)} rows, val={len(df_val)} rows")

    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_fit, "train_v2", device=DEVICE)
    Xv, Mv, Dv, Pv, yv = prepare_temporal_tensors_v2(df_val, "train_v2", device=DEVICE)
    yf_np, yv_np = yf.cpu().numpy(), yv.cpu().numpy()

    Sf = torch.tensor(prepare_static_features_v2(df_fit), device=DEVICE)
    Sv = torch.tensor(prepare_static_features_v2(df_val), device=DEVICE)

    # ── Stage 2A: static MLP architecture ───────────────────────────────────
    print(f"\n{'='*60}\nStage 2A: {len(GRID_2A)} configs x {len(TUNE_SEEDS)} seeds\n{'='*60}")
    trace_2a = []
    best_2a_cfg, best_2a_auroc = None, -1.0
    for hidden, dropout in GRID_2A:
        val_aurocs = []
        for seed in TUNE_SEEDS:
            encoder = load_frozen_encoder_v2(seed)
            head, _ = train_on_frozen_encoder_v2(
                encoder, Xf, Mf, Df, Pf, yf_np, Xv, Mv, Dv, Pv, yv_np, seed=seed,
                S_fit=Sf, S_val=Sv, static_hidden=hidden, static_dropout=dropout,
                fusion_dropout=HOLDING_FUSION_DROPOUT, lr=HOLDING_LR,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            )
            p_val = predict_proba_frozen_v2(encoder, head, Xv, Mv, Dv, Pv, S=Sv)
            auroc_val = roc_auc_score(yv_np, p_val)
            val_aurocs.append(auroc_val)
            trace_2a.append({"static_hidden": hidden, "static_dropout": dropout, "seed": seed, "val_auroc": auroc_val})
        mean_auroc = float(np.mean(val_aurocs))
        print(f"  static_hidden={hidden:>2} static_dropout={dropout}: val AUROC = {mean_auroc:.4f} +/- {np.std(val_aurocs):.4f}")
        if mean_auroc > best_2a_auroc:
            best_2a_auroc, best_2a_cfg = mean_auroc, (hidden, dropout)
    pd.DataFrame(trace_2a).to_csv(TRACE_2A_CSV, index=False)
    static_hidden, static_dropout = best_2a_cfg
    print(f"\nStage 2A selected: static_hidden={static_hidden}, static_dropout={static_dropout} "
          f"(val AUROC {best_2a_auroc:.4f})")

    # ── Stage 2B: fusion head, using the frozen Stage 2A architecture ──────
    print(f"\n{'='*60}\nStage 2B: {len(GRID_2B)} configs x {len(TUNE_SEEDS)} seeds\n{'='*60}")
    trace_2b = []
    best_2b_cfg, best_2b_auroc = None, -1.0
    for fusion_dropout, lr in GRID_2B:
        val_aurocs = []
        for seed in TUNE_SEEDS:
            encoder = load_frozen_encoder_v2(seed)
            head, _ = train_on_frozen_encoder_v2(
                encoder, Xf, Mf, Df, Pf, yf_np, Xv, Mv, Dv, Pv, yv_np, seed=seed,
                S_fit=Sf, S_val=Sv, static_hidden=static_hidden, static_dropout=static_dropout,
                fusion_dropout=fusion_dropout, lr=lr,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            )
            p_val = predict_proba_frozen_v2(encoder, head, Xv, Mv, Dv, Pv, S=Sv)
            auroc_val = roc_auc_score(yv_np, p_val)
            val_aurocs.append(auroc_val)
            trace_2b.append({"fusion_dropout": fusion_dropout, "lr": lr, "seed": seed, "val_auroc": auroc_val})
        mean_auroc = float(np.mean(val_aurocs))
        print(f"  fusion_dropout={fusion_dropout} lr={lr}: val AUROC = {mean_auroc:.4f} +/- {np.std(val_aurocs):.4f}")
        if mean_auroc > best_2b_auroc:
            best_2b_auroc, best_2b_cfg = mean_auroc, (fusion_dropout, lr)
    pd.DataFrame(trace_2b).to_csv(TRACE_2B_CSV, index=False)
    fusion_dropout, lr = best_2b_cfg
    print(f"\nStage 2B selected: fusion_dropout={fusion_dropout}, lr={lr} (val AUROC {best_2b_auroc:.4f})")

    config = {
        "static_hidden": static_hidden, "static_dropout": static_dropout,
        "fusion_dropout": fusion_dropout, "lr": lr, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    }
    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nWrote {CONFIG_JSON}")


if __name__ == "__main__":
    main()
