"""
M2 Hybrid Guide v3.8 -- Hybrid GRU-D (Stage 3, Sec 2/6.2/6.3)
========================================================================
Primary model: frozen temporal encoder (one per seed, from
m2_temporal_only_grud.py) + a static MLP branch (Sec 2.2, 4 Guide static
features) via a two-branch fusion architecture. Fusion: 32-d temporal +
32-d static = 64-d -> Linear(64,1) -> sigmoid (Sec 2.3, resolved in
M2_Hybrid_Guide_Clarification_Request.md Item 2). The temporal encoder's
parameters never receive gradients; only the static-MLP-encoder and fusion
head are trained (Sec 6.5's gradient constraint).

Sequential bounded tuning (Sec 6.2/6.3), matching the guide's own staged
order:
  Stage 2A: static MLP architecture -- hidden in {16,32,64} x dropout in
    {0.1,0.3} (6 configs), evaluated with a HELD-FIXED fusion_dropout=0.1,
    lr=1e-3 (Sec 6.3's own grid's first/most-common point, and what
    Frozen-Temporal Head Control's own independent Stage-2B-style search
    happened to select too -- a reasonable, documented holding value,
    since Stage 2B hasn't run yet at this point). Winning (hidden,dropout)
    is frozen as an ARCHITECTURE choice -- Sec 2.3's explicit clarification
    that this locks the hidden-dim/dropout *choice*, not the weights,
    which stay trainable in Stage 2B.
  Stage 2B: fusion_dropout in {0.1,0.3} x lr in {1e-3,5e-4} (4 configs),
    using the frozen Stage-2A architecture (fresh weights each run).

Both stages select by mean validation AUROC across TUNE_SEEDS (matching
the convention already used for Static-Only MLP and Frozen-Temporal Head
Control). Final 5-seed run trains a fresh Hybrid (frozen per-seed encoder
+ fresh static-MLP + fresh fusion head, at the two frozen architecture/
hyperparameter choices) per FINAL_SEEDS -- no cross-seed encoder sharing
(Guide Sec 7).

Per Guide Sec 8, this model gets full conformal reporting (MAPIE/ECI) --
computed separately in Phase L, applied uniformly across all 3 GRU-D-
family models -- plus discrimination/calibration here.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR = os.path.join(DATA_DIR, "m2_hybrid_seed_results", "hybrid_grud")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
TRACE_2A_CSV = os.path.join(DATA_DIR, "m2_hybrid_stage2a_tuning_trace.csv")
TRACE_2B_CSV = os.path.join(DATA_DIR, "m2_hybrid_stage2b_tuning_trace.csv")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    TARGET, HYBRID_STATIC_COLS, add_hybrid_static_features, prepare_static_features,
    prepare_temporal_tensors, build_fixed_dev_split, load_frozen_encoder,
    train_on_frozen_encoder, predict_proba_frozen,
    expected_calibration_error, calibration_slope_intercept,
    save_predictions, save_seed_metrics, save_manifest,
)

TUNE_SEEDS = [42, 123, 7]
FINAL_SEEDS = [42, 123, 7, 456, 789]
GRID_2A = [(h, d) for h in (16, 32, 64) for d in (0.1, 0.3)]  # Sec 6.2
GRID_2B = [(fd, lr) for fd in (0.1, 0.3) for lr in (1e-3, 5e-4)]  # Sec 6.3
HOLDING_FUSION_DROPOUT, HOLDING_LR = 0.1, 1e-3  # for Stage 2A's search only
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 100, 10  # Sec 6.5


def main():
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_train = add_hybrid_static_features(df_train)
    df_test = add_hybrid_static_features(df_test)

    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())
    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]
    print(f"Fixed partitions: training={len(df_fit)} rows, validation={len(df_val)} rows")

    X_fit, M_fit, D_fit, y_fit = prepare_temporal_tensors(df_fit)
    X_val, M_val, D_val, y_val = prepare_temporal_tensors(df_val)
    X_test, M_test, D_test, y_test = prepare_temporal_tensors(df_test)
    # Standardised via the shared train-only scaler (Deviation 7 / S5):
    # previously these were raw values, while Static-Only MLP standardised
    # the same 4 columns -- confounding Delta_static_contribution.
    S_fit = prepare_static_features(df_fit)
    S_val = prepare_static_features(df_val)
    S_test = prepare_static_features(df_test)
    subj_test = df_test["subject_id"].values
    stay_test = df_test["stay_id"].values

    # ── Stage 2A: static MLP architecture ───────────────────────────────────
    print(f"\n{'='*60}\nStage 2A: {len(GRID_2A)} configs x {len(TUNE_SEEDS)} seeds\n{'='*60}")
    trace_2a = []
    best_2a_cfg, best_2a_auroc = None, -1.0
    for hidden, dropout in GRID_2A:
        val_aurocs = []
        for seed in TUNE_SEEDS:
            encoder = load_frozen_encoder(seed)
            head, _ = train_on_frozen_encoder(
                encoder, X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed=seed,
                S_fit=S_fit, S_val=S_val, static_hidden=hidden, static_dropout=dropout,
                fusion_dropout=HOLDING_FUSION_DROPOUT, lr=HOLDING_LR,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            )
            p_val = predict_proba_frozen(encoder, head, X_val, M_val, D_val, S=S_val)
            auroc_val = roc_auc_score(y_val, p_val)
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
            encoder = load_frozen_encoder(seed)
            head, _ = train_on_frozen_encoder(
                encoder, X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed=seed,
                S_fit=S_fit, S_val=S_val, static_hidden=static_hidden, static_dropout=static_dropout,
                fusion_dropout=fusion_dropout, lr=lr,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            )
            p_val = predict_proba_frozen(encoder, head, X_val, M_val, D_val, S=S_val)
            auroc_val = roc_auc_score(y_val, p_val)
            val_aurocs.append(auroc_val)
            trace_2b.append({"fusion_dropout": fusion_dropout, "lr": lr, "seed": seed, "val_auroc": auroc_val})
        mean_auroc = float(np.mean(val_aurocs))
        print(f"  fusion_dropout={fusion_dropout} lr={lr}: val AUROC = {mean_auroc:.4f} +/- {np.std(val_aurocs):.4f}")
        if mean_auroc > best_2b_auroc:
            best_2b_auroc, best_2b_cfg = mean_auroc, (fusion_dropout, lr)
    pd.DataFrame(trace_2b).to_csv(TRACE_2B_CSV, index=False)
    fusion_dropout, lr = best_2b_cfg
    print(f"\nStage 2B selected: fusion_dropout={fusion_dropout}, lr={lr} (val AUROC {best_2b_auroc:.4f})")

    # ── Final 5-seed run ─────────────────────────────────────────────────
    print(f"\n{'='*60}\nFinal 5-seed run at frozen config\n{'='*60}")
    config = {
        "static_hidden": static_hidden, "static_dropout": static_dropout,
        "fusion_dropout": fusion_dropout, "lr": lr, "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "fusion_width": "32+32=64",
    }
    for seed in FINAL_SEEDS:
        encoder = load_frozen_encoder(seed)
        head, hist = train_on_frozen_encoder(
            encoder, X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed=seed,
            S_fit=S_fit, S_val=S_val, static_hidden=static_hidden, static_dropout=static_dropout,
            fusion_dropout=fusion_dropout, lr=lr,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        stopped_early = len(hist) < MAX_EPOCHS
        y_prob = predict_proba_frozen(encoder, head, X_test, M_test, D_test, S=S_test)

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test, y_prob)
        slope, intercept = calibration_slope_intercept(y_test, y_prob)
        print(f"  seed={seed}: {len(hist)}/{MAX_EPOCHS} epochs ({'early-stopped' if stopped_early else 'hit cap'})  "
              f"AUROC={auroc:.4f} AUPRC={auprc:.4f} Brier={brier:.4f} ECE={ece:.4f}")

        metrics = {
            "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
            "ece": float(ece), "calibration_slope": slope, "calibration_intercept": intercept,
        }
        result = {
            "seed": seed, "model": "Hybrid GRU-D", "config": config,
            "epochs_trained": len(hist), "stopped_early": stopped_early, "metrics": metrics,
        }
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)

        torch.save(head.state_dict(), os.path.join(CHECKPOINT_DIR, f"m2_hybrid_grud_head_seed{seed}.pt"))
        save_predictions(subj_test, "Hybrid GRU-D", seed, y_test, y_prob, stay_id=stay_test)
        save_seed_metrics("Hybrid GRU-D", seed, metrics)
        save_manifest("Hybrid GRU-D", seed, config)

    df_res = pd.read_csv(os.path.join(DATA_DIR, "results_hybrid.csv"))
    df_res = df_res[df_res["model"] == "Hybrid GRU-D"]
    print(f"\n5-seed summary (Hybrid GRU-D): "
          f"AUROC {df_res['auroc'].mean():.4f} +/- {df_res['auroc'].std():.4f}  "
          f"AUPRC {df_res['auprc'].mean():.4f} +/- {df_res['auprc'].std():.4f}  "
          f"Brier {df_res['brier'].mean():.4f} +/- {df_res['brier'].std():.4f}  "
          f"ECE {df_res['ece'].mean():.4f} +/- {df_res['ece'].std():.4f}")


if __name__ == "__main__":
    main()
