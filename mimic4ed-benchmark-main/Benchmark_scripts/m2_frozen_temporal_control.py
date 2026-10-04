"""
M2 Hybrid Guide v3.8 -- Frozen-Temporal Head Control (Stage 2, Sec 4.2)
========================================================================
"This model is essential for a clean interpretation of Delta_static." Uses
the SAME frozen temporal encoder as Hybrid GRU-D (one per seed, from
m2_temporal_only_grud.py -- never cross-seed-shared), but trains a fresh
prediction head with NO static branch. Any Hybrid-vs-this-control
difference is then attributable only to the static MLP branch itself, not
to head-training differences (Sec 4.2: "trained using the same
training/validation protocol as the Hybrid head... but with no static
branch").

"Same protocol" is read here as: same optimizer, early-stopping patience,
epoch budget, and training/validation subjects as Hybrid's head -- AND its
own independent Stage-2B-style tuning (fusion_dropout in {0.1,0.3} x lr in
{1e-3,5e-4}, Sec 6.3's grid, same validation-based selection rule), not a
reuse of Hybrid's specific *selected* values. Two things support this
reading: the guide's own enumerated definition of "same protocol" never
lists "same selected hyperparameters," and the guide's own Day-3-before-
Day-4 execution order (Sec 17) only makes sense this way -- this script
(Day 3) cannot depend on a Hybrid GRU-D that Day 4's script hasn't built
yet. train_on_frozen_encoder() (m2_hybrid_common.py) is the exact same
function m2_hybrid_experiment.py calls for Hybrid's head, parameterized by
static_hidden=None here vs. a real value there -- so the two models' heads
are trained by verifiably identical code.

Per Guide Sec 8, this model gets full conformal reporting (MAPIE/ECI) --
computed separately, in Phase L, applied uniformly across all 3 GRU-D-
family models -- plus discrimination/calibration here.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR = os.path.join(DATA_DIR, "m2_hybrid_seed_results", "frozen_temporal_control")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
TRACE_CSV = os.path.join(DATA_DIR, "m2_frozen_temporal_control_tuning_trace.csv")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    TARGET, prepare_temporal_tensors, build_fixed_dev_split, load_frozen_encoder,
    train_on_frozen_encoder, predict_proba_frozen,
    expected_calibration_error, calibration_slope_intercept,
    save_predictions, save_seed_metrics, save_manifest,
)
import torch

TUNE_SEEDS = [42, 123, 7]
FINAL_SEEDS = [42, 123, 7, 456, 789]
GRID = [(fd, lr) for fd in (0.1, 0.3) for lr in (1e-3, 5e-4)]  # Sec 6.3
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 100, 10  # Sec 6.5


def main():
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())
    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]
    print(f"Fixed partitions: training={len(df_fit)} rows, validation={len(df_val)} rows")

    X_fit, M_fit, D_fit, y_fit = prepare_temporal_tensors(df_fit)
    X_val, M_val, D_val, y_val = prepare_temporal_tensors(df_val)
    X_test, M_test, D_test, y_test = prepare_temporal_tensors(df_test)
    subj_test = df_test["subject_id"].values
    stay_test = df_test["stay_id"].values

    # ── Stage 2B-style tuning (own independent grid, Sec 6.3) ──────────────
    print(f"\n{'='*60}\nLight tuning: {len(GRID)} configs x {len(TUNE_SEEDS)} seeds\n{'='*60}")
    trace_rows = []
    best_cfg, best_mean_auroc = None, -1.0
    for fusion_dropout, lr in GRID:
        val_aurocs = []
        for seed in TUNE_SEEDS:
            encoder = load_frozen_encoder(seed)
            head, hist = train_on_frozen_encoder(
                encoder, X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed=seed,
                static_hidden=None, fusion_dropout=fusion_dropout, lr=lr,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
            )
            p_val = predict_proba_frozen(encoder, head, X_val, M_val, D_val)
            auroc_val = roc_auc_score(y_val, p_val)
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

    # ── Final 5-seed run: own frozen encoder per seed, frozen config ───────
    print(f"\n{'='*60}\nFinal 5-seed run at frozen config\n{'='*60}")
    config = {"fusion_dropout": fusion_dropout, "lr": lr, "batch_size": BATCH_SIZE,
              "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "static_hidden": None}
    for seed in FINAL_SEEDS:
        encoder = load_frozen_encoder(seed)
        head, hist = train_on_frozen_encoder(
            encoder, X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed=seed,
            static_hidden=None, fusion_dropout=fusion_dropout, lr=lr,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        stopped_early = len(hist) < MAX_EPOCHS
        y_prob = predict_proba_frozen(encoder, head, X_test, M_test, D_test)

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
            "seed": seed, "model": "Frozen-Temporal Head Control", "config": config,
            "epochs_trained": len(hist), "stopped_early": stopped_early, "metrics": metrics,
        }
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)

        torch.save(head.state_dict(), os.path.join(CHECKPOINT_DIR, f"m2_frozen_temporal_control_head_seed{seed}.pt"))
        save_predictions(subj_test, "Frozen-Temporal Head Control", seed, y_test, y_prob, stay_id=stay_test)
        save_seed_metrics("Frozen-Temporal Head Control", seed, metrics)
        save_manifest("Frozen-Temporal Head Control", seed, config)

    df_res = pd.read_csv(os.path.join(DATA_DIR, "results_hybrid.csv"))
    df_res = df_res[df_res["model"] == "Frozen-Temporal Head Control"]
    print(f"\n5-seed summary (Frozen-Temporal Head Control): "
          f"AUROC {df_res['auroc'].mean():.4f} +/- {df_res['auroc'].std():.4f}  "
          f"AUPRC {df_res['auprc'].mean():.4f} +/- {df_res['auprc'].std():.4f}  "
          f"Brier {df_res['brier'].mean():.4f} +/- {df_res['brier'].std():.4f}  "
          f"ECE {df_res['ece'].mean():.4f} +/- {df_res['ece'].std():.4f}")


if __name__ == "__main__":
    main()
