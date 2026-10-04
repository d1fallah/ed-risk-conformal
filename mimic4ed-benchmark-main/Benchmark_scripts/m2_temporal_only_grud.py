"""
M2 Hybrid Guide v3.8 -- Temporal-Only GRU-D (Stage 1, Sec 2.1/6.1)
========================================================================
"Clean temporal baseline." Vital-sign sequence only (2-timestep, irregular)
-- NO static features (static_size=0). Trains an encoder + a temporal
prediction head jointly; the head is discarded and only the frozen encoder
is reused by Frozen-Temporal Head Control and Hybrid GRU-D (Sec 2.1).

Critical rule (Guide Sec 0): the legacy 0.9009 checkpoint contains 24
static features baked into its classifier head and is NOT a valid
temporal-only comparator -- it is never loaded, reused, or used to
initialize anything here. This encoder is trained from scratch.

Fixed configuration, no grid (Sec 6.1): hidden dim = 32, dropout = 0.2,
weight decay = 1e-4, learning rate = 0.01, batch size = 128, early
stopping on validation loss with patience = 10. This is now Sec 6.1's
locked table followed EXACTLY, with no deviations.

That is a change from the previous version of this script, and the reason
matters. Deviation 3 recorded that lr=0.01+batch_size=128 "never learns"
(200 epochs pinned at chance AUROC) and Deviation 5 recorded that
patience=10 stopped ~1500 epochs short of the optimum, forcing a 2000-epoch
full-batch budget. Both diagnoses were correct about what the code did, but
both were measured against RAW inputs: vitals reached the GRU in Fahrenheit
and with outlier removal bypassed (temperature max 986, SpO2 max 9322, SBP
max 151103), unscaled, into tanh/sigmoid gates. Those were saturation
artifacts, not properties of the guide's configuration.

With M2_Hybrid_Deviation_Log.md Deviation 7's fix in place (Celsius +
outlier removal restored, per-vital standardisation fit on the `training`
subset only, and a GRU-D decay mechanism that is actually differentiable),
the locked configuration trains cleanly on the first attempt --
re-diagnosed directly at seed 42, both arms, in
data/m2_temporal_only_grud_convergence_diagnostic_dev7_seed42.json:

  lr=0.01, batch_size=128, patience=10 -> early stopping fires normally at
      epoch 31, best val_loss 0.1510 at epoch 20, val AUROC 0.8736.
  lr=0.01, full-batch, 600 epochs      -> best val_loss 0.1455 at epoch
      582, val AUROC 0.8840.

The full-batch arm scores slightly higher, and is deliberately NOT adopted:
selecting it would mean keeping two protocol deviations in order to gain
validation AUROC, which is exactly the post-hoc configuration selection
this project's anti-p-hacking design exists to prevent. The guide's locked
configuration works, so the guide's locked configuration is what runs.
Deviations 3 and 5 are retracted in M2_Hybrid_Deviation_Log.md.

Per seed: trains on the FIXED `training` subject subset, early-stops using
the FIXED `validation` subject subset (Guide Sec 5.1/Sec 7 -- identical
partition across all 5 seeds, only weight init/data order vary), reports
Temporal-Only GRU-D's own discrimination + probabilistic-calibration
metrics (conformal MAPIE/ECI are computed separately, in Phase L, applied
uniformly across all 3 GRU-D-family models), then checkpoints the FULL
model (encoder+head) to data/checkpoints/ for Phases J/K to load just the
encoder from (TemporalEncoder.load_state_dict(..., strict=False) picks out
only the cell.* recurrent tensors, ignoring the discarded head;
load_frozen_encoder() asserts none of them were silently missing).
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR = os.path.join(DATA_DIR, "m2_hybrid_seed_results", "temporal_only_grud")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(SEED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    TARGET, prepare_temporal_tensors, build_fixed_dev_split,
    train_temporal_only, predict_proba_temporal_only,
    expected_calibration_error, calibration_slope_intercept,
    save_predictions, save_seed_metrics, save_manifest,
)

FINAL_SEEDS = [42, 123, 7, 456, 789]

# Guide Sec 6.1's locked table, followed exactly. See the module docstring
# for why the previous full-batch / 2000-epoch / patience-disabled settings
# (Deviations 3 and 5) are retracted rather than carried forward.
HIDDEN, DROPOUT, WEIGHT_DECAY, LR, BATCH_SIZE = 32, 0.2, 1e-4, 0.01, 128
MAX_EPOCHS = 100
PATIENCE = 10


def main():
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())
    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]
    print(f"Fixed partitions: training={len(df_fit)} rows / {len(train_ids)} patients, "
          f"validation={len(df_val)} rows / {len(val_ids)} patients")

    X_fit, M_fit, D_fit, y_fit = prepare_temporal_tensors(df_fit)
    X_val, M_val, D_val, y_val = prepare_temporal_tensors(df_val)
    X_test, M_test, D_test, y_test = prepare_temporal_tensors(df_test)
    subj_test = df_test["subject_id"].values
    stay_test = df_test["stay_id"].values

    config = {
        "hidden": HIDDEN, "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY, "lr": LR,
        "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "stat": 0,
    }

    for seed in FINAL_SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        t0 = time.time()
        model, history = train_temporal_only(
            X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed=seed,
            hidden=HIDDEN, dropout=DROPOUT, weight_decay=WEIGHT_DECAY, lr=LR,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        elapsed = time.time() - t0
        stopped_early = len(history) < MAX_EPOCHS
        print(f"  Trained {len(history)}/{MAX_EPOCHS} epochs ({'early-stopped' if stopped_early else 'hit epoch cap'}) "
              f"in {elapsed:.1f}s; best val_loss={min(h['val_loss'] for h in history):.4f}")

        y_prob = predict_proba_temporal_only(model, X_test, M_test, D_test)
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test, y_prob)
        slope, intercept = calibration_slope_intercept(y_test, y_prob)
        print(f"  AUROC={auroc:.4f} AUPRC={auprc:.4f} Brier={brier:.4f} ECE={ece:.4f} "
              f"slope={slope:.3f} intercept={intercept:.3f}")

        metrics = {
            "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
            "ece": float(ece), "calibration_slope": slope, "calibration_intercept": intercept,
        }
        result = {
            "seed": seed, "model": "Temporal-Only GRU-D", "config": config,
            "epochs_trained": len(history), "stopped_early": stopped_early,
            "training_seconds": elapsed, "metrics": metrics,
            "history_tail": history[-5:],
        }
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)

        # Checkpoint the FULL model (encoder+head). Phases J/K load just
        # the encoder's 5 recurrent-weight tensors via
        # TemporalEncoder.load_state_dict(..., strict=False).
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"m2_temporal_only_grud_seed{seed}.pt"))

        save_predictions(subj_test, "Temporal-Only GRU-D", seed, y_test, y_prob, stay_id=stay_test)
        save_seed_metrics("Temporal-Only GRU-D", seed, metrics)
        save_manifest("Temporal-Only GRU-D", seed, config)

    df_res = pd.read_csv(os.path.join(DATA_DIR, "results_hybrid.csv"))
    df_res = df_res[df_res["model"] == "Temporal-Only GRU-D"]
    print(f"\n5-seed summary (Temporal-Only GRU-D): "
          f"AUROC {df_res['auroc'].mean():.4f} +/- {df_res['auroc'].std():.4f}  "
          f"AUPRC {df_res['auprc'].mean():.4f} +/- {df_res['auprc'].std():.4f}  "
          f"Brier {df_res['brier'].mean():.4f} +/- {df_res['brier'].std():.4f}  "
          f"ECE {df_res['ece'].mean():.4f} +/- {df_res['ece'].std():.4f}")


if __name__ == "__main__":
    main()
