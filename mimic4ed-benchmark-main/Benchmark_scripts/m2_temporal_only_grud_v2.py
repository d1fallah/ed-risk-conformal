"""
M2 -- Temporal-Only GRU-D, irregular-series (v2) cohort, FULL 5-SEED RUN
==============================================================================
Full confirmatory run (5 seeds {42,123,7,456,789}) following the pilot's
single-seed result (AUROC 0.9234, seed 42) that motivated committing to
the full v2 grid. Same architecture/hyperparameters as the T=2 pipeline's
m2_temporal_only_grud.py (HIDDEN=32, DROPOUT=0.2, WEIGHT_DECAY=1e-4,
LR=0.01, BATCH_SIZE=128, MAX_EPOCHS=100, PATIENCE=10) -- unchanged, so any
difference from the T=2 result is attributable to the cohort/representation
change alone. Checkpoints saved per seed (data/checkpoints/
m2_temporal_only_grud_v2_seed<N>.pt), required by Frozen-Temporal Head
Control v2 and Hybrid GRU-D v2, which freeze and reuse this encoder.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    _minibatch_early_stop_loop, expected_calibration_error, calibration_slope_intercept, DEVICE,
)
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, GRUDWithEncoderV2,
    save_checkpoint_v2, save_predictions_v2, save_seed_metrics_v2,
    HORIZON, TARGET, SUFFIX, apply_horizon_filter,
)

SEED_DIR = os.path.join(DATA_DIR, f"m2_v2_seed_results{SUFFIX}", "temporal_only_grud")
os.makedirs(SEED_DIR, exist_ok=True)

FINAL_SEEDS = [42, 123, 7, 456, 789]
HIDDEN, DROPOUT, WEIGHT_DECAY, LR, BATCH_SIZE = 32, 0.2, 1e-4, 0.01, 128
MAX_EPOCHS = 100
PATIENCE = 10
MODEL_NAME = "Temporal-Only GRU-D v2"


def main():
    print(f"Device: {DEVICE}")
    df_train = load_split_frame("train_v2")
    df_test = load_split_frame("test_v2")
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())

    df_fit = apply_horizon_filter(df_train[df_train["subject_id"].isin(train_ids)])
    df_val = apply_horizon_filter(df_train[df_train["subject_id"].isin(val_ids)])
    df_test = apply_horizon_filter(df_test)
    print(f"target={TARGET} (horizon={HORIZON or 'composite'})  fit={len(df_fit)}, val={len(df_val)}, test={len(df_test)}")
    for _n, _d in (("fit", df_fit), ("val", df_val), ("test", df_test)):
        _pos = int(_d[TARGET].sum())
        print(f"  {_n}: {_pos} positives ({_pos/len(_d):.5f})")
        # roc_auc_score (called every epoch in the shared loop, and at evaluation) raises on a single-class fold.
        assert 0 < _pos < len(_d), f"{_n} fold has no positives (or no negatives) for {TARGET}: cannot train/evaluate. STOP."

    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_fit, "train_v2", device=DEVICE)
    Xv, Mv, Dv, Pv, yv = prepare_temporal_tensors_v2(df_val, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_test, "test_v2", device=DEVICE)
    y_test_np = yt.cpu().numpy()

    seed_metrics = []
    for seed in FINAL_SEEDS:
        torch.manual_seed(seed)
        model = GRUDWithEncoderV2(inp=7, hid=HIDDEN, stat=0, clf_hidden=32, dropout=DROPOUT).to(DEVICE)

        t0 = time.perf_counter()
        history = _minibatch_early_stop_loop(
            param_groups_fn=lambda: model.parameters(),
            forward_fn=lambda ts: model(ts[0], ts[1], ts[2], ts[3]),
            X_fit_tensors=[Xf, Mf, Df, Pf], y_fit=yf.cpu().numpy(),
            X_val_tensors=[Xv, Mv, Dv, Pv], y_val=yv.cpu().numpy(),
            seed=seed, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, train_modules=[model],
        )
        train_s = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            test_prob = torch.sigmoid(model(Xt, Mt, Dt, Pt)).cpu().numpy().ravel()

        auroc = roc_auc_score(y_test_np, test_prob)
        auprc = average_precision_score(y_test_np, test_prob)
        brier = brier_score_loss(y_test_np, test_prob)
        ece = expected_calibration_error(y_test_np, test_prob)
        slope, intercept = calibration_slope_intercept(y_test_np, test_prob)

        m = {"seed": seed, "epochs_trained": len(history), "stopped_early": len(history) < MAX_EPOCHS,
             "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
             "ece": float(ece), "cal_slope": float(slope), "cal_intercept": float(intercept),
             "train_seconds": train_s}
        seed_metrics.append(m)
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(m, f, indent=2)

        save_checkpoint_v2(model, "m2_temporal_only_grud", seed)
        save_predictions_v2(df_test["subject_id"].values, MODEL_NAME, seed, y_test_np, test_prob,
                             stay_id=df_test["stay_id"].values)
        save_seed_metrics_v2(MODEL_NAME, seed, {
            "auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece,
            "cal_slope": slope, "cal_intercept": intercept,
        })

        print(f"seed {seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f} epochs={len(history)} "
              f"train={train_s:.1f}s", flush=True)

    aurocs = [m["auroc"] for m in seed_metrics]
    auprcs = [m["auprc"] for m in seed_metrics]
    print(f"\n{MODEL_NAME}, 5-seed: AUROC={np.mean(aurocs):.4f}+/-{np.std(aurocs):.4f}  "
          f"AUPRC={np.mean(auprcs):.4f}+/-{np.std(auprcs):.4f}")


if __name__ == "__main__":
    main()
