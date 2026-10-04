"""
M2 -- Hybrid GRU-D (primary model), irregular-series (v2) cohort, 5-SEED
==============================================================================
Trains a StaticEncoder + FusionHead on top of Temporal-Only GRU-D v2's
FROZEN encoder, fused with the 4-feature static branch -- v2 counterpart
of m2_hybrid_experiment.py. Requires m2_temporal_only_grud_v2.py's 5-seed
checkpoints.

Loads the tuned architecture/head config selected by
m2_hybrid_head_tuning_v2.py (Phase 2b of the session's "Clearing the
Runway to the Manuscript" plan, data/m2_v2_hybrid_head_tuned_config.json)
-- REPLACES the earlier disclosed reuse of the T=2 pipeline's own
validation-selected winner (Deviation 14). Refuses to silently fall back
to the untuned/reused values (same discipline as m2_mtan_final_v2.py).
"""
import os, sys, json, time, warnings
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CONFIG_JSON = os.path.join(DATA_DIR, "m2_v2_hybrid_head_tuned_config.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import HYBRID_STATIC_COLS, add_hybrid_static_features, expected_calibration_error, calibration_slope_intercept, DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
    load_frozen_encoder_v2, train_on_frozen_encoder_v2, save_checkpoint_v2,
    save_predictions_v2, save_seed_metrics_v2,
    HORIZON, TARGET, SUFFIX, apply_horizon_filter,
)

SEED_DIR = os.path.join(DATA_DIR, f"m2_v2_seed_results{SUFFIX}", "hybrid_grud")
os.makedirs(SEED_DIR, exist_ok=True)

FINAL_SEEDS = [42, 123, 7, 456, 789]
MODEL_NAME = "Hybrid GRU-D v2"


def main():
    assert os.path.exists(CONFIG_JSON), (
        f"{CONFIG_JSON} not found -- run m2_hybrid_head_tuning_v2.py first. "
        f"Refusing to run the confirmatory grid on the untuned/reused config."
    )
    with open(CONFIG_JSON) as f:
        cfg = json.load(f)
    print(f"Using tuned config: {cfg}")
    STATIC_HIDDEN, STATIC_DROPOUT = cfg["static_hidden"], cfg["static_dropout"]
    FUSION_DROPOUT, LR = cfg["fusion_dropout"], cfg["lr"]
    BATCH_SIZE, MAX_EPOCHS, PATIENCE = cfg["batch_size"], cfg["max_epochs"], cfg["patience"]

    print(f"Device: {DEVICE}")
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())

    df_fit = apply_horizon_filter(df_train[df_train["subject_id"].isin(train_ids)])
    df_val = apply_horizon_filter(df_train[df_train["subject_id"].isin(val_ids)])
    df_test = apply_horizon_filter(df_test)
    print(f"target={TARGET} (horizon={HORIZON or 'composite'})  fit={len(df_fit)}, val={len(df_val)}, test={len(df_test)}")

    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_fit, "train_v2", device=DEVICE)
    Xv, Mv, Dv, Pv, yv = prepare_temporal_tensors_v2(df_val, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_test, "test_v2", device=DEVICE)
    y_test_np = yt.cpu().numpy()

    Sf = torch.tensor(prepare_static_features_v2(df_fit), device=DEVICE)
    Sv = torch.tensor(prepare_static_features_v2(df_val), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_test), device=DEVICE)

    seed_metrics = []
    for seed in FINAL_SEEDS:
        encoder = load_frozen_encoder_v2(seed)

        t0 = time.perf_counter()
        head, history = train_on_frozen_encoder_v2(
            encoder, Xf, Mf, Df, Pf, yf.cpu().numpy(), Xv, Mv, Dv, Pv, yv.cpu().numpy(),
            seed=seed, S_fit=Sf, S_val=Sv, static_hidden=STATIC_HIDDEN, static_dropout=STATIC_DROPOUT,
            fusion_dropout=FUSION_DROPOUT, lr=LR,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        train_s = time.perf_counter() - t0

        encoder.eval(); head.eval()
        with torch.no_grad():
            h_test = encoder(Xt, Mt, Dt, Pt)
            test_prob = torch.sigmoid(head(h_test, St)).cpu().numpy().ravel()

        auroc = roc_auc_score(y_test_np, test_prob)
        auprc = average_precision_score(y_test_np, test_prob)
        brier = brier_score_loss(y_test_np, test_prob)
        ece = expected_calibration_error(y_test_np, test_prob)
        slope, intercept = calibration_slope_intercept(y_test_np, test_prob)

        m = {"seed": seed, "epochs_trained": len(history), "auroc": float(auroc),
             "auprc": float(auprc), "brier": float(brier), "ece": float(ece),
             "cal_slope": float(slope), "cal_intercept": float(intercept), "train_seconds": train_s}
        seed_metrics.append(m)
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(m, f, indent=2)

        save_checkpoint_v2(head, "m2_hybrid_grud_head", seed)
        save_predictions_v2(df_test["subject_id"].values, MODEL_NAME, seed, y_test_np, test_prob,
                             stay_id=df_test["stay_id"].values)
        save_seed_metrics_v2(MODEL_NAME, seed, {
            "auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece,
            "cal_slope": slope, "cal_intercept": intercept,
        })
        print(f"seed {seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f} epochs={len(history)} "
              f"train={train_s:.1f}s", flush=True)

    aurocs = [m["auroc"] for m in seed_metrics]
    print(f"\n{MODEL_NAME}, 5-seed: AUROC={np.mean(aurocs):.4f}+/-{np.std(aurocs):.4f}")


if __name__ == "__main__":
    main()
