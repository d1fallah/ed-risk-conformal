"""
M2 -- mTAN, irregular-series (v2) cohort, FULL 5-SEED RUN (tuned config)
==============================================================================
Loads the tuned config selected by m2_mtan_tuning_matched_v2.py
(data/m2_v2_mtan_tuned_config_matched.json -- the matched-rigor pass that closes
Deviation 15; the earlier lighter-protocol config is retained for the record), confirms it exists (this script
refuses to silently fall back to defaults -- an untuned mTAN result must
never be reported as if the tuning pass had informed it), and runs the
full 5-seed confirmatory protocol under the SAME batch size/early-
stopping discipline as every other v2 model.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MTAN_SRC = os.path.join(SCRIPT_DIR, "..", "..", "mTAN", "src")
SEED_DIR = os.path.join(DATA_DIR, "m2_v2_seed_results", "mtan")
CONFIG_JSON = os.path.join(DATA_DIR, "m2_v2_mtan_tuned_config_matched.json")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, MTAN_SRC)
from m2_hybrid_common import _minibatch_early_stop_loop, expected_calibration_error, calibration_slope_intercept, DEVICE
from m2_v2_common import load_split_frame, load_dev_pool_split_v2, save_predictions_v2, save_seed_metrics_v2, save_checkpoint_v2
from m2_mtan_v2 import MTANClassifierHead, prepare_mtan_tensors, BATCH_SIZE, MAX_EPOCHS, PATIENCE

FINAL_SEEDS = [42, 123, 7, 456, 789]
MODEL_NAME = "mTAN v2"


def main():
    assert os.path.exists(CONFIG_JSON), (
        f"{CONFIG_JSON} not found -- run m2_mtan_tuning_matched_v2.py first. "
        f"Refusing to run the confirmatory grid on an untuned/default config."
    )
    with open(CONFIG_JSON) as f:
        cfg = json.load(f)
    print(f"Using tuned config: {cfg}")
    assert not cfg.get("smoke"), "config was produced by a --smoke run -- refusing to use it"
    for k in ("nhidden", "embed_time", "num_ref_points", "num_heads", "lr", "weight_decay"):
        assert k in cfg, f"tuned config is missing {k!r}: it must come from m2_mtan_tuning_matched_v2.py (which also tunes lr and weight_decay)"

    print(f"Device: {DEVICE}")
    df_train = load_split_frame("train_v2")
    df_test = load_split_frame("test_v2")
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())

    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]

    Xf, Tf, yf = prepare_mtan_tensors(df_fit, "train_v2", DEVICE)
    Xv, Tv, yv = prepare_mtan_tensors(df_val, "train_v2", DEVICE)
    Xt, Tt, yt = prepare_mtan_tensors(df_test, "test_v2", DEVICE)
    y_test_np = yt.cpu().numpy()

    seed_metrics = []
    for seed in FINAL_SEEDS:
        torch.manual_seed(seed)
        query = torch.linspace(0, 1, cfg["num_ref_points"])
        model = MTANClassifierHead(
            input_dim=7, query=query, nhidden=cfg["nhidden"], embed_time=cfg["embed_time"],
            num_heads=cfg["num_heads"], device=DEVICE,
        ).to(DEVICE)

        t0 = time.perf_counter()
        history = _minibatch_early_stop_loop(
            param_groups_fn=lambda: model.parameters(),
            forward_fn=lambda ts: model(ts[0], ts[1]),
            X_fit_tensors=[Xf, Tf], y_fit=yf.cpu().numpy(),
            X_val_tensors=[Xv, Tv], y_val=yv.cpu().numpy(),
            seed=seed, lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, train_modules=[model],
        )
        train_s = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            test_prob = torch.sigmoid(model(Xt, Tt)).cpu().numpy().ravel()

        auroc = roc_auc_score(y_test_np, test_prob)
        auprc = average_precision_score(y_test_np, test_prob)
        brier = brier_score_loss(y_test_np, test_prob)
        ece = expected_calibration_error(y_test_np, test_prob)
        slope, intercept = calibration_slope_intercept(y_test_np, test_prob)

        m = {"seed": seed, "epochs_trained": len(history), "auroc": float(auroc),
             "auprc": float(auprc), "brier": float(brier), "ece": float(ece),
             "cal_slope": float(slope), "cal_intercept": float(intercept),
             "train_seconds": train_s, "config": cfg}
        seed_metrics.append(m)
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(m, f, indent=2)

        save_checkpoint_v2(model, "m2_mtan", seed)
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
    print("Note: mTAN checkpoints saved (data/checkpoints/m2_mtan_v2_seed<N>.pt) but mTAN is "
          "NOT included in this session's v2 conformal re-run (m2_conformal_v2.py) -- scoped to "
          "the 3 GRU-D-family models, matching the T=2 pipeline's own conformal scope "
          "(m2_hybrid_conformal.py: 'Static-Only MLP and XGBoost-Matched are untouched here'). "
          "Extending conformal reporting to mTAN is a reasonable future addition, not done here.")


if __name__ == "__main__":
    main()
