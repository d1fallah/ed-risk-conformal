"""
M2 -- Static-Only MLP, irregular-series (v2) cohort, PILOT (Phase 3)
==============================================================================
Pilot run (1-2 seeds, not the full 5-seed protocol -- see session status:
user chose "reduce to a 1-2 seed pilot first" over the ~5h full-grid
commit). Fixed config (hidden=32, dropout=0.2, matching GRU-D v2's hidden
size for comparability) rather than the T=2 pipeline's 6-config x 3-seed
tuning grid -- if the pilot motivates the full 5-seed run, the tuning grid
should be added back at that point, not skipped permanently.

Static features are unaffected by the irregular-series rebuild (same 4
Guide-mandated columns: age, sex, triage_acuity, comorbidity_elixhauser_
count), so this reuses add_hybrid_static_features() unchanged; only the
cohort/split (train_v2/test_v2, chronological primary) differs from the
T=2 pipeline's Static-Only MLP.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR = os.path.join(DATA_DIR, "m2_v2_seed_results", "static_only_mlp")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    HYBRID_STATIC_COLS, add_hybrid_static_features,
    _minibatch_early_stop_loop, expected_calibration_error, calibration_slope_intercept, DEVICE,
)
from m2_v2_common import load_split_frame, load_dev_pool_split_v2, save_predictions_v2, save_seed_metrics_v2

HIDDEN, DROPOUT, LR = 32, 0.2, 1e-3
MAX_EPOCHS = 3000  # T=2 pipeline's own diagnosed epoch budget for this 4-input MLP (see m2_static_only_mlp.py)
PATIENCE = 10
FINAL_SEEDS = [42, 123, 7, 456, 789]
MODEL_NAME = "Static-Only MLP v2"


class StaticMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


def main():
    print(f"Device: {DEVICE}")
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())

    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]

    scaler = StandardScaler().fit(df_fit[HYBRID_STATIC_COLS].values)
    Xf = scaler.transform(df_fit[HYBRID_STATIC_COLS].values)
    Xv = scaler.transform(df_val[HYBRID_STATIC_COLS].values)
    Xt = scaler.transform(df_test[HYBRID_STATIC_COLS].values)
    yf = df_fit["outcome_critical"].astype(float).values
    yv = df_val["outcome_critical"].astype(float).values
    yt = df_test["outcome_critical"].astype(float).values

    Xf_t = torch.tensor(Xf, dtype=torch.float32, device=DEVICE)
    Xv_t = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
    Xt_t = torch.tensor(Xt, dtype=torch.float32, device=DEVICE)

    results = []
    for seed in FINAL_SEEDS:
        torch.manual_seed(seed)
        model = StaticMLP(in_dim=len(HYBRID_STATIC_COLS), hidden=HIDDEN, dropout=DROPOUT).to(DEVICE)
        t0 = time.perf_counter()
        history = _minibatch_early_stop_loop(
            param_groups_fn=lambda: model.parameters(),
            forward_fn=lambda ts: model(ts[0]),
            X_fit_tensors=[Xf_t], y_fit=yf, X_val_tensors=[Xv_t], y_val=yv,
            seed=seed, lr=LR, weight_decay=0.0, batch_size=None,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, train_modules=[model],
        )
        train_s = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            test_prob = torch.sigmoid(model(Xt_t)).cpu().numpy().ravel()
        auroc = roc_auc_score(yt, test_prob)
        auprc = average_precision_score(yt, test_prob)
        brier = brier_score_loss(yt, test_prob)
        ece = expected_calibration_error(yt, test_prob)
        slope, intercept = calibration_slope_intercept(yt, test_prob)
        m = {"seed": seed, "epochs_trained": len(history), "auroc": float(auroc),
             "auprc": float(auprc), "brier": float(brier), "ece": float(ece),
             "cal_slope": float(slope), "cal_intercept": float(intercept), "train_seconds": train_s}
        results.append(m)
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(m, f, indent=2)
        save_predictions_v2(df_test["subject_id"].values, MODEL_NAME, seed, yt, test_prob,
                             stay_id=df_test["stay_id"].values)
        save_seed_metrics_v2(MODEL_NAME, seed, {
            "auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece,
            "cal_slope": slope, "cal_intercept": intercept,
        })
        print(f"seed {seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f} epochs={len(history)} "
              f"train={train_s:.1f}s", flush=True)

    aurocs = [r["auroc"] for r in results]
    print(f"\nStatic-Only MLP v2 pilot ({len(results)} seeds): "
          f"AUROC mean={np.mean(aurocs):.4f} std={np.std(aurocs):.4f}")


if __name__ == "__main__":
    main()
